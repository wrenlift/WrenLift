/// A string interner backed by a contiguous arena.
///
/// All identifier strings, field names, method names, and literals are stored
/// once and referenced by a `SymbolId` (a thin `u32` index). This gives:
///
/// - **Deduplication**: identical strings share one allocation
/// - **O(1) comparison**: compare `u32` indices, not string bytes
/// - **Cache-friendly**: all strings packed in one allocation
///
/// The interner is append-only — strings are never removed during compilation.
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard};

/// An interned string identifier. Cheap to copy and compare.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct SymbolId(u32);

impl SymbolId {
    /// Get the raw index.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }

    /// Create from a raw index (for tests and internal use).
    #[inline]
    pub fn from_raw(index: u32) -> Self {
        Self(index)
    }
}

impl std::fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SymbolId({})", self.0)
    }
}

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.0)
    }
}

/// Append-only string interner.
///
/// Several threads of a running program read it while the parser or
/// a native adds a name: `resolve` is lock-free over storage that
/// never moves, and the name map is behind a lock only `intern` and
/// `lookup` take.
pub struct Interner {
    /// Maps string content → symbol id for dedup lookup.
    map: RwLock<HashMap<String, SymbolId>>,
    /// Indexed storage: `strings.get(id.0)` is the interned string.
    strings: Strings,
}

const CHUNK: usize = 1024;
const MAX_CHUNKS: usize = 4096;

/// Strings in fixed chunks reached through a fixed table, so a slot
/// once published never moves. One writer at a time (the map lock),
/// any number of readers.
struct Strings {
    chunks: Box<[AtomicPtr<[MaybeUninit<String>; CHUNK]>]>,
    len: AtomicUsize,
}

impl Strings {
    fn new() -> Strings {
        Strings {
            chunks: (0..MAX_CHUNKS)
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            len: AtomicUsize::new(0),
        }
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    fn get(&self, i: usize) -> &str {
        assert!(i < self.len(), "symbol {i} is not interned");
        let chunk = self.chunks[i / CHUNK].load(Ordering::Acquire);
        // SAFETY: a slot below `len` was written before `len` was
        // published, and never moves or changes.
        unsafe { (*chunk)[i % CHUNK].assume_init_ref().as_str() }
    }

    /// Append `s`; the caller holds the map's write lock.
    fn push(&self, s: String) -> usize {
        let i = self.len.load(Ordering::Relaxed);
        assert!(i / CHUNK < MAX_CHUNKS, "interner is full");
        let slot = &self.chunks[i / CHUNK];
        let mut chunk = slot.load(Ordering::Acquire);
        if chunk.is_null() {
            let fresh: Box<[MaybeUninit<String>; CHUNK]> =
                Box::new(std::array::from_fn(|_| MaybeUninit::uninit()));
            chunk = Box::into_raw(fresh);
            slot.store(chunk, Ordering::Release);
        }
        // SAFETY: the slot is past `len`, so no reader touches it.
        unsafe { (*chunk)[i % CHUNK].write(s) };
        self.len.store(i + 1, Ordering::Release);
        i
    }
}

impl Drop for Strings {
    fn drop(&mut self) {
        let len = self.len();
        for (c, slot) in self.chunks.iter().enumerate() {
            let chunk = slot.load(Ordering::Acquire);
            if chunk.is_null() {
                break;
            }
            let filled = len.saturating_sub(c * CHUNK).min(CHUNK);
            unsafe {
                for i in 0..filled {
                    (*chunk)[i].assume_init_drop();
                }
                drop(Box::from_raw(chunk));
            }
        }
    }
}

// SAFETY: readers see only published slots; writes happen under the
// map lock.
unsafe impl Send for Strings {}
unsafe impl Sync for Strings {}

impl Interner {
    /// Create a new empty interner.
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            strings: Strings::new(),
        }
    }

    /// Create an interner with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(cap)),
            strings: Strings::new(),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, HashMap<String, SymbolId>> {
        self.map.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Intern a string, returning its `SymbolId`.
    /// If the string was already interned, returns the existing id.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        self.intern_shared(s)
    }

    /// `intern` from a shared reference: several threads may call it.
    pub fn intern_shared(&self, s: &str) -> SymbolId {
        if let Some(&id) = self.read().get(s) {
            return id;
        }
        let mut map = self.map.write().unwrap_or_else(|e| e.into_inner());
        if let Some(&id) = map.get(s) {
            return id;
        }
        let owned = s.to_owned();
        let id = SymbolId(self.strings.push(owned.clone()) as u32);
        map.insert(owned, id);
        id
    }

    /// Look up a string without interning it. Returns `None` if not found.
    pub fn lookup(&self, s: &str) -> Option<SymbolId> {
        self.read().get(s).copied()
    }

    /// Resolve a `SymbolId` back to its string.
    #[inline]
    pub fn resolve(&self, id: SymbolId) -> &str {
        self.strings.get(id.0 as usize)
    }

    /// Number of interned strings.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Is the interner empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.len() == 0
    }

    /// Every interned string, in id order.
    pub fn strings(&self) -> Vec<String> {
        (0..self.len())
            .map(|i| self.strings.get(i).to_owned())
            .collect()
    }
}

impl Clone for Interner {
    fn clone(&self) -> Self {
        let out = Interner::with_capacity(self.len());
        for i in 0..self.len() {
            out.intern_shared(self.strings.get(i));
        }
        out
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

/// The wire shape of an interner: the same two fields the derived
/// form wrote, so existing artifacts still load.
#[derive(serde::Serialize, serde::Deserialize)]
struct InternerWire {
    map: HashMap<String, SymbolId>,
    strings: Vec<String>,
}

impl serde::Serialize for Interner {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let strings = self.strings();
        let map = strings
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), SymbolId(i as u32)))
            .collect();
        InternerWire { map, strings }.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Interner {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = InternerWire::deserialize(deserializer)?;
        let out = Interner::with_capacity(wire.strings.len());
        for s in wire.strings {
            out.strings.push(s.clone());
            out.map
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(s, SymbolId((out.len() - 1) as u32));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_new_string() {
        let mut interner = Interner::new();
        let id = interner.intern("hello");
        assert_eq!(interner.resolve(id), "hello");
    }

    #[test]
    fn test_intern_dedup() {
        let mut interner = Interner::new();
        let a = interner.intern("foo");
        let b = interner.intern("foo");
        assert_eq!(a, b);
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn test_intern_different_strings() {
        let mut interner = Interner::new();
        let a = interner.intern("foo");
        let b = interner.intern("bar");
        assert_ne!(a, b);
        assert_eq!(interner.len(), 2);
        assert_eq!(interner.resolve(a), "foo");
        assert_eq!(interner.resolve(b), "bar");
    }

    #[test]
    fn test_intern_empty_string() {
        let mut interner = Interner::new();
        let id = interner.intern("");
        assert_eq!(interner.resolve(id), "");
    }

    #[test]
    fn test_intern_ordering() {
        let mut interner = Interner::new();
        let a = interner.intern("alpha");
        let b = interner.intern("beta");
        let c = interner.intern("gamma");
        assert!(a.index() < b.index());
        assert!(b.index() < c.index());
    }

    #[test]
    fn test_symbol_id_copy_eq() {
        let mut interner = Interner::new();
        let id = interner.intern("test");
        let copy = id;
        assert_eq!(id, copy);
        assert_eq!(id.index(), 0);
    }

    #[test]
    fn test_symbol_id_hash() {
        use std::collections::HashSet;
        let mut interner = Interner::new();
        let a = interner.intern("a");
        let b = interner.intern("b");

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(a); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_intern_many() {
        let mut interner = Interner::with_capacity(1000);
        for i in 0..1000 {
            let s = format!("sym_{}", i);
            let id = interner.intern(&s);
            assert_eq!(interner.resolve(id), s);
        }
        assert_eq!(interner.len(), 1000);

        // Re-intern all — should return same ids
        for i in 0..1000 {
            let s = format!("sym_{}", i);
            let id = interner.intern(&s);
            assert_eq!(id.index(), i as u32);
        }
        assert_eq!(interner.len(), 1000); // no growth
    }

    #[test]
    fn test_intern_debug_display() {
        let id = SymbolId(42);
        assert_eq!(format!("{:?}", id), "SymbolId(42)");
        assert_eq!(format!("{}", id), "$42");
    }
}
