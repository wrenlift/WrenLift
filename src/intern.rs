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

/// An interned string identifier. Cheap to copy and compare.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
pub struct Interner {
    /// Maps string content → symbol id for dedup lookup.
    map: HashMap<String, SymbolId>,
    /// Indexed storage: `strings[id.0]` is the interned string.
    strings: Vec<String>,
}

impl Interner {
    /// Create a new empty interner.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    /// Create an interner with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            strings: Vec::with_capacity(cap),
        }
    }

    /// Intern a string, returning its `SymbolId`.
    /// If the string was already interned, returns the existing id.
    pub fn intern(&mut self, s: &str) -> SymbolId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = SymbolId(self.strings.len() as u32);
        let owned = s.to_owned();
        self.map.insert(owned.clone(), id);
        self.strings.push(owned);
        id
    }

    /// Resolve a `SymbolId` back to its string.
    #[inline]
    pub fn resolve(&self, id: SymbolId) -> &str {
        &self.strings[id.0 as usize]
    }

    /// Number of interned strings.
    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Is the interner empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
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
