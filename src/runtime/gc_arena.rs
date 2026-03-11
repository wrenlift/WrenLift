/// Arena garbage collector — allocates, never collects.
///
/// All objects are Box-allocated and freed when the ArenaGc is dropped.
/// This is ideal for:
/// - Short-lived scripts where GC pauses are wasted work
/// - Benchmarks where you want to measure execution speed without GC noise
/// - Server request handlers where each request gets a fresh VM
///
/// Trade-off: unbounded memory growth. Suitable only when total allocation
/// is bounded (e.g., a single request, a short script, a benchmark).
use super::gc::GcStats;
use super::gc_trait::GcAllocator;
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Arena GC
// ---------------------------------------------------------------------------

pub struct ArenaGc {
    /// All allocated objects (Box-allocated, freed on drop).
    objects: Vec<*mut ObjHeader>,
    /// String intern table.
    intern_table: HashMap<u64, Vec<*mut ObjString>>,
    pub stats: GcStats,
}

impl Default for ArenaGc {
    fn default() -> Self {
        Self::new()
    }
}

impl ArenaGc {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            intern_table: HashMap::new(),
            stats: GcStats::default(),
        }
    }

    fn alloc_boxed<T>(&mut self, obj: T) -> *mut T {
        let size = std::mem::size_of::<T>();
        let ptr = Box::into_raw(Box::new(obj));
        self.objects.push(ptr as *mut ObjHeader);
        self.stats.total_allocated += size;
        self.stats.objects_allocated += 1;
        let total = self.objects.len();
        if total > self.stats.peak_objects {
            self.stats.peak_objects = total;
        }
        ptr
    }
}

impl Drop for ArenaGc {
    fn drop(&mut self) {
        for &obj in &self.objects {
            unsafe {
                drop_object(obj);
            }
        }
        self.objects.clear();
    }
}

impl GcAllocator for ArenaGc {
    fn alloc_string(&mut self, s: String) -> *mut ObjString {
        self.alloc_boxed(ObjString::new(s))
    }

    fn alloc_list(&mut self) -> *mut ObjList {
        self.alloc_boxed(ObjList::new())
    }

    fn alloc_map(&mut self) -> *mut ObjMap {
        self.alloc_boxed(ObjMap::new())
    }

    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        self.alloc_boxed(ObjRange::new(from, to, inclusive))
    }

    fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        self.alloc_boxed(ObjFn::new(name, arity, upvalue_count, fn_id))
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        let uv_count = if function.is_null() {
            0
        } else {
            unsafe { (*function).upvalue_count as usize }
        };
        self.alloc_boxed(ObjClosure::new(function, uv_count))
    }

    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        self.alloc_boxed(ObjUpvalue::new(location))
    }

    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        self.alloc_boxed(ObjFiber::new())
    }

    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        self.alloc_boxed(ObjClass::new(name, superclass))
    }

    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        self.alloc_boxed(ObjInstance::new(class))
    }

    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        self.alloc_boxed(ObjForeign::new(data))
    }

    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        self.alloc_boxed(ObjModule::new(name))
    }

    fn intern_string(&mut self, s: String) -> *mut ObjString {
        let hash = fnv1a_hash_bytes(s.as_bytes());
        if let Some(ptrs) = self.intern_table.get(&hash) {
            for &ptr in ptrs {
                if unsafe { (*ptr).value == s } {
                    return ptr;
                }
            }
        }
        let ptr = self.alloc_string(s);
        self.intern_table.entry(hash).or_default().push(ptr);
        ptr
    }

    fn write_barrier(&mut self, _source: *mut ObjHeader, _value: Value) {
        // No-op: arena GC doesn't collect, so no barriers needed.
    }

    fn collect(&mut self, _roots: &mut [Value]) {
        // No-op: arena GC never collects.
    }

    fn should_collect(&self) -> bool {
        false // Never collect.
    }

    fn stats(&self) -> &GcStats {
        &self.stats
    }
}

/// Reconstruct Box and drop.
unsafe fn drop_object(header: *mut ObjHeader) {
    match (*header).obj_type {
        ObjType::String => {
            let _ = Box::from_raw(header as *mut ObjString);
        }
        ObjType::List => {
            let _ = Box::from_raw(header as *mut ObjList);
        }
        ObjType::Map => {
            let _ = Box::from_raw(header as *mut ObjMap);
        }
        ObjType::Range => {
            let _ = Box::from_raw(header as *mut ObjRange);
        }
        ObjType::Fn => {
            let _ = Box::from_raw(header as *mut ObjFn);
        }
        ObjType::Closure => {
            let _ = Box::from_raw(header as *mut ObjClosure);
        }
        ObjType::Upvalue => {
            let _ = Box::from_raw(header as *mut ObjUpvalue);
        }
        ObjType::Fiber => {
            let _ = Box::from_raw(header as *mut ObjFiber);
        }
        ObjType::Class => {
            let _ = Box::from_raw(header as *mut ObjClass);
        }
        ObjType::Instance => {
            let _ = Box::from_raw(header as *mut ObjInstance);
        }
        ObjType::Foreign => {
            let _ = Box::from_raw(header as *mut ObjForeign);
        }
        ObjType::Module => {
            let _ = Box::from_raw(header as *mut ObjModule);
        }
    }
}

fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc_and_drop() {
        let mut gc = ArenaGc::new();
        gc.alloc_string("hello".into());
        gc.alloc_string("world".into());
        gc.alloc_list();
        assert_eq!(gc.stats.objects_allocated, 3);
        // drop frees everything
    }

    #[test]
    fn test_arena_intern() {
        let mut gc = ArenaGc::new();
        let a = gc.intern_string("hello".into());
        let b = gc.intern_string("hello".into());
        assert_eq!(a, b);
        assert_eq!(gc.stats.objects_allocated, 1);
    }

    #[test]
    fn test_arena_collect_is_noop() {
        let mut gc = ArenaGc::new();
        gc.alloc_string("keep".into());
        gc.collect(&mut []);
        assert!(!gc.should_collect());
        assert_eq!(gc.stats.objects_allocated, 1);
    }
}
