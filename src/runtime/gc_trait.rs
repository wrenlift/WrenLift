//! The collector behind the VM. `GcAllocator` is the interface the
//! VM programs against; `GcImpl` is the collector a build runs with,
//! Immix geometry with non-moving mark-sweep and a conservative scan
//! of native stacks. Nothing moves, so there is no write barrier, no
//! stack map and no pointer writeback.
use super::gc::GcStats;
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;

/// The VM's collector.
pub type GcImpl = super::gc_immix::ImmixGc;

/// The allocation interface the VM programs against; a host that
/// brings its own collector implements it.
pub trait GcAllocator {
    // -- Typed allocation ---------------------------------------------------

    fn alloc_string(&mut self, s: String) -> *mut ObjString;
    fn alloc_list(&mut self) -> *mut ObjList;
    /// A list with room for `cap` elements before it grows; zero
    /// leaves the room to the allocator.
    fn alloc_list_sized(&mut self, cap: usize) -> *mut ObjList {
        let _ = cap;
        self.alloc_list()
    }
    fn alloc_map(&mut self) -> *mut ObjMap;
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange;
    fn alloc_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> *mut ObjTypedArray;
    fn alloc_simd(&mut self, kind: SimdKind, lanes: [u32; 4]) -> *mut ObjSimd;
    fn alloc_fn(&mut self, name: SymbolId, arity: u8, upvalue_count: u16, fn_id: u32)
    -> *mut ObjFn;
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure;
    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue;
    fn alloc_fiber(&mut self) -> *mut ObjFiber;
    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass;
    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance;
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign;
    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule;

    // -- String interning ---------------------------------------------------

    fn intern_string(&mut self, s: String) -> *mut ObjString;

    // -- Collection ---------------------------------------------------------

    /// Run a collection with `roots` as the roots beyond the
    /// collector's own. Objects never move, so roots are read, not
    /// written.
    fn collect(&mut self, roots: &[Value]);

    /// Should the VM trigger a GC safepoint?
    fn should_collect(&self) -> bool;

    // -- Statistics ---------------------------------------------------------

    /// A snapshot of the counters.
    fn stats(&self) -> GcStats;
}
