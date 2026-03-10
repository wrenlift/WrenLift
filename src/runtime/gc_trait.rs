/// Trait abstracting garbage collector implementations.
///
/// The VM is generic over `GcAllocator`, allowing different GC strategies:
/// - `Gc` (default): generational nursery + old gen mark-sweep
/// - Future: Immix mark-region, concurrent GC, arena GC, etc.
///
/// Using a trait (with static dispatch via generics) gives zero-cost
/// abstraction — no vtable overhead on the allocation hot path.
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;

use super::gc::GcStats;

/// Core GC interface. Every GC implementation must provide these operations.
pub trait GcAllocator {
    // -- Typed allocation ---------------------------------------------------

    fn alloc_string(&mut self, s: String) -> *mut ObjString;
    fn alloc_list(&mut self) -> *mut ObjList;
    fn alloc_map(&mut self) -> *mut ObjMap;
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange;
    fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn;
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure;
    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue;
    fn alloc_fiber(&mut self) -> *mut ObjFiber;
    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass;
    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance;
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign;
    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule;

    // -- String interning ---------------------------------------------------

    fn intern_string(&mut self, s: String) -> *mut ObjString;

    // -- Write barrier (generational/incremental GCs) -----------------------

    /// Notify the GC that `source` now references `value`.
    /// No-op for non-generational GCs.
    fn write_barrier(&mut self, source: *mut ObjHeader, value: Value);

    // -- Collection ---------------------------------------------------------

    /// Run a GC cycle. `roots` are mutable because the GC may relocate objects.
    fn collect(&mut self, roots: &mut [Value]);

    /// Should the VM trigger a GC safepoint?
    fn should_collect(&self) -> bool;

    // -- Statistics ---------------------------------------------------------

    fn stats(&self) -> &GcStats;
}
