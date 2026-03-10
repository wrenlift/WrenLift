/// Trait abstracting garbage collector implementations.
///
/// The VM uses `GcImpl` (enum dispatch) so different GC strategies can be
/// selected at runtime without infecting the entire codebase with generics.
///
/// Available implementations:
/// - `Gc` (default): generational nursery + old gen mark-sweep
/// - `ArenaGc`: allocate-only, free on drop (short-lived scripts, benchmarks)
/// - `SemispaceGc`: copying collector with excellent locality
use super::gc::GcStats;
use super::gc_arena::ArenaGc;
use super::gc_marksweep::MarkSweepGc;
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;

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

// ---------------------------------------------------------------------------
// Enum-based dispatch: runtime GC selection without generic pollution
// ---------------------------------------------------------------------------

/// Runtime-selectable GC implementation.
///
/// Uses enum dispatch instead of trait objects or generics to avoid:
/// - Generic parameter pollution across 30+ files
/// - Vtable indirection on every allocation
///
/// The match dispatch is effectively free: one predictable branch per call,
/// and the branch predictor learns the pattern after the first few calls.
pub enum GcImpl {
    /// Generational: nursery bump alloc + old gen mark-sweep.
    Generational(super::gc::Gc),
    /// Arena: allocate-only, free everything on drop. For short-lived scripts.
    Arena(ArenaGc),
    /// Mark-sweep: simple non-generational stop-the-world collector.
    MarkSweep(MarkSweepGc),
}

/// Which GC strategy to use. Selectable via CLI `--gc` flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcStrategy {
    /// Generational nursery + old gen mark-sweep (default).
    Generational,
    /// Allocate-only, free on drop. For benchmarks / short-lived scripts.
    Arena,
    /// Simple non-generational mark-sweep.
    MarkSweep,
}

impl Default for GcStrategy {
    fn default() -> Self {
        GcStrategy::Generational
    }
}

/// Macro to dispatch a method call to the inner GC implementation.
macro_rules! gc_dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            GcImpl::Generational(gc) => gc.$method($($arg),*),
            GcImpl::Arena(gc) => gc.$method($($arg),*),
            GcImpl::MarkSweep(gc) => gc.$method($($arg),*),
        }
    };
}

impl GcImpl {
    /// Create a new GC instance for the given strategy.
    pub fn new(strategy: GcStrategy) -> Self {
        match strategy {
            GcStrategy::Generational => GcImpl::Generational(super::gc::Gc::new()),
            GcStrategy::Arena => GcImpl::Arena(ArenaGc::new()),
            GcStrategy::MarkSweep => GcImpl::MarkSweep(MarkSweepGc::new()),
        }
    }

    // -- Direct methods (bypass trait dispatch for zero-overhead enum dispatch) --

    #[inline(always)]
    pub fn alloc_string(&mut self, s: String) -> *mut ObjString {
        gc_dispatch!(self, alloc_string, s)
    }
    #[inline(always)]
    pub fn alloc_list(&mut self) -> *mut ObjList {
        gc_dispatch!(self, alloc_list)
    }
    #[inline(always)]
    pub fn alloc_map(&mut self) -> *mut ObjMap {
        gc_dispatch!(self, alloc_map)
    }
    #[inline(always)]
    pub fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        gc_dispatch!(self, alloc_range, from, to, inclusive)
    }
    #[inline(always)]
    pub fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        gc_dispatch!(self, alloc_fn, name, arity, upvalue_count, fn_id)
    }
    #[inline(always)]
    pub fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        gc_dispatch!(self, alloc_closure, function)
    }
    #[inline(always)]
    pub fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        gc_dispatch!(self, alloc_upvalue, location)
    }
    #[inline(always)]
    pub fn alloc_fiber(&mut self) -> *mut ObjFiber {
        gc_dispatch!(self, alloc_fiber)
    }
    #[inline(always)]
    pub fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        gc_dispatch!(self, alloc_class, name, superclass)
    }
    #[inline(always)]
    pub fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        gc_dispatch!(self, alloc_instance, class)
    }
    #[inline(always)]
    pub fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        gc_dispatch!(self, alloc_foreign, data)
    }
    #[inline(always)]
    pub fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        gc_dispatch!(self, alloc_module, name)
    }
    #[inline(always)]
    pub fn intern_string(&mut self, s: String) -> *mut ObjString {
        gc_dispatch!(self, intern_string, s)
    }
    #[inline(always)]
    pub fn write_barrier(&mut self, source: *mut ObjHeader, value: Value) {
        gc_dispatch!(self, write_barrier, source, value)
    }
    #[inline(always)]
    pub fn collect(&mut self, roots: &mut [Value]) {
        gc_dispatch!(self, collect, roots)
    }
    #[inline(always)]
    pub fn should_collect(&self) -> bool {
        gc_dispatch!(self, should_collect)
    }
    #[inline(always)]
    pub fn stats(&self) -> &GcStats {
        gc_dispatch!(self, stats)
    }
}

impl GcAllocator for GcImpl {
    #[inline(always)]
    fn alloc_string(&mut self, s: String) -> *mut ObjString {
        gc_dispatch!(self, alloc_string, s)
    }
    #[inline(always)]
    fn alloc_list(&mut self) -> *mut ObjList {
        gc_dispatch!(self, alloc_list)
    }
    #[inline(always)]
    fn alloc_map(&mut self) -> *mut ObjMap {
        gc_dispatch!(self, alloc_map)
    }
    #[inline(always)]
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        gc_dispatch!(self, alloc_range, from, to, inclusive)
    }
    #[inline(always)]
    fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        gc_dispatch!(self, alloc_fn, name, arity, upvalue_count, fn_id)
    }
    #[inline(always)]
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        gc_dispatch!(self, alloc_closure, function)
    }
    #[inline(always)]
    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        gc_dispatch!(self, alloc_upvalue, location)
    }
    #[inline(always)]
    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        gc_dispatch!(self, alloc_fiber)
    }
    #[inline(always)]
    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        gc_dispatch!(self, alloc_class, name, superclass)
    }
    #[inline(always)]
    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        gc_dispatch!(self, alloc_instance, class)
    }
    #[inline(always)]
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        gc_dispatch!(self, alloc_foreign, data)
    }
    #[inline(always)]
    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        gc_dispatch!(self, alloc_module, name)
    }
    #[inline(always)]
    fn intern_string(&mut self, s: String) -> *mut ObjString {
        gc_dispatch!(self, intern_string, s)
    }
    #[inline(always)]
    fn write_barrier(&mut self, source: *mut ObjHeader, value: Value) {
        gc_dispatch!(self, write_barrier, source, value)
    }
    #[inline(always)]
    fn collect(&mut self, roots: &mut [Value]) {
        gc_dispatch!(self, collect, roots)
    }
    #[inline(always)]
    fn should_collect(&self) -> bool {
        gc_dispatch!(self, should_collect)
    }
    #[inline(always)]
    fn stats(&self) -> &GcStats {
        gc_dispatch!(self, stats)
    }
}
