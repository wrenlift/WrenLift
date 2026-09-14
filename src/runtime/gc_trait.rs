/// Trait abstracting garbage collector implementations.
///
/// The VM uses `GcImpl` (enum dispatch) so different GC strategies can be
/// selected at runtime without infecting the entire codebase with generics.
///
/// Available implementations:
/// - `ImmixGc` (default): block/line bump allocation, non-moving
///   mark-sweep, conservative native stack scanning
/// - `Gc`: generational nursery + old gen mark-sweep
/// - `ArenaGc`: allocate-only, free on drop (short-lived scripts, benchmarks)
/// - `MarkSweepGc`: simple non-generational mark-sweep
use super::gc::GcStats;
use super::gc_arena::ArenaGc;
use super::gc_immix::ImmixGc;
use super::gc_marksweep::MarkSweepGc;
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;

/// Core GC interface. Every GC implementation must provide these operations.
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

    /// A snapshot of the counters.
    fn stats(&self) -> GcStats;
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
    /// Immix geometry: block/line bump allocation, non-moving mark-sweep.
    Immix(ImmixGc),
}

/// Which GC strategy to use. Selectable via CLI `--gc` flag or the
/// `WLIFT_GC` env var (generational | arena | marksweep | immix).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GcStrategy {
    /// Generational nursery + old gen mark-sweep.
    Generational,
    /// Allocate-only, free on drop. For benchmarks / short-lived scripts.
    Arena,
    /// Simple non-generational mark-sweep.
    MarkSweep,
    /// Block/line bump allocation with non-moving mark-sweep and
    /// conservative native stack scanning (default).
    #[default]
    Immix,
}

impl GcStrategy {
    /// Strategy named by `WLIFT_GC`, if set and recognised. Safe to
    /// run with any value; unknown names fall back to the default.
    pub fn from_env() -> Option<Self> {
        let v = std::env::var("WLIFT_GC").ok()?;
        match v.trim().to_ascii_lowercase().as_str() {
            "generational" => Some(GcStrategy::Generational),
            "arena" => Some(GcStrategy::Arena),
            "marksweep" | "mark-sweep" => Some(GcStrategy::MarkSweep),
            "immix" => Some(GcStrategy::Immix),
            _ => None,
        }
    }
}

/// Macro to dispatch a method call to the inner GC implementation.
macro_rules! gc_dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            GcImpl::Generational(gc) => gc.$method($($arg),*),
            GcImpl::Arena(gc) => gc.$method($($arg),*),
            GcImpl::MarkSweep(gc) => gc.$method($($arg),*),
            GcImpl::Immix(gc) => gc.$method($($arg),*),
        }
    };
}

/// VMs created with a collector that needs write barriers. Compiled
/// code emits barriers while this is non-zero; a process that only
/// ever runs Immix VMs leaves it at zero and skips them. Read on the
/// compile broker thread, so it is process-wide.
static BARRIER_VMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// True when JIT code compiled now must carry write barriers.
pub fn jit_needs_write_barriers() -> bool {
    BARRIER_VMS.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// True when JIT code compiled now must record its live values for
/// the collector: Immix scans native frames conservatively and reads
/// no stack map.
pub fn jit_needs_stack_maps() -> bool {
    jit_needs_write_barriers()
}

impl GcImpl {
    /// Create a new GC instance for the given strategy.
    pub fn new(strategy: GcStrategy) -> Self {
        if strategy != GcStrategy::Immix {
            BARRIER_VMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        match strategy {
            GcStrategy::Generational => GcImpl::Generational(super::gc::Gc::new()),
            GcStrategy::Arena => GcImpl::Arena(ArenaGc::new()),
            GcStrategy::MarkSweep => GcImpl::MarkSweep(MarkSweepGc::new()),
            GcStrategy::Immix => GcImpl::Immix(ImmixGc::new()),
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
    pub fn alloc_list_sized(&mut self, cap: usize) -> *mut ObjList {
        gc_dispatch!(self, alloc_list_sized, cap)
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
    pub fn alloc_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> *mut ObjTypedArray {
        gc_dispatch!(self, alloc_typed_array, count, kind)
    }
    #[inline(always)]
    pub fn alloc_simd(&mut self, kind: SimdKind, lanes: [u32; 4]) -> *mut ObjSimd {
        gc_dispatch!(self, alloc_simd, kind, lanes)
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
    /// Charge off-heap bytes against the generational GC's pressure
    /// counter. No-op on arena / mark-sweep backends — they don't
    /// have a separate nursery threshold to push against. Used by
    /// the `Fiber.new` foreign method so krio mmap stacks actually
    /// drive `should_collect`.
    #[inline(always)]
    pub fn track_external(&mut self, bytes: usize) {
        match self {
            GcImpl::Generational(gc) => gc.track_external(bytes),
            GcImpl::Immix(gc) => gc.track_external(bytes),
            _ => {}
        }
    }
    #[inline(always)]
    pub fn stats(&self) -> GcStats {
        gc_dispatch!(self, stats)
    }

    /// Pre-collect sanity check on the remembered set. No-op for
    /// Arena / MarkSweep (those don't have one). Generational runs
    /// both directions — missed barriers + stale sources. Release-safe
    /// because every call site is gated on `WLIFT_VALIDATE_BARRIERS`.
    #[inline(always)]
    pub fn validate_write_barriers(&self) {
        if let GcImpl::Generational(gc) = self {
            gc.validate_write_barriers();
        }
    }

    /// Diagnostic: detect a `wren_write_barrier` source pointer that
    /// reads `GEN_OLD` but isn't in the live old_objects list — a
    /// stale-source bug from AOT codegen holding a freed pointer.
    /// No-op for Arena / MarkSweep (no concept of sweep-after-promotion).
    #[inline(always)]
    pub fn is_stale_old_source(&self, header: *mut ObjHeader) -> bool {
        match self {
            GcImpl::Generational(gc) => gc.is_stale_old_source(header),
            _ => false,
        }
    }

    /// Whether `ptr` lies within the generational nursery arena.
    /// Always `false` for Arena / MarkSweep — those allocators don't
    /// have a separate nursery. Used by the conservative FP-chain
    /// forwarding fixup to gate writes on the source being a nursery
    /// object (an old-gen object's `gc_mark==FORWARDED` would only
    /// fire transiently mid-collection and the `next` field there
    /// is the intrusive sweep list, not a forwarding pointer).
    #[inline(always)]
    pub fn nursery_contains(&self, ptr: *const u8) -> bool {
        match self {
            GcImpl::Generational(gc) => gc.nursery_contains(ptr),
            _ => false,
        }
    }

    /// Whether `ptr` lies inside the live old-gen arena (any chunk).
    /// Always `false` for Arena / MarkSweep. Used by the same
    /// FP-chain forwarding fixup to validate the *target* of a
    /// FORWARDED redirect before dereferencing it.
    #[inline(always)]
    pub fn old_arena_contains(&self, ptr: *const u8) -> bool {
        match self {
            GcImpl::Generational(gc) => gc.old_arena_contains(ptr),
            _ => false,
        }
    }

    /// Iterate every currently-allocated `ObjFiber` in the heap.
    ///
    /// Used by the krio-fiber AOT integration's GC Pass 3 to find
    /// every fiber whose stack might hold Wren-Value roots. Going
    /// through the GC's own object list (rather than a separate
    /// side table) avoids dangling-pointer issues: every fiber the
    /// closure sees is, by definition, currently allocated.
    ///
    /// The callback runs synchronously while we hold a `&self`
    /// borrow on the GC, so the closure mustn't trigger an
    /// allocation or another GC pass.
    /// True when the collector scans native stacks conservatively and
    /// never moves objects.
    #[inline(always)]
    pub fn is_immix(&self) -> bool {
        matches!(self, GcImpl::Immix(_))
    }

    /// The Immix bump region compiled code may allocate from; 0 for
    /// any other collector or a host-provided heap.
    pub fn bump_region_ptr(&self) -> usize {
        match self {
            GcImpl::Immix(gc) => gc.bump_region_ptr(),
            _ => 0,
        }
    }

    /// The program is about to run on more than one thread: compiled
    /// code stops bumping the heap's own region. Only Immix serves
    /// several threads; the other collectors ignore this.
    pub fn set_multithreaded(&mut self) {
        if let GcImpl::Immix(gc) = self {
            gc.set_multithreaded();
        }
    }

    /// Whether address-keyed caches (method cache, inline caches) must
    /// be dropped after the last collection. Collectors that cannot
    /// say answer yes.
    pub fn take_freed_code_objects(&mut self) -> bool {
        match self {
            GcImpl::Immix(gc) => gc.take_freed_code_objects(),
            _ => true,
        }
    }

    pub fn for_each_fiber<F: FnMut(*mut super::object::ObjFiber)>(&self, f: F) {
        gc_dispatch!(self, for_each_fiber, f)
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
    fn alloc_list_sized(&mut self, cap: usize) -> *mut ObjList {
        gc_dispatch!(self, alloc_list_sized, cap)
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
    fn alloc_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> *mut ObjTypedArray {
        gc_dispatch!(self, alloc_typed_array, count, kind)
    }
    #[inline(always)]
    fn alloc_simd(&mut self, kind: SimdKind, lanes: [u32; 4]) -> *mut ObjSimd {
        gc_dispatch!(self, alloc_simd, kind, lanes)
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
    fn stats(&self) -> GcStats {
        gc_dispatch!(self, stats)
    }
}
