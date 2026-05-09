//! Wasm-side tier-up broker.
//!
//! `tier.rs` (the host build) wraps `beadie`, which spawns a
//! background promoter thread to drive tier-up. Neither beadie
//! nor `std::thread::spawn` works on `wasm32-unknown-unknown`,
//! so the wasm path runs the broker inline.
//!
//! ## Tier-up to wasm
//!
//! Hot Wren methods get lowered to wasm via
//! `codegen::wasm::emit_mir`, instantiated by the JS host
//! (`wlift_wasm` crate, `tier_up.rs`), and dispatched through a
//! per-function slot table — same shape as the native JIT path,
//! different backend.
//!
//! The runtime crate stays wasm-bindgen-agnostic by exposing
//! **callback hooks**:
//!
//!   * [`set_compile_callback`] — wlift_wasm registers a
//!     function that takes a `&MirFunction` and returns
//!     `Option<u32>` (a slot index in the JIT instance table,
//!     or `None` if compilation failed).
//!
//!   * [`set_dispatch_callback`] — wlift_wasm registers a
//!     function that takes `(slot, fn_export_name, args)` and
//!     returns the i64 result of invoking the slot's function.
//!
//! Plain Rust function pointers — no `wasm-bindgen` types
//! crossing into the runtime crate. The wasm-bindgen-side glue
//! lives in `wlift_wasm::tier_up`.
//!
//! Most of the `TierManager` methods stay no-ops because wasm
//! tier-up tracking happens via `engine.wasm_jit_slots` (parallel
//! to the host's `jit_code`), not through beadie's per-bead
//! state. The `TierManager` shape is preserved purely so the
//! engine compiles cleanly under either build.

use std::sync::OnceLock;

use crate::mir::MirFunction;
use crate::runtime::engine::FuncId;

// ---------------------------------------------------------------------------
// Callback hooks
// ---------------------------------------------------------------------------
//
// `OnceLock` lets the host register exactly once at module init.
// `Send + Sync` is required by the lock; our function pointers
// are zero-state so it's automatic.

/// Compile a MIR function to wasm bytecode + instantiate it
/// + return the slot index in the host's JIT instance table.
///
/// `None` signals compile / instantiate failure (the dispatch
/// path then falls through to the BC interpreter as if no JIT
/// had been registered).
pub type CompileCallback = fn(&MirFunction) -> Option<u32>;

/// Invoke a previously-compiled JIT slot with the given args.
/// `args` is the NaN-boxed `u64` representation of each
/// argument. `fn_export_name` is the wasm export name the
/// emitted module uses (`emit_mir` formats it as
/// `fn_<symbol_index>`).
pub type DispatchCallback = fn(slot: u32, fn_export_name: &str, args: &[u64]) -> u64;

static COMPILE_CALLBACK: OnceLock<CompileCallback> = OnceLock::new();
static DISPATCH_CALLBACK: OnceLock<DispatchCallback> = OnceLock::new();

/// Register the host-provided compile hook. Called once from
/// `wlift_wasm::tier_up` at module init. Subsequent calls are
/// silently dropped.
pub fn set_compile_callback(cb: CompileCallback) {
    let _ = COMPILE_CALLBACK.set(cb);
}

/// Register the host-provided dispatch hook. See [`set_compile_callback`].
pub fn set_dispatch_callback(cb: DispatchCallback) {
    let _ = DISPATCH_CALLBACK.set(cb);
}

/// Compile `mir` via the registered host callback. Returns
/// `None` if no host has registered yet (e.g. wasi smoke
/// binary) or if compilation failed.
pub fn try_compile(mir: &MirFunction) -> Option<u32> {
    COMPILE_CALLBACK.get().and_then(|cb| cb(mir))
}

/// Dispatch through a previously-compiled slot. Caller must
/// have registered a dispatch callback first (verified by the
/// debug assertion); production paths only reach here after
/// `try_compile` succeeded, which implies registration.
pub fn dispatch(slot: u32, fn_export_name: &str, args: &[u64]) -> u64 {
    DISPATCH_CALLBACK
        .get()
        .map(|cb| cb(slot, fn_export_name, args))
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Thread-local current VM
// ---------------------------------------------------------------------------
//
// JIT'd wasm functions need to call back into the runtime —
// `wren_call` for closure dispatch, `wren_make_list` for GC
// allocation, etc. — but they don't have a VM pointer in scope.
// We can't pass it through every wasm import (each helper
// would need an extra arg, ballooning the wasm size), and we
// can't store it on the JIT module's memory because that's
// instance-scoped.
//
// On wasm32 there's only one thread. A `static` cell works as
// a thread-local: `vm.interpret` and `vm_interp::run_fiber`
// set it on entry, restore the prior value on exit. Re-entrant
// `run_fiber` calls (e.g. the scheduler resuming a parked
// fiber) save/restore the cell so nested calls all see the
// right VM.
//
// `current_vm()` returns a raw pointer rather than a reference
// because callers operate inside FFI boundaries where reference
// lifetimes can't be expressed. Discipline: only deref while
// the host thread is inside `vm.interpret` or a re-entrant
// helper called *from* `vm.interpret`. After interpret returns
// the cell is null and deref is UB — which is fine because
// the `static mut`'s only legitimate writers are the entry/exit
// pairs that wlift_wasm controls.

#[cfg(target_arch = "wasm32")]
mod current_vm_cell {
    use crate::runtime::vm::VM;
    use std::cell::UnsafeCell;

    /// Single-threaded wasm cell — `Sync` is purely cosmetic,
    /// the runtime never runs from more than one thread.
    pub(super) struct VmCell(UnsafeCell<*mut VM>);
    unsafe impl Sync for VmCell {}

    pub(super) static CURRENT: VmCell = VmCell(UnsafeCell::new(std::ptr::null_mut()));

    impl VmCell {
        pub(super) fn get(&self) -> *mut VM {
            unsafe { *self.0.get() }
        }
        pub(super) fn set(&self, v: *mut VM) {
            unsafe { *self.0.get() = v }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod current_closure_cell {
    use crate::runtime::object::ObjClosure;
    use std::cell::UnsafeCell;

    /// Receiver closure of the currently-running JIT'd wasm
    /// function. Set by every dispatch boundary that crosses
    /// into a JIT'd module — the BC interpreter's tier-up
    /// dispatch, plus the JIT-to-JIT recursive fast path in
    /// `wren_call_<N>`. Read by `wren_get_upvalue` /
    /// `wren_set_upvalue` to resolve the closure's upvalue
    /// array without an extra wasm-ABI argument per JIT'd
    /// call. Mirrors the host's
    /// `read_jit_ctx().closure` pattern.
    pub(super) struct ClosureCell(UnsafeCell<*mut ObjClosure>);
    unsafe impl Sync for ClosureCell {}
    pub(super) static CURRENT: ClosureCell = ClosureCell(UnsafeCell::new(std::ptr::null_mut()));
    impl ClosureCell {
        pub(super) fn get(&self) -> *mut ObjClosure {
            unsafe { *self.0.get() }
        }
        pub(super) fn set(&self, c: *mut ObjClosure) {
            unsafe { *self.0.get() = c }
        }
    }
}

/// Read the current VM pointer. Returns null on host builds and
/// outside any `interpret` / `run_fiber` window. JIT'd helpers
/// like `wren_call` rely on this to find the live runtime
/// without taking a VM ref through the wasm ABI.
pub fn current_vm() -> *mut crate::runtime::vm::VM {
    #[cfg(target_arch = "wasm32")]
    {
        current_vm_cell::CURRENT.get()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::ptr::null_mut()
    }
}

/// Read the closure currently executing in JIT'd wasm. Null
/// outside any tier-up dispatch window. Returns the live
/// `ObjClosure` ptr so upvalue helpers can index into its
/// upvalue array.
#[cfg(target_arch = "wasm32")]
pub fn current_closure() -> *mut crate::runtime::object::ObjClosure {
    current_closure_cell::CURRENT.get()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn current_closure() -> *mut crate::runtime::object::ObjClosure {
    std::ptr::null_mut()
}

#[cfg(target_arch = "wasm32")]
mod current_module_vars_cell {
    use std::cell::UnsafeCell;

    /// Data pointer of the currently-executing module's
    /// `vars: Vec<Value>`. Set by every dispatch boundary that
    /// crosses into JIT'd wasm (BC → tier and tier → tier across
    /// modules); read inline by JIT'd `Instruction::GetModuleVar`
    /// / `SetModuleVar` lowerings via an `i32.const ADDR;
    /// i32.load; i64.load offset=idx*8` triple — replacing the
    /// per-access wasm-bindgen call into `wren_get_module_var`.
    /// The static lives at a stable address in the cdylib's
    /// linear memory; codegen bakes that address as an i32.const
    /// literal in the JIT'd module.
    pub(super) struct VarsCell(UnsafeCell<*mut u64>);
    unsafe impl Sync for VarsCell {}
    pub(super) static CURRENT: VarsCell = VarsCell(UnsafeCell::new(std::ptr::null_mut()));
    impl VarsCell {
        pub(super) fn get(&self) -> *mut u64 {
            unsafe { *self.0.get() }
        }
        pub(super) fn set(&self, p: *mut u64) {
            unsafe { *self.0.get() = p }
        }
    }
}

/// Read the data pointer for the currently-executing module's
/// `vars` Vec. Null outside any tier-up dispatch window. JIT'd
/// modules read this via inline `i32.load` against the static's
/// fixed address — see [`current_module_vars_addr`].
#[cfg(target_arch = "wasm32")]
pub fn current_module_vars() -> *mut u64 {
    current_module_vars_cell::CURRENT.get()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn current_module_vars() -> *mut u64 {
    std::ptr::null_mut()
}

/// Address of the `current_module_vars_cell::CURRENT` static
/// inside the cdylib's linear memory, returned as a wasm32 i32.
/// Used by the wasm codegen to bake an `i32.const ADDR` into
/// JIT'd modules so `Instruction::GetModuleVar(idx)` lowers to
/// `(i32.const ADDR) (i32.load) (i64.load offset=idx*8)` — three
/// inline instructions instead of an `(import "wren"
/// "wren_get_module_var")` wasm-bindgen boundary call. Returns
/// `None` on host builds (no cdylib memory to point at).
#[cfg(target_arch = "wasm32")]
pub fn current_module_vars_addr() -> Option<u32> {
    Some(&current_module_vars_cell::CURRENT as *const _ as usize as u32)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn current_module_vars_addr() -> Option<u32> {
    None
}

/// Address of the `current_closure_cell::CURRENT` static inside
/// the cdylib's linear memory. Same role as
/// [`current_module_vars_addr`] but for the in-flight closure
/// pointer — JIT'd `Instruction::GetUpvalue(idx)` reads through
/// this address (`i32.load → closure ptr → upvalues data ptr →
/// upvalues[idx] → location → *location`) instead of crossing
/// the wasm-bindgen boundary into `wren_get_upvalue`.
#[cfg(target_arch = "wasm32")]
pub fn current_closure_addr() -> Option<u32> {
    Some(&current_closure_cell::CURRENT as *const _ as usize as u32)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn current_closure_addr() -> Option<u32> {
    None
}

/// Byte offset within `ObjClosure` at which `Vec::as_ptr()`
/// (the upvalue array's heap data pointer) is stored. Discovered
/// once at first call by constructing a dummy `ObjClosure` and
/// linearly probing for the matching `usize` value, then cached.
///
/// Rust's `Vec` doesn't have a stable repr(C) layout, so we
/// can't assume `(ptr, cap, len)` order — the host JIT pins this
/// against `Vec::as_ptr()` at runtime via the
/// `verify_closure_layout` test, and the wasm tier-up does the
/// same here at startup. Single allocation + one Vec scan; cost
/// amortizes immediately.
#[cfg(target_arch = "wasm32")]
pub fn closure_upvalues_data_offset() -> u32 {
    use crate::runtime::object::ObjClosure;
    use std::sync::OnceLock;
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let dummy = ObjClosure::new(std::ptr::null_mut(), 1);
        let base = &dummy as *const _ as usize;
        let target = dummy.upvalues.as_ptr() as usize;
        let max = std::mem::size_of::<ObjClosure>();
        let step = std::mem::align_of::<usize>();
        let mut offset = 0;
        while offset + std::mem::size_of::<usize>() <= max {
            // SAFETY: `base + offset` is inside the live
            // `dummy` ObjClosure (offset bounded by sizeof);
            // alignment is `align_of::<usize>()` which
            // satisfies `*const usize` requirements.
            let probed = unsafe { *(base.wrapping_add(offset) as *const usize) };
            if probed == target {
                return offset as u32;
            }
            offset += step;
        }
        panic!(
            "could not find Vec data ptr offset in ObjClosure; \
             Rust's Vec internal layout may have changed"
        );
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn closure_upvalues_data_offset() -> u32 {
    0
}

/// Save / install the current module-vars data pointer; return
/// the previous value for the caller to restore on exit. Same
/// RAII pattern as `enter_closure` / `enter_vm`. The dispatcher
/// computes the new pointer by walking
/// `closure.function.fn_id → engine.func_module(id) →
/// engine.modules[name].vars.as_mut_ptr()` once per BC→JIT or
/// cross-module JIT→JIT entry. In-module call_indirect calls
/// inside a tier'd-up function don't go through the dispatcher,
/// so the cell stays valid for the whole nested-call chain
/// without per-call updates.
/// Save / install the module-vars data pointer. The hot path
/// in a steady-state outer loop (BC repeatedly calling into the
/// same tier'd-up function) writes the same pointer the cell
/// already holds, so skip the store when `prev == p`. The
/// returned `prev` is what the matching `restore_module_vars`
/// will compare against to decide whether to reinstall.
#[allow(unused_variables)]
pub fn enter_module_vars(p: *mut u64) -> *mut u64 {
    #[cfg(target_arch = "wasm32")]
    {
        let prev = current_module_vars_cell::CURRENT.get();
        if prev != p {
            current_module_vars_cell::CURRENT.set(p);
        }
        prev
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::ptr::null_mut()
    }
}

#[allow(unused_variables)]
pub fn restore_module_vars(prev: *mut u64) {
    #[cfg(target_arch = "wasm32")]
    {
        if current_module_vars_cell::CURRENT.get() != prev {
            current_module_vars_cell::CURRENT.set(prev);
        }
    }
}

/// Walk `closure → function → fn_id → engine.func_module(id) →
/// engine.modules[name].vars.as_mut_ptr()` to find the module
/// vars data pointer for a closure that's about to enter JIT'd
/// code. Returns null if any link is missing (closure outside
/// any registered module — should never happen in practice but
/// matches the existing `current_vm` null-on-not-set ergonomics).
///
/// Steady-state hot path (BC repeatedly dispatching the same
/// tier'd-up closure — fib's recursion, Adder.add's outer loop)
/// is short-circuited by a single-entry "last closure → vars
/// ptr" cache: ~13.5k recursive `fib(20)` calls all hit the
/// same closure, so the second call onward returns the cached
/// pointer without a HashMap probe. Cache invalidates whenever
/// a different closure is seen.
///
/// # Safety
/// `closure` must be a live `ObjClosure` and `vm` must be a live
/// VM. Both invariants hold whenever this is called from the
/// tier dispatch boundaries.
#[cfg(target_arch = "wasm32")]
pub unsafe fn module_vars_ptr_for_closure(
    vm: &mut crate::runtime::vm::VM,
    closure: *mut crate::runtime::object::ObjClosure,
) -> *mut u64 {
    if closure.is_null() {
        return std::ptr::null_mut();
    }
    // Fast path: did we look up this exact closure last time?
    // wasm32 is single-threaded so the cell is a plain static
    // UnsafeCell — no atomics, no thread-locals.
    {
        let cached = last_module_vars_cache::CURRENT.get();
        if cached.0 == closure {
            return cached.1;
        }
    }
    let fn_ptr = (*closure).function;
    if fn_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let func_id = FuncId((*fn_ptr).fn_id);
    let module_name = match vm.engine.func_module(func_id) {
        Some(name) => name.clone(),
        None => return std::ptr::null_mut(),
    };
    // `engine.modules` keys on `String` (not `Rc<String>`), so
    // pass an `&str` view of the Rc for the lookup.
    let result = match vm.engine.modules.get_mut(module_name.as_str()) {
        Some(entry) => entry.vars.as_mut_ptr() as *mut u64,
        None => std::ptr::null_mut(),
    };
    last_module_vars_cache::CURRENT.set((closure, result));
    result
}

#[cfg(target_arch = "wasm32")]
mod last_module_vars_cache {
    use crate::runtime::object::ObjClosure;
    use std::cell::UnsafeCell;

    /// Single-entry "last closure → module vars data ptr"
    /// cache. Invalidates when a different closure is seen.
    /// Steady-state outer loops where the same closure is
    /// dispatched each iteration get a cache hit on every
    /// subsequent call, skipping the per-dispatch HashMap probe
    /// in `module_vars_ptr_for_closure`.
    pub(super) struct Cache(UnsafeCell<(*mut ObjClosure, *mut u64)>);
    unsafe impl Sync for Cache {}
    pub(super) static CURRENT: Cache = Cache(UnsafeCell::new((
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    )));
    impl Cache {
        pub(super) fn get(&self) -> (*mut ObjClosure, *mut u64) {
            unsafe { *self.0.get() }
        }
        pub(super) fn set(&self, v: (*mut ObjClosure, *mut u64)) {
            unsafe { *self.0.get() = v }
        }
    }
}

/// Save / install the current closure pointer, return the
/// previous value for the caller to restore on exit. Same RAII
/// rationale as `enter_vm`: dispatch boundaries are short and
/// nested, so an explicit save / restore pair is simpler than
/// a guard with a closing borrow.
/// Save / install the receiver closure pointer. Steady-state
/// dispatch into the same tier'd-up closure repeatedly writes
/// the same pointer the cell already holds; skip the store on
/// match.
#[allow(unused_variables)]
pub fn enter_closure(
    c: *mut crate::runtime::object::ObjClosure,
) -> *mut crate::runtime::object::ObjClosure {
    #[cfg(target_arch = "wasm32")]
    {
        let prev = current_closure_cell::CURRENT.get();
        if prev != c {
            current_closure_cell::CURRENT.set(c);
        }
        prev
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::ptr::null_mut()
    }
}

#[allow(unused_variables)]
pub fn restore_closure(prev: *mut crate::runtime::object::ObjClosure) {
    #[cfg(target_arch = "wasm32")]
    {
        if current_closure_cell::CURRENT.get() != prev {
            current_closure_cell::CURRENT.set(prev);
        }
    }
}

/// Save the current value, install `vm`, return the saved ptr
/// for the caller to restore on exit. RAII would be cleaner
/// but the borrow against `vm: &mut VM` makes it awkward — the
/// scope guard would need a `'_` lifetime tied to the VM ref,
/// which then collides with `interpret`'s mutable borrow. Plain
/// save / restore keeps the type plumbing simple.
#[allow(unused_variables)]
pub fn enter_vm(vm: *mut crate::runtime::vm::VM) -> *mut crate::runtime::vm::VM {
    #[cfg(target_arch = "wasm32")]
    {
        let prev = current_vm_cell::CURRENT.get();
        current_vm_cell::CURRENT.set(vm);
        prev
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::ptr::null_mut()
    }
}

/// Restore a previously-saved VM pointer. Pair with [`enter_vm`].
#[allow(unused_variables)]
pub fn exit_vm(prev: *mut crate::runtime::vm::VM) {
    #[cfg(target_arch = "wasm32")]
    {
        current_vm_cell::CURRENT.set(prev);
    }
}

// ---------------------------------------------------------------------------
// Raw dispatch-hook counter
// ---------------------------------------------------------------------------
//
// Bumped on every call into `dispatch_closure_bc_inner`'s wasm
// hook, **before** any slot lookup or tier-up trigger. Lets the
// host distinguish "hook never runs" from "hook runs but
// `try_compile` silently fails." If this counter stays 0 while
// fib(20) executes, the BC interpreter is dispatching closure
// calls through a path that bypasses the hook entirely.
//
// Cost: one relaxed atomic increment per Wren method call. <1 ns
// on wasm32. Negligible vs. dispatch overhead.

static DISPATCH_HOOK_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Increment the hook-hit counter. Called from the wasm
/// dispatch hook in `vm_interp.rs::dispatch_closure_bc_inner`.
#[inline(always)]
pub fn bump_dispatch_hook_hits() {
    DISPATCH_HOOK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Read the total number of dispatch-hook entries. Exposed via
/// `wlift_wasm::tier_up::jit_dispatch_hook_hits()`.
pub fn dispatch_hook_hits() -> u64 {
    DISPATCH_HOOK_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

pub const BASELINE_THRESHOLD: u32 = 100;
pub const OPTIMIZED_THRESHOLD: u32 = 1_000;
pub const DEFAULT_DEOPT_LIMIT: u32 = 3;

/// Same osr-site packing as the host so downstream hash maps that
/// key on this u64 stay layout-compatible across builds.
#[inline]
pub fn encode_osr_site(block_id: u32, param_count: u16) -> u64 {
    ((block_id as u64) << 16) | (param_count as u64)
}

/// Stub-only Bead. Returned from `register` so callers' `Vec<Option<Arc<Bead>>>`
/// fields keep their host shape without dragging beadie in.
pub struct Bead;

/// Stub bead state. Wasm always reports `Interpreted` because there
/// is no JIT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeadState {
    Interpreted,
}

/// Mirror of `beadie::SubmitResult`. Wasm submissions always
/// return `AlreadyCompiling` to keep the broker-level retry path
/// quiet — no compile thread will ever actually pick the work up.
#[derive(Debug)]
pub enum SubmitResult {
    Accepted,
    AlreadyCompiling,
}

impl SubmitResult {
    pub fn is_accepted(&self) -> bool {
        matches!(self, SubmitResult::Accepted)
    }
}

/// Mirror of `beadie::DeoptDecision`. Wasm has no JIT so bailouts
/// can't actually happen, but the type has to exist for return
/// signatures to line up with the host build.
pub enum DeoptDecision {
    Recompile,
    Blacklist,
}

/// Empty osr-entry record. Stays compatible with the host
/// `beadie::OsrEntry` shape so engine code that builds `Vec<OsrEntry>`
/// for `install_or_swap_osr` keeps compiling.
pub struct OsrEntry {
    pub site: u64,
    pub addr: usize,
}

/// Empty marker — wasm has no broker thread, no beads, no tier
/// machinery. The methods callers reach for are no-ops or
/// `Interpreted`/`None`/`false` sentinels; the BC interpreter
/// never asks the tier manager for work because
/// `ExecutionMode::Interpreter` short-circuits the `record_call`
/// /  `request_tier_up` path entirely on wasm.
#[derive(Default)]
pub struct TierManager;

impl TierManager {
    pub fn with_thresholds(_baseline: u32, _optimized: u32) -> Self {
        Self
    }

    pub fn register(&mut self, _id: FuncId, _core: *mut ()) -> std::sync::Arc<Bead> {
        std::sync::Arc::new(Bead)
    }

    pub fn tick(&self, _id: FuncId) {}

    pub fn should_promote_on_tick(&self, _id: FuncId, _threshold: u32) -> bool {
        false
    }

    pub fn osr_entry(&self, _id: FuncId, _site: u64) -> Option<*mut ()> {
        None
    }

    pub fn install_or_swap(&self, _id: FuncId, _code: *mut ()) -> bool {
        false
    }

    pub fn install_or_swap_osr(&self, _id: FuncId, _code: *mut (), _osr: Vec<OsrEntry>) -> bool {
        false
    }

    pub fn submit_compile_osr<F>(&self, _id: FuncId, _compile: F) -> SubmitResult
    where
        F: Send + 'static,
    {
        SubmitResult::AlreadyCompiling
    }

    pub fn record_bailout(
        &self,
        _id: FuncId,
        _guard_id: u32,
        _pc_offset: u32,
    ) -> Option<DeoptDecision> {
        None
    }

    pub fn is_blacklisted(&self, _id: FuncId) -> bool {
        false
    }

    pub fn invocations(&self, _id: FuncId) -> u32 {
        0
    }

    pub fn state(&self, _id: FuncId) -> Option<BeadState> {
        Some(BeadState::Interpreted)
    }

    pub fn invalidate(&self, _id: FuncId) {}

    pub fn update_core(&self, _id: FuncId, _core: *mut ()) {}

    pub fn bailouts(&self, _id: FuncId) -> u32 {
        0
    }

    pub fn snapshot(&self) -> Vec<TierSnapshot> {
        Vec::new()
    }
}

/// Mirror of the host `TierSnapshot`. Carries no live state on
/// wasm — `snapshot()` would always return an empty list.
#[derive(Clone, Debug)]
pub struct TierSnapshot {
    pub func_id: FuncId,
    pub state: BeadState,
    pub invocations: u32,
    pub bailouts: u32,
}
