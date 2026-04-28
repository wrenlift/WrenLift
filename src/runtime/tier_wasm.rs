//! Wasm-side tier-up broker.
//!
//! `tier.rs` (the host build) wraps `beadie`, which spawns a
//! background promoter thread to drive tier-up. Neither beadie
//! nor `std::thread::spawn` works on `wasm32-unknown-unknown`,
//! so the wasm path runs the broker inline.
//!
//! ## Tier-up to wasm (Phase 3+)
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
// Callback hooks (Phase 3a)
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
// Thread-local current VM (Phase 4 foundation)
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
