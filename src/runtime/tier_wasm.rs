//! Wasm-side stub for the tier-up broker.
//!
//! `tier.rs` (the host build) wraps `beadie`, which spawns a
//! background promoter thread to drive tier-up. Neither beadie
//! nor `std::thread::spawn` works on `wasm32-unknown-unknown` —
//! and the Phase 1.0 wasm runtime defaults to
//! `ExecutionMode::Interpreter` anyway, so tier-up is
//! conceptually a no-op for the first cut.
//!
//! Phase 2 will swap this stub for a real wasm-aware tier
//! manager: hot Wren functions get lowered through the existing
//! `codegen::wasm::emit_mir` path, the resulting bytes get fed
//! to `WebAssembly.compile()` (browser host) or
//! `wasmtime::Module::new` (wasi host), and the broker installs
//! the returned function reference / table index in place of
//! the BC interpreter dispatch — same shape as the native JIT
//! path, different backend. The single-threaded story (no
//! `std::thread`) means the broker has to run inline, which
//! beadie 0.4+ already supports via its sync-policy variant.
//!
//! Until that lands, the methods here are no-ops; tier-up
//! callers (`request_tier_up`, `tier_up`, `submit_compile`)
//! live behind `#[cfg(feature = "host")]` so they never see
//! this module.

use crate::runtime::engine::FuncId;

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

    pub fn register(
        &mut self,
        _id: FuncId,
        _core: *mut (),
    ) -> std::sync::Arc<Bead> {
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

    pub fn install_or_swap_osr(
        &self,
        _id: FuncId,
        _code: *mut (),
        _osr: Vec<OsrEntry>,
    ) -> bool {
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
