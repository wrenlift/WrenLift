//! Tier-up promotion manager — thin wrapper over the `beadie` crate.
//!
//! WrenLift's tiered mode used to carry its own per-function call-count
//! fields (`FuncBody::Interpreted { call_count }` / `FuncBody::Native
//! { opt_call_count }`), a `Vec<FuncId>` compile queue, and a synchronous
//! `tier_up` path scattered through `engine.rs`. `TierManager` consolidates
//! that into a typed, observable state machine:
//!
//! ```text
//!   Interpreted ──tick──► Queued ──broker──► Compiling ──install──► Compiled
//!        ▲                                                             │
//!        └────────── invalidate / deopt ──────────────────────────────┘
//! ```
//!
//! ## Migration status
//!
//! The cutover ran in phases. What's live now:
//!
//! - **Phase 1** — adapter module added, shadow-only unit tests.
//! - **Phase 2** — `TierManager` field on `ExecutionEngine`; bead
//!   ticks on every `record_call` alongside the legacy counter.
//! - **Phase 3** — `FuncBody::Interpreted.call_count` removed;
//!   baseline tier-up drives off `should_promote_on_tick`.
//! - **Phase 4** — install sites notify the bead via
//!   `install_or_swap` so `BeadState` matches engine `TierState`.
//! - **Phase 5** — `WLIFT_TIER_TRACE` prints both views side by side;
//!   `TierManager::snapshot()` for end-of-run dumps.
//! - **Phase 6** — `FuncBody::Native.opt_call_count` removed; both
//!   tiers share the single bead counter using absolute thresholds.
//! - **Phase 7** — `request_tier_up`'s `std::thread::spawn` swapped
//!   for `Beadie::submit`; the broker's bounded channel replaces the
//!   ad-hoc `compile_queue: Vec<FuncId>` plus `compile_handle`.
//!
//! ## What stays custom
//!
//! Beadie doesn't model WrenLift-specific compile-time context:
//! inline-cache snapshots, back-edge counts for OSR, on-stack-
//! replacement safepoints, the rich post-compile install that
//! touches `jit_code`/`tier_states`/`code_ranges`. All of that lives
//! engine-side; the compile closure captures what it needs and
//! returns the raw native pointer for beadie's state machine, while
//! the richer `ExecutableFunction` travels back to the main thread
//! through an mpsc channel that `poll_compilations` drains at
//! safepoints.

use std::sync::Arc;

use beadie::{
    BailoutInfo, Bead, BeadState, Beadie, CoreHandle, DeoptDecision, DeoptPolicy, OsrCompileResult,
    OsrEntry, SubmitResult, ThresholdDeoptPolicy, ThresholdPolicy, TieredPolicy,
};

use crate::runtime::engine::FuncId;

// ── Defaults ───────────────────────────────────────────────────────────────

/// Baseline promotion threshold — matches the prior `jit_threshold = 100`
/// default in `Engine::new`.
pub const BASELINE_THRESHOLD: u32 = 100;

/// Optimized promotion threshold — matches `opt_threshold`.
pub const OPTIMIZED_THRESHOLD: u32 = 1_000;

/// Default deopt recompile budget. After this many bailouts a function
/// is blacklisted from further JIT compilation for the remainder of the
/// process — assumed pathologically polymorphic / unstable.
pub const DEFAULT_DEOPT_LIMIT: u32 = 3;

/// Encode a WrenLift OSR site into beadie's u64 site key.
///
/// OSR entries are keyed on `(target_block, param_count)` — the MIR
/// block we're transferring INTO plus the number of block parameters
/// the compiled entry expects (so leaf / tail variants of the same
/// block land in the right place). We pack both into the 64-bit key
/// beadie stores:
///
/// ```text
///   +-------------------+-------------------+
///   |  block_id (upper 32 bits)  |  param_count (lower 16) ignored rest  |
///   +-------------------+-------------------+
/// ```
#[inline]
pub fn encode_osr_site(block_id: u32, param_count: u16) -> u64 {
    ((block_id as u64) << 16) | (param_count as u64)
}

// ── TierManager ───────────────────────────────────────────────────────────

/// Per-engine tier-up manager. One instance, lives for the engine's
/// lifetime. Beadie's broker thread runs under the hood.
pub struct TierManager {
    beadie: Beadie<TieredPolicy>,
    /// `beads[idx]` is the [`Bead`] for [`FuncId`] `idx`, or `None` for
    /// gaps (slots that belong to modules, host fns, etc. with no JIT).
    /// Grown lazily; same index space as `Engine.functions`.
    beads: Vec<Option<Arc<Bead>>>,
    /// Policy consulted on every bailout from JIT'd code. Defaults to
    /// a simple threshold (recompile up to N times, then blacklist).
    /// Swappable via [`Self::set_deopt_policy`] for custom strategies.
    deopt_policy: Arc<dyn DeoptPolicy>,
}

impl Default for TierManager {
    fn default() -> Self {
        Self::with_thresholds(BASELINE_THRESHOLD, OPTIMIZED_THRESHOLD)
    }
}

impl TierManager {
    pub fn with_thresholds(baseline: u32, optimized: u32) -> Self {
        Self {
            beadie: Beadie::with_policy(TieredPolicy {
                tier1_threshold: baseline,
                tier2_threshold: optimized,
            }),
            beads: Vec::new(),
            deopt_policy: Arc::new(ThresholdDeoptPolicy::new(DEFAULT_DEOPT_LIMIT)),
        }
    }

    /// Swap out the default `ThresholdDeoptPolicy` for a different
    /// strategy (e.g. `AlwaysRecompilePolicy` for development builds
    /// that want every bailout to recompile, or `ExponentialBackoffPolicy`
    /// for long-running servers that want to be more conservative).
    pub fn set_deopt_policy(&mut self, policy: Arc<dyn DeoptPolicy>) {
        self.deopt_policy = policy;
    }

    /// For tests / tools that want plain threshold semantics without a
    /// second tier.
    pub fn with_baseline_only(threshold: u32) -> Beadie<ThresholdPolicy> {
        Beadie::with_policy(ThresholdPolicy::new(threshold))
    }

    /// Register a function's bead. Call once when the function is
    /// compiled into the engine. `core` is an opaque pointer the
    /// runtime uses to round-trip back to its own function object
    /// inside the compile closure — beadie stores but never
    /// dereferences it.
    pub fn register(&mut self, id: FuncId, core: CoreHandle) -> Arc<Bead> {
        let idx = id.0 as usize;
        if idx >= self.beads.len() {
            self.beads.resize_with(idx + 1, || None);
        }
        let bead = self.beadie.register(core, None);
        self.beads[idx] = Some(Arc::clone(&bead));
        bead
    }

    /// Hot-path entry: called for every invocation of `id`. Returns a
    /// pointer to the installed native code when one exists, or `None`
    /// to keep interpreting. When the bead crosses its promotion
    /// threshold, the compile closure is submitted to beadie's broker
    /// thread — the closure runs exactly once per tier transition.
    #[inline]
    pub fn on_invoke<F>(&self, id: FuncId, compile: F) -> Option<*mut ()>
    where
        F: FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    {
        let bead = self.beads.get(id.0 as usize)?.as_ref()?;
        self.beadie.on_invoke(bead, compile)
    }

    /// Shadow-mode tick: bump the bead's invocation counter without
    /// running the promotion policy or submitting a compile closure.
    /// Used by Phase 2 to run the bead state machine alongside the
    /// engine's existing `record_call` counter so the two can be
    /// compared before the interpreter is switched over.
    pub fn tick(&self, id: FuncId) {
        if let Some(Some(bead)) = self.beads.get(id.0 as usize) {
            let _ = bead.tick();
        }
    }

    /// Tick the bead AND return true iff this tick's count equals
    /// `threshold` exactly — matches the legacy `count == threshold`
    /// fire-once semantics used for both baseline and optimized tier-up.
    ///
    /// The bead's invocation counter is monotonically increasing and
    /// never decrements, so exact equality fires once (and only once)
    /// across a function's lifetime. Callers layer their own state
    /// guards on top (e.g. `FuncBody::Interpreted` for baseline,
    /// `TierState::BaselineNative` for optimized) to decide when a
    /// tier-up is actually valid.
    ///
    /// `threshold` is passed at call-time rather than stored on the
    /// manager so [`Engine::jit_threshold`] / `opt_threshold` stay
    /// the single source of truth.
    pub fn should_promote_on_tick(&self, id: FuncId, threshold: u32) -> bool {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return false;
        };
        let (count, _) = bead.tick();
        count == threshold
    }

    /// Ignore the broker; slam a compiled code pointer into the bead
    /// immediately. Used by the synchronous tier-up path and by tests
    /// that want to verify tier transitions without spinning up the
    /// broker's timing. Returns true if the install took effect.
    pub fn eager_install(&self, id: FuncId, code: *mut ()) -> bool {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return false;
        };
        bead.eager_install(code)
    }

    /// Submit a bead to beadie's broker for background compilation.
    /// The closure runs on the broker thread and must return the
    /// compiled native pointer (or null on failure) — beadie's state
    /// machine stores it and transitions the bead to `Compiled`.
    ///
    /// Returns `SubmitResult::AlreadyQueued` if the bead has already
    /// moved past `Interpreted` (race between ticks), `QueueFull` if
    /// the broker's bounded channel is saturated, `Accepted` on success.
    pub fn submit_compile<F>(&self, id: FuncId, compile: F) -> SubmitResult
    where
        F: FnOnce(&Arc<Bead>) -> *mut () + Send + 'static,
    {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return SubmitResult::AlreadyQueued;
        };
        self.beadie.submit(bead, compile)
    }

    /// OSR-aware submit: the closure returns an `OsrCompileResult` with
    /// both the normal entry pointer and a vec of `OsrEntry`s (one per
    /// hot loop header). Beadie installs both atomically — readers that
    /// see the new generation see matching code + OSR table.
    pub fn submit_compile_osr<F>(&self, id: FuncId, compile: F) -> SubmitResult
    where
        F: FnOnce(&Arc<Bead>) -> OsrCompileResult + Send + 'static,
    {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return SubmitResult::AlreadyQueued;
        };
        self.beadie.submit_osr(bead, compile)
    }

    /// Look up an OSR entry point for a given back-edge site. O(log N)
    /// binary search on the bead's sorted OSR table. Returns `None` if
    /// the bead isn't compiled, has no OSR table, or no entry matches
    /// the site key.
    #[inline]
    pub fn osr_entry(&self, id: FuncId, site: u64) -> Option<*mut ()> {
        self.beads.get(id.0 as usize)?.as_ref()?.osr_entry(site)
    }

    /// Install a new compiled pointer, choosing the right beadie API
    /// based on current state: `eager_install` when the bead is still
    /// `Interpreted` (first-time install), `swap_compiled` when it's
    /// already `Compiled` (tier-up from baseline to optimized). Returns
    /// true if either path accepted the pointer.
    pub fn install_or_swap(&self, id: FuncId, code: *mut ()) -> bool {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return false;
        };
        match bead.state() {
            BeadState::Compiled => bead.swap_compiled(code).is_some(),
            _ => bead.eager_install(code),
        }
    }

    /// Install a compiled pointer + OSR entry table, state-aware.
    ///
    /// - `Interpreted` → `eager_install` the code (bead goes to
    ///   `Compiled`), then `swap_compiled_with_osr` to publish the
    ///   OSR table on top. Two-step because beadie's public install
    ///   API for the first-time-install path doesn't take an OSR
    ///   vec directly (the broker-submitted path does, so normal
    ///   tier-up via `submit_compile_osr` avoids this dance).
    /// - `Compiled` → `swap_compiled_with_osr` directly. Used for
    ///   tier-up from baseline to optimized; both code and OSR
    ///   table get replaced atomically under a new generation.
    ///
    /// Returns true if the bead ended up with the supplied pointer
    /// and OSR table.
    pub fn install_or_swap_osr(&self, id: FuncId, code: *mut (), osr: Vec<OsrEntry>) -> bool {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return false;
        };
        match bead.state() {
            BeadState::Compiled => bead.swap_compiled_with_osr(code, osr).is_some(),
            _ => {
                if !bead.eager_install(code) {
                    return false;
                }
                bead.swap_compiled_with_osr(code, osr).is_some()
            }
        }
    }

    /// Update the opaque core pointer after the runtime moves the
    /// function object (e.g. GC compaction).
    pub fn update_core(&self, id: FuncId, core: CoreHandle) {
        if let Some(Some(bead)) = self.beads.get(id.0 as usize) {
            bead.update_core(core);
        }
    }

    /// Mark the function unusable — closures live on the bead but the
    /// compiled code will no longer be dispatched to. Called on
    /// module unload / engine shutdown.
    pub fn invalidate(&self, id: FuncId) {
        if let Some(Some(bead)) = self.beads.get(id.0 as usize) {
            bead.invalidate();
        }
    }

    /// Current state for diagnostics. Returns `None` if the id was
    /// never registered.
    pub fn state(&self, id: FuncId) -> Option<BeadState> {
        self.beads.get(id.0 as usize)?.as_ref().map(|b| b.state())
    }

    /// Invocation count so far. Useful for `WLIFT_TIER_TRACE`-style
    /// observability and for test assertions.
    pub fn invocations(&self, id: FuncId) -> u32 {
        self.beads
            .get(id.0 as usize)
            .and_then(|b| b.as_ref())
            .map(|b| b.invocation_count())
            .unwrap_or(0)
    }

    /// Raw bead access for advanced use — profile sampling, bailout
    /// recording, introspection. Avoid on the hot path; prefer
    /// [`Self::on_invoke`] which already threads the bead internally.
    pub fn bead(&self, id: FuncId) -> Option<&Arc<Bead>> {
        self.beads.get(id.0 as usize)?.as_ref()
    }

    /// Number of registered beads. For diagnostics only.
    pub fn registered(&self) -> usize {
        self.beads.iter().filter(|b| b.is_some()).count()
    }

    /// Record a bailout from JIT'd code. The configured `DeoptPolicy`
    /// decides whether to allow recompilation (the bead reverts to
    /// `Interpreted` and the invocation counter eventually re-promotes)
    /// or blacklist the function permanently.
    ///
    /// Returns the policy decision so the caller can log, emit a trace,
    /// or take additional engine-side action (e.g. clearing the
    /// `jit_code` slot so subsequent dispatches fall back to the
    /// interpreter immediately).
    ///
    /// `guard_id` and `pc_offset` are runtime-defined — beadie stores
    /// them on the `BailoutInfo` but doesn't inspect them. Pass any
    /// stable identifier the compiler backend emits at the guard site.
    /// `generation` is `bead.generation()` at the time of the bailout
    /// — distinguishes a tier-1 bailout from a tier-2 bailout after a
    /// swap.
    pub fn record_bailout(
        &self,
        id: FuncId,
        guard_id: u32,
        pc_offset: u32,
    ) -> Option<DeoptDecision> {
        let bead = self.beads.get(id.0 as usize)?.as_ref()?;
        let info = BailoutInfo {
            guard_id,
            pc_offset,
            generation: bead.generation(),
        };
        Some(bead.on_bailout(info, self.deopt_policy.as_ref()))
    }

    /// Check whether a function has been permanently blacklisted by
    /// the deopt policy. Hot path hint — callers can skip tier-up
    /// submission and re-compile requests for blacklisted functions.
    pub fn is_blacklisted(&self, id: FuncId) -> bool {
        self.beads
            .get(id.0 as usize)
            .and_then(|b| b.as_ref())
            .map(|b| b.is_blacklisted())
            .unwrap_or(false)
    }

    /// Total bailouts seen by a specific function. Useful for
    /// `WLIFT_TIER_STATS` reporting and test assertions.
    pub fn bailouts(&self, id: FuncId) -> u32 {
        self.beads
            .get(id.0 as usize)
            .and_then(|b| b.as_ref())
            .map(|b| b.bailout_count())
            .unwrap_or(0)
    }

    /// Flat snapshot of every bead's state + invocation count, keyed by
    /// FuncId raw index. For end-of-run diagnostics and test assertions;
    /// not a hot-path API (walks the whole chain).
    pub fn snapshot(&self) -> Vec<TierSnapshot> {
        self.beads
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                slot.as_ref().map(|bead| TierSnapshot {
                    func_id: FuncId(idx as u32),
                    state: bead.state(),
                    invocations: bead.invocation_count(),
                    bailouts: bead.bailout_count(),
                })
            })
            .collect()
    }
}

/// One row of [`TierManager::snapshot`]. Cheap to clone; intended for
/// passing across thread or module boundaries in diagnostics output.
#[derive(Debug, Clone)]
pub struct TierSnapshot {
    pub func_id: FuncId,
    pub state: BeadState,
    pub invocations: u32,
    pub bailouts: u32,
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn fake_code_ptr() -> *mut () {
        // Any non-null address works; beadie stores but never calls it
        // in these tests. 0xdeadbeef keeps it obvious in debug output.
        0xdead_beef_usize as *mut ()
    }

    #[test]
    fn register_assigns_bead_indexed_by_func_id() {
        let mut m = TierManager::default();
        m.register(FuncId(0), std::ptr::null_mut());
        m.register(FuncId(3), std::ptr::null_mut());
        assert_eq!(m.registered(), 2);
        assert!(m.bead(FuncId(0)).is_some());
        assert!(m.bead(FuncId(1)).is_none()); // gap
        assert!(m.bead(FuncId(3)).is_some());
    }

    #[test]
    fn on_invoke_ticks_until_threshold_then_submits_compile() {
        let mut m = TierManager::with_thresholds(5, 50);
        m.register(FuncId(0), std::ptr::null_mut());
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..4 {
            let calls = Arc::clone(&calls);
            let result = m.on_invoke(FuncId(0), move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                fake_code_ptr()
            });
            assert!(result.is_none(), "should still be interpreting");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no compile yet");

        // Fifth call crosses the threshold — broker submission happens.
        let calls2 = Arc::clone(&calls);
        let _ = m.on_invoke(FuncId(0), move |_| {
            calls2.fetch_add(1, Ordering::SeqCst);
            fake_code_ptr()
        });

        // Broker is async — wait a moment for the closure to run.
        let start = Instant::now();
        while calls.load(Ordering::SeqCst) == 0 {
            if start.elapsed() > Duration::from_millis(500) {
                panic!("compile closure never ran");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn eager_install_bypasses_broker() {
        let mut m = TierManager::default();
        m.register(FuncId(7), std::ptr::null_mut());
        assert!(m.eager_install(FuncId(7), fake_code_ptr()));
        // Subsequent on_invoke should hit the fast path and return Some
        // without running the compile closure.
        let got = m.on_invoke(FuncId(7), |_| panic!("compile must not run"));
        assert_eq!(got, Some(fake_code_ptr()));
    }

    #[test]
    fn unregistered_func_id_is_noop() {
        let m = TierManager::default();
        assert_eq!(m.on_invoke(FuncId(99), |_| fake_code_ptr()), None);
        assert_eq!(m.state(FuncId(99)), None);
        assert_eq!(m.invocations(FuncId(99)), 0);
        // tick on an unregistered id is also a no-op (doesn't allocate, doesn't panic).
        m.tick(FuncId(99));
    }

    #[test]
    fn record_bailout_blacklists_after_limit() {
        // DEFAULT_DEOPT_LIMIT = 3 → recompile for bailouts 1..=3,
        // blacklist on the 4th.
        let mut mgr = TierManager::default();
        mgr.register(FuncId(0), std::ptr::null_mut());
        // Need a compiled bead to bail out of — fake-install a pointer.
        assert!(mgr.eager_install(FuncId(0), fake_code_ptr()));
        assert!(!mgr.is_blacklisted(FuncId(0)));

        for n in 1..=3 {
            let d = mgr.record_bailout(FuncId(0), 0, 0).unwrap();
            assert_eq!(d, DeoptDecision::Recompile, "bailout #{} recompiles", n);
            assert!(!mgr.is_blacklisted(FuncId(0)));
            // After Recompile the bead reverts to Interpreted; re-install
            // so the next bailout has something to bail from.
            assert!(mgr.eager_install(FuncId(0), fake_code_ptr()));
        }
        let d = mgr.record_bailout(FuncId(0), 0, 0).unwrap();
        assert_eq!(d, DeoptDecision::Blacklist);
        assert!(mgr.is_blacklisted(FuncId(0)));
        assert_eq!(mgr.bailouts(FuncId(0)), 4);
    }

    #[test]
    fn record_bailout_on_unregistered_id_is_noop() {
        let mgr = TierManager::default();
        assert!(mgr.record_bailout(FuncId(42), 0, 0).is_none());
    }

    #[test]
    fn tick_bumps_counter_without_submitting() {
        let mut m = TierManager::with_thresholds(3, 30);
        m.register(FuncId(0), std::ptr::null_mut());
        for _ in 0..100 {
            m.tick(FuncId(0));
        }
        // The bead has been ticked 100 times but no promotion policy
        // ran, so state is still Interpreted.
        assert_eq!(m.state(FuncId(0)), Some(BeadState::Interpreted));
        assert_eq!(m.invocations(FuncId(0)), 100);
    }

    #[test]
    fn invalidate_marks_bead_as_invalid() {
        let mut m = TierManager::default();
        m.register(FuncId(1), std::ptr::null_mut());
        assert!(m.bead(FuncId(1)).unwrap().is_valid());
        m.invalidate(FuncId(1));
        assert!(!m.bead(FuncId(1)).unwrap().is_valid());
    }
}
