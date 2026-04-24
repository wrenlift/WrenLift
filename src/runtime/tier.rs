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
//! ## Phase 1 scope (this commit)
//!
//! This module is **shadow-only**: it compiles, has unit tests, and is
//! callable, but nothing in `engine.rs` or `vm_interp.rs` routes through
//! it yet. The old path still owns all live tier decisions.
//!
//! Subsequent commits will:
//! - Phase 2: replace `record_call` / `call_count` fields with `on_invoke`
//! - Phase 3: retire `compile_queue: Vec<FuncId>` in favour of the broker
//! - Phase 4: wire tier 2 (optimized) through the same broker + TieredPolicy
//!
//! ## What stays custom
//!
//! Beadie doesn't know about WrenLift-specific compile-time context:
//! inline-cache snapshots, back-edge counts for OSR, or on-stack-replacement
//! safepoints. Those stay in the engine and travel with the MIR that the
//! compile closure captures — beadie just owns the state transitions.

use std::sync::Arc;

use beadie::{Bead, Beadie, BeadState, CoreHandle, ThresholdPolicy, TieredPolicy};

use crate::runtime::engine::FuncId;

// ── Defaults ───────────────────────────────────────────────────────────────

/// Baseline promotion threshold — matches the prior `jit_threshold = 100`
/// default in `Engine::new`.
pub const BASELINE_THRESHOLD: u32 = 100;

/// Optimized promotion threshold — matches `opt_threshold`.
pub const OPTIMIZED_THRESHOLD: u32 = 1_000;

// ── TierManager ───────────────────────────────────────────────────────────

/// Per-engine tier-up manager. One instance, lives for the engine's
/// lifetime. Beadie's broker thread runs under the hood.
pub struct TierManager {
    beadie: Beadie<TieredPolicy>,
    /// `beads[idx]` is the [`Bead`] for [`FuncId`] `idx`, or `None` for
    /// gaps (slots that belong to modules, host fns, etc. with no JIT).
    /// Grown lazily; same index space as `Engine.functions`.
    beads: Vec<Option<Arc<Bead>>>,
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
        }
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

    /// Tick the bead AND return true iff this invocation is the one
    /// that crosses `threshold` for the first time (i.e. matches the
    /// legacy `call_count == jit_threshold` fire-once semantics).
    ///
    /// `threshold` is passed at call-time rather than stored on the
    /// manager so [`Engine::jit_threshold`] stays the single source
    /// of truth while tests mutate it post-construction.
    ///
    /// Returns false for unregistered ids, beads that have already
    /// moved past `Interpreted` (Queued/Compiling/Compiled), and
    /// counters that haven't reached threshold yet.
    pub fn should_promote_on_tick(&self, id: FuncId, threshold: u32) -> bool {
        let Some(bead) = self.beads.get(id.0 as usize).and_then(|b| b.as_ref()) else {
            return false;
        };
        let (count, state) = bead.tick();
        state == BeadState::Interpreted && count == threshold
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
        self.beads
            .get(id.0 as usize)?
            .as_ref()
            .map(|b| b.state())
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
