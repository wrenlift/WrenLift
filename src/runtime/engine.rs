/// Tiered execution engine for WrenLift.
///
/// Manages function bodies across three execution tiers:
/// - Interpreter: walk MIR directly
/// - Tiered: start interpreted, JIT-compile hot functions
/// - JIT: compile everything to native before execution
///
/// The engine owns the function registry (MIR + compiled code) and
/// handles tier-up decisions based on call-count profiling.
///
/// In Tiered mode, compilation happens asynchronously on a background
/// thread. The interpreter continues using bytecode until compilation
/// finishes, then swaps to native code on the next dispatch.
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::OnceLock;

use crate::codegen::native_meta::NativeFrameMetadata;
use crate::codegen::{CompileTier, ExecutableFunction, NativeOsrEntry};
use crate::intern::SymbolId;
use crate::mir::bytecode::{BytecodeFunction, CallSiteIC};
use crate::mir::MirFunction;

/// Class-hierarchy-analysis index: for each method symbol, the
/// list of `(class_ptr, func_id, closure_ptr)` triples implementing
/// it. Consumed by both the JIT (to plant IC entries before any
/// runtime miss) and the interpreter (to dispatch monomorphic
/// CHA-known sites without a method-cache lookup).
pub type ChaMap = HashMap<SymbolId, Vec<(usize, u32, usize)>>;

/// Counters baseline code keeps for the tier above it. `countdown` is
/// decremented on every entry and outermost-loop iteration and calls
/// `wren_tier_tick` when it reaches zero; the helper reloads it with
/// the next interval and keeps the running total, so the call happens
/// only when a decision can change. `retier` is polled at outermost
/// loop headers and set once top-tier code with OSR entries is
/// installed. Offsets are baked into compiled code
/// (`cranelift_backend::TIER_CELL_*`).
#[repr(C)]
#[derive(Default)]
pub struct TierCell {
    pub countdown: std::sync::atomic::AtomicU32,
    pub retier: std::sync::atomic::AtomicU32,
    /// Entries counted so far, updated by the helper.
    pub total: std::sync::atomic::AtomicU32,
    /// The interval the countdown was last loaded with.
    pub interval: std::sync::atomic::AtomicU32,
}

impl TierCell {
    /// Call the helper again after `interval` more entries.
    fn tick_after(&self, interval: u32) {
        use std::sync::atomic::Ordering::Relaxed;
        self.interval.store(interval, Relaxed);
        self.countdown.store(interval, Relaxed);
    }

    /// Credit the interval that just elapsed and return the total.
    fn tick(&self) -> u32 {
        use std::sync::atomic::Ordering::Relaxed;
        let total = self
            .total
            .load(Relaxed)
            .saturating_add(self.interval.load(Relaxed));
        self.total.store(total, Relaxed);
        total
    }
}

/// One thread with a bounded queue for top-tier compiles, the shape of
/// beadie's promotion broker: a full queue rejects the proposal and the
/// function is re-proposed later.
#[cfg(feature = "host")]
struct Promoter {
    tx: Option<mpsc::SyncSender<Box<dyn FnOnce() + Send>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    /// Set when the engine goes away; jobs still queued are dropped
    /// unrun.
    stop: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "host")]
impl Promoter {
    fn start() -> Self {
        let (tx, rx) = mpsc::sync_channel::<Box<dyn FnOnce() + Send>>(256);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let worker = std::thread::Builder::new()
            .name("wlift-promoter".into())
            .spawn(move || {
                for job in rx {
                    if stopped.load(std::sync::atomic::Ordering::Acquire) {
                        drop(job);
                        continue;
                    }
                    job();
                }
            })
            .expect("spawn promoter thread");
        Self {
            tx: Some(tx),
            worker: Some(worker),
            stop,
        }
    }

    fn submit(&self, job: Box<dyn FnOnce() + Send>) -> bool {
        self.tx
            .as_ref()
            .map(|tx| tx.try_send(job).is_ok())
            .unwrap_or(false)
    }
}

/// A job in flight writes into engine memory (the function's tier cell,
/// the install channel), so the engine waits for it before its tables
/// go; jobs still queued are dropped unrun.
#[cfg(feature = "host")]
impl Drop for Promoter {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        drop(self.tx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// `WLIFT_RESULT_SPEC=0` stops the top tier guarding call results the
/// inline caches only ever saw as Num. Safe to run with either way.
fn result_speculation_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("WLIFT_RESULT_SPEC")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// `WLIFT_PROMOTE_GATE=0` proposes every hot function to the top tier;
/// unset keeps the worth gate. Safe to run with either way.
fn promotion_gate_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("WLIFT_PROMOTE_GATE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// What the top tier could win on a function, judged from its MIR
/// before a compile is spent: nothing for a loop-free body that is
/// mostly a call, little for a loop whose body is mostly calls, and
/// the rest for a loop with real work between the calls. Only `High`
/// is promoted: the baseline tier already serves the others, and a
/// top-tier compile there costs more than it returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopTierCeiling {
    None,
    Low,
    High,
}

/// `field_access_site(i)` says whether the `i`-th call site (in block
/// order, `Call` and `SuperCall` only) resolves to a trivial getter or
/// setter, which the tiers lower to a field access rather than a call.
pub fn top_tier_ceiling(
    mir: &MirFunction,
    field_access_site: &dyn Fn(usize) -> bool,
) -> TopTierCeiling {
    use crate::mir::opt::licm::{compute_dominators, compute_rpo, detect_loops};
    use crate::mir::Instruction;
    let instrs: usize = mir.blocks.iter().map(|b| b.instructions.len()).sum();
    let mut site = 0usize;
    let mut calls = 0usize;
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                Instruction::Call { .. } | Instruction::SuperCall { .. } => {
                    if !field_access_site(site) {
                        calls += 1;
                    }
                    site += 1;
                }
                Instruction::CallKnownFunc { .. } | Instruction::CallStaticSelf { .. } => {
                    calls += 1;
                }
                _ => {}
            }
        }
    }
    if mir.blocks.is_empty() {
        return TopTierCeiling::None;
    }
    let mut with_preds = mir.clone();
    with_preds.compute_predecessors();
    let rpo = compute_rpo(&with_preds);
    let idom = compute_dominators(&with_preds, &rpo);
    if detect_loops(&with_preds, &idom).is_empty() {
        if instrs.saturating_sub(calls) <= 8 {
            return TopTierCeiling::None;
        }
        return TopTierCeiling::Low;
    }
    if calls * 3 >= instrs {
        TopTierCeiling::Low
    } else {
        TopTierCeiling::High
    }
}

/// Optional shared CHA snapshot threaded through codegen.
pub type SharedCha = Option<Arc<ChaMap>>;

// ---------------------------------------------------------------------------
// Execution mode
// ---------------------------------------------------------------------------

/// How the VM should execute Wren code.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ExecutionMode {
    /// Walk the MIR directly. Never JIT-compile.
    Interpreter,
    /// Start interpreted, JIT-compile hot functions (default).
    #[default]
    Tiered,
    /// Compile everything to native before execution.
    Jit,
}

// ---------------------------------------------------------------------------
// Function identity
// ---------------------------------------------------------------------------

/// Unique identifier for a function within the engine.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FuncId(pub u32);

// ---------------------------------------------------------------------------
// Function body (the heart of tiered dispatch)
// ---------------------------------------------------------------------------

/// The current executable form of a function.
/// Runtime type profile for function parameters. Built from sampling
/// argument types during the first 100 interpreted calls.
#[derive(Clone, Default, Debug)]
pub struct TypeProfile {
    /// Observed type per param index (0 = receiver/this, 1+ = args).
    /// 0=unseen, 1=Num, 2=Bool, 3=Null, 4=String, 5=Object, 6=Mixed
    pub param_types: [u8; 8],
    pub sample_count: u32,
}

pub const PROFILE_UNSEEN: u8 = 0;
pub const PROFILE_NUM: u8 = 1;
pub const PROFILE_BOOL: u8 = 2;
pub const PROFILE_NULL: u8 = 3;
pub const PROFILE_STRING: u8 = 4;
pub const PROFILE_OBJECT: u8 = 5;
pub const PROFILE_MIXED: u8 = 6;

/// Classify a runtime Value into a profile type tag.
#[inline(always)]
pub fn classify_value(v: crate::runtime::value::Value) -> u8 {
    if v.is_num() {
        PROFILE_NUM
    } else if v.as_bool().is_some() {
        PROFILE_BOOL
    } else if v.is_null() {
        PROFILE_NULL
    } else if v.is_object() {
        PROFILE_OBJECT
    } else {
        PROFILE_MIXED
    }
}

/// Backing storage for a registered function.
///
/// The `Native` variant dominates size because `ExecutableFunction` inlines
/// the backend buffer. Boxing would double heap allocations per installed
/// function without reducing peak memory — each `FuncBody` already lives
/// inside a `Vec<FuncBody>` that grows linearly with the program.
#[allow(clippy::large_enum_variant)]
pub enum FuncBody {
    /// MIR available for interpretation. Not yet compiled to native.
    Interpreted {
        mir: Arc<MirFunction>,
        /// Lazily-lowered compact bytecode for the bytecode VM.
        bytecode: Option<Arc<BytecodeFunction>>,
    },
    /// Baseline native code is resident; optimized code may be installed later.
    Native {
        baseline_executable: ExecutableFunction,
        optimized_executable: Option<ExecutableFunction>,
        mir: Arc<MirFunction>,
        bytecode: Option<Arc<BytecodeFunction>>,
    },
}

impl FuncBody {
    /// Get a shared reference to the MIR (available in all tiers).
    /// Returns an Arc so callers can hold the MIR independently of the engine.
    pub fn mir(&self) -> &Arc<MirFunction> {
        match self {
            FuncBody::Interpreted { mir, .. } | FuncBody::Native { mir, .. } => mir,
        }
    }

    /// Whether this function has been compiled to native code.
    pub fn is_compiled(&self) -> bool {
        matches!(self, FuncBody::Native { .. })
    }
}

/// Currently active execution tier for a function.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TierState {
    #[default]
    Interpreted,
    BaselineNative,
    OptimizedNative,
}

/// Per-function tiering and dispatch counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FuncTierStats {
    pub interpreted_entries: u64,
    pub baseline_entries: u64,
    pub optimized_entries: u64,
    pub compile_attempts: u64,
    pub compile_successes: u64,
    pub ic_hits: u64,
    pub ic_misses: u64,
    pub native_to_native_calls: u64,
    pub osr_entries: u64,
    pub deopts_to_baseline: u64,
    pub fallbacks_to_interpreter: u64,
}

/// Coarse runtime-call counters used to find dispatch overhead in benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeCallStats {
    pub wren_call_entries: u64,
    pub wren_call_noframe_fastpath: u64,
    pub dispatch_call_entries: u64,
    pub dispatch_call_fn_fastpath: u64,
    pub dispatch_call_ic_attempts: u64,
    pub dispatch_call_ic_class_misses: u64,
    pub dispatch_call_method_cache_hits: u64,
    pub dispatch_call_method_cache_misses: u64,
    pub dispatch_call_list_native_fastpath: u64,
    pub dispatch_method_native: u64,
    pub dispatch_method_closure: u64,
    pub dispatch_method_constructor: u64,
    pub dispatch_method_trivial_getter: u64,
    pub dispatch_method_trivial_setter: u64,
    pub call_closure_entries: u64,
    pub call_closure_native_candidates: u64,
    pub call_closure_native_entries: u64,
    pub call_closure_interpreter_fallbacks: u64,
    pub jit_context_save_restore_pairs: u64,
    pub ic_kind1_hits: u64,
    pub ic_kind2_hits: u64,
    pub ic_kind3_hits: u64,
    pub ic_kind4_hits: u64,
    pub ic_kind5_hits: u64,
    pub ic_kind6_hits: u64,
    pub ic_invalidations: u64,
}

// ---------------------------------------------------------------------------
// Compiled module
// ---------------------------------------------------------------------------

/// A module's worth of compiled functions, ready for execution.
pub struct CompiledModule {
    /// The top-level function (executed when the module loads).
    pub top_level: FuncId,
    /// Module name.
    pub name: SymbolId,
}

// ---------------------------------------------------------------------------
// Execution engine
// ---------------------------------------------------------------------------

/// A completed background compilation ready to be installed.
///
/// `Compiled` is the dominant variant; the two sit next to each other on a
/// one-shot mpsc channel, so a single allocation per compile is the cheapest
/// shape. Boxing just to quiet the lint would add a heap hop on every
/// tier-up install.
#[allow(clippy::large_enum_variant)]
enum CompilationResult {
    Compiled {
        id: FuncId,
        tier: CompileTier,
        executable: ExecutableFunction,
        native_meta: Option<Arc<NativeFrameMetadata>>,
        inline_safe: bool,
    },
    Failed {
        id: FuncId,
    },
}

/// Run the optimization pipeline on MIR for JIT compilation.
/// Same passes as the AOT pipeline: ConstFold, DCE, CSE, TypeSpecialize, LICM, SRA.
fn run_jit_opt_pipeline(mir: &mut MirFunction, interner: &crate::intern::Interner) {
    use crate::mir::opt::{
        self, constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm,
        range_loop::RangeLoop, sra::Sra, MirPass,
    };
    let range_loop = RangeLoop { interner };
    // WLIFT_DISABLE_MATH_GUARD=1 leaves math methods on unknown
    // receivers as calls; safe to run with.
    if std::env::var_os("WLIFT_DISABLE_MATH_GUARD").is_none() {
        crate::mir::opt::math_guard::MathGuard::new(interner).run(mir);
    }
    let constfold = ConstFold;
    let dce = Dce;
    let cse = Cse::default();
    let type_spec = TypeSpecialize::with_math(interner);
    let licm = Licm;
    let sra = Sra;

    let passes: Vec<&dyn MirPass> = vec![
        &range_loop,
        &constfold,
        &dce,
        &cse,
        &type_spec,
        &constfold,
        &dce,
        &licm,
        &sra,
        &dce,
    ];
    opt::run_to_fixpoint(mir, &passes, 10);
    // WLIFT_DISABLE_UNBOX_PARAMS keeps loop-carried Nums boxed; safe to run with.
    if std::env::var_os("WLIFT_DISABLE_UNBOX_PARAMS").is_none() {
        crate::mir::opt::unbox_params::UnboxParams.run(mir);
        // WLIFT_DISABLE_INT_SPEC keeps proven-integral values as f64; safe to run with.
        if std::env::var_os("WLIFT_DISABLE_INT_SPEC").is_none() {
            crate::mir::opt::int_loop::IntSpecialize.run(mir);
        }
    }
}

/// Insert speculative type guards for function parameters based on runtime
/// profile data. When profile is available, only guard params that were
/// observed as Num (avoiding instant deopt for object params like
/// stronger(s1, s2) where s1/s2 are Strength objects).
/// When no profile: blind GuardNum for all non-this params (legacy behavior).
fn insert_speculative_guards(mir: &mut MirFunction, profile: Option<&TypeProfile>) {
    use crate::mir::Instruction;
    if mir.blocks.is_empty() {
        return;
    }
    let params: Vec<(crate::mir::ValueId, u16)> = mir.blocks[0]
        .instructions
        .iter()
        .filter_map(|(vid, inst)| {
            if let Instruction::BlockParam(idx) = inst {
                Some((*vid, *idx))
            } else {
                None
            }
        })
        .collect();

    let insert_pos = mir.blocks[0]
        .instructions
        .iter()
        .position(|(_, inst)| !matches!(inst, Instruction::BlockParam(_)))
        .unwrap_or(mir.blocks[0].instructions.len());

    let mut guards = Vec::new();
    for (vid, idx) in &params {
        if *idx == 0 {
            continue; // Skip 'this'
        }
        // Profile-guided: only guard params observed as Num or Bool.
        // Object/Mixed/Unseen params → no guard → keep boxed → avoid deopt.
        let observed = profile
            .and_then(|p| p.param_types.get(*idx as usize).copied())
            .unwrap_or(PROFILE_UNSEEN);

        match observed {
            PROFILE_NUM => {
                guards.push((mir.new_value(), Instruction::GuardNum(*vid)));
            }
            PROFILE_BOOL => {
                guards.push((mir.new_value(), Instruction::GuardBool(*vid)));
            }
            // Object, Mixed, Null, String, or never observed: no guard.
            // Guards are enforced, so a guess would only deopt.
            _ => {}
        }
    }

    for (i, guard) in guards.into_iter().enumerate() {
        mir.blocks[0].instructions.insert(insert_pos + i, guard);
    }
}

/// Translate WrenLift's per-function `NativeOsrEntry` list into the
/// beadie `OsrEntry` vec that gets installed on the bead. Skips
/// entries with null pointers (non-native / WASM fallback tiers).
#[cfg(feature = "host")]
fn encode_osr_entries(entries: &[NativeOsrEntry]) -> Vec<beadie::OsrEntry> {
    entries
        .iter()
        .filter(|e| !e.ptr.is_null())
        .map(|e| beadie::OsrEntry {
            site: super::tier::encode_osr_site(e.target_block.0, e.param_count),
            code: e.ptr as *mut (),
        })
        .collect()
}

fn tier_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WLIFT_TIER_TRACE").is_some())
}

/// Milliseconds since the first trace line, for ordering trace output.
pub fn trace_clock_ms() -> f64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1e3
}

/// Returns true if `mir` directly invokes a method that drives a
/// fiber switch — `Fiber.yield` / `Fiber.suspend`, or one of the
/// instance-side fiber controls (`fiber.try / call / transfer /
/// transferError`). The JIT's `handle_jit_fiber_action` keeps a
/// caller-side stub for `Yield`/`Suspend` (it just returns the
/// yielded value to the caller without actually suspending the
/// fiber), and even the `Call`/`Transfer` cases reach into the
/// caller's `mir_frames` and re-enter the bytecode interpreter
/// — both routes assume the JIT'd caller has not stowed any live
/// state in CPU registers across the dispatch. In practice that
/// assumption breaks on `Subscription_.receive` (busy-spins on
/// yield) and `Scheduler_.tick`'s `f.try()` loop (segfaults
/// during the cross-fiber switch when the inner fiber yields back
/// past a JIT'd `tick`). The bytecode interpreter handles all of
/// these correctly via `handle_fiber_action_bc`, so we keep any
/// function that *directly* dispatches a fiber control method off
/// the JIT until the native unwinder grows proper yield-safepoint
/// support.
///
/// `call()` / `call(_)` is intentionally NOT in this list:
/// `Fn.call` shares the same method name and is one of the most
/// common methods in user code (every closure invocation, every
/// `iter.each {|x| ...}.call(x)`). Excluding it would prevent
/// most hot loops from JIT'ing. Functions that genuinely call
/// `someFiber.call(_)` instead of `someFn.call(_)` are rare; if
/// they show up, we'll need a receiver-class check at compile
/// time rather than this name-based filter.
fn mir_calls_jit_unsafe_fiber_method(
    mir: &MirFunction,
    interner: &crate::intern::Interner,
) -> bool {
    direct_yield_method_names()
        .iter()
        .any(|n| mir_calls_method_named(mir, interner, n))
}

/// Method names that directly suspend the running fiber via
/// `Fiber.yield` / `.suspend` / `.try` / `.transfer*`. The JIT can't
/// currently unwind across these — the C-stack frame holding the
/// JIT'd function's locals isn't part of the saved fiber state, so
/// after the resume the spilled receiver register decays to the
/// uninitialized-slot sentinel (`TAG_UNDEFINED`) and the next method
/// call surfaces as `Object does not implement '...'`.
///
/// `call()` / `call(_)` is intentionally NOT here — see the doc
/// comment on `mir_calls_jit_unsafe_fiber_method`.
fn direct_yield_method_names() -> &'static [&'static str] {
    &[
        "yield()",
        "yield(_)",
        "suspend()",
        "try()",
        "try(_)",
        "transfer()",
        "transfer(_)",
        "transferError(_)",
    ]
}

fn mir_calls_method_named(
    mir: &MirFunction,
    interner: &crate::intern::Interner,
    name: &str,
) -> bool {
    use crate::mir::Instruction;
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            if let Instruction::Call { method, .. } = inst {
                if interner.resolve(*method) == name {
                    return true;
                }
            }
        }
    }
    false
}

/// Walk a single function's MIR and return true if it calls any
/// method whose symbol appears in `tainted`. Used to extend the
/// direct-yield refusal into a transitive one — if `f` calls
/// `g.method`, and any function named `method` may itself yield, `f`
/// is also unsafe to JIT.
fn mir_touches_module_vars(mir: &MirFunction) -> bool {
    use crate::mir::Instruction;
    mir.blocks.iter().any(|b| {
        b.instructions.iter().any(|(_, i)| {
            matches!(
                i,
                Instruction::GetModuleVar(_) | Instruction::SetModuleVar(..)
            )
        })
    })
}

fn mir_calls_any_tainted_method(
    mir: &MirFunction,
    tainted: &std::collections::HashSet<crate::intern::SymbolId>,
) -> bool {
    use crate::mir::Instruction;
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            if let Instruction::Call { method, .. } = inst {
                if tainted.contains(method) {
                    return true;
                }
            }
        }
    }
    false
}

/// Determine leaf status from compiled code metadata. A function is leaf if
/// its only safepoints are self-calls (CallLocal or wren_call_static_self_*).
/// This replaces MIR-level analysis which can't predict conditional CallRuntime
/// emission (e.g., SetField with/without write barrier).
#[allow(dead_code)]
fn is_compiled_leaf(
    native_meta: &Option<std::sync::Arc<crate::codegen::native_meta::NativeFrameMetadata>>,
) -> bool {
    native_meta
        .as_ref()
        .map(|meta| {
            meta.safepoints
                .iter()
                .all(|sp| sp.kind == crate::codegen::native_meta::SafepointKind::CallLocal)
        })
        .unwrap_or(true)
}

/// Check if a MIR function can stay on the direct native fast path for the
/// given compilation tier.
fn is_mir_inline_safe(mir: &MirFunction, compile_tier: CompileTier) -> bool {
    use crate::mir::Instruction;
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                // These generate runtime calls (allocation, dispatch, context access):
                Instruction::Call { .. }
                | Instruction::CallKnownFunc { .. }
                | Instruction::SuperCall { .. }
                | Instruction::MakeList(_)
                | Instruction::MakeMap(_)
                | Instruction::MakeRange { .. }
                | Instruction::MakeClosure { .. }
                | Instruction::StringConcat(_)
                | Instruction::ToString(_)
                | Instruction::GetUpvalue(_)
                | Instruction::SetUpvalue(_, _)
                | Instruction::GetModuleVar(_)
                | Instruction::SetModuleVar(_, _)
                | Instruction::GetStaticField(_)
                | Instruction::SetStaticField(_, _)
                | Instruction::GuardNum(_)
                | Instruction::GuardBool(_) => return false,
                Instruction::CallStaticSelf { .. } if compile_tier == CompileTier::Optimized => {
                    // Optimized-tier recursive direct calls still deopt through a
                    // runtime helper today, so keep them off the leaf fast path.
                    return false;
                }
                // Everything else is inline (field access, arithmetic, guards, etc.)
                _ => {}
            }
        }
    }
    true
}

/// The execution engine: owns all function bodies, dispatches calls,
/// and manages tiered compilation.
///
/// In Tiered mode, hot functions are compiled asynchronously — each
/// request spawns a thread that compiles and sends the result back
/// via a channel. The interpreter continues using bytecode until
/// `poll_compilations` installs the native code at the next safepoint.
pub struct ExecutionEngine {
    /// Current execution mode.
    pub mode: ExecutionMode,
    /// Whether to collect per-call tier telemetry.
    pub collect_tier_stats: bool,
    /// All registered functions, indexed by FuncId.
    pub functions: Vec<FuncBody>,
    /// Profiling threshold: interpreted calls before baseline compilation triggers.
    pub jit_threshold: u32,
    /// Profiling threshold: baseline-native calls before optimize tier-up triggers.
    pub opt_threshold: u32,
    /// Module registry: module name → (top_level FuncId, module var storage).
    pub modules: HashMap<String, ModuleEntry>,
    /// Sender cloned into each background compilation thread.
    compilation_tx: mpsc::Sender<CompilationResult>,
    /// Receiver polled at safepoints to install completed compilations.
    compilation_rx: mpsc::Receiver<CompilationResult>,
    /// Per-function compile tier currently in flight, if any.
    compiling_tier: Vec<Option<CompileTier>>,
    /// Number of compilations currently in flight (fast check for
    /// `poll_compilations`). Incremented per `request_tier_up` submission
    /// accepted by beadie's broker; decremented per mpsc install.
    pending_count: u32,
    /// Callee FuncIds discovered from IC data after a baseline install,
    /// waiting to be submitted for predictive pre-compile. Drained at
    /// the next `drain_compile_queue` call where we have interner access.
    pending_callee_precompile: Vec<FuncId>,
    /// JIT native code pointers indexed by FuncId. O(1) lookup for fast dispatch.
    /// null_ptr entries mean the function is not yet compiled.
    pub jit_code: Vec<*const u8>,
    /// Wasm-tier-up dispatch slots indexed by `FuncId`. `0` = not
    /// compiled; otherwise the value is `slot + 1` (the actual
    /// slot index is `value - 1`). Lives parallel to `jit_code`
    /// because wasm doesn't deal in raw native function pointers
    /// — slots index into the host's `WebAssembly.Instance` table
    /// (see `wlift_wasm::tier_up`). Always empty on host builds;
    /// the BC interpreter's wasm dispatch hook reads it under
    /// `cfg(target_arch = "wasm32")` only.
    pub wasm_jit_slots: Vec<u32>,
    /// Wasm-tier-up call counters indexed by `FuncId`. Bumped on
    /// each invocation under `cfg(target_arch = "wasm32")` only;
    /// when a counter crosses `WASM_JIT_THRESHOLD` the dispatch
    /// hook calls `runtime::tier::try_compile` and stashes the
    /// returned slot in `wasm_jit_slots`. Separate from
    /// `tier_stats` so a tier-up event in one path doesn't
    /// confuse the other.
    pub wasm_call_counts: Vec<u32>,
    /// Whether the currently active native tier can use the direct fast path.
    pub jit_leaf: Vec<bool>,
    /// Preserved native-frame metadata for compiled functions.
    pub jit_metadata: Vec<Option<Arc<NativeFrameMetadata>>>,
    /// Active execution tier for each function.
    pub tier_states: Vec<TierState>,
    /// Per-function tier and dispatch statistics.
    pub tier_stats: Vec<FuncTierStats>,
    /// Defining module for each function indexed by FuncId.
    /// `GetModuleVar @idx` inside the function's body is bound to
    /// *this* module's variable slots (where the function was
    /// compiled), not the caller's. Call dispatch reads this to set
    /// each pushed frame's `module_name`, so a class imported from
    /// module A keeps resolving its sibling classes against A even
    /// when invoked from module B. `None` is only used by isolated
    /// unit-test paths that don't run through the full VM pipeline.
    pub func_modules: Vec<Option<Rc<String>>>,
    /// Per function, the address of its module's variable cell once
    /// resolved (0 until then); the cell is leaked per module, so the
    /// address never goes stale.
    func_modvars_cell: std::cell::RefCell<Vec<usize>>,
    pub runtime_call_stats: RuntimeCallStats,
    /// Cached field index for trivial getters of the form `{ _field }`.
    pub trivial_getter_fields: Vec<Option<u16>>,
    /// Cached field index for trivial setters of the form `{ _field = value }`.
    pub trivial_setter_fields: Vec<Option<u16>>,
    /// Baseline native code pointers indexed by FuncId.
    pub baseline_code: Vec<*const u8>,
    /// Baseline native OSR entry points indexed by FuncId.
    pub baseline_osr_entries: Vec<Vec<NativeOsrEntry>>,
    /// Loop headers whose body had no inline-cache data when the
    /// installed code was compiled, so its call sites are generic. The
    /// interpreter keeps running such a loop and asks for a recompile
    /// once it is hot, instead of entering code that will never improve.
    pub cold_osr_blocks: Vec<std::collections::HashSet<crate::mir::BlockId>>,
    /// Cold sets of compiles in flight, applied at install.
    pending_cold_osr: HashMap<usize, std::collections::HashSet<crate::mir::BlockId>>,
    /// Probes of a cold entry per (function, block), for pacing recompile requests.
    cold_osr_probes: HashMap<(u32, u32), u32>,
    /// Whether baseline-native code can use the direct fast path.
    pub baseline_leaf: Vec<bool>,
    /// Baseline native metadata indexed by FuncId.
    pub baseline_metadata: Vec<Option<Arc<NativeFrameMetadata>>>,
    /// Optimized native code pointers indexed by FuncId.
    pub optimized_code: Vec<*const u8>,
    /// Optimized native OSR entry points indexed by FuncId.
    pub optimized_osr_entries: Vec<Vec<NativeOsrEntry>>,
    /// Whether optimized-native code can use the direct fast path.
    pub optimized_leaf: Vec<bool>,
    /// Optimized native metadata indexed by FuncId.
    pub optimized_metadata: Vec<Option<Arc<NativeFrameMetadata>>>,
    /// Per function, the counters its baseline code reads and writes;
    /// boxed so the address baked into the code stays valid.
    #[allow(clippy::vec_box)]
    tier_cells: Vec<Box<TierCell>>,
    /// Invocation count before which the top tier is not proposed again
    /// for the function (a declined proposal doubles it).
    promote_retry_at: Vec<u32>,
    /// Functions the top tier will never take: the worth gate refused
    /// them or their compile failed.
    promote_refused: Vec<bool>,
    /// Functions a speculative guard failed in; their compiles carry
    /// no speculation from then on.
    speculation_failed: Vec<bool>,
    /// The closure and defining class each method was bound with, so
    /// a failed mid-body guard can resume the method in the
    /// interpreter. Null for functions that are not class methods.
    pub method_binding: Vec<(
        *mut crate::runtime::object::ObjClosure,
        *mut crate::runtime::object::ObjClass,
    )>,
    /// Mid-body guards that failed and resumed the interpreter.
    pub deopt_exits: u32,
    /// Native code replaced by a recompile. Frames may still be running
    /// it, so it is kept for the engine's lifetime.
    retired_code: Vec<ExecutableFunction>,
    /// The thread top-tier compiles run on, started on first use.
    #[cfg(feature = "host")]
    promoter: Option<Promoter>,
    /// Bytecode pointers indexed by FuncId. O(1) lookup for bytecode dispatch.
    /// null entries mean bytecode not yet compiled for this function.
    pub bc_cache: Vec<*const crate::mir::bytecode::BytecodeFunction>,
    /// Type profiles indexed by FuncId. Persists across tier transitions.
    pub type_profiles: Vec<Option<TypeProfile>>,
    /// Code ranges for compiled functions, sorted by start address.
    /// Used by GC stack walker to map return addresses → safepoint metadata.
    pub code_ranges: Vec<CodeRange>,
    /// AOT-binary modvars / consts data regions registered at
    /// startup. Each `(addr, count)` is a slice of `count` u64
    /// `Value`-bits the GC scans + writes back forwarded
    /// pointers to. `engine.modules` is empty under AOT (the
    /// bootstrap doesn't go through `interpret`'s install loop),
    /// so without these the const strings + closure pointers in
    /// `wlift_modvars_<n>` and `wlift_consts_<n>` aren't roots
    /// and a minor GC sweeps them while the AOT body still
    /// references them via `GetModuleVar`.
    pub aot_root_regions: Vec<(*mut u64, usize)>,
    /// Per-FuncId flag — `true` when the function was AOT-emitted
    /// as a state machine (tainted by `Fiber.yield`). Dispatch
    /// reads this off the function pointer the closure resolves
    /// to and routes through `wlift_aot_invoke_state_machine`
    /// (calls the poll fn with `(fiber, resume_v)`) instead of
    /// the regular `wren_call_*` path.
    pub aot_state_machine: Vec<bool>,
    /// Threaded-code cache indexed by FuncId. Lazily populated on first
    /// interpreter dispatch — faster than bytecode for hot functions.
    /// `None` = not yet checked. `Some(None)` = checked, not eligible.
    /// `Some(Some(tc))` = eligible, threaded code ready.
    #[cfg(feature = "host")]
    pub threaded_code: Vec<Option<Option<crate::mir::threaded::ThreadedCode>>>,
    /// Tier-up promotion broker.
    pub tier: super::tier::TierManager,
    /// Cached per-callee purity map. Recomputed only when the
    /// function table changes — JIT submissions can fire 70+ times
    /// for a single benchmark, so eagerly walking every body on
    /// every submit was an O(N²) hot spot.
    cached_purity_map: Option<Arc<std::collections::HashMap<u32, bool>>>,
    /// `functions.len()` snapshot at the point `cached_purity_map`
    /// was built. The map invalidates when the function table grows.
    cached_purity_map_len: usize,
    /// Cached "this function transitively doesn't allocate" map. Used
    /// by the IC kind=1 inline JIT-leaf fast path to guarantee that
    /// the callee can't fire a GC, so register-passed args stay
    /// valid without JIT-frame stack maps.
    cached_alloc_free_map: Option<Arc<std::collections::HashMap<u32, bool>>>,
    /// Vec form of `cached_alloc_free_map` indexed by `FuncId.0` for
    /// O(1) lookup on the IC hot path. Both share the same
    /// invalidation key (`functions.len()`).
    cached_alloc_free_vec: Vec<bool>,
}

// Cache for the transitive may-yield method-name set. Held in a
// thread_local rather than as an `ExecutionEngine` field because
// plugins (cdylibs) are compiled against the host's published
// `VM` / `ExecutionEngine` layout — adding fields would drift the
// struct size and the next foreign call from a stale plugin would
// SIGSEGV. Plugins must enable the `wren_lift/host` feature to keep the
// `VM` / `ExecutionEngine` struct layout in sync with the runtime they
// load against.
//
// Invalidates when the `ExecutionEngine` whose pointer it last saw
// changes, or when its function-table length grew. The pointer
// guard handles process restarts that recreate the VM at the same
// `functions.len()`; the length guard handles new functions being
// registered after a previous compute.
thread_local! {
    static MAY_YIELD_CACHE: std::cell::RefCell<MayYieldCache> =
        std::cell::RefCell::new(MayYieldCache::default());
}

#[derive(Default)]
struct MayYieldCache {
    last_engine: usize,
    last_len: usize,
    set: Option<Arc<std::collections::HashSet<crate::intern::SymbolId>>>,
}

// FuncId → "calls remaining before re-tier-up is allowed" map for
// auto-deopted functions. When `vm_interp` detects a corruption
// signal (receiver class resolves to bare `Object`) inside a JIT
// frame and demotes via `note_deopt_to_baseline`, it also seeds an
// entry here at `jit_threshold`. `record_call` decrements on every
// subsequent invocation; once the entry hits zero, the call site
// returns `true` (forcing a re-tier-up submission) and the entry
// is cleared. This is what lets a function recover after a
// transient JIT mis-compile — without it, the bead's monotonic
// invocation counter has already passed the fire-once threshold,
// so the regular tier-up path would never re-fire and the
// function stays at the lower tier indefinitely.
//
// Lives in a thread_local so the engine struct layout doesn't
// drift for already-loaded plugin cdylibs.
thread_local! {
    static AUTO_DEOPT_RETRY: std::cell::RefCell<std::collections::HashMap<u32, u32>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Mark a function as eligible for re-tier-up after `delay`
/// further invocations. Used by the auto-deopt path in
/// `vm_interp` so a function demoted because of suspected JIT
/// corruption gets a clean observation window before retrying
/// JIT compilation.
pub fn schedule_auto_deopt_retry(id: FuncId, delay: u32) {
    AUTO_DEOPT_RETRY.with(|m| {
        m.borrow_mut().insert(id.0, delay);
    });
}

/// Tick the auto-deopt retry counter for `id`. Returns `true`
/// once when the counter reaches zero, signalling that the
/// caller should re-issue tier-up. The entry is cleared on the
/// triggering call so subsequent calls don't keep re-firing.
fn auto_deopt_retry_tick(id: FuncId) -> bool {
    AUTO_DEOPT_RETRY.with(|m| {
        let mut map = m.borrow_mut();
        let Some(remaining) = map.get_mut(&id.0) else {
            return false;
        };
        if *remaining == 0 {
            map.remove(&id.0);
            return true;
        }
        *remaining = remaining.saturating_sub(1);
        if *remaining == 0 {
            map.remove(&id.0);
            return true;
        }
        false
    })
}

/// Address range of a compiled function's native code.
#[derive(Debug, Clone)]
pub struct CodeRange {
    pub start: usize,
    pub end: usize,
    pub func_id: FuncId,
    pub metadata: Arc<NativeFrameMetadata>,
}

/// Stable view of a module's variable storage for compiled code.
///
/// JIT bodies bake the cell's address and load the array pointer and
/// length through it, so a module variable access is three loads and
/// a bounds check instead of a helper call. The cell is leaked, one
/// per module entry, and zeroed when its entry drops so a stale body
/// reads null rather than freed memory. Every mutation of the vector
/// that may reallocate must be followed by `ModuleEntry::sync_cell`.
#[repr(C)]
pub struct ModuleVarsCell {
    pub ptr: std::sync::atomic::AtomicPtr<u64>,
    pub len: std::sync::atomic::AtomicUsize,
}

/// Per-module execution state.
pub struct ModuleEntry {
    /// The top-level function for this module.
    pub top_level: FuncId,
    /// Module-level variable storage (indexed by slot number).
    pub vars: Vec<super::value::Value>,
    /// Variable names corresponding to each slot (for C API lookup).
    pub var_names: Vec<String>,
    /// See [`ModuleVarsCell`].
    pub cell: &'static ModuleVarsCell,
}

impl ModuleEntry {
    pub fn new(top_level: FuncId, vars: Vec<super::value::Value>, var_names: Vec<String>) -> Self {
        let cell: &'static ModuleVarsCell = Box::leak(Box::new(ModuleVarsCell {
            ptr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            len: std::sync::atomic::AtomicUsize::new(0),
        }));
        let entry = ModuleEntry {
            top_level,
            vars,
            var_names,
            cell,
        };
        entry.sync_cell();
        entry
    }

    /// Publish the vector's current buffer to compiled code. Call after
    /// any push, resize or replacement of `vars`.
    pub fn sync_cell(&self) {
        use std::sync::atomic::Ordering;
        // Length first so a reader never sees a new length with the old
        // pointer; the pointer store makes both current.
        self.cell.len.store(0, Ordering::Release);
        self.cell
            .ptr
            .store(self.vars.as_ptr() as *mut u64, Ordering::Release);
        self.cell.len.store(self.vars.len(), Ordering::Release);
    }
}

impl Drop for ModuleEntry {
    fn drop(&mut self) {
        self.cell.len.store(0, std::sync::atomic::Ordering::Release);
    }
}

/// Result of interpreting Wren source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterpretResult {
    Success,
    CompileError,
    RuntimeError,
}

impl ExecutionEngine {
    /// Create a new engine with the given mode.
    pub fn new(mode: ExecutionMode) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            mode,
            collect_tier_stats: std::env::var_os("WLIFT_TIER_STATS").is_some(),
            functions: Vec::new(),
            jit_threshold: 100,
            opt_threshold: 1000,
            modules: HashMap::new(),
            compilation_tx: tx,
            compilation_rx: rx,
            compiling_tier: Vec::new(),
            pending_count: 0,
            pending_callee_precompile: Vec::new(),
            jit_code: Vec::new(),
            wasm_jit_slots: Vec::new(),
            wasm_call_counts: Vec::new(),
            jit_leaf: Vec::new(),
            jit_metadata: Vec::new(),
            tier_states: Vec::new(),
            tier_stats: Vec::new(),
            func_modules: Vec::new(),
            func_modvars_cell: std::cell::RefCell::new(Vec::new()),
            runtime_call_stats: RuntimeCallStats::default(),
            trivial_getter_fields: Vec::new(),
            trivial_setter_fields: Vec::new(),
            baseline_code: Vec::new(),
            baseline_osr_entries: Vec::new(),
            cold_osr_blocks: Vec::new(),
            pending_cold_osr: HashMap::new(),
            cold_osr_probes: HashMap::new(),
            baseline_leaf: Vec::new(),
            baseline_metadata: Vec::new(),
            optimized_code: Vec::new(),
            optimized_osr_entries: Vec::new(),
            optimized_leaf: Vec::new(),
            optimized_metadata: Vec::new(),
            tier_cells: Vec::new(),
            promote_retry_at: Vec::new(),
            promote_refused: Vec::new(),
            speculation_failed: Vec::new(),
            method_binding: Vec::new(),
            deopt_exits: 0,
            retired_code: Vec::new(),
            #[cfg(feature = "host")]
            promoter: None,
            bc_cache: Vec::new(),
            type_profiles: Vec::new(),
            code_ranges: Vec::new(),
            aot_root_regions: Vec::new(),
            aot_state_machine: Vec::new(),
            #[cfg(feature = "host")]
            threaded_code: Vec::new(),
            // Thresholds mirror the legacy jit_threshold / opt_threshold
            // defaults above so shadow-mode tick counts stay in step with
            // the existing counters they'll eventually replace.
            tier: super::tier::TierManager::with_thresholds(
                super::tier::BASELINE_THRESHOLD,
                super::tier::OPTIMIZED_THRESHOLD,
            ),
            cached_purity_map: None,
            cached_purity_map_len: 0,
            cached_alloc_free_map: None,
            cached_alloc_free_vec: Vec::new(),
        }
    }

    /// Register a MIR function and return its FuncId. The function is
    /// recorded as module-less — only suitable for isolated tests.
    /// Production paths should use [`Self::register_function_in`] so call
    /// dispatch can bind frames to the function's defining module.
    pub fn register_function(&mut self, mir: MirFunction) -> FuncId {
        self.register_function_in(mir, None)
    }

    /// Register a MIR function bound to the given defining module.
    pub fn register_function_in(&mut self, mir: MirFunction, module: Option<Rc<String>>) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        let trivial_getter = Self::mir_trivial_getter_field(&mir);
        let trivial_setter = Self::mir_trivial_setter_field(&mir);
        self.functions.push(FuncBody::Interpreted {
            mir: Arc::new(mir),
            bytecode: None,
        });
        self.compiling_tier.push(None);
        self.jit_code.push(std::ptr::null());
        self.wasm_jit_slots.push(0);
        self.wasm_call_counts.push(0);
        self.jit_leaf.push(false);
        self.jit_metadata.push(None);
        self.tier_states.push(TierState::Interpreted);
        self.tier_stats.push(FuncTierStats::default());
        self.func_modules.push(module);
        self.trivial_getter_fields.push(trivial_getter);
        self.trivial_setter_fields.push(trivial_setter);
        self.baseline_code.push(std::ptr::null());
        self.baseline_osr_entries.push(Vec::new());
        self.cold_osr_blocks.push(std::collections::HashSet::new());
        self.baseline_leaf.push(false);
        self.baseline_metadata.push(None);
        self.optimized_code.push(std::ptr::null());
        self.optimized_osr_entries.push(Vec::new());
        self.optimized_leaf.push(false);
        self.optimized_metadata.push(None);
        self.tier_cells.push(Box::new(TierCell::default()));
        self.promote_retry_at.push(0);
        self.promote_refused.push(false);
        self.speculation_failed.push(false);
        self.method_binding
            .push((std::ptr::null_mut(), std::ptr::null_mut()));
        self.bc_cache.push(std::ptr::null());
        #[cfg(feature = "host")]
        self.threaded_code.push(None); // None = not yet checked
        self.type_profiles.push(None);
        self.aot_state_machine.push(false);
        // Register the bead with a null core pointer; tier decisions
        // are driven entirely from Rust state, beadie just tracks
        // per-function transitions.
        self.tier.register(id, std::ptr::null_mut());
        id
    }

    /// Defining module for a function, if recorded. Callers push new
    /// frames using this rather than propagating the caller's module
    /// so `GetModuleVar` binds against the slots the function was
    /// compiled against.
    pub fn func_module(&self, id: FuncId) -> Option<&Rc<String>> {
        self.func_modules
            .get(id.0 as usize)
            .and_then(|m| m.as_ref())
    }

    /// Register an AOT-compiled function — claim a `FuncId` and pre-fill
    /// `jit_code[id]` with the native pointer so the existing JIT
    /// dispatch path drives the call without ever walking MIR. The
    /// stub `MirFunction` keeps the engine's per-function vectors
    /// (tier_states, jit_metadata, …) in lock-step; runtime helpers
    /// consult the slot via fn_id but never read the empty body.
    pub fn register_aot_function(
        &mut self,
        name: SymbolId,
        arity: u8,
        fn_ptr: *const u8,
        module: Option<Rc<String>>,
    ) -> FuncId {
        let mir = MirFunction::new(name, arity);
        let id = self.register_function_in(mir, module);
        self.jit_code[id.0 as usize] = fn_ptr;
        id
    }

    /// Variant of [`Self::register_aot_function`] that flags `id` as
    /// AOT state-machine-backed. Dispatch reads `aot_state_machine[
    /// id]` to route through the poll-fn invocation path
    /// (`(fiber, resume_v) -> result`) instead of
    /// `wren_call_*`. The fn pointer's signature must already
    /// match — the AOT emit pass picks it based on the same
    /// taint set.
    pub fn register_aot_state_machine_function(
        &mut self,
        name: SymbolId,
        fn_ptr: *const u8,
        module: Option<Rc<String>>,
    ) -> FuncId {
        // arity = 0 from Wren's perspective — fiber bodies are
        // 0-arg blocks; the (fiber, resume_v) signature is an
        // implementation detail of the poll path.
        let id = self.register_aot_function(name, 0, fn_ptr, module);
        self.aot_state_machine[id.0 as usize] = true;
        id
    }

    /// Install a wasm-tier-up dispatch slot for `id`. Stored as
    /// `slot + 1` internally so `0` means "not compiled" and we
    /// can use a plain `Vec<u32>` instead of `Vec<Option<u32>>`.
    /// See `runtime::tier::set_compile_callback`.
    pub fn install_wasm_jit_slot(&mut self, id: FuncId, slot: u32) {
        let idx = id.0 as usize;
        if idx >= self.wasm_jit_slots.len() {
            self.wasm_jit_slots.resize(idx + 1, 0);
        }
        self.wasm_jit_slots[idx] = slot.saturating_add(1);
    }

    /// Look up the wasm-tier-up slot for `id`. `None` if the
    /// function hasn't been JIT-compiled to wasm yet. Always
    /// returns `None` on host builds.
    pub fn wasm_jit_slot(&self, id: FuncId) -> Option<u32> {
        let raw = *self.wasm_jit_slots.get(id.0 as usize)?;
        if raw == 0 {
            None
        } else {
            Some(raw - 1)
        }
    }

    /// Bump the wasm-tier-up call count for `id` and return the
    /// new value. The dispatch hook uses this on every invocation
    /// of an interpreted function to decide when to tier up.
    pub fn bump_wasm_call_count(&mut self, id: FuncId) -> u32 {
        let idx = id.0 as usize;
        if idx >= self.wasm_call_counts.len() {
            self.wasm_call_counts.resize(idx + 1, 0);
        }
        self.wasm_call_counts[idx] = self.wasm_call_counts[idx].saturating_add(1);
        self.wasm_call_counts[idx]
    }
}

/// Threshold at which a Wren function gets compiled to wasm.
/// Aggressive (low) on purpose — wasm-AOT compilation is cheap
/// per-function (a few KB of bytes via `emit_mir` + a sync
/// `WebAssembly.Module` constructor) and the interpreter floor
/// is high enough that even a few-thousand-call function
/// benefits.
pub const WASM_JIT_THRESHOLD: u32 = 50;

impl ExecutionEngine {
    /// Install per-function `NativeFrameMetadata` into the slot
    /// the GC stack walker reads. Used by the AOT install path:
    /// AOT functions are registered via `register_aot_function`
    /// (which pushes a `None` at `jit_metadata[func_id]`); the
    /// follow-up `register_code_range` call needs the metadata
    /// slot populated for `scan_native_stack_roots` to find the
    /// safepoint live-roots when it walks the fp chain.
    ///
    /// JIT functions take a different path: their metadata is
    /// staged in `baseline_metadata` / `optimized_metadata` and
    /// promoted to `jit_metadata` by `set_tier_state`.
    pub fn set_aot_metadata(&mut self, func_id: FuncId, metadata: Arc<NativeFrameMetadata>) {
        let idx = func_id.0 as usize;
        if idx < self.jit_metadata.len() {
            self.jit_metadata[idx] = Some(metadata);
        }
    }

    /// Register a compiled function's code range for GC stack walking.
    pub fn register_code_range(
        &mut self,
        func_id: FuncId,
        code_start: usize,
        code_end: usize,
        metadata: Arc<NativeFrameMetadata>,
    ) {
        self.code_ranges.push(CodeRange {
            start: code_start,
            end: code_end,
            func_id,
            metadata,
        });
        // Keep sorted by start address for binary search.
        self.code_ranges.sort_by_key(|r| r.start);
    }

    /// Find the compiled function containing a given return address.
    /// Used by GC stack walker to look up safepoint metadata.
    pub fn find_code_range(&self, addr: usize) -> Option<&CodeRange> {
        // Binary search: find the last range whose start <= addr
        let idx = self.code_ranges.partition_point(|r| r.start <= addr);
        if idx == 0 {
            return None;
        }
        let range = &self.code_ranges[idx - 1];
        if addr < range.end {
            Some(range)
        } else {
            None
        }
    }

    /// Get the MIR for a function by ID.
    /// Returns an Arc clone so the caller can hold it without borrowing the engine.
    pub fn get_mir(&self, id: FuncId) -> Option<Arc<MirFunction>> {
        self.functions
            .get(id.0 as usize)
            .map(|body| Arc::clone(body.mir()))
    }

    /// Number of registered functions across all modules. Every id in
    /// `0..function_count()` has an entry in `self.functions`.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Get or lazily create threaded code for a function.
    /// Returns a reference to the ThreadedCode if available.
    /// Only creates threaded code for functions where ALL instructions
    /// are supported — otherwise falls through to bytecode interpreter.
    #[cfg(feature = "host")]
    pub fn ensure_threaded_code(
        &mut self,
        id: FuncId,
        interner: &crate::intern::Interner,
    ) -> Option<&crate::mir::threaded::ThreadedCode> {
        let idx = id.0 as usize;
        if idx >= self.threaded_code.len() {
            return None;
        }
        if self.threaded_code[idx].is_none() {
            let mir = self.get_mir(id)?;
            if !crate::mir::threaded::can_use_threaded(&mir, Some(interner)) {
                // Cache negative result so we don't re-check.
                self.threaded_code[idx] = Some(None);
                return None;
            }
            let tc = crate::mir::threaded::lower_mir_to_threaded(&mir, Some(interner));
            self.threaded_code[idx] = Some(Some(tc));
        }
        self.threaded_code[idx]
            .as_ref()
            .and_then(|opt| opt.as_ref())
    }

    /// Get a function body by ID.
    pub fn get_function(&self, id: FuncId) -> Option<&FuncBody> {
        self.functions.get(id.0 as usize)
    }

    /// Peek at already-compiled bytecode without triggering lazy compilation.
    pub fn peek_bytecode(&self, id: FuncId) -> Option<Arc<BytecodeFunction>> {
        match self.functions.get(id.0 as usize)? {
            FuncBody::Interpreted { bytecode, .. } => bytecode.clone(),
            FuncBody::Native { bytecode, .. } => bytecode.clone(),
        }
    }

    /// Get (or lazily lower) the bytecode for a function.
    /// Works for both Interpreted and Compiled functions (bytecode is preserved
    /// across tier-up so the interpreter can continue as a fallback).
    pub fn get_bytecode(&mut self, id: FuncId) -> Option<Arc<BytecodeFunction>> {
        match self.functions.get_mut(id.0 as usize)? {
            FuncBody::Interpreted { mir, bytecode, .. } => {
                if bytecode.is_none() {
                    let bc = crate::mir::bytecode::lower(mir);
                    *bytecode = Some(Arc::new(bc));
                }
                bytecode.clone()
            }
            FuncBody::Native { bytecode, mir, .. } => {
                if bytecode.is_none() {
                    let bc = crate::mir::bytecode::lower(mir);
                    *bytecode = Some(Arc::new(bc));
                }
                bytecode.clone()
            }
        }
    }

    /// Ensure bytecode is compiled and return a raw pointer to it.
    /// The pointer is stable as long as the engine's function table is not modified
    /// (bytecode is never freed once compiled). Use this in the hot interpreter loop
    /// to avoid Arc clone overhead.
    #[inline]
    pub fn ensure_bytecode(&mut self, id: FuncId) -> Option<*const BytecodeFunction> {
        let idx = id.0 as usize;
        // Fast path: check bc_cache first (O(1), no enum match)
        if idx < self.bc_cache.len() {
            let cached = self.bc_cache[idx];
            if !cached.is_null() {
                return Some(cached);
            }
        }
        self.ensure_bytecode_slow(id)
    }

    #[inline(never)]
    fn ensure_bytecode_slow(&mut self, id: FuncId) -> Option<*const BytecodeFunction> {
        let idx = id.0 as usize;
        // Slow path: compile bytecode if needed
        let ptr = match self.functions.get_mut(idx)? {
            FuncBody::Interpreted { mir, bytecode, .. } => {
                if bytecode.is_none() {
                    let bc = crate::mir::bytecode::lower(mir);
                    *bytecode = Some(Arc::new(bc));
                }
                bytecode.as_ref().map(Arc::as_ptr)
            }
            FuncBody::Native { bytecode, mir, .. } => {
                if bytecode.is_none() {
                    let bc = crate::mir::bytecode::lower(mir);
                    *bytecode = Some(Arc::new(bc));
                }
                bytecode.as_ref().map(Arc::as_ptr)
            }
        };
        // Populate bc_cache for future O(1) lookups
        if let Some(p) = ptr {
            if idx >= self.bc_cache.len() {
                self.bc_cache.resize(idx + 1, std::ptr::null());
            }
            self.bc_cache[idx] = p;
        }
        ptr
    }

    /// Snapshot IC entries for compilation AND collect live IC entry pointers.
    /// The snapshot is used by the compiler to decide WHAT to inline (kind=5 getter).
    /// The live pointers are embedded in JIT code as constants for indirect IC:
    /// the generated code loads class/jit_ptr from the live entry at runtime,
    /// so it always uses the most recent IC data without going stale.
    fn callsite_ic_data_for_compile(
        &mut self,
        id: FuncId,
    ) -> Option<(Vec<CallSiteIC>, Vec<usize>)> {
        let bc_ptr = self.ensure_bytecode(id)?;
        let bc = unsafe { &*bc_ptr };
        let ic_table = unsafe { &mut *bc.ic_table.get() };

        // Snapshot for the compiler (used on background thread).
        let snapshot: Vec<CallSiteIC> = ic_table
            .iter()
            .map(|ic| CallSiteIC {
                class: ic.class,
                jit_ptr: ic.jit_ptr,
                closure: ic.closure,
                func_id: ic.func_id,
                kind: ic.kind,
            })
            .collect();
        // Live pointers: address of each IC entry in the live table.
        // These are stable because the Vec doesn't reallocate after creation.
        let live_ptrs: Vec<usize> = ic_table
            .iter_mut()
            .map(|ic| ic as *mut CallSiteIC as usize)
            .collect();
        if tier_trace_enabled() && !snapshot.is_empty() {
            let k5 = snapshot.iter().filter(|ic| ic.kind == 5).count();
            eprintln!(
                "tier-trace: [{:.2}ms] ic_ptrs FuncId({}) total={} kind5={}",
                trace_clock_ms(),
                id.0,
                snapshot.len(),
                k5
            );
        }
        Some((snapshot, live_ptrs))
    }

    /// Compute per-IC-entry devirtualization hints.
    /// For each monomorphic IC entry (kind=1 with a known func_id):
    /// - If the callee is a trivial getter, record the field index
    ///   so Cranelift inlines a direct field load.
    /// - If the callee has no internal calls (pure leaf), mark it so
    ///   Cranelift can emit a pure direct call_indirect with zero FFI.
    fn compute_devirt_hints(&self, ic_snapshot: &[CallSiteIC]) -> Vec<crate::codegen::DevirtHint> {
        ic_snapshot
            .iter()
            .map(|ic| {
                let mut hint = crate::codegen::DevirtHint::default();
                if ic.kind != 1 || ic.class == 0 || ic.func_id == 0 {
                    return hint;
                }
                let callee_id = FuncId(ic.func_id as u32);
                let Some(mir) = self.get_mir(callee_id) else {
                    return hint;
                };
                hint.getter_field = Self::mir_trivial_getter_field(&mir);
                hint.pure_leaf = Self::mir_is_pure_leaf(&mir);
                hint
            })
            .collect()
    }

    /// JIT-side class-hierarchy snapshot. Mirrors AOT's `AotCha`:
    /// each entry maps a method symbol to every
    /// `(class_ptr, FuncId, closure_ptr)` triple that implements
    /// it. Built fresh per compile from the live module-var
    /// table. Class and closure pointers travel to the broker
    /// thread for IC fill — both interp and JIT consume the
    /// resulting IC entries with a class-check guard before
    /// dispatch, so a stale pointer just falls through to the
    /// slow path. Including `closure_ptr` lets the interpreter
    /// dispatch directly from a CHA-planted IC entry instead of
    /// falling back to a method-table lookup.
    pub fn build_jit_cha(&self) -> ChaMap {
        let mut by_method: ChaMap = HashMap::new();
        for entry in self.modules.values() {
            for var in &entry.vars {
                if !var.is_object() {
                    continue;
                }
                let Some(ptr) = var.as_object() else {
                    continue;
                };
                let header = ptr as *const crate::runtime::object::ObjHeader;
                let obj_type = unsafe { (*header).obj_type };
                if obj_type != crate::runtime::object::ObjType::Class {
                    continue;
                }
                let class_ptr = ptr as *mut crate::runtime::object::ObjClass;
                let class = unsafe { &*class_ptr };
                for (idx, slot) in class.methods.iter().enumerate() {
                    let Some(method) = slot else { continue };
                    let closure_ptr = match method {
                        crate::runtime::object::Method::Closure(p) => *p,
                        // Constructors / native / foreign methods need
                        // their own dispatch flavour; only plain
                        // closures are safe to plant as a kind=1 IC.
                        _ => continue,
                    };
                    if closure_ptr.is_null() {
                        continue;
                    }
                    let closure = unsafe { &*closure_ptr };
                    if closure.function.is_null() {
                        continue;
                    }
                    let func_id = unsafe { (*closure.function).fn_id };
                    if let Some(mir) = self.get_mir(FuncId(func_id)) {
                        if Self::mir_uses_defining_class(&mir) {
                            continue;
                        }
                    }
                    let sym = crate::intern::SymbolId::from_raw(idx as u32);
                    by_method.entry(sym).or_default().push((
                        class_ptr as usize,
                        func_id,
                        closure_ptr as usize,
                    ));
                }
            }
        }
        by_method
    }

    fn mir_uses_defining_class(mir: &crate::mir::MirFunction) -> bool {
        use crate::mir::Instruction;
        for block in &mir.blocks {
            for (_, inst) in &block.instructions {
                if matches!(
                    inst,
                    Instruction::SuperCall { .. }
                        | Instruction::GetStaticField(_)
                        | Instruction::SetStaticField(_, _)
                ) {
                    return true;
                }
            }
        }
        false
    }

    /// Eligibility for the trivial-method inliner. The body is
    /// inlinable when it's a single block, fits in a small budget,
    /// and contains only ops the codegen can lower without spawning
    /// helpers, allocations, or further dispatch. The class-check
    /// guard wraps the inlined body at the call site, so a polymorphic
    /// receiver still falls back through `wren_call_N`.
    pub fn mir_inlinable_single_block(mir: &crate::mir::MirFunction) -> bool {
        use crate::mir::{Instruction, Terminator};
        if mir.blocks.len() != 1 {
            return false;
        }
        let block = &mir.blocks[0];
        let n_real = block
            .instructions
            .iter()
            .filter(|(_, inst)| !matches!(inst, Instruction::BlockParam(_)))
            .count();
        // Small enough that splicing it at every monomorphic site
        // costs less than the call it replaces; a straight-line
        // arithmetic method with a few temporaries fits.
        if n_real > 32 {
            return false;
        }
        for (_, inst) in &block.instructions {
            match inst {
                // Disallowed: anything that needs dispatch, allocation,
                // module/static/upvalue context, or a helper call. The
                // inliner emits the body directly into the caller's
                // Cranelift function with no surrounding frame.
                Instruction::Call { .. }
                | Instruction::CallKnownFunc { .. }
                | Instruction::CallStaticSelf { .. }
                | Instruction::SuperCall { .. }
                | Instruction::GetModuleVar(_)
                | Instruction::SetModuleVar(_, _)
                | Instruction::GetStaticField(_)
                | Instruction::SetStaticField(_, _)
                | Instruction::GetUpvalue(_)
                | Instruction::SetUpvalue(_, _)
                | Instruction::MakeList(_)
                | Instruction::MakeMap(_)
                | Instruction::MakeRange(..)
                | Instruction::MakeClosure { .. }
                | Instruction::StringConcat(_)
                | Instruction::ToString(_)
                | Instruction::SubscriptGet { .. }
                | Instruction::SubscriptSet { .. }
                | Instruction::GuardNum(_) => return false,
                _ => {}
            }
        }
        matches!(
            block.terminator,
            Terminator::Return(_) | Terminator::ReturnNull
        )
    }

    /// Build a `func_id → MIR` table of every callee whose body is
    /// safe to inline at a JIT call site. Used by the codegen to
    /// substitute the body in place of a `wren_known_call_N` helper
    /// when the speculative class check matches.
    pub fn compute_inline_bodies(
        &self,
    ) -> Arc<std::collections::HashMap<u32, Arc<crate::mir::MirFunction>>> {
        let mut map: std::collections::HashMap<u32, Arc<crate::mir::MirFunction>> =
            std::collections::HashMap::new();
        for (idx, body) in self.functions.iter().enumerate() {
            let mir = body.mir();
            if Self::mir_inlinable_single_block(mir) && !Self::mir_uses_defining_class(mir) {
                map.insert(idx as u32, Arc::clone(mir));
            }
        }
        Arc::new(map)
    }

    /// Mutate `ic_snapshot` in place: for any empty entry, plant a
    /// kind=1 IC when CHA shows exactly one impl for the call's
    /// method symbol. The downstream `compute_devirt_hints` +
    /// `devirt_calls_with_ic` then converts those Call sites into
    /// class-checked direct calls (or trivial-getter inlines).
    /// Walks MIR Call/SuperCall in the same order
    /// `callsite_ic_data_for_compile` produces the snapshot.
    ///
    /// Polymorphic methods (multiple `(class, fn)` impls in CHA)
    /// stay empty: planting an arbitrary first impl turns class-
    /// check misses into a wren_call_N detour for every receiver
    /// that doesn't match, which on benchmarks like delta_blue
    /// regresses 3-5%. A multi-class dispatch tree (mirroring
    /// AOT-CHA's emit) is the proper fix and will land separately.
    /// Re-order a bytecode-ordered inline-cache snapshot to the call
    /// order of an optimised MIR. Both the bytecode emitter and the
    /// backends number call sites by walking blocks in order, but the
    /// JIT compiles a clone whose passes drop, add and move calls, so
    /// positions no longer agree. A call's destination value id does
    /// survive optimisation, so entries are matched on it; calls the
    /// optimiser introduced get an empty entry.
    fn align_callsite_ics(
        authoritative: &MirFunction,
        optimised: &MirFunction,
        ics: Vec<CallSiteIC>,
        live: Vec<usize>,
        hints: Option<Vec<crate::codegen::DevirtHint>>,
    ) -> (
        Vec<CallSiteIC>,
        Vec<usize>,
        Option<Vec<crate::codegen::DevirtHint>>,
    ) {
        use crate::mir::Instruction;
        let mut by_dst: HashMap<crate::mir::ValueId, usize> = HashMap::new();
        let mut idx = 0usize;
        for block in &authoritative.blocks {
            for (dst, inst) in &block.instructions {
                if matches!(
                    inst,
                    Instruction::Call { .. } | Instruction::SuperCall { .. }
                ) {
                    by_dst.insert(*dst, idx);
                    idx += 1;
                }
            }
        }
        let mut out_ics = Vec::new();
        let mut out_live = Vec::new();
        let mut out_hints = hints.as_ref().map(|_| Vec::new());
        for block in &optimised.blocks {
            for (dst, inst) in &block.instructions {
                if matches!(
                    inst,
                    Instruction::Call { .. } | Instruction::SuperCall { .. }
                ) {
                    match by_dst.get(dst) {
                        Some(&i) if i < ics.len() => {
                            out_ics.push(ics[i]);
                            out_live.push(live.get(i).copied().unwrap_or(0));
                            if let (Some(out), Some(h)) = (out_hints.as_mut(), hints.as_ref()) {
                                out.push(h.get(i).copied().unwrap_or_default());
                            }
                        }
                        _ => {
                            out_ics.push(CallSiteIC::default());
                            out_live.push(0);
                            if let Some(out) = out_hints.as_mut() {
                                out.push(Default::default());
                            }
                        }
                    }
                }
            }
        }
        (out_ics, out_live, out_hints)
    }

    fn fill_ic_with_cha(
        &self,
        mir: &crate::mir::MirFunction,
        ic_snapshot: &mut [CallSiteIC],
        cha: &ChaMap,
    ) {
        use crate::mir::Instruction;
        let mut ic_idx = 0usize;
        for block in &mir.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Call { method, .. } => {
                        if let Some(slot) = ic_snapshot.get_mut(ic_idx) {
                            if slot.kind == 0 {
                                if let Some(impls) = cha.get(method) {
                                    if impls.len() == 1 {
                                        let (class_ptr, fid, closure_ptr) = impls[0];
                                        slot.class = class_ptr;
                                        slot.func_id = fid as u64;
                                        slot.closure = closure_ptr as *const u8;
                                        slot.kind = 1;
                                    }
                                }
                            }
                        }
                        ic_idx += 1;
                    }
                    Instruction::SuperCall { .. } => {
                        ic_idx += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Compute a per-function "no observable side effects" map keyed
    /// by `FuncId.0`. Used by the post-devirt CSE pass in codegen so
    /// `CallKnownFunc { func_id }` calls into pure user methods don't
    /// flush the memory-read cache.
    ///
    /// Cached on the engine — `delta_blue` submits 70+ JIT compiles
    /// during a run, and recomputing the map for every submission is
    /// O(N²). Invalidates when the function table grows; new
    /// installations (hot reload, late-bound classes) bump
    /// `functions.len()` so `cached_purity_map_len != self.functions.len()`
    /// triggers a refresh.
    ///
    /// In-place MIR rewrites for already-installed functions do *not*
    /// invalidate the cache. That's safe so far because the only
    /// post-install rewrite — JIT-time devirt — runs on a *clone* of
    /// the MIR; the engine's authoritative copy is unchanged.
    pub fn compute_callee_purity_map(&mut self) -> Arc<std::collections::HashMap<u32, bool>> {
        self.refresh_purity_caches();
        Arc::clone(self.cached_purity_map.as_ref().expect("just refreshed"))
    }

    /// O(1) lookup: "the function with this id can't fire a GC during
    /// its body". Used by the IC kind=1 inline JIT-leaf fast path so
    /// register-passed args stay valid without JIT-frame stack maps.
    /// Returns false (conservative) for ids past the cached vec.
    pub fn func_is_alloc_free(&self, func_id: u32) -> bool {
        self.cached_alloc_free_vec
            .get(func_id as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Refresh the purity / alloc-free caches if the function table
    /// has grown since the last build. Both caches share the same
    /// invalidation key (`functions.len()`) so a single check covers
    /// both. In-place MIR rewrites for already-installed functions
    /// don't invalidate; the engine's authoritative copy stays
    /// unchanged because JIT-time devirt operates on a clone.
    pub fn refresh_purity_caches(&mut self) {
        if self.cached_purity_map_len == self.functions.len() && self.cached_purity_map.is_some() {
            return;
        }
        let funcs: Vec<(u32, &crate::mir::MirFunction)> = self
            .functions
            .iter()
            .enumerate()
            .map(|(idx, body)| (idx as u32, body.mir().as_ref()))
            .collect();
        let purity = Arc::new(crate::mir::opt::purity::compute_purity_map(
            funcs.iter().copied(),
        ));
        let alloc_free = Arc::new(crate::mir::opt::purity::compute_alloc_free_map(
            funcs.iter().copied(),
        ));
        let n = self.functions.len();
        self.cached_alloc_free_vec.clear();
        self.cached_alloc_free_vec.reserve(n);
        for i in 0..n {
            self.cached_alloc_free_vec
                .push(alloc_free.get(&(i as u32)).copied().unwrap_or(false));
        }
        self.cached_purity_map = Some(purity);
        self.cached_alloc_free_map = Some(alloc_free);
        self.cached_purity_map_len = n;
    }

    /// Build (or refresh) the cached set of method-name symbols whose
    /// implementation transitively touches a JIT-unsafe fiber control
    /// method. Seeded with the direct names, then iterated to a
    /// fixed point: a function `f` named `m` is added to the set
    /// whenever its MIR contains a `Call` whose method is already
    /// in the set. Cache lives in a thread_local so the
    /// `ExecutionEngine` struct layout stays binary-compatible with
    /// already-loaded native plugins (see `MAY_YIELD_CACHE`).
    pub fn compute_may_yield_methods(
        &self,
        interner: &crate::intern::Interner,
    ) -> Arc<std::collections::HashSet<crate::intern::SymbolId>> {
        let engine_id = self as *const ExecutionEngine as usize;
        let len = self.functions.len();
        // Fast path: cache hit on both engine identity and table length.
        if let Some(arc) = MAY_YIELD_CACHE.with(|cell| {
            let c = cell.borrow();
            if c.last_engine == engine_id && c.last_len == len {
                c.set.as_ref().map(Arc::clone)
            } else {
                None
            }
        }) {
            return arc;
        }

        use std::collections::HashSet;
        let mut tainted: HashSet<crate::intern::SymbolId> = HashSet::new();
        for n in direct_yield_method_names() {
            if let Some(sym) = interner.lookup(n) {
                tainted.insert(sym);
            }
        }
        // Fixed-point iteration. Each pass marks any function whose
        // body calls a tainted method as itself tainted (under its
        // mir.name). Bound by the function count + 1 so we always
        // terminate even if every iteration adds exactly one new
        // entry.
        let bound = len.saturating_add(1);
        for _ in 0..bound {
            let mut changed = false;
            for body in &self.functions {
                let name_sym = body.mir().name;
                if tainted.contains(&name_sym) {
                    continue;
                }
                if mir_calls_any_tainted_method(body.mir(), &tainted) {
                    tainted.insert(name_sym);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let arc = Arc::new(tainted);
        MAY_YIELD_CACHE.with(|cell| {
            let mut c = cell.borrow_mut();
            c.last_engine = engine_id;
            c.last_len = len;
            c.set = Some(Arc::clone(&arc));
        });
        arc
    }

    /// A "pure leaf" function has no internal method calls — Cranelift
    /// can emit a zero-FFI direct `call_indirect` because the callee
    /// doesn't need its own JIT context (current_func_id, etc.).
    fn mir_is_pure_leaf(mir: &crate::mir::MirFunction) -> bool {
        use crate::mir::Instruction;
        for block in &mir.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Call { .. }
                    | Instruction::CallKnownFunc { .. }
                    | Instruction::CallStaticSelf { .. }
                    | Instruction::SuperCall { .. } => return false,
                    // Runtime calls that require context (module_vars, etc.)
                    // — any of these prevent pure_leaf status.
                    Instruction::GetModuleVar(_)
                    | Instruction::SetModuleVar(_, _)
                    | Instruction::GetStaticField(_)
                    | Instruction::SetStaticField(_, _)
                    | Instruction::GetUpvalue(_)
                    | Instruction::SetUpvalue(_, _)
                    | Instruction::MakeList(_)
                    | Instruction::MakeMap(_)
                    | Instruction::MakeRange { .. }
                    | Instruction::MakeClosure { .. }
                    | Instruction::StringConcat(_)
                    | Instruction::ToString(_)
                    | Instruction::SubscriptGet { .. }
                    | Instruction::SubscriptSet { .. } => return false,
                    _ => {}
                }
            }
        }
        true
    }

    /// Check if a MIR function is a trivial getter: `get_field this, #N; return`.
    pub fn mir_trivial_getter_field(mir: &crate::mir::MirFunction) -> Option<u16> {
        use crate::mir::{Instruction, Terminator};
        if mir.blocks.len() != 1 {
            return None;
        }
        let block = &mir.blocks[0];
        let mut self_param = None;
        let mut getter = None;
        for (vid, inst) in &block.instructions {
            match inst {
                Instruction::BlockParam(0) if self_param.is_none() => self_param = Some(*vid),
                Instruction::GetField(recv, idx)
                    if getter.is_none() && Some(*recv) == self_param =>
                {
                    getter = Some((*vid, *idx));
                }
                _ => return None,
            }
        }
        match (getter, &block.terminator) {
            (Some((ret_vid, field_idx)), Terminator::Return(v)) if *v == ret_vid => Some(field_idx),
            _ => None,
        }
    }

    /// Check if a MIR function is a trivial setter: `set_field this, #N, value; return`.
    fn mir_trivial_setter_field(mir: &crate::mir::MirFunction) -> Option<u16> {
        use crate::mir::{Instruction, Terminator};
        if mir.blocks.len() != 1 || mir.arity != 2 {
            return None;
        }
        let block = &mir.blocks[0];
        let mut self_param = None;
        let mut value_param = None;
        let mut value_alias = None;
        let mut setter = None;
        for (vid, inst) in &block.instructions {
            match inst {
                Instruction::BlockParam(0) if self_param.is_none() => self_param = Some(*vid),
                Instruction::BlockParam(1) if value_param.is_none() => value_param = Some(*vid),
                Instruction::Move(src) if Some(*src) == value_param && value_alias.is_none() => {
                    value_alias = Some(*vid);
                }
                Instruction::SetField(recv, idx, val)
                    if setter.is_none()
                        && Some(*recv) == self_param
                        && (Some(*val) == value_param || Some(*val) == value_alias) =>
                {
                    setter = Some((*vid, *idx));
                }
                _ => return None,
            }
        }
        match (setter, &block.terminator) {
            (Some((ret_vid, field_idx)), Terminator::Return(v)) if *v == ret_vid => Some(field_idx),
            _ => None,
        }
    }

    /// Whether any background compilations are waiting to be installed.
    /// Beadie's broker owns its own queue internally, so all the engine
    /// needs to know is whether there are compiles whose results haven't
    /// yet been drained from the install-side mpsc channel.
    #[inline(always)]
    pub fn has_pending_compilations(&self) -> bool {
        self.pending_count != 0
    }

    /// Invalidate all monomorphic call-site inline caches.
    ///
    /// This must run after moving GC because cached class/closure pointers
    /// may have been relocated.
    /// Invalidate IC entries that reference a specific function.
    /// Called after recompilation so stale jit_ptr/closure entries get refreshed.
    fn invalidate_ic_entries_for(&mut self, target: FuncId) {
        let target_id = target.0 as u64;
        for body in &self.functions {
            let bytecode = match body {
                FuncBody::Interpreted { bytecode, .. } | FuncBody::Native { bytecode, .. } => {
                    bytecode.as_ref()
                }
            };
            let Some(bytecode) = bytecode else { continue };
            let ic_table = unsafe { &mut *bytecode.ic_table.get() };
            for entry in ic_table.iter_mut() {
                if entry.func_id == target_id {
                    *entry = CallSiteIC::default();
                }
            }
        }
    }

    pub fn invalidate_inline_caches(&mut self) {
        for body in &self.functions {
            let bytecode = match body {
                FuncBody::Interpreted { bytecode, .. } | FuncBody::Native { bytecode, .. } => {
                    bytecode.as_ref()
                }
            };
            let Some(bytecode) = bytecode else {
                continue;
            };
            let ic_table = unsafe { &mut *bytecode.ic_table.get() };
            for entry in ic_table.iter_mut() {
                *entry = CallSiteIC::default();
            }
        }
    }

    #[cfg(feature = "host")]
    fn sync_active_tier_cache(&mut self, idx: usize) {
        let state = self
            .tier_states
            .get(idx)
            .copied()
            .unwrap_or(TierState::Interpreted);
        // The active OSR table lives on the beadie bead; when a tier
        // switch revives a previously-compiled tier (e.g. deopt back
        // to baseline) we re-publish that tier's stashed OSR entries
        // via `install_or_swap_osr` so the bead's table matches the
        // code pointer we just swapped in.
        let id = FuncId(idx as u32);
        match state {
            TierState::Interpreted => {
                self.jit_code[idx] = std::ptr::null();
                self.jit_leaf[idx] = false;
                self.jit_metadata[idx] = None;
                // Bead's OSR table is cleared by `reload()` / `blacklist()`
                // which the deopt path in record_bailout already drives.
            }
            TierState::BaselineNative => {
                self.jit_code[idx] = self.baseline_code[idx];
                self.jit_leaf[idx] = self.baseline_leaf[idx];
                self.jit_metadata[idx] = self.baseline_metadata[idx].clone();
                if !self.baseline_code[idx].is_null() {
                    let entries = encode_osr_entries(&self.baseline_osr_entries[idx]);
                    self.tier
                        .install_or_swap_osr(id, self.baseline_code[idx] as *mut (), entries);
                }
            }
            TierState::OptimizedNative => {
                self.jit_code[idx] = self.optimized_code[idx];
                self.jit_leaf[idx] = self.optimized_leaf[idx];
                self.jit_metadata[idx] = self.optimized_metadata[idx].clone();
                if !self.optimized_code[idx].is_null() {
                    let entries = encode_osr_entries(&self.optimized_osr_entries[idx]);
                    self.tier
                        .install_or_swap_osr(id, self.optimized_code[idx] as *mut (), entries);
                }
            }
        }
    }

    /// The installed OSR entry for `target_block`, if any. Candidates
    /// come from the compiled tiers' descriptors (which carry the
    /// live-in register list); beadie's table decides which one is
    /// active by pointer.
    pub fn active_osr_entry(
        &self,
        id: FuncId,
        target_block: crate::mir::BlockId,
    ) -> Option<NativeOsrEntry> {
        let idx = id.0 as usize;
        let candidates = self
            .optimized_osr_entries
            .get(idx)
            .into_iter()
            .chain(self.baseline_osr_entries.get(idx))
            .flat_map(|v| v.iter())
            .filter(|e| e.target_block == target_block);
        for entry in candidates {
            let site = super::tier::encode_osr_site(target_block.0, entry.param_count);
            let Some(ptr) = self.tier.osr_entry(id, site) else {
                if tier_trace_enabled() {
                    eprintln!(
                        "tier-trace: [{:.2}ms] osr site bb{} params={} not in bead table for FuncId({})", trace_clock_ms(),
                        target_block.0, entry.param_count, id.0
                    );
                }
                continue;
            };
            if ptr.is_null() || ptr as *const u8 != entry.ptr {
                if tier_trace_enabled() {
                    eprintln!(
                        "tier-trace: [{:.2}ms] osr site bb{} pointer mismatch for FuncId({})",
                        trace_clock_ms(),
                        target_block.0,
                        id.0
                    );
                }
                continue;
            }
            return Some(NativeOsrEntry {
                target_block,
                param_count: entry.param_count,
                ptr: entry.ptr,
                live_in_regs: entry.live_in_regs.clone(),
                live_in_num: entry.live_in_num.clone(),
                live_in_field: entry.live_in_field.clone(),
                live_in_int: entry.live_in_int.clone(),
            });
        }
        None
    }

    pub fn tier_state(&self, id: FuncId) -> TierState {
        self.tier_states
            .get(id.0 as usize)
            .copied()
            .unwrap_or(TierState::Interpreted)
    }

    pub fn note_interpreted_entry(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(id.0 as usize) {
            stats.interpreted_entries += 1;
        }
    }

    pub fn note_native_entry(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        let idx = id.0 as usize;
        if let Some(stats) = self.tier_stats.get_mut(idx) {
            match self
                .tier_states
                .get(idx)
                .copied()
                .unwrap_or(TierState::Interpreted)
            {
                TierState::BaselineNative => stats.baseline_entries += 1,
                TierState::OptimizedNative => stats.optimized_entries += 1,
                TierState::Interpreted => {}
            }
        }
    }

    pub fn note_native_to_native_call(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(id.0 as usize) {
            stats.native_to_native_calls += 1;
        }
    }

    pub fn note_osr_entry(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(id.0 as usize) {
            stats.osr_entries += 1;
        }
    }

    pub fn note_fallback_to_interpreter(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(id.0 as usize) {
            stats.fallbacks_to_interpreter += 1;
        }
    }

    pub fn note_deopt_to_baseline(&mut self, id: FuncId) {
        let idx = id.0 as usize;
        if idx >= self.tier_states.len() {
            return;
        }
        self.tier_states[idx] = if self.baseline_code[idx].is_null() {
            TierState::Interpreted
        } else {
            TierState::BaselineNative
        };
        #[cfg(feature = "host")]
        self.sync_active_tier_cache(idx);
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(idx) {
            stats.deopts_to_baseline += 1;
        }
    }

    /// A speculative guard failed in `id`'s compiled code. The function
    /// runs its baseline until a compile without speculation replaces
    /// it; the top tier may be proposed again for the unspeculated body.
    #[cfg(feature = "host")]
    pub fn note_speculation_failed(&mut self, id: FuncId, interner: &crate::intern::Interner) {
        let idx = id.0 as usize;
        if idx >= self.functions.len() || self.speculation_failed[idx] {
            return;
        }
        self.speculation_failed[idx] = true;
        if tier_trace_enabled() {
            eprintln!(
                "tier-trace: [{:.2}ms] speculation failed FuncId({}), recompiling",
                trace_clock_ms(),
                id.0
            );
        }
        if let Some(FuncBody::Native {
            optimized_executable,
            ..
        }) = self.functions.get_mut(idx)
        {
            if let Some(old) = optimized_executable.take() {
                self.retired_code.push(old);
            }
        }
        self.optimized_code[idx] = std::ptr::null();
        self.optimized_osr_entries[idx].clear();
        self.baseline_code[idx] = std::ptr::null();
        self.baseline_osr_entries[idx].clear();
        self.tier_states[idx] = TierState::Interpreted;
        self.sync_active_tier_cache(idx);
        self.invalidate_ic_entries_for(id);
        self.tier_cells[idx]
            .retier
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.promote_retry_at[idx] = 0;
        if self.compiling_tier[idx].is_none() {
            self.request_compile(id, CompileTier::Baseline, interner);
        }
    }

    pub fn note_ic_hit(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(id.0 as usize) {
            stats.ic_hits += 1;
        }
    }

    pub fn note_ic_miss(&mut self, id: FuncId) {
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(id.0 as usize) {
            stats.ic_misses += 1;
        }
    }

    #[inline(always)]
    pub fn note_runtime_call_stats(&mut self, f: impl FnOnce(&mut RuntimeCallStats)) {
        if !self.collect_tier_stats {
            return;
        }
        f(&mut self.runtime_call_stats);
    }

    pub fn dump_tier_stats(&self, interner: &crate::intern::Interner) {
        let runtime = self.runtime_call_stats;
        if runtime != RuntimeCallStats::default() {
            eprintln!("=== WLIFT runtime call stats ===");
            eprintln!(
                "wren_call={} noframe_fastpath={} dispatch_call={} fn_fastpath={} list_native_fastpath={} ic_attempts={} ic_class_misses={} method_cache={}/{}",
                runtime.wren_call_entries,
                runtime.wren_call_noframe_fastpath,
                runtime.dispatch_call_entries,
                runtime.dispatch_call_fn_fastpath,
                runtime.dispatch_call_list_native_fastpath,
                runtime.dispatch_call_ic_attempts,
                runtime.dispatch_call_ic_class_misses,
                runtime.dispatch_call_method_cache_hits,
                runtime.dispatch_call_method_cache_misses,
            );
            eprintln!(
                "dispatch_method native={} closure={} ctor={} trivial_getter={} trivial_setter={} call_closure={} native_candidates={} native_entries={} interp_fallbacks={} ctx_save_restore={}",
                runtime.dispatch_method_native,
                runtime.dispatch_method_closure,
                runtime.dispatch_method_constructor,
                runtime.dispatch_method_trivial_getter,
                runtime.dispatch_method_trivial_setter,
                runtime.call_closure_entries,
                runtime.call_closure_native_candidates,
                runtime.call_closure_native_entries,
                runtime.call_closure_interpreter_fallbacks,
                runtime.jit_context_save_restore_pairs,
            );
            eprintln!(
                "ic_hits kind1={} kind2={} kind3={} kind4={} kind5={} kind6={} invalidations={}",
                runtime.ic_kind1_hits,
                runtime.ic_kind2_hits,
                runtime.ic_kind3_hits,
                runtime.ic_kind4_hits,
                runtime.ic_kind5_hits,
                runtime.ic_kind6_hits,
                runtime.ic_invalidations,
            );
        }
        eprintln!("=== WLIFT tier stats ===");
        for (idx, body) in self.functions.iter().enumerate() {
            let stats = self.tier_stats.get(idx).copied().unwrap_or_default();
            if stats == FuncTierStats::default() {
                continue;
            }
            let name = interner.resolve(body.mir().name);
            eprintln!(
                "FuncId({idx}) {name} tier={:?} interp={} baseline={} opt={} compiles={}/{} ic={}/{} native2native={} osr={} deopts={} fallbacks={}",
                self.tier_states.get(idx).copied().unwrap_or(TierState::Interpreted),
                stats.interpreted_entries,
                stats.baseline_entries,
                stats.optimized_entries,
                stats.compile_successes,
                stats.compile_attempts,
                stats.ic_hits,
                stats.ic_misses,
                stats.native_to_native_calls,
                stats.osr_entries,
                stats.deopts_to_baseline,
                stats.fallbacks_to_interpreter,
            );
        }
    }

    fn next_compile_tier(&self, idx: usize) -> Option<CompileTier> {
        if self.mode != ExecutionMode::Tiered && self.mode != ExecutionMode::Jit {
            return None;
        }
        match self.functions.get(idx)? {
            FuncBody::Interpreted { .. } => Some(CompileTier::Baseline),
            FuncBody::Native {
                optimized_executable,
                ..
            } if optimized_executable.is_none() => Some(CompileTier::Optimized),
            _ => None,
        }
    }

    /// Address of the module variable cell for `id`'s defining module,
    /// or 0 when the module is not recorded.
    fn modvars_cell_addr(&self, id: FuncId) -> usize {
        let idx = id.0 as usize;
        if let Some(&addr) = self.func_modvars_cell.borrow().get(idx) {
            if addr != 0 {
                return addr;
            }
        }
        let addr = self
            .func_modules
            .get(idx)
            .and_then(|m| m.as_ref())
            .and_then(|name| self.modules.get(name.as_str()))
            .map(|e| e.cell as *const ModuleVarsCell as usize)
            .unwrap_or(0);
        if addr != 0 {
            let mut cache = self.func_modvars_cell.borrow_mut();
            if cache.len() <= idx {
                cache.resize(idx + 1, 0);
            }
            cache[idx] = addr;
        }
        addr
    }

    /// The module variable table of `id`'s module as (pointer, length),
    /// read from the module's cell without a name lookup.
    #[inline]
    pub fn module_vars_for(&self, id: FuncId) -> (*mut u64, u32) {
        use std::sync::atomic::Ordering;
        let addr = self.modvars_cell_addr(id);
        if addr == 0 {
            return (std::ptr::null_mut(), 0);
        }
        let cell = unsafe { &*(addr as *const ModuleVarsCell) };
        let len = cell.len.load(Ordering::Acquire);
        let ptr = cell.ptr.load(Ordering::Acquire);
        (ptr, len as u32)
    }

    /// Shape of the class held by module variable `idx` of `module`
    /// for scalar replacement, or None when the class is not eligible:
    /// the slot is reassigned somewhere in the module, the value is not
    /// a class, or the class has no trivial constructor.
    fn scalar_class_for_modvar(
        &self,
        interner: &crate::intern::Interner,
        module: &str,
        idx: u32,
    ) -> Option<Arc<crate::mir::opt::sroa_loop::ScalarClass>> {
        use crate::mir::opt::sroa_loop::{
            trivial_ctor_field_map, trivial_getter_field, ScalarClass,
        };
        use crate::mir::Instruction;
        use crate::runtime::object::{Method, ObjClass, ObjHeader, ObjType};
        // A class slot is written once by the VM at install; any
        // SetModuleVar on it in the module's own code makes the baked
        // class unsound.
        for (fid, m) in self.func_modules.iter().enumerate() {
            let Some(m) = m else { continue };
            if m.as_str() != module {
                continue;
            }
            let Some(body) = self.functions.get(fid) else {
                continue;
            };
            for block in &body.mir().blocks {
                for (_, inst) in &block.instructions {
                    if let Instruction::SetModuleVar(slot, _) = inst {
                        if *slot as u32 == idx {
                            return None;
                        }
                    }
                }
            }
        }
        let entry = self.modules.get(module)?;
        let value = *entry.vars.get(idx as usize)?;
        if std::env::var_os("WLIFT_SROA_TRACE").is_some() {
            eprintln!(
                "sroa-trace: module {} slot {} is_object={}",
                module,
                idx,
                value.is_object()
            );
        }
        let ptr = value.as_object()?;
        let header = ptr as *const ObjHeader;
        if unsafe { (*header).obj_type } != ObjType::Class {
            return None;
        }
        let class = unsafe { &*(ptr as *const ObjClass) };
        let num_fields = class.num_fields as usize;
        let mut ctors = std::collections::HashMap::new();
        let mut getters = std::collections::HashMap::new();
        for (sym_idx, slot) in class.methods.iter().enumerate() {
            let Some(method) = slot else { continue };
            let sym = crate::intern::SymbolId::from_raw(sym_idx as u32);
            match method {
                Method::Constructor(closure) => {
                    if closure.is_null() {
                        continue;
                    }
                    let fn_id = unsafe { (*(**closure).function).fn_id };
                    let Some(mir) = self.get_mir(FuncId(fn_id)) else {
                        continue;
                    };
                    let Some(map) = trivial_ctor_field_map(&mir) else {
                        continue;
                    };
                    let nargs = mir.arity.saturating_sub(1) as usize;
                    let mut per_arg: Vec<Option<usize>> = vec![None; nargs];
                    let mut ok = true;
                    for (field, arg) in map {
                        if field >= num_fields || arg >= nargs {
                            ok = false;
                            break;
                        }
                        per_arg[arg] = Some(field);
                    }
                    // Constructors are bound under `static:<sig>`; call
                    // sites use the plain signature.
                    let plain = interner
                        .resolve(sym)
                        .strip_prefix("static:")
                        .and_then(|p| interner.lookup(p))
                        .unwrap_or(sym);
                    if ok {
                        ctors.insert(plain, per_arg);
                    }
                }
                Method::Closure(closure) => {
                    if closure.is_null() {
                        continue;
                    }
                    let fn_id = unsafe { (*(**closure).function).fn_id };
                    let Some(mir) = self.get_mir(FuncId(fn_id)) else {
                        continue;
                    };
                    if let Some(field) = trivial_getter_field(&mir) {
                        if field < num_fields {
                            getters.insert(sym, field);
                        }
                    }
                }
                _ => {}
            }
        }
        if std::env::var_os("WLIFT_SROA_TRACE").is_some() {
            eprintln!(
                "sroa-trace: class at slot {} fields={} ctors={} getters={}",
                idx,
                num_fields,
                ctors.len(),
                getters.len()
            );
        }
        if ctors.is_empty() {
            return None;
        }
        Some(Arc::new(ScalarClass {
            num_fields,
            ctors,
            getters,
        }))
    }

    /// Clone of `mir` with loop-carried objects of trivial classes
    /// replaced by scalars, or the original when nothing applies.
    fn scalar_replaced(
        &self,
        id: FuncId,
        mir: &Arc<MirFunction>,
        interner: &crate::intern::Interner,
    ) -> Arc<MirFunction> {
        if std::env::var_os("WLIFT_DISABLE_SROA").is_some() {
            return Arc::clone(mir);
        }
        let Some(module) = self.func_modules.get(id.0 as usize).and_then(|m| m.clone()) else {
            return Arc::clone(mir);
        };
        let resolver = |idx: u32| self.scalar_class_for_modvar(interner, module.as_str(), idx);
        let mut clone = (**mir).clone();
        if std::env::var_os("WLIFT_SROA_TRACE").is_some() {
            eprintln!("sroa-trace: FuncId({}) module {}", id.0, module);
        }
        if crate::mir::opt::sroa_loop::scalar_replace_loop_objects(&mut clone, &resolver) {
            Arc::new(clone)
        } else {
            Arc::clone(mir)
        }
    }

    /// Callees the inline cache resolved for the caller's call sites,
    /// keyed by the call's destination value, restricted to bodies the
    /// MIR inliner may splice: same module (or no module-variable
    /// access), and nothing the JIT refuses to compile.
    fn known_call_sites(
        &self,
        caller: FuncId,
        mir: &MirFunction,
        ics: &[CallSiteIC],
        interner: &crate::intern::Interner,
    ) -> HashMap<crate::mir::ValueId, crate::mir::opt::inline_calls::KnownCallee> {
        use crate::mir::opt::inline_calls::{CalleeGuard, KnownCallee};
        use crate::mir::Instruction;
        let mut sites = HashMap::new();
        let tainted = self.compute_may_yield_methods(interner);
        let caller_module = self.func_modules.get(caller.0 as usize).cloned().flatten();
        let mut ic_idx = 0usize;
        for block in &mir.blocks {
            for (dst, inst) in &block.instructions {
                let method = match inst {
                    Instruction::Call { method, .. } => Some(*method),
                    Instruction::SuperCall { .. } => None,
                    _ => continue,
                };
                let ic = ics.get(ic_idx).copied();
                ic_idx += 1;
                let (Some(method), Some(ic)) = (method, ic) else {
                    continue;
                };
                if std::env::var_os("WLIFT_INLINE_TRACE").is_some() {
                    eprintln!(
                        "inline-trace: FuncId({}) site {} kind={} class={:#x} func_id={}",
                        caller.0, dst, ic.kind, ic.class, ic.func_id
                    );
                }
                if ic.class == 0 {
                    continue;
                }
                let callee = FuncId(ic.func_id as u32);
                if callee == caller {
                    continue;
                }
                let guard = match ic.kind {
                    7 => CalleeGuard::ClosureFn(ic.class),
                    // Method kinds use function id 0 as "unset".
                    1 | 2 | 6 if ic.func_id != 0 => {
                        // A class receiving a static call is cached under its
                        // own pointer, so the guard is identity.
                        let class = ic.class as *const crate::runtime::object::ObjClass;
                        let name = interner.resolve(method).to_string();
                        let static_sym = interner.lookup(&format!("static:{}", name));
                        let is_static = static_sym
                            .and_then(|sym| unsafe { (*class).find_method(sym) })
                            .map(|m| match m {
                                crate::runtime::object::Method::Closure(c) => {
                                    let fn_id = unsafe { (*(**c).function).fn_id };
                                    fn_id == ic.func_id as u32
                                }
                                _ => false,
                            })
                            .unwrap_or(false);
                        if is_static {
                            CalleeGuard::Object(ic.class)
                        } else {
                            CalleeGuard::Class(ic.class)
                        }
                    }
                    _ => continue,
                };
                let Some(body) = self.get_mir(callee) else {
                    continue;
                };
                if !crate::mir::opt::inline_calls::inlinable_body(&body) {
                    continue;
                }
                // The backend already splices small bodies behind a guard;
                // MIR inlining earns its keep only where the caller's types
                // can reach arithmetic in the body.
                if !crate::mir::opt::inline_calls::body_has_arithmetic(&body) {
                    continue;
                }
                let callee_module = self.func_modules.get(callee.0 as usize).cloned().flatten();
                if callee_module != caller_module && mir_touches_module_vars(&body) {
                    continue;
                }
                if mir_calls_jit_unsafe_fiber_method(&body, interner)
                    || mir_calls_any_tainted_method(&body, &tainted)
                {
                    continue;
                }
                sites.insert(*dst, KnownCallee { guard, body });
            }
        }
        sites
    }

    /// Loop headers whose body has call sites but no inline-cache data
    /// yet: the loop has not run, so compiling it now bakes in generic
    /// dispatch.
    /// Each cold header maps to the registers the interpreter needs to
    /// resume there: the header's live-ins in the bytecode's own terms.
    fn cold_loop_headers(
        mir: &MirFunction,
        ics: &[CallSiteIC],
    ) -> HashMap<crate::mir::BlockId, Vec<crate::mir::ValueId>> {
        use crate::mir::opt::licm::{
            compute_dominators, compute_rpo, detect_loops, merge_loops_by_header,
        };
        use crate::mir::Instruction;
        let mut cold = HashMap::new();
        // WLIFT_COLD_LOOP_RECOMPILE=1 turns on cold-loop exits and the
        // recompile they request; safe to run with, but the recompile
        // costs more than it returns on short loops, so it is off.
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("WLIFT_COLD_LOOP_RECOMPILE").is_some()) {
            return cold;
        }
        if mir.blocks.is_empty() {
            return cold;
        }
        // Call-site index per block, in the snapshot's order.
        let mut ic_idx = 0usize;
        let mut site_kinds: Vec<Vec<u64>> = vec![Vec::new(); mir.blocks.len()];
        for (bi, block) in mir.blocks.iter().enumerate() {
            for (_, inst) in &block.instructions {
                if matches!(
                    inst,
                    Instruction::Call { .. } | Instruction::SuperCall { .. }
                ) {
                    site_kinds[bi].push(ics.get(ic_idx).map(|ic| ic.kind).unwrap_or(0));
                    ic_idx += 1;
                }
            }
        }
        let rpo = compute_rpo(mir);
        let idom = compute_dominators(mir, &rpo);
        for lp in merge_loops_by_header(&detect_loops(mir, &idom)) {
            let kinds: Vec<u64> = lp
                .body
                .iter()
                .flat_map(|b| site_kinds[b.0 as usize].iter().copied())
                .collect();
            if !kinds.is_empty() && kinds.iter().all(|k| *k == 0) {
                let live: Vec<crate::mir::ValueId> =
                    crate::mir::osr_external_live_values(mir, lp.header)
                        .into_iter()
                        .chain(
                            mir.blocks[lp.header.0 as usize]
                                .params
                                .iter()
                                .map(|(p, _)| *p),
                        )
                        .collect();
                cold.insert(lp.header, live);
            }
        }
        cold
    }

    /// Whether the installed OSR entry for `block` was compiled before
    /// its loop ever ran. Every 256th probe asks for the next tier, so
    /// a loop that is really hot gets code with its caches filled.
    pub fn osr_entry_is_cold(
        &mut self,
        id: FuncId,
        block: crate::mir::BlockId,
        interner: &crate::intern::Interner,
    ) -> bool {
        let idx = id.0 as usize;
        if !self
            .cold_osr_blocks
            .get(idx)
            .map(|s| s.contains(&block))
            .unwrap_or(false)
        {
            return false;
        }
        let probes = self.cold_osr_probes.entry((id.0, block.0)).or_insert(0);
        *probes += 1;
        if probes.is_multiple_of(256) {
            self.request_tier_up(id, interner);
        }
        true
    }

    /// The compile clone with a `GuardNumAt` after every call whose
    /// inline cache only ever produced a Num, where the interpreter can
    /// take over if the guard fails: the function is a bound method,
    /// the bytecode records the offset past the call, and every
    /// register live there is a value the clone still defines.
    fn speculate_call_results(
        &mut self,
        id: FuncId,
        authoritative: &MirFunction,
        mir: Arc<MirFunction>,
    ) -> Arc<MirFunction> {
        use crate::mir::bytecode::RESULT_NUM;
        use crate::mir::{live_in_sets, DeoptReg, DeoptSource, Instruction, ValueId};
        use std::collections::HashSet;
        let idx = id.0 as usize;
        if self
            .method_binding
            .get(idx)
            .map(|(c, _)| c.is_null())
            .unwrap_or(true)
        {
            return mir;
        }
        let Some(bc) = self.ensure_bytecode(id) else {
            return mir;
        };
        let resume_after_call = unsafe { &(*bc).resume_after_call };
        let result_kinds: Vec<u8> = unsafe { (*(*bc).result_kinds.get()).clone() };
        let live_in = live_in_sets(authoritative);
        let mut sites: Vec<(ValueId, u32, Vec<ValueId>)> = Vec::new();
        for block in &authoritative.blocks {
            for (i, (dst, inst)) in block.instructions.iter().enumerate() {
                if !matches!(inst, Instruction::Call { .. }) {
                    continue;
                }
                let seen = result_kinds.get(dst.0 as usize).copied().unwrap_or(0);
                if seen != RESULT_NUM {
                    continue;
                }
                let Some(&pc) = resume_after_call.get(dst) else {
                    continue;
                };
                let mut live: HashSet<ValueId> = HashSet::new();
                for succ in block.terminator.successors() {
                    live.extend(live_in.iter(succ.0 as usize));
                }
                live.extend(block.terminator.operands());
                for (later_dst, later) in block.instructions[i + 1..].iter().rev() {
                    live.remove(later_dst);
                    live.extend(later.operands());
                }
                live.insert(*dst);
                let mut live: Vec<ValueId> = live.into_iter().collect();
                live.sort_by_key(|v| v.0);
                sites.push((*dst, pc, live));
            }
        }
        if sites.is_empty() {
            return mir;
        }
        let defined: HashSet<ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| {
                b.params
                    .iter()
                    .map(|(v, _)| *v)
                    .chain(b.instructions.iter().map(|(v, _)| *v))
            })
            .collect();
        // A range the interpreter iterates is rebuilt from its bounds
        // at the exit, so the compiled body need not keep it.
        let ranges: HashMap<ValueId, (ValueId, ValueId, bool)> = authoritative
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|(v, inst)| match inst {
                Instruction::MakeRange(from, to, inclusive) => Some((*v, (*from, *to, *inclusive))),
                _ => None,
            })
            .collect();
        let source_of = |v: ValueId| -> Option<DeoptSource> {
            if let Some(&(from, to, inclusive)) = ranges.get(&v) {
                if defined.contains(&from) && defined.contains(&to) {
                    return Some(DeoptSource::Range {
                        from,
                        to,
                        inclusive,
                    });
                }
            }
            defined.contains(&v).then_some(DeoptSource::Value(v))
        };
        // A result nothing reads (a setter used as a statement) needs
        // no guard.
        let used: HashSet<ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| {
                b.instructions
                    .iter()
                    .flat_map(|(_, inst)| inst.operands())
                    .chain(b.terminator.operands())
            })
            .collect();
        let mut out = (*mir).clone();
        let mut placed: Vec<(usize, usize, Instruction, ValueId)> = Vec::new();
        for (dst, pc, live) in sites {
            if !used.contains(&dst) {
                continue;
            }
            let live: Option<Vec<DeoptReg>> = live
                .iter()
                .map(|v| source_of(*v).map(|source| DeoptReg { reg: v.0, source }))
                .collect();
            let Some(live) = live else {
                continue;
            };
            let Some((bi, pos)) = out.blocks.iter().enumerate().find_map(|(bi, b)| {
                b.instructions
                    .iter()
                    .position(|(v, inst)| *v == dst && matches!(inst, Instruction::Call { .. }))
                    .map(|pos| (bi, pos))
            }) else {
                continue;
            };
            let guard = Instruction::GuardNumAt {
                value: dst,
                pc,
                live,
            };
            placed.push((bi, pos, guard, dst));
        }
        // Arguments the baseline only ever saw as Num take an entry
        // guard, which the entry deopt can re-run the call for.
        let already_guarded: HashSet<ValueId> = out.blocks[0]
            .instructions
            .iter()
            .filter_map(|(_, inst)| match inst {
                Instruction::GuardNum(v) => Some(*v),
                _ => None,
            })
            .collect();
        let mut entry_guards: Vec<Instruction> = Vec::new();
        for (vid, inst) in &out.blocks[0].instructions {
            if let Instruction::BlockParam(idx) = inst {
                if *idx > 0
                    && !already_guarded.contains(vid)
                    && result_kinds.get(vid.0 as usize).copied() == Some(RESULT_NUM)
                {
                    entry_guards.push(Instruction::GuardNum(*vid));
                    if !out.speculated_num_params.contains(vid) {
                        out.speculated_num_params.push(*vid);
                    }
                }
            }
        }
        if placed.is_empty() && entry_guards.is_empty() {
            return mir;
        }
        // Later positions first so earlier inserts do not shift them.
        placed.sort_by_key(|p| std::cmp::Reverse((p.0, p.1)));
        let count = placed.len();
        for (bi, pos, guard, dst) in placed {
            let g = out.new_value();
            out.blocks[bi].instructions.insert(pos + 1, (g, guard));
            if !out.speculated_num_params.contains(&dst) {
                out.speculated_num_params.push(dst);
            }
        }
        let insert_at = out.blocks[0]
            .instructions
            .iter()
            .position(|(_, inst)| !matches!(inst, Instruction::BlockParam(_)))
            .unwrap_or(out.blocks[0].instructions.len());
        for (i, guard) in entry_guards.into_iter().enumerate() {
            let g = out.new_value();
            out.blocks[0].instructions.insert(insert_at + i, (g, guard));
        }
        if tier_trace_enabled() {
            eprintln!(
                "tier-trace: [{:.2}ms] result speculation FuncId({}) guards={}",
                trace_clock_ms(),
                id.0,
                count
            );
        }
        Arc::new(out)
    }

    /// The compile clone with known calls inlined. `WLIFT_DISABLE_MIR_INLINE`
    /// turns it off; safe to run with.
    fn inline_known(
        &self,
        caller: FuncId,
        mir: &MirFunction,
        clone: Arc<MirFunction>,
        ics: Option<&[CallSiteIC]>,
        interner: &crate::intern::Interner,
    ) -> Arc<MirFunction> {
        if std::env::var_os("WLIFT_DISABLE_MIR_INLINE").is_some() {
            return clone;
        }
        // Comma-separated caller ids to leave alone; a bisection aid.
        if let Ok(skip) = std::env::var("WLIFT_MIR_INLINE_SKIP") {
            if skip.split(',').any(|s| s.trim() == caller.0.to_string()) {
                return clone;
            }
        }
        let Some(ics) = ics else {
            return clone;
        };
        let sites = self.known_call_sites(caller, mir, ics, interner);
        if sites.is_empty() {
            return clone;
        }
        let mut out = (*clone).clone();
        if crate::mir::opt::inline_calls::inline_known_calls(&mut out, &sites) {
            if std::env::var_os("WLIFT_INLINE_TRACE").is_some() {
                eprintln!(
                    "inline-trace: FuncId({}) inlined {} site(s)",
                    caller.0,
                    sites.len()
                );
            }
            Arc::new(out)
        } else {
            clone
        }
    }

    /// The MIR a tier compiles: the profile's type guards when
    /// speculation is allowed, then the JIT pipeline.
    fn build_compile_mir(
        mir: &Arc<MirFunction>,
        tier: CompileTier,
        interner: &crate::intern::Interner,
        profile: Option<&TypeProfile>,
        speculate: bool,
    ) -> Arc<MirFunction> {
        let _ = tier;
        let mut out = (**mir).clone();
        if speculate && profile.is_some() {
            insert_speculative_guards(&mut out, profile);
        }
        run_jit_opt_pipeline(&mut out, interner);
        Arc::new(out)
    }

    /// WLIFT_TIER_TRACE helper — prints the engine's TierState alongside
    /// the beadie bead's BeadState and invocation count at a given event.
    ///
    /// (Helpers like encode_osr_entries live outside this impl.)
    /// Used to cross-check that the two views of the tier state machine
    /// stay in sync; drift shows up immediately as a mismatch in the
    /// trace stream rather than as a hard-to-debug flake.
    fn emit_bead_trace(&self, idx: usize, tier: CompileTier, event: &str) {
        let id = FuncId(idx as u32);
        let engine_tier = self.tier_states.get(idx).copied().unwrap_or_default();
        let bead_state = self
            .tier
            .state(id)
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|| "Unregistered".to_string());
        let invocations = self.tier.invocations(id);
        eprintln!(
            "tier-trace: [{:.2}ms] {event} {tier:?} FuncId({}) engine={engine_tier:?} bead={bead_state} invocations={invocations}", trace_clock_ms(),
            idx
        );
    }

    #[cfg(feature = "host")]
    fn install_compiled_tier(
        &mut self,
        idx: usize,
        tier: CompileTier,
        executable: ExecutableFunction,
        native_meta: Option<Arc<NativeFrameMetadata>>,
        inline_safe: bool,
    ) {
        let native_ptr = if executable.is_native() {
            executable.native_ptr()
        } else {
            std::ptr::null()
        };
        let osr_entries = executable.osr_entries().to_vec();
        let installed_code_size = executable.code_size();
        if std::env::var_os("WLIFT_TRACE_INSTALL").is_some() {
            eprintln!(
                "INSTALL: idx={} tier={:?} native={} ptr={:p}",
                idx,
                tier,
                executable.is_native(),
                native_ptr
            );
        }

        match tier {
            CompileTier::Baseline => {
                let body = match std::mem::replace(
                    &mut self.functions[idx],
                    FuncBody::Interpreted {
                        mir: Arc::new(MirFunction::new(SymbolId::from_raw(0), 0)),
                        bytecode: None,
                    },
                ) {
                    FuncBody::Interpreted { mir, bytecode, .. } => FuncBody::Native {
                        baseline_executable: executable,
                        optimized_executable: None,
                        mir,
                        bytecode,
                    },
                    // A recompile: the old code may still be on the
                    // stack, so it is retired rather than dropped.
                    FuncBody::Native {
                        baseline_executable,
                        optimized_executable,
                        mir,
                        bytecode,
                    } => {
                        self.retired_code.push(baseline_executable);
                        FuncBody::Native {
                            baseline_executable: executable,
                            optimized_executable,
                            mir,
                            bytecode,
                        }
                    }
                };
                self.functions[idx] = body;
                self.baseline_code[idx] = native_ptr;
                self.baseline_osr_entries[idx] = osr_entries.clone();
                self.cold_osr_blocks[idx] = self.pending_cold_osr.remove(&idx).unwrap_or_default();
                self.baseline_leaf[idx] = inline_safe;
                self.baseline_metadata[idx] = native_meta;
                self.tier_states[idx] = TierState::BaselineNative;
                // Publish code + OSR table to the bead atomically.
                // `install_or_swap_osr` picks eager-install or swap
                // based on current state. Non-native tiers (WASM
                // fallback) have a null pointer and stay Interpreted
                // on the bead side — the helper rejects nulls.
                if !native_ptr.is_null() {
                    self.tier.install_or_swap_osr(
                        FuncId(idx as u32),
                        native_ptr as *mut (),
                        encode_osr_entries(&osr_entries),
                    );
                }
                if tier_trace_enabled() {
                    self.emit_bead_trace(idx, tier, "install-done");
                }
            }
            CompileTier::Optimized => {
                if let Some(FuncBody::Native {
                    optimized_executable,
                    ..
                }) = self.functions.get_mut(idx)
                {
                    *optimized_executable = Some(executable);
                }
                self.optimized_code[idx] = native_ptr;
                self.optimized_osr_entries[idx] = osr_entries.clone();
                self.cold_osr_blocks[idx] = self.pending_cold_osr.remove(&idx).unwrap_or_default();
                self.optimized_leaf[idx] = inline_safe;
                self.optimized_metadata[idx] = native_meta;
                self.tier_states[idx] = TierState::OptimizedNative;
                if !native_ptr.is_null() && !osr_entries.is_empty() {
                    self.tier_cells[idx]
                        .retier
                        .store(1, std::sync::atomic::Ordering::Release);
                }
                // Swap both pointer and OSR table atomically to
                // optimized tier. Beadie bumps the bead's generation
                // so stale baseline OSR lookups can't race.
                if !native_ptr.is_null() {
                    self.tier.install_or_swap_osr(
                        FuncId(idx as u32),
                        native_ptr as *mut (),
                        encode_osr_entries(&osr_entries),
                    );
                }
                if tier_trace_enabled() {
                    self.emit_bead_trace(idx, tier, "install-done");
                }
            }
        }

        if let Some(bytecode) = self.peek_bytecode(FuncId(idx as u32)) {
            self.bc_cache[idx] = Arc::as_ptr(&bytecode);
        }
        #[cfg(feature = "host")]
        self.sync_active_tier_cache(idx);
        if std::env::var_os("WLIFT_TRACE_INSTALL").is_some() {
            let jit_ptr = self.jit_code.get(idx).copied().unwrap_or(std::ptr::null());
            let leaf = self.jit_leaf.get(idx).copied().unwrap_or(false);
            let state = self
                .tier_states
                .get(idx)
                .copied()
                .unwrap_or(TierState::Interpreted);
            eprintln!(
                "SYNC: idx={} jit_code={:p} leaf={} state={:?}",
                idx, jit_ptr, leaf, state
            );
        }
        // Dump machine code hex for any installed native function.
        if std::env::var_os("WLIFT_DUMP_HEX").is_some() && !native_ptr.is_null() {
            let code_size = installed_code_size;
            if code_size > 0 {
                eprint!("HEX:f{}:{}:", idx, code_size);
                let code_bytes = unsafe { std::slice::from_raw_parts(native_ptr, code_size) };
                for b in code_bytes {
                    eprint!("{:02x}", b);
                }
                eprintln!();
            }
        }
        // Register code range for GC stack walking.
        if !native_ptr.is_null() {
            if let Some(meta) = self.jit_metadata.get(idx).and_then(|m| m.clone()) {
                let func_id = FuncId(idx as u32);
                let start = native_ptr as usize;
                // Get code size from the executable stored in functions.
                let code_size = match self.functions.get(idx) {
                    Some(FuncBody::Native {
                        baseline_executable,
                        optimized_executable,
                        ..
                    }) => {
                        if let Some(opt) = optimized_executable {
                            opt.code_size()
                        } else {
                            baseline_executable.code_size()
                        }
                    }
                    _ => 0,
                };
                if code_size > 0 {
                    self.code_ranges.retain(|r| r.func_id != func_id);
                    if std::env::var_os("WLIFT_TRACE_CODE_RANGE").is_some() {
                        eprintln!(
                            "code-range: register f{} {:#x}-{:#x} size={}",
                            func_id.0,
                            start,
                            start + code_size,
                            code_size
                        );
                    }
                    // Dump raw machine code hex for offline disassembly.
                    if std::env::var_os("WLIFT_DUMP_HEX").is_some() {
                        eprint!("HEX:f{}:{}:", func_id.0, code_size);
                        let code_bytes =
                            unsafe { std::slice::from_raw_parts(start as *const u8, code_size) };
                        for b in code_bytes {
                            eprint!("{:02x}", b);
                        }
                        eprintln!();
                    }
                    self.register_code_range(func_id, start, start + code_size, meta);
                }
            }
        }
        // Selective IC invalidation: only clear IC entries that reference
        // the recompiled function. This allows other functions' ICs (including
        // recursive constructor call sites) to remain populated.
        self.invalidate_ic_entries_for(FuncId(idx as u32));
        if self.collect_tier_stats {
            if let Some(stats) = self.tier_stats.get_mut(idx) {
                stats.compile_successes += 1;
            }
        }
    }

    /// Record a call to a function. Returns true if the next tier should be
    /// requested now.
    pub fn record_call(&mut self, id: FuncId) -> bool {
        if self.mode != ExecutionMode::Tiered {
            return false;
        }
        // Auto-deopt retry: if this function was demoted by the
        // corruption-signal auto-deopt in `vm_interp` and has been
        // observed for `jit_threshold` more calls in the lower
        // tier, force a re-tier-up. This bypasses the bead's
        // fire-once invocation counter so a function with a
        // transient JIT mis-compile gets to retry rather than
        // staying at the lower tier forever.
        if auto_deopt_retry_tick(id) {
            return true;
        }
        let idx = id.0 as usize;
        match self.functions.get_mut(idx) {
            Some(FuncBody::Interpreted { .. }) => {
                // Baseline tier-up counting lives on the beadie bead.
                // Fires true exactly once when the bead's invocation
                // count first equals `jit_threshold`.
                self.tier.should_promote_on_tick(id, self.jit_threshold)
            }
            Some(FuncBody::Native { .. }) => {
                self.tier.tick(id);
                // Only the baseline code's own ticks count towards the
                // top tier: its result profile is what the top tier
                // speculates on, so it must have run first.
                let count = self.tier_cells[idx]
                    .total
                    .load(std::sync::atomic::Ordering::Relaxed);
                self.propose_top_tier(idx, count)
            }
            _ => false,
        }
    }

    /// Invocation count at which the top tier is proposed: the
    /// threshold less a fifth, so the compile is queued ahead of it.
    pub fn top_tier_queue_at(&self) -> u32 {
        self.opt_threshold
            .saturating_sub(self.opt_threshold / 5)
            .max(1)
    }

    /// Whether `count` invocations of a baseline-compiled function
    /// propose the top tier now. The proposal is compared with `>=`, so
    /// it fires until the compile is queued; a function whose baseline
    /// is not installed, whose compile is in flight, or which the tier
    /// refused is never proposed.
    fn propose_top_tier(&self, idx: usize, count: u32) -> bool {
        if crate::codegen::top_tier() == crate::codegen::TopTier::Off {
            return false;
        }
        if self.promote_refused.get(idx).copied().unwrap_or(true) {
            return false;
        }
        if self.tier_states.get(idx).copied() != Some(TierState::BaselineNative)
            || self.compiling_tier.get(idx).copied().flatten().is_some()
        {
            return false;
        }
        match self.functions.get(idx) {
            Some(FuncBody::Native {
                optimized_executable,
                ..
            }) if optimized_executable.is_none() => {}
            _ => return false,
        }
        count >= self.top_tier_queue_at() && count >= self.promote_retry_at[idx]
    }

    /// The top tier's ceiling for `id`, with call sites the inline
    /// caches resolve to trivial getters and setters counted as field
    /// accesses.
    fn ceiling(&mut self, id: FuncId, mir: &MirFunction) -> TopTierCeiling {
        let ics = self.callsite_ic_data_for_compile(id).map(|(s, _)| s);
        let field_access = |site: usize| -> bool {
            let Some(ics) = ics.as_ref() else {
                return false;
            };
            let Some(ic) = ics.get(site) else {
                return false;
            };
            if ic.kind == 5 {
                return true;
            }
            if ic.func_id == 0 || ic.class == 0 {
                return false;
            }
            let callee = ic.func_id as usize;
            self.trivial_getter_fields
                .get(callee)
                .map(|f| f.is_some())
                .unwrap_or(false)
                || self
                    .trivial_setter_fields
                    .get(callee)
                    .map(|f| f.is_some())
                    .unwrap_or(false)
        };
        top_tier_ceiling(mir, &field_access)
    }

    /// Baseline code's own entry count crossing a sampling point.
    #[cfg(feature = "host")]
    pub fn native_tick(&mut self, id: FuncId, interner: &crate::intern::Interner) {
        if self.mode != ExecutionMode::Tiered {
            return;
        }
        let Some(count) = self.tier_cells.get(id.0 as usize).map(|c| c.tick()) else {
            return;
        };
        if tier_trace_enabled() {
            eprintln!(
                "tier-trace: [{:.2}ms] tick FuncId({}) count={}",
                trace_clock_ms(),
                id.0,
                count
            );
        }
        let idx = id.0 as usize;
        // Native code is a safepoint for installs too; nothing running
        // is dropped, the lower tier's code stays owned by the function.
        self.poll_compilations();
        if self.propose_top_tier(idx, count) {
            self.request_tier_up(id, interner);
        }
        let Some(cell) = self.tier_cells.get(idx) else {
            return;
        };
        let settled = self.promote_refused[idx]
            || self.tier_states[idx] == TierState::OptimizedNative
            || crate::codegen::top_tier() == crate::codegen::TopTier::Off;
        if settled {
            cell.tick_after(u32::MAX);
        } else if self.compiling_tier[idx].is_some() {
            // Compile in flight: the finished compile brings the tick
            // forward itself; this is the fallback.
            cell.tick_after(1 << 20);
        } else {
            let at = self.promote_retry_at[idx]
                .max(self.top_tier_queue_at())
                .max(count.saturating_add(64));
            cell.tick_after(at - count);
        }
    }

    /// The top tier's OSR entry for a loop header, once installed.
    pub fn top_tier_osr_entry(
        &self,
        id: FuncId,
        header: crate::mir::BlockId,
    ) -> Option<NativeOsrEntry> {
        let idx = id.0 as usize;
        if self.tier_states.get(idx).copied() != Some(TierState::OptimizedNative) {
            return None;
        }
        let entry = self
            .optimized_osr_entries
            .get(idx)?
            .iter()
            .find(|e| e.target_block == header)?;
        // Only the bytecode's own registers mean the same thing in both
        // bodies; a value the JIT pipeline created is numbered per
        // compile.
        let registers = self.functions.get(idx)?.mir().next_value;
        if entry.live_in_regs.iter().any(|r| *r >= registers) {
            return None;
        }
        Some(entry.clone())
    }

    /// Stop baseline code polling for a transfer into the top tier.
    pub fn stop_retier(&self, id: FuncId) {
        if let Some(cell) = self.tier_cells.get(id.0 as usize) {
            cell.retier.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Outermost loop headers of `mir`, the points baseline code polls
    /// for a transfer into the top tier.
    /// Every loop header of `mir`.
    fn loop_headers(mir: &MirFunction) -> std::collections::HashSet<crate::mir::BlockId> {
        use crate::mir::opt::licm::{
            compute_dominators, compute_rpo, detect_loops, merge_loops_by_header,
        };
        if mir.blocks.is_empty() {
            return std::collections::HashSet::new();
        }
        let mut with_preds = mir.clone();
        with_preds.compute_predecessors();
        let rpo = compute_rpo(&with_preds);
        let idom = compute_dominators(&with_preds, &rpo);
        merge_loops_by_header(&detect_loops(&with_preds, &idom))
            .iter()
            .map(|lp| lp.header)
            .collect()
    }

    fn retier_headers(mir: &MirFunction) -> std::collections::HashSet<crate::mir::BlockId> {
        use crate::mir::opt::licm::{
            compute_dominators, compute_rpo, detect_loops, merge_loops_by_header,
        };
        let mut out = std::collections::HashSet::new();
        if mir.blocks.is_empty() {
            return out;
        }
        let mut with_preds = mir.clone();
        with_preds.compute_predecessors();
        let rpo = compute_rpo(&with_preds);
        let idom = compute_dominators(&with_preds, &rpo);
        let loops = merge_loops_by_header(&detect_loops(&with_preds, &idom));
        for lp in &loops {
            let nested = loops
                .iter()
                .any(|other| other.header != lp.header && other.body.contains(&lp.header));
            if !nested {
                out.insert(lp.header);
            }
        }
        out
    }

    /// Sample argument types during interpretation for profile-guided compilation.
    pub fn sample_arg_types(&mut self, id: FuncId, args: &[crate::runtime::value::Value]) {
        let idx = id.0 as usize;
        if idx >= self.type_profiles.len() {
            return;
        }
        let profile = self.type_profiles[idx].get_or_insert_with(TypeProfile::default);
        if profile.sample_count >= 32 {
            return;
        }
        profile.sample_count += 1;
        for (i, arg) in args.iter().enumerate().take(8) {
            let observed = classify_value(*arg);
            let slot = &mut profile.param_types[i];
            if *slot == PROFILE_UNSEEN {
                *slot = observed;
            } else if *slot != observed {
                *slot = PROFILE_MIXED;
            }
        }
    }

    /// Get the type profile for a function (persists across tier transitions).
    pub fn get_type_profile(&self, id: FuncId) -> Option<&TypeProfile> {
        self.type_profiles.get(id.0 as usize)?.as_ref()
    }

    /// Compile the next tier synchronously and install it.
    /// Wasm builds run BC-only — there is no native code to compile,
    /// no broker thread to drive, and `tier_up` on the host pulls in
    /// `compile_function_artifact_*` which is itself host-only. The
    /// wasm stub returns `false` so any caller that opportunistically
    /// asked "did this function get promoted?" sees the same shape.
    #[cfg(not(feature = "host"))]
    pub fn tier_up(&mut self, _id: FuncId, _interner: &crate::intern::Interner) -> bool {
        false
    }

    #[cfg(feature = "host")]
    pub fn tier_up(&mut self, id: FuncId, interner: &crate::intern::Interner) -> bool {
        let idx = id.0 as usize;
        let Some(tier) = self.next_compile_tier(idx) else {
            return self.tier_state(id) != TierState::Interpreted;
        };
        if idx >= self.functions.len() {
            return false;
        }
        // Skip JIT for functions that directly call `Fiber.yield()` /
        // `Fiber.suspend()`. The JIT's `handle_jit_fiber_action`
        // Yield/Suspend branches are stubs that simply return the
        // yielded value — the compiled function keeps running as if
        // yield were a regular call, so the fiber never suspends.
        // Cooperative loops like `Subscription_.receive` (which polls
        // `Fiber.yield()` until a message arrives) busy-spin forever
        // once tiered, starving every other fiber on the scheduler.
        // Until JIT can unwind out of native code at a yield, refuse
        // to compile yielding functions and let the bytecode
        // interpreter (whose `handle_fiber_action_bc` does suspend
        // properly) handle them.
        //
        // Direct check first, then transitive: a function that calls
        // a method whose implementation may itself yield is just as
        // unsafe as one that yields directly. The taint propagates
        // through the call graph (e.g. `Http_.readRequest` calls
        // `buf.readLine` which calls `fill_()` which calls
        // `Fiber.yield()` — readRequest is now correctly excluded).
        // Compute the tainted set first (mutable borrow), then take
        // an immutable borrow of `self.functions[idx]` for the
        // checks — keeping these on separate lines avoids the
        // borrow conflict.
        let tainted = self.compute_may_yield_methods(interner);
        let body = &self.functions[idx];
        if mir_calls_jit_unsafe_fiber_method(body.mir(), interner) {
            return false;
        }
        if mir_calls_any_tainted_method(body.mir(), &tainted) {
            return false;
        }
        // Skip JIT compilation for functions named in WLIFT_SKIP_JIT env var
        if let Ok(skip) = std::env::var("WLIFT_SKIP_JIT") {
            let name = interner.resolve(body.mir().name);
            if skip.split(',').any(|s| name == s || name.contains(s)) {
                return false;
            }
        }
        // Only JIT functions named in WLIFT_ONLY_JIT env var
        if let Ok(only) = std::env::var("WLIFT_ONLY_JIT") {
            let name = interner.resolve(body.mir().name);
            if !only.split(',').any(|s| name == s || name.contains(s)) {
                return false;
            }
        }
        let mir = Arc::clone(body.mir());
        let profile = self.get_type_profile(id).cloned();
        if self.collect_tier_stats {
            if let Some(stats) = self.tier_stats.get_mut(idx) {
                stats.compile_attempts += 1;
            }
        }
        let sroa_mir = self.scalar_replaced(id, &mir, interner);
        let (mut callsite_ic_ptrs, callsite_ic_live_ptrs) = self
            .callsite_ic_data_for_compile(id)
            .map(|(s, l)| (Some(s), Some(l)))
            .unwrap_or((None, None));
        let cha_disabled = std::env::var_os("WLIFT_DISABLE_JIT_CHA").is_some();
        let cha_for_codegen: SharedCha = if cha_disabled {
            None
        } else {
            let cha = self.build_jit_cha();
            if let Some(ref mut ics) = callsite_ic_ptrs {
                self.fill_ic_with_cha(&mir, ics, &cha);
            }
            Some(Arc::new(cha))
        };
        let sroa_mir = self.inline_known(id, &mir, sroa_mir, callsite_ic_ptrs.as_deref(), interner);
        let speculate = !self.speculation_failed[idx];
        let compile_mir =
            Self::build_compile_mir(&sroa_mir, tier, interner, profile.as_ref(), speculate);
        let devirt_hints = callsite_ic_ptrs
            .as_ref()
            .map(|ics| self.compute_devirt_hints(ics));
        let (callsite_ic_ptrs, callsite_ic_live_ptrs, devirt_hints) =
            match (callsite_ic_ptrs, callsite_ic_live_ptrs) {
                (Some(ics), Some(live)) => {
                    let (a, b, h) =
                        Self::align_callsite_ics(&mir, &compile_mir, ics, live, devirt_hints);
                    (Some(a), Some(b), h)
                }
                _ => (None, None, None),
            };
        let jit_code_base = Some(self.jit_code.as_ptr());
        let callee_purity = Some(self.compute_callee_purity_map());
        let inline_bodies = if std::env::var_os("WLIFT_DISABLE_JIT_INLINE").is_none() {
            Some(self.compute_inline_bodies())
        } else {
            None
        };
        if std::env::var("WLIFT_JIT_DUMP").is_ok() {
            eprintln!("=== {:?} compile FuncId({}) ===", tier, id.0);
            eprintln!("{}", compile_mir.pretty_print(interner));
        }
        let target = Self::native_target();
        let modvars_cell = self.modvars_cell_addr(id);
        crate::codegen::cranelift_backend::cl::set_jit_modvars_cell(modvars_cell);
        let compiled_result =
            crate::codegen::compile_function_artifact_with_interner_and_callsite_ics(
                &compile_mir,
                target,
                interner,
                tier,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                devirt_hints,
                jit_code_base,
                callee_purity,
                inline_bodies,
                cha_for_codegen,
            );
        crate::codegen::cranelift_backend::cl::set_jit_modvars_cell(0);
        let compiled = match compiled_result {
            Ok(compiled) => compiled,
            Err(_) => return false,
        };
        let native_meta = compiled.native_meta;
        // Use MIR analysis for leaf classification. Shadow frame push/pop
        // is handled by each dispatch path via metadata checks, so even if
        // a "leaf" function needs shadow stores, they'll be set up correctly.
        // Leaf = MIR says leaf OR compiled code has no shadow stores.
        // Profile-guided guards (GuardNum) make is_mir_inline_safe return false,
        // but if the guards + TypeSpecialize eliminate all CallRuntime → no shadow
        // stores → the function IS safe for leaf dispatch.
        let inline_safe = is_mir_inline_safe(&compile_mir, tier) || !compiled.needs_shadow_frame;
        let executable = match compiled.code.into_executable() {
            Ok(executable) => executable,
            Err(_) => return false,
        };
        self.install_compiled_tier(idx, tier, executable, native_meta, inline_safe);
        true
    }

    /// Submit the next tier for background compilation.
    /// The interpreter keeps running bytecode; the compiled result is installed
    /// when `poll_compilations` is called at the next safepoint.
    #[cfg(not(feature = "host"))]
    pub fn request_tier_up(&mut self, _id: FuncId, _interner: &crate::intern::Interner) {}

    #[cfg(feature = "host")]
    pub fn request_tier_up(&mut self, id: FuncId, interner: &crate::intern::Interner) {
        let idx = id.0 as usize;
        let Some(tier) = self.next_compile_tier(idx) else {
            return;
        };
        self.request_compile(id, tier, interner);
    }

    /// Compile `id` at `tier` in the background; the install lands
    /// through `poll_compilations`.
    #[cfg(feature = "host")]
    fn request_compile(
        &mut self,
        id: FuncId,
        tier: CompileTier,
        interner: &crate::intern::Interner,
    ) {
        if self.mode != ExecutionMode::Tiered {
            return;
        }
        let prep_started = std::time::Instant::now();
        // Respect the deopt policy: functions blacklisted after too many
        // bailouts stay in the interpreter for the rest of the run.
        if self.tier.is_blacklisted(id) {
            return;
        }
        let idx = id.0 as usize;
        // Direct + transitive yield-method check — same shape as the
        // gate in `should_request_compile`; both refuse the JIT for
        // any function whose call graph reaches `Fiber.yield`.
        let direct_unsafe = self
            .functions
            .get(idx)
            .map(|body| mir_calls_jit_unsafe_fiber_method(body.mir(), interner))
            .unwrap_or(false);
        if direct_unsafe {
            return;
        }
        let tainted = self.compute_may_yield_methods(interner);
        if let Some(body) = self.functions.get(idx) {
            if mir_calls_any_tainted_method(body.mir(), &tainted) {
                return;
            }
        }
        // Skip JIT for functions named in WLIFT_SKIP_JIT env var
        if let Ok(skip) = std::env::var("WLIFT_SKIP_JIT") {
            if let Some(body) = self.functions.get(idx) {
                let name = interner.resolve(body.mir().name);
                if skip.split(',').any(|s| name == s || name.contains(s)) {
                    return;
                }
            }
        }
        // Only JIT functions named in WLIFT_ONLY_JIT env var
        if let Ok(only) = std::env::var("WLIFT_ONLY_JIT") {
            if let Some(body) = self.functions.get(idx) {
                let name = interner.resolve(body.mir().name);
                if !only.split(',').any(|s| name == s || name.contains(s)) {
                    return;
                }
            }
        }
        if idx >= self.compiling_tier.len() || self.compiling_tier[idx].is_some() {
            return;
        }
        let Some(body) = self.functions.get(idx) else {
            return;
        };
        let mir = Arc::clone(body.mir());
        let top_tier_on = crate::codegen::top_tier() != crate::codegen::TopTier::Off;
        let worth_top_tier = top_tier_on
            && (!promotion_gate_enabled() || self.ceiling(id, &mir) == TopTierCeiling::High);
        if tier == CompileTier::Optimized {
            if !top_tier_on {
                return;
            }
            if !worth_top_tier {
                self.promote_refused[idx] = true;
                if tier_trace_enabled() {
                    eprintln!(
                        "tier-trace: [{:.2}ms] skip Optimized FuncId({}) reason=no-headroom",
                        trace_clock_ms(),
                        id.0
                    );
                }
                return;
            }
        }
        // Only a body the top tier could take carries the counter and
        // the re-tier polls; the rest is never proposed.
        if tier == CompileTier::Baseline && !worth_top_tier {
            self.promote_refused[idx] = true;
        }
        let tier_hook = if tier == CompileTier::Baseline && worth_top_tier {
            self.tier_cells[idx].tick_after(self.top_tier_queue_at());
            Some(crate::codegen::cranelift_backend::cl::TierHook {
                func_id: id.0,
                cell: self.tier_cells[idx].as_ref() as *const TierCell as usize,
                retier_headers: Self::retier_headers(&mir),
                tick_headers: Self::loop_headers(&mir),
                result_kinds: self
                    .ensure_bytecode(id)
                    .map(|bc| unsafe { (*(*bc).result_kinds.get()).as_ptr() as usize })
                    .unwrap_or(0),
            })
        } else {
            None
        };
        let sroa_mir = self.scalar_replaced(id, &mir, interner);
        let profile = self.get_type_profile(id).cloned();
        let speculate = !self.speculation_failed[idx];
        let trace_name = self
            .functions
            .get(idx)
            .map(|body| interner.resolve(body.mir().name).to_string())
            .unwrap_or_else(|| format!("FuncId({})", id.0));
        if self.collect_tier_stats {
            if let Some(stats) = self.tier_stats.get_mut(idx) {
                stats.compile_attempts += 1;
            }
        }

        self.compiling_tier[idx] = Some(tier);
        self.pending_count += 1;
        let tx = self.compilation_tx.clone();
        let target = Self::native_target();
        let interner_clone = interner.clone();
        let trace_name_clone = trace_name.clone();
        let (mut callsite_ic_ptrs, callsite_ic_live_ptrs) = self
            .callsite_ic_data_for_compile(id)
            .map(|(s, l)| (Some(s), Some(l)))
            .unwrap_or((None, None));
        let cha_disabled = std::env::var_os("WLIFT_DISABLE_JIT_CHA").is_some();
        let cha_for_codegen: SharedCha = if cha_disabled {
            None
        } else {
            let cha = self.build_jit_cha();
            if let Some(ref mut ics) = callsite_ic_ptrs {
                self.fill_ic_with_cha(&mir, ics, &cha);
            }
            Some(Arc::new(cha))
        };
        let devirt_hints = callsite_ic_ptrs
            .as_ref()
            .map(|ics| self.compute_devirt_hints(ics));
        let cold = callsite_ic_ptrs
            .as_deref()
            .map(|ics| Self::cold_loop_headers(&mir, ics))
            .unwrap_or_default();
        self.pending_cold_osr
            .insert(idx, cold.keys().copied().collect());
        let sroa_mir = self.inline_known(id, &mir, sroa_mir, callsite_ic_ptrs.as_deref(), interner);
        let sroa_mir =
            if tier == CompileTier::Optimized && speculate && result_speculation_enabled() {
                self.speculate_call_results(id, &mir, sroa_mir)
            } else {
                sroa_mir
            };
        let jit_code_base_raw = self.jit_code.as_ptr() as usize;
        // The finished compile brings the baseline code's next tick
        // forward so the install lands at its next entry or outermost
        // iteration instead of at the interpreter's next safepoint.
        let tier_cell_addr = if tier == CompileTier::Optimized {
            self.tier_cells[idx].as_ref() as *const TierCell as usize
        } else {
            0
        };
        let modvars_cell = self.modvars_cell_addr(id);
        let callee_purity = self.compute_callee_purity_map();
        let inline_bodies = if std::env::var_os("WLIFT_DISABLE_JIT_INLINE").is_none() {
            Some(self.compute_inline_bodies())
        } else {
            None
        };

        if tier_trace_enabled() {
            let ic_count = callsite_ic_ptrs.as_ref().map(|v| v.len()).unwrap_or(0);
            eprintln!(
                "tier-trace: [{:.2}ms] queue {:?} FuncId({}) {} ic_ptrs={} prep={:?}",
                trace_clock_ms(),
                tier,
                id.0,
                trace_name,
                ic_count,
                prep_started.elapsed()
            );
        }

        // Compile closure — runs on beadie's broker thread. Returns an
        // `OsrCompileResult` so beadie installs the native entry
        // pointer AND the OSR table atomically under one generation
        // bump, closing the race where a back-edge probe could see a
        // fresh compiled pointer but a stale (empty) OSR table.
        //
        // The richer install artifact (ExecutableFunction ownership,
        // native_meta, tier) still travels back to the interpreter
        // thread through `compilation_tx` so `poll_compilations` can
        // finish the engine-side install at a safepoint.
        let compile_fn = move || -> beadie::OsrCompileResult {
            if tier_trace_enabled() {
                eprintln!(
                    "tier-trace: [{:.2}ms] start {:?} FuncId({}) {}",
                    trace_clock_ms(),
                    tier,
                    id.0,
                    trace_name_clone
                );
            }
            let compile_started = std::time::Instant::now();
            let mut compile_mir = Self::build_compile_mir(
                &sroa_mir,
                tier,
                &interner_clone,
                profile.as_ref(),
                speculate,
            );
            if !cold.is_empty() {
                Arc::make_mut(&mut compile_mir)
                    .osr_excluded
                    .extend(cold.keys().copied());
            }
            let mir_ready = compile_started.elapsed();
            let (callsite_ic_ptrs, callsite_ic_live_ptrs, devirt_hints) =
                match (callsite_ic_ptrs, callsite_ic_live_ptrs) {
                    (Some(ics), Some(live)) => {
                        let (a, b, h) =
                            Self::align_callsite_ics(&mir, &compile_mir, ics, live, devirt_hints);
                        (Some(a), Some(b), h)
                    }
                    _ => (None, None, None),
                };
            if std::env::var("WLIFT_JIT_DUMP").is_ok() {
                eprintln!("=== {:?} compile FuncId({}) ===", tier, id.0);
                eprintln!("{}", compile_mir.pretty_print(&interner_clone));
            }
            crate::codegen::cranelift_backend::cl::set_jit_modvars_cell(modvars_cell);
            crate::codegen::cranelift_backend::cl::set_jit_cold_headers(cold);
            crate::codegen::cranelift_backend::cl::set_jit_tier_hook(tier_hook);
            crate::codegen::cranelift_backend::cl::set_jit_func_id(id.0);
            let result = crate::codegen::compile_function_artifact_with_interner_and_callsite_ics(
                &compile_mir,
                target,
                &interner_clone,
                tier,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                devirt_hints,
                Some(jit_code_base_raw as *const *const u8),
                Some(callee_purity.clone()),
                inline_bodies.clone(),
                cha_for_codegen.clone(),
            );
            crate::codegen::cranelift_backend::cl::set_jit_cold_headers(Default::default());
            crate::codegen::cranelift_backend::cl::set_jit_tier_hook(None);
            crate::codegen::cranelift_backend::cl::set_jit_modvars_cell(0);
            let result = result
                .map_err(|e| {
                    if std::env::var_os("WLIFT_JIT_DEBUG").is_some() {
                        eprintln!("COMPILE ERR FuncId({}): {}", id.0, e);
                    }
                    e
                })
                .ok()
                .and_then(|artifact| {
                    let native_meta = artifact.native_meta;
                    let inline_safe =
                        is_mir_inline_safe(&compile_mir, tier) || !artifact.needs_shadow_frame;
                    artifact
                        .code
                        .into_executable()
                        .map_err(|e| {
                            if std::env::var_os("WLIFT_JIT_DEBUG").is_some() {
                                eprintln!("EXEC ERR FuncId({}): {}", id.0, e);
                            }
                            e
                        })
                        .ok()
                        .map(|executable| CompilationResult::Compiled {
                            id,
                            tier,
                            executable,
                            native_meta,
                            inline_safe,
                        })
                });
            if tier_trace_enabled() {
                eprintln!(
                    "tier-trace: [{:.2}ms] finish {:?} FuncId({}) {} success={} mir={:?} total={:?}",
                    trace_clock_ms(),
                    tier,
                    id.0,
                    trace_name_clone,
                    result.is_some(),
                    mir_ready,
                    compile_started.elapsed()
                );
            }
            // Extract the native entry + OSR entries for beadie BEFORE
            // sending the executable across the mpsc boundary. The
            // pointers refer into the heap-allocated mmap owned by the
            // executable — the mmap's address is stable when the
            // ExecutableFunction moves through the channel, so the
            // pointers remain valid after the engine-side install via
            // `poll_compilations` takes ownership.
            let (native_ptr, osr) = match &result {
                Some(CompilationResult::Compiled { executable, .. }) if executable.is_native() => {
                    let entries = encode_osr_entries(executable.osr_entries());
                    (executable.native_ptr() as *mut (), entries)
                }
                _ => (std::ptr::null_mut(), Vec::new()),
            };
            let _ = tx.send(result.unwrap_or(CompilationResult::Failed { id }));
            if tier_cell_addr != 0 {
                // SAFETY: the cell is boxed for the engine's lifetime and
                // only ever read through atomics on other threads; the
                // compiled code's own countdown update is a plain
                // read-modify-write, so this store can lose to it, costing
                // one more iteration before the tick.
                let cell = unsafe { &*(tier_cell_addr as *const TierCell) };
                cell.tick_after(1);
            }
            beadie::OsrCompileResult {
                entry: native_ptr,
                osr,
            }
        };

        // Submit to beadie's broker. The bead goes Interpreted → Queued
        // → Compiling → Compiled (with OSR table) as the closure
        // progresses. If the bead has already been promoted (race),
        // `AlreadyQueued` is returned and we roll back the pending_count.
        // The broker only takes a bead that is still interpreted; a
        // recompile of compiled code runs on a runtime thread and lands
        // through the same channel, where the install swaps the bead's
        // code and OSR table.
        // The broker takes interpreted beads only; the top tier and
        // recompiles of a compiled bead go through the promoter.
        let bead_interpreted = self.tier.state(id) == Some(beadie::BeadState::Interpreted);
        if tier == CompileTier::Optimized || !bead_interpreted {
            let promoter = self.promoter.get_or_insert_with(Promoter::start);
            if !promoter.submit(Box::new(move || {
                let _ = compile_fn();
            })) {
                // Nothing is in flight; propose again at double the count.
                self.compiling_tier[idx] = None;
                self.pending_count = self.pending_count.saturating_sub(1);
                let cell = &self.tier_cells[idx];
                let count = self
                    .tier
                    .invocations(id)
                    .max(cell.total.load(std::sync::atomic::Ordering::Relaxed));
                self.promote_retry_at[idx] = count.saturating_mul(2).max(count + 1);
                cell.tick_after(self.promote_retry_at[idx] - count);
            }
            return;
        }
        let submit_result = self
            .tier
            .submit_compile_osr(id, move |_bead: &std::sync::Arc<beadie::Bead>| compile_fn());
        if !submit_result.is_accepted() {
            self.compiling_tier[idx] = None;
            self.pending_count = self.pending_count.saturating_sub(1);
            if tier_trace_enabled() {
                eprintln!(
                    "tier-trace: [{:.2}ms] submit-rejected {:?} FuncId({}) {:?}",
                    trace_clock_ms(),
                    tier,
                    id.0,
                    submit_result
                );
            }
        }
    }

    /// Install any completed background compilations.
    #[inline]
    #[cfg(not(feature = "host"))]
    pub fn poll_compilations(&mut self) {}

    #[cfg(feature = "host")]
    pub fn poll_compilations(&mut self) {
        if self.pending_count == 0 {
            return;
        }
        while let Ok(result) = self.compilation_rx.try_recv() {
            let idx = match result {
                CompilationResult::Compiled { id, .. } | CompilationResult::Failed { id, .. } => {
                    id.0 as usize
                }
            };
            if tier_trace_enabled() {
                match &result {
                    CompilationResult::Compiled { id, tier, .. } => {
                        eprintln!(
                            "tier-trace: [{:.2}ms] install {:?} FuncId({})",
                            trace_clock_ms(),
                            tier,
                            id.0
                        );
                    }
                    CompilationResult::Failed { id } => {
                        eprintln!(
                            "tier-trace: [{:.2}ms] install failed FuncId({})",
                            trace_clock_ms(),
                            id.0
                        );
                    }
                }
            }
            if idx < self.functions.len() {
                if matches!(result, CompilationResult::Failed { .. })
                    && self.compiling_tier.get(idx).copied().flatten()
                        == Some(CompileTier::Optimized)
                {
                    // The same compile fails the same way; never re-propose.
                    self.promote_refused[idx] = true;
                }
                if let CompilationResult::Compiled {
                    tier,
                    executable,
                    native_meta,
                    inline_safe,
                    ..
                } = result
                {
                    self.install_compiled_tier(idx, tier, executable, native_meta, inline_safe);
                    // Stash callees for predictive pre-compile. The
                    // actual submits happen later in `drain_compile_queue`
                    // where we have interner access for env-var filtering.
                    self.record_pending_callees(FuncId(idx as u32));
                }
            }
            if idx < self.compiling_tier.len() {
                self.compiling_tier[idx] = None;
            }
            self.pending_count = self.pending_count.saturating_sub(1);
        }
    }

    /// Walk the caller's IC table and stash any uncompiled callees
    /// for predictive pre-compile. Runs during `poll_compilations`
    /// which doesn't have interner access, so the actual broker
    /// submissions happen later in [`Self::drain_compile_queue`].
    fn record_pending_callees(&mut self, caller_id: FuncId) {
        let bc_ptr = self
            .bc_cache
            .get(caller_id.0 as usize)
            .copied()
            .unwrap_or(std::ptr::null());
        if bc_ptr.is_null() {
            return;
        }
        let bc = unsafe { &*bc_ptr };
        let ic_table = unsafe { &*bc.ic_table.get() };
        for ic in ic_table.iter() {
            if ic.kind == 1 && ic.func_id != 0 {
                let callee_id = FuncId(ic.func_id as u32);
                let callee_idx = callee_id.0 as usize;
                let already_compiled = self
                    .jit_code
                    .get(callee_idx)
                    .map(|p| !p.is_null())
                    .unwrap_or(false);
                let already_queued = self
                    .compiling_tier
                    .get(callee_idx)
                    .map(|t| t.is_some())
                    .unwrap_or(false);
                if !already_compiled
                    && !already_queued
                    && callee_idx < self.functions.len()
                    && !self.pending_callee_precompile.contains(&callee_id)
                {
                    self.pending_callee_precompile.push(callee_id);
                }
            }
        }
    }

    /// Legacy entry point kept for vm_interp callers. Beadie's broker
    /// owns the compile queue proper now; the engine's job here is to
    /// (1) drain the install-side mpsc channel, and (2) submit any
    /// callees we stashed during the last install so they get compiled
    /// before the caller reaches them.
    pub fn drain_compile_queue(&mut self, interner: &crate::intern::Interner) {
        self.poll_compilations();
        let pending = std::mem::take(&mut self.pending_callee_precompile);
        for callee_id in pending {
            // request_tier_up runs env-var filters and re-checks state —
            // safe to call even if the callee has since been compiled.
            self.request_tier_up(callee_id, interner);
        }
    }

    /// Determine the native compilation target for the current platform.
    pub fn native_target() -> crate::codegen::Target {
        #[cfg(target_arch = "x86_64")]
        {
            crate::codegen::Target::X86_64
        }
        #[cfg(target_arch = "aarch64")]
        {
            crate::codegen::Target::Aarch64
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            crate::codegen::Target::Wasm
        }
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        #[cfg(feature = "host")]
        drop(self.promoter.take());
        // Beadie's broker owns the worker thread now; its Drop impl sends
        // a shutdown signal and joins when TierManager drops. Any in-flight
        // compile results that never reached `poll_compilations` get
        // drained here so their ExecutableFunction buffers are freed.
        while self.compilation_rx.try_recv().is_ok() {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;

    fn make_mir() -> MirFunction {
        let mut interner = Interner::new();
        let name = interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        f.block_mut(bb).terminator = crate::mir::Terminator::ReturnNull;
        f
    }

    #[test]
    fn test_engine_creation() {
        let engine = ExecutionEngine::new(ExecutionMode::Tiered);
        assert_eq!(engine.mode, ExecutionMode::Tiered);
        assert!(engine.functions.is_empty());
    }

    #[test]
    fn test_register_function() {
        let mut engine = ExecutionEngine::new(ExecutionMode::Interpreter);
        let mir = make_mir();
        let id = engine.register_function(mir);
        assert_eq!(id, FuncId(0));
        assert!(engine.get_function(id).is_some());
        assert!(engine.baseline_osr_entries[id.0 as usize].is_empty());
        assert!(engine.optimized_osr_entries[id.0 as usize].is_empty());
        assert!(engine
            .active_osr_entry(id, crate::mir::BlockId(0))
            .is_none());
    }

    #[test]
    fn test_bytecode_cache_lazy() {
        let mut engine = ExecutionEngine::new(ExecutionMode::Interpreter);
        let mir = make_mir();
        let id = engine.register_function(mir);

        // First call lazily compiles bytecode
        let bc1 = engine.get_bytecode(id);
        assert!(bc1.is_some());

        // Second call returns the same Arc (cached)
        let bc2 = engine.get_bytecode(id);
        assert!(Arc::ptr_eq(bc1.as_ref().unwrap(), bc2.as_ref().unwrap()));
    }

    #[test]
    fn test_call_counting() {
        let mut engine = ExecutionEngine::new(ExecutionMode::Tiered);
        engine.jit_threshold = 3;
        let mir = make_mir();
        let id = engine.register_function(mir);

        assert!(!engine.record_call(id)); // 1
        assert!(!engine.record_call(id)); // 2
        assert!(engine.record_call(id)); // 3 = threshold
    }

    #[test]
    fn test_bead_should_promote_fires_at_each_absolute_threshold() {
        // Phase 6 invariant: because the bead counter is monotonically
        // increasing and `should_promote_on_tick` tests exact equality,
        // the SAME bead fires true at multiple absolute thresholds in
        // succession — once when count first equals jit_threshold, again
        // when count first equals (jit_threshold + opt_threshold). That's
        // how baseline and optimized tier-up share one counter.
        let mut mgr = super::super::tier::TierManager::with_thresholds(3, 5);
        mgr.register(FuncId(0), std::ptr::null_mut());

        // Baseline: fires at tick #3.
        assert!(!mgr.should_promote_on_tick(FuncId(0), 3)); // count=1
        assert!(!mgr.should_promote_on_tick(FuncId(0), 3)); // count=2
        assert!(mgr.should_promote_on_tick(FuncId(0), 3)); // count=3 ← fires
        assert!(!mgr.should_promote_on_tick(FuncId(0), 3)); // count=4 — exact-eq doesn't re-fire

        // Optimized absolute threshold = 3 + 5 = 8. Queries above are
        // called with threshold=3, so they didn't fire at 5,6,7. Now
        // switch to the optimized threshold and count up to it.
        assert!(!mgr.should_promote_on_tick(FuncId(0), 8)); // count=5
        assert!(!mgr.should_promote_on_tick(FuncId(0), 8)); // count=6
        assert!(!mgr.should_promote_on_tick(FuncId(0), 8)); // count=7
        assert!(mgr.should_promote_on_tick(FuncId(0), 8)); // count=8 ← fires
        assert!(!mgr.should_promote_on_tick(FuncId(0), 8)); // count=9
    }

    #[test]
    fn test_bead_counts_interpreted_calls() {
        // Phase 3 invariant: record_call ticks the bead on each invocation
        // of an Interpreted function. With a threshold higher than our
        // loop, no tier-up fires and the bead stays in Interpreted state.
        let mut engine = ExecutionEngine::new(ExecutionMode::Tiered);
        engine.jit_threshold = 1_000;
        let mir = make_mir();
        let id = engine.register_function(mir);

        for _ in 0..7 {
            assert!(!engine.record_call(id));
        }
        assert_eq!(engine.tier.invocations(id), 7);
        assert_eq!(engine.tier.state(id), Some(beadie::BeadState::Interpreted));
    }

    #[test]
    fn test_shadow_tier_no_tick_in_interpreter_mode() {
        // Interpreter mode short-circuits record_call before the tick,
        // so the bead never sees any invocations either.
        let mut engine = ExecutionEngine::new(ExecutionMode::Interpreter);
        let mir = make_mir();
        let id = engine.register_function(mir);
        for _ in 0..5 {
            engine.record_call(id);
        }
        assert_eq!(engine.tier.invocations(id), 0);
    }

    #[test]
    fn test_call_counting_interpreter_mode() {
        let mut engine = ExecutionEngine::new(ExecutionMode::Interpreter);
        engine.jit_threshold = 1;
        let mir = make_mir();
        let id = engine.register_function(mir);

        // In interpreter mode, never triggers JIT
        assert!(!engine.record_call(id));
        assert!(!engine.record_call(id));
    }

    #[test]
    fn test_default_execution_mode() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::Tiered);
    }

    #[test]
    fn test_tier_up_compiles_function() {
        let mut interner = Interner::new();
        let name = interner.intern("add_f64");
        // Use unboxed f64 arithmetic (ConstF64 + AddF64) which compiles
        // inline without needing CallRuntime ABI support.
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions
                .push((v0, crate::mir::Instruction::ConstF64(10.0)));
            b.instructions
                .push((v1, crate::mir::Instruction::ConstF64(32.0)));
            b.instructions
                .push((v2, crate::mir::Instruction::AddF64(v0, v1)));
            b.terminator = crate::mir::Terminator::Return(v2);
        }

        let mut engine = ExecutionEngine::new(ExecutionMode::Tiered);
        engine.jit_threshold = 2;
        let id = engine.register_function(f);

        assert!(!engine.get_function(id).unwrap().is_compiled());
        assert!(!engine.record_call(id)); // 1
        assert!(engine.record_call(id)); // 2 = threshold

        let result = engine.tier_up(id, &interner);
        assert!(result, "tier_up should succeed for simple f64 arithmetic");
        assert!(engine.get_function(id).unwrap().is_compiled());
        // MIR should still be accessible after compilation
        assert!(engine.get_mir(id).is_some());

        // Phase 4 invariant: after a successful baseline install the bead
        // should be in Compiled state, so beadie's view of tier transitions
        // matches the engine's.
        assert_eq!(engine.tier.state(id), Some(beadie::BeadState::Compiled));
    }

    // -----------------------------------------------------------------------
    // JIT execution tests (compile + actually call native code)
    // -----------------------------------------------------------------------

    // JIT execution tests — compile MIR to native code and call it.
    // Only enabled on the native architecture.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    mod jit_exec {
        use super::*;
        use crate::codegen::runtime_fns::{self, JitContext};
        use crate::runtime::value::Value;

        /// Compile MIR to native code for the current platform and execute it.
        fn compile_and_exec(f: MirFunction, interner: &Interner) -> Option<u64> {
            let target = ExecutionEngine::native_target();
            let compiled =
                crate::codegen::compile_function_with_interner(&f, target, interner).ok()?;
            let executable = compiled.into_executable().ok()?;
            if !executable.is_native() {
                return None;
            }
            // SAFETY: executable stays alive while func runs (same scope).
            let func: fn() -> u64 = unsafe { executable.as_fn() };
            let result = func();
            drop(executable); // explicit: release mmap after call
            Some(result)
        }

        #[test]
        fn test_jit_exec_f64_add() {
            let mut interner = Interner::new();
            let name = interner.intern("f64_add");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            let v1 = f.new_value();
            let v2 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::ConstF64(10.0)));
                b.instructions
                    .push((v1, crate::mir::Instruction::ConstF64(32.0)));
                b.instructions
                    .push((v2, crate::mir::Instruction::AddF64(v0, v1)));
                b.terminator = crate::mir::Terminator::Return(v2);
            }

            let result = compile_and_exec(f, &interner);
            assert!(result.is_some(), "JIT compilation should succeed");
            let val = f64::from_bits(result.unwrap());
            assert_eq!(val, 42.0);
        }

        #[test]
        fn test_jit_exec_boxed_add() {
            // Test boxed (NaN-boxed) arithmetic via CallRuntime → wren_num_add
            let mut interner = Interner::new();
            let name = interner.intern("boxed_add");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            let v1 = f.new_value();
            let v2 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::ConstNum(10.0)));
                b.instructions
                    .push((v1, crate::mir::Instruction::ConstNum(32.0)));
                b.instructions
                    .push((v2, crate::mir::Instruction::Add(v0, v1)));
                b.terminator = crate::mir::Terminator::Return(v2);
            }

            let result = compile_and_exec(f, &interner);
            assert!(
                result.is_some(),
                "JIT compilation with CallRuntime should succeed"
            );
            let val = Value::from_bits(result.unwrap());
            let n = val.as_num().expect("should be a number");
            assert_eq!(n, 42.0);
        }

        #[test]
        fn test_jit_exec_boxed_cmp() {
            let mut interner = Interner::new();
            let name = interner.intern("boxed_cmp");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            let v1 = f.new_value();
            let v2 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::ConstNum(10.0)));
                b.instructions
                    .push((v1, crate::mir::Instruction::ConstNum(32.0)));
                b.instructions
                    .push((v2, crate::mir::Instruction::CmpLt(v0, v1)));
                b.terminator = crate::mir::Terminator::Return(v2);
            }

            let result = compile_and_exec(f, &interner);
            assert!(result.is_some());
            let val = Value::from_bits(result.unwrap());
            assert_eq!(val.as_bool(), Some(true));
        }

        #[test]
        fn test_jit_exec_module_var() {
            let mut interner = Interner::new();
            let name = interner.intern("read_modvar");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::GetModuleVar(0)));
                b.terminator = crate::mir::Terminator::Return(v0);
            }

            let expected = Value::num(99.0);
            let mut vars: Vec<u64> = vec![expected.to_bits()];
            let mut dummy_vm = [0u8; 8192];
            runtime_fns::set_jit_context(JitContext {
                module_vars: vars.as_mut_ptr(),
                module_var_count: 1,
                vm: dummy_vm.as_mut_ptr(),
                module_name: std::ptr::null(),
                module_name_len: 0,
                current_func_id: u32::MAX as u64,
                closure: std::ptr::null_mut(),
                defining_class: std::ptr::null_mut(),
                jit_code_base: std::ptr::null(),
                jit_code_len: 0,
            });

            let result = compile_and_exec(f, &interner);
            assert!(result.is_some());
            let val = Value::from_bits(result.unwrap());
            assert_eq!(val.as_num(), Some(99.0));

            runtime_fns::set_jit_context(JitContext::default());
        }

        #[test]
        fn test_jit_exec_const_bool() {
            let mut interner = Interner::new();
            let name = interner.intern("const_bool");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::ConstBool(true)));
                b.terminator = crate::mir::Terminator::Return(v0);
            }

            let result = compile_and_exec(f, &interner);
            assert!(result.is_some());
            let val = Value::from_bits(result.unwrap());
            assert_eq!(val.as_bool(), Some(true));
        }

        #[test]
        fn test_jit_exec_boxed_neg() {
            let mut interner = Interner::new();
            let name = interner.intern("boxed_neg");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            let v1 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::ConstNum(7.0)));
                b.instructions.push((v1, crate::mir::Instruction::Neg(v0)));
                b.terminator = crate::mir::Terminator::Return(v1);
            }

            let result = compile_and_exec(f, &interner);
            assert!(result.is_some());
            let val = Value::from_bits(result.unwrap());
            assert_eq!(val.as_num(), Some(-7.0));
        }

        #[test]
        fn test_jit_exec_boxed_not() {
            let mut interner = Interner::new();
            let name = interner.intern("boxed_not");
            let mut f = MirFunction::new(name, 0);
            let bb = f.new_block();
            let v0 = f.new_value();
            let v1 = f.new_value();
            {
                let b = f.block_mut(bb);
                b.instructions
                    .push((v0, crate::mir::Instruction::ConstBool(false)));
                b.instructions.push((v1, crate::mir::Instruction::Not(v0)));
                b.terminator = crate::mir::Terminator::Return(v1);
            }

            let result = compile_and_exec(f, &interner);
            assert!(result.is_some());
            let val = Value::from_bits(result.unwrap());
            assert_eq!(val.as_bool(), Some(true));
        }
    }
}
