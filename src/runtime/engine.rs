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
    let constfold = ConstFold;
    let dce = Dce;
    let cse = Cse;
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
            PROFILE_UNSEEN if profile.is_none() => {
                // No profile at all → blind speculation (legacy)
                guards.push((mir.new_value(), Instruction::GuardNum(*vid)));
            }
            _ => {
                // Object, Mixed, Null, String, or Unseen-with-profile → no guard
            }
        }
    }

    for (i, guard) in guards.into_iter().enumerate() {
        mir.blocks[0].instructions.insert(insert_pos + i, guard);
    }
}

fn tier_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WLIFT_TIER_TRACE").is_some())
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
    /// Active native OSR entry points indexed by FuncId.
    pub jit_osr_entries: Vec<Vec<NativeOsrEntry>>,
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
    pub runtime_call_stats: RuntimeCallStats,
    /// Cached field index for trivial getters of the form `{ _field }`.
    pub trivial_getter_fields: Vec<Option<u16>>,
    /// Cached field index for trivial setters of the form `{ _field = value }`.
    pub trivial_setter_fields: Vec<Option<u16>>,
    /// Baseline native code pointers indexed by FuncId.
    pub baseline_code: Vec<*const u8>,
    /// Baseline native OSR entry points indexed by FuncId.
    pub baseline_osr_entries: Vec<Vec<NativeOsrEntry>>,
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
    /// Bytecode pointers indexed by FuncId. O(1) lookup for bytecode dispatch.
    /// null entries mean bytecode not yet compiled for this function.
    pub bc_cache: Vec<*const crate::mir::bytecode::BytecodeFunction>,
    /// Type profiles indexed by FuncId. Persists across tier transitions.
    pub type_profiles: Vec<Option<TypeProfile>>,
    /// Code ranges for compiled functions, sorted by start address.
    /// Used by GC stack walker to map return addresses → safepoint metadata.
    pub code_ranges: Vec<CodeRange>,
    /// Threaded-code cache indexed by FuncId. Lazily populated on first
    /// interpreter dispatch — faster than bytecode for hot functions.
    /// `None` = not yet checked. `Some(None)` = checked, not eligible.
    /// `Some(Some(tc))` = eligible, threaded code ready.
    pub threaded_code: Vec<Option<Option<crate::mir::threaded::ThreadedCode>>>,
    /// Tier-up promotion broker. Phase 2 wires this up in shadow mode
    /// alongside the existing call_count / compile_queue machinery — it
    /// ticks on every recorded call but doesn't yet drive any tier
    /// decision. Later phases will retire the legacy path in its favour.
    pub tier: super::tier::TierManager,
}

/// Address range of a compiled function's native code.
#[derive(Debug, Clone)]
pub struct CodeRange {
    pub start: usize,
    pub end: usize,
    pub func_id: FuncId,
    pub metadata: Arc<NativeFrameMetadata>,
}

/// Per-module execution state.
pub struct ModuleEntry {
    /// The top-level function for this module.
    pub top_level: FuncId,
    /// Module-level variable storage (indexed by slot number).
    pub vars: Vec<super::value::Value>,
    /// Variable names corresponding to each slot (for C API lookup).
    pub var_names: Vec<String>,
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
            jit_osr_entries: Vec::new(),
            jit_leaf: Vec::new(),
            jit_metadata: Vec::new(),
            tier_states: Vec::new(),
            tier_stats: Vec::new(),
            func_modules: Vec::new(),
            runtime_call_stats: RuntimeCallStats::default(),
            trivial_getter_fields: Vec::new(),
            trivial_setter_fields: Vec::new(),
            baseline_code: Vec::new(),
            baseline_osr_entries: Vec::new(),
            baseline_leaf: Vec::new(),
            baseline_metadata: Vec::new(),
            optimized_code: Vec::new(),
            optimized_osr_entries: Vec::new(),
            optimized_leaf: Vec::new(),
            optimized_metadata: Vec::new(),
            bc_cache: Vec::new(),
            type_profiles: Vec::new(),
            code_ranges: Vec::new(),
            threaded_code: Vec::new(),
            // Thresholds mirror the legacy jit_threshold / opt_threshold
            // defaults above so shadow-mode tick counts stay in step with
            // the existing counters they'll eventually replace.
            tier: super::tier::TierManager::with_thresholds(
                super::tier::BASELINE_THRESHOLD,
                super::tier::OPTIMIZED_THRESHOLD,
            ),
        }
    }

    /// Register a MIR function and return its FuncId. The function is
    /// recorded as module-less — only suitable for isolated tests.
    /// Production paths should use [`register_function_in`] so call
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
        self.jit_osr_entries.push(Vec::new());
        self.jit_leaf.push(false);
        self.jit_metadata.push(None);
        self.tier_states.push(TierState::Interpreted);
        self.tier_stats.push(FuncTierStats::default());
        self.func_modules.push(module);
        self.trivial_getter_fields.push(trivial_getter);
        self.trivial_setter_fields.push(trivial_setter);
        self.baseline_code.push(std::ptr::null());
        self.baseline_osr_entries.push(Vec::new());
        self.baseline_leaf.push(false);
        self.baseline_metadata.push(None);
        self.optimized_code.push(std::ptr::null());
        self.optimized_osr_entries.push(Vec::new());
        self.optimized_leaf.push(false);
        self.optimized_metadata.push(None);
        self.bc_cache.push(std::ptr::null());
        self.threaded_code.push(None); // None = not yet checked
        self.type_profiles.push(None);
        // Shadow-mode registration. The core pointer stays null for now —
        // we don't need round-trip-to-Wren-fn yet. The bead exists purely
        // to track per-function state transitions in parallel with the
        // legacy call_count field. Phase 3 will hand the compile closure
        // real context and start driving tier decisions through this.
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
            if !crate::mir::threaded::can_use_threaded(&mir) {
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
    pub fn ensure_bytecode(&mut self, id: FuncId) -> Option<*const BytecodeFunction> {
        let idx = id.0 as usize;
        // Fast path: check bc_cache first (O(1), no enum match)
        if idx < self.bc_cache.len() {
            let cached = self.bc_cache[idx];
            if !cached.is_null() {
                return Some(cached);
            }
        }
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
                "tier-trace: ic_ptrs FuncId({}) total={} kind5={}",
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
    fn mir_trivial_getter_field(mir: &crate::mir::MirFunction) -> Option<u16> {
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

    fn sync_active_tier_cache(&mut self, idx: usize) {
        let state = self
            .tier_states
            .get(idx)
            .copied()
            .unwrap_or(TierState::Interpreted);
        match state {
            TierState::Interpreted => {
                self.jit_code[idx] = std::ptr::null();
                self.jit_osr_entries[idx].clear();
                self.jit_leaf[idx] = false;
                self.jit_metadata[idx] = None;
            }
            TierState::BaselineNative => {
                self.jit_code[idx] = self.baseline_code[idx];
                self.jit_osr_entries[idx] = self.baseline_osr_entries[idx].clone();
                self.jit_leaf[idx] = self.baseline_leaf[idx];
                self.jit_metadata[idx] = self.baseline_metadata[idx].clone();
            }
            TierState::OptimizedNative => {
                self.jit_code[idx] = self.optimized_code[idx];
                self.jit_osr_entries[idx] = self.optimized_osr_entries[idx].clone();
                self.jit_leaf[idx] = self.optimized_leaf[idx];
                self.jit_metadata[idx] = self.optimized_metadata[idx].clone();
            }
        }
    }

    pub fn active_osr_entry(
        &self,
        id: FuncId,
        target_block: crate::mir::BlockId,
        param_count: usize,
    ) -> Option<NativeOsrEntry> {
        self.jit_osr_entries
            .get(id.0 as usize)?
            .iter()
            .copied()
            .find(|entry| {
                entry.target_block == target_block
                    && entry.param_count as usize == param_count
                    && !entry.ptr.is_null()
            })
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
        self.sync_active_tier_cache(idx);
        if !self.collect_tier_stats {
            return;
        }
        if let Some(stats) = self.tier_stats.get_mut(idx) {
            stats.deopts_to_baseline += 1;
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

    fn build_compile_mir(
        mir: &Arc<MirFunction>,
        tier: CompileTier,
        interner: &crate::intern::Interner,
        profile: Option<&TypeProfile>,
    ) -> Arc<MirFunction> {
        match tier {
            CompileTier::Baseline => {
                let mut baseline_mir = (**mir).clone();
                // Insert speculative guards at baseline when profile data is
                // available (collected during the 100 interpreted calls).
                // This enables TypeSpecialize to convert boxed arith to
                // native f64 ops, giving 10-40x speedup on numeric code.
                if profile.is_some() {
                    insert_speculative_guards(&mut baseline_mir, profile);
                }
                run_jit_opt_pipeline(&mut baseline_mir, interner);
                Arc::new(baseline_mir)
            }
            CompileTier::Optimized => {
                let mut spec = (**mir).clone();
                insert_speculative_guards(&mut spec, profile);
                run_jit_opt_pipeline(&mut spec, interner);
                Arc::new(spec)
            }
        }
    }

    /// WLIFT_TIER_TRACE helper — prints the engine's TierState alongside
    /// the beadie bead's BeadState and invocation count at a given event.
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
            "tier-trace: {event} {tier:?} FuncId({}) engine={engine_tier:?} bead={bead_state} invocations={invocations}",
            idx
        );
    }

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
                    native @ FuncBody::Native { .. } => native,
                };
                self.functions[idx] = body;
                self.baseline_code[idx] = native_ptr;
                self.baseline_osr_entries[idx] = osr_entries;
                self.baseline_leaf[idx] = inline_safe;
                self.baseline_metadata[idx] = native_meta;
                self.tier_states[idx] = TierState::BaselineNative;
                // Tell the bead a compiled artifact is live so its
                // state machine tracks reality. Non-native tiers (WASM
                // fallback) have a null pointer and stay Interpreted
                // on the bead side — install_or_swap rejects nulls.
                if !native_ptr.is_null() {
                    self.tier
                        .install_or_swap(FuncId(idx as u32), native_ptr as *mut ());
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
                self.optimized_osr_entries[idx] = osr_entries;
                self.optimized_leaf[idx] = inline_safe;
                self.optimized_metadata[idx] = native_meta;
                self.tier_states[idx] = TierState::OptimizedNative;
                // Swap the bead's pointer over to the optimized tier.
                // install_or_swap picks `swap_compiled` when already
                // Compiled, `eager_install` otherwise.
                if !native_ptr.is_null() {
                    self.tier
                        .install_or_swap(FuncId(idx as u32), native_ptr as *mut ());
                }
                if tier_trace_enabled() {
                    self.emit_bead_trace(idx, tier, "install-done");
                }
            }
        }

        if let Some(bytecode) = self.peek_bytecode(FuncId(idx as u32)) {
            self.bc_cache[idx] = Arc::as_ptr(&bytecode);
        }
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
        let idx = id.0 as usize;
        match self.functions.get_mut(idx) {
            Some(FuncBody::Interpreted { .. }) => {
                // Baseline tier-up counting lives on the beadie bead.
                // Fires true exactly once when the bead's invocation
                // count first equals `jit_threshold`.
                self.tier.should_promote_on_tick(id, self.jit_threshold)
            }
            Some(FuncBody::Native {
                optimized_executable,
                ..
            }) if optimized_executable.is_none()
                && self.tier_states.get(idx).copied() == Some(TierState::BaselineNative)
                && self.compiling_tier.get(idx).copied().flatten().is_none() =>
            {
                // Optimized tier-up uses the SAME bead counter, just with
                // an absolute threshold of (baseline_threshold + opt_threshold).
                // Since the counter is monotonically increasing and
                // comparison is exact equality, this fires exactly once
                // per function — matching the legacy `opt_call_count ==
                // opt_threshold` semantics without a second counter.
                let total = self.jit_threshold.saturating_add(self.opt_threshold);
                self.tier.should_promote_on_tick(id, total)
            }
            _ => false,
        }
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
    pub fn tier_up(&mut self, id: FuncId, interner: &crate::intern::Interner) -> bool {
        let idx = id.0 as usize;
        let Some(tier) = self.next_compile_tier(idx) else {
            return self.tier_state(id) != TierState::Interpreted;
        };
        let Some(body) = self.functions.get(idx) else {
            return false;
        };
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
        let compile_mir = Self::build_compile_mir(&mir, tier, interner, profile.as_ref());
        let (callsite_ic_ptrs, callsite_ic_live_ptrs) = self
            .callsite_ic_data_for_compile(id)
            .map(|(s, l)| (Some(s), Some(l)))
            .unwrap_or((None, None));
        let devirt_hints = callsite_ic_ptrs
            .as_ref()
            .map(|ics| self.compute_devirt_hints(ics));
        let jit_code_base = Some(self.jit_code.as_ptr());
        if std::env::var("WLIFT_JIT_DUMP").is_ok() {
            eprintln!("=== {:?} compile FuncId({}) ===", tier, id.0);
            eprintln!("{}", compile_mir.pretty_print(interner));
        }
        let target = Self::native_target();
        let compiled =
            match crate::codegen::compile_function_artifact_with_interner_and_callsite_ics(
                &compile_mir,
                target,
                interner,
                tier,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                devirt_hints,
                jit_code_base,
            ) {
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
    pub fn request_tier_up(&mut self, id: FuncId, interner: &crate::intern::Interner) {
        if self.mode != ExecutionMode::Tiered {
            return;
        }
        // Respect the deopt policy: functions blacklisted after too many
        // bailouts stay in the interpreter for the rest of the run.
        if self.tier.is_blacklisted(id) {
            return;
        }
        let idx = id.0 as usize;
        let Some(tier) = self.next_compile_tier(idx) else {
            return;
        };
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
        // Don't start optimized-tier compilation while any other compile
        // is in flight — the optimizer can hang on complex functions and
        // would block all progress. `pending_count` is bumped per submit
        // and decremented in `poll_compilations`, so it reflects all
        // baseline submissions currently walking the broker's pipeline.
        if tier == CompileTier::Optimized && self.pending_count != 0 {
            return;
        }
        if idx >= self.compiling_tier.len() || self.compiling_tier[idx].is_some() {
            return;
        }
        let Some(body) = self.functions.get(idx) else {
            return;
        };
        let mir = Arc::clone(body.mir());
        let profile = self.get_type_profile(id).cloned();
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
        let (callsite_ic_ptrs, callsite_ic_live_ptrs) = self
            .callsite_ic_data_for_compile(id)
            .map(|(s, l)| (Some(s), Some(l)))
            .unwrap_or((None, None));
        let devirt_hints = callsite_ic_ptrs
            .as_ref()
            .map(|ics| self.compute_devirt_hints(ics));
        let jit_code_base_raw = self.jit_code.as_ptr() as usize;

        if tier_trace_enabled() {
            let ic_count = callsite_ic_ptrs.as_ref().map(|v| v.len()).unwrap_or(0);
            eprintln!(
                "tier-trace: queue {:?} FuncId({}) {} ic_ptrs={}",
                tier, id.0, trace_name, ic_count
            );
        }

        // Compile closure — runs on beadie's broker thread. Returns the
        // raw native pointer for beadie's state machine AND sends the
        // richer install artifact (executable, metadata, tier) back to
        // the interpreter thread through `compilation_tx` so
        // `poll_compilations` can finish the install at a safepoint.
        let compile_fn = move |_bead: &std::sync::Arc<beadie::Bead>| -> *mut () {
            if tier_trace_enabled() {
                eprintln!(
                    "tier-trace: start {:?} FuncId({}) {}",
                    tier, id.0, trace_name_clone
                );
            }
            let compile_mir =
                Self::build_compile_mir(&mir, tier, &interner_clone, profile.as_ref());
            if std::env::var("WLIFT_JIT_DUMP").is_ok() {
                eprintln!("=== {:?} compile FuncId({}) ===", tier, id.0);
                eprintln!("{}", compile_mir.pretty_print(&interner_clone));
            }
            let result = crate::codegen::compile_function_artifact_with_interner_and_callsite_ics(
                &compile_mir,
                target,
                &interner_clone,
                tier,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                devirt_hints,
                Some(jit_code_base_raw as *const *const u8),
            )
            .map_err(|e| {
                eprintln!("COMPILE ERR FuncId({}): {}", id.0, e);
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
                        eprintln!("EXEC ERR FuncId({}): {}", id.0, e);
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
                    "tier-trace: finish {:?} FuncId({}) {} success={}",
                    tier,
                    id.0,
                    trace_name_clone,
                    result.is_some()
                );
            }
            // Extract the native pointer for beadie BEFORE sending the
            // executable across the mpsc boundary. The pointer refers
            // into a heap-allocated mmap owned by the executable — the
            // mmap's address doesn't change when the ExecutableFunction
            // moves through the channel, so the pointer remains valid
            // through install_or_swap at the receiver.
            let native_ptr = match &result {
                Some(CompilationResult::Compiled { executable, .. }) if executable.is_native() => {
                    executable.native_ptr() as *mut ()
                }
                _ => std::ptr::null_mut(),
            };
            let _ = tx.send(result.unwrap_or(CompilationResult::Failed { id }));
            native_ptr
        };

        // Submit to beadie's broker. The bead goes Interpreted → Queued
        // → Compiling → Compiled as the closure progresses. If the bead
        // has already been promoted (race), `AlreadyQueued` is returned
        // and we roll back the pending_count bookkeeping.
        let submit_result = self.tier.submit_compile(id, compile_fn);
        if !submit_result.is_accepted() {
            self.compiling_tier[idx] = None;
            self.pending_count = self.pending_count.saturating_sub(1);
            if tier_trace_enabled() {
                eprintln!(
                    "tier-trace: submit-rejected {:?} FuncId({}) {:?}",
                    tier, id.0, submit_result
                );
            }
        }
    }

    /// Install any completed background compilations.
    #[inline]
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
                        eprintln!("tier-trace: install {:?} FuncId({})", tier, id.0);
                    }
                    CompilationResult::Failed { id } => {
                        eprintln!("tier-trace: install failed FuncId({})", id.0);
                    }
                }
            }
            if idx < self.functions.len() {
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
        assert!(engine.jit_osr_entries[id.0 as usize].is_empty());
        assert!(engine.baseline_osr_entries[id.0 as usize].is_empty());
        assert!(engine.optimized_osr_entries[id.0 as usize].is_empty());
        assert!(engine
            .active_osr_entry(id, crate::mir::BlockId(0), 0)
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
