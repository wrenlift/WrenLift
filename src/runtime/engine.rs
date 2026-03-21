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
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::OnceLock;

use crate::codegen::native_meta::NativeFrameMetadata;
use crate::codegen::{CompileTier, ExecutableFunction};
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

pub enum FuncBody {
    /// MIR available for interpretation. Not yet compiled to native.
    Interpreted {
        mir: Arc<MirFunction>,
        /// Lazily-lowered compact bytecode for the bytecode VM.
        bytecode: Option<Arc<BytecodeFunction>>,
        /// Number of interpreted calls seen before baseline tier-up.
        call_count: u32,
    },
    /// Baseline native code is resident; optimized code may be installed later.
    Native {
        baseline_executable: ExecutableFunction,
        optimized_executable: Option<ExecutableFunction>,
        mir: Arc<MirFunction>,
        bytecode: Option<Arc<BytecodeFunction>>,
        /// Number of baseline-native calls seen before optimize tier-up.
        opt_call_count: u32,
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
    pub deopts_to_baseline: u64,
    pub fallbacks_to_interpreter: u64,
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
        &range_loop, &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
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
fn is_compiled_leaf(
    native_meta: &Option<std::sync::Arc<crate::codegen::native_meta::NativeFrameMetadata>>,
) -> bool {
    native_meta
        .as_ref()
        .map(|meta| {
            meta.safepoints.iter().all(|sp| {
                sp.kind == crate::codegen::native_meta::SafepointKind::CallLocal
            })
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
    /// Number of compilations currently in flight (fast check for poll_compilations).
    pending_count: u32,
    /// Join handle for the current compilation thread (at most 1 at a time).
    compile_handle: Option<std::thread::JoinHandle<()>>,
    /// JIT native code pointers indexed by FuncId. O(1) lookup for fast dispatch.
    /// null_ptr entries mean the function is not yet compiled.
    pub jit_code: Vec<*const u8>,
    /// Whether the currently active native tier can use the direct fast path.
    pub jit_leaf: Vec<bool>,
    /// Preserved native-frame metadata for compiled functions.
    pub jit_metadata: Vec<Option<Arc<NativeFrameMetadata>>>,
    /// Active execution tier for each function.
    pub tier_states: Vec<TierState>,
    /// Per-function tier and dispatch statistics.
    pub tier_stats: Vec<FuncTierStats>,
    /// Baseline native code pointers indexed by FuncId.
    pub baseline_code: Vec<*const u8>,
    /// Whether baseline-native code can use the direct fast path.
    pub baseline_leaf: Vec<bool>,
    /// Baseline native metadata indexed by FuncId.
    pub baseline_metadata: Vec<Option<Arc<NativeFrameMetadata>>>,
    /// Optimized native code pointers indexed by FuncId.
    pub optimized_code: Vec<*const u8>,
    /// Whether optimized-native code can use the direct fast path.
    pub optimized_leaf: Vec<bool>,
    /// Optimized native metadata indexed by FuncId.
    pub optimized_metadata: Vec<Option<Arc<NativeFrameMetadata>>>,
    /// Bytecode pointers indexed by FuncId. O(1) lookup for bytecode dispatch.
    /// null entries mean bytecode not yet compiled for this function.
    pub bc_cache: Vec<*const crate::mir::bytecode::BytecodeFunction>,
    /// Type profiles indexed by FuncId. Persists across tier transitions.
    pub type_profiles: Vec<Option<TypeProfile>>,
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
            compile_handle: None,
            jit_code: Vec::new(),
            jit_leaf: Vec::new(),
            jit_metadata: Vec::new(),
            tier_states: Vec::new(),
            tier_stats: Vec::new(),
            baseline_code: Vec::new(),
            baseline_leaf: Vec::new(),
            baseline_metadata: Vec::new(),
            optimized_code: Vec::new(),
            optimized_leaf: Vec::new(),
            optimized_metadata: Vec::new(),
            bc_cache: Vec::new(),
            type_profiles: Vec::new(),
        }
    }

    /// Register a MIR function and return its FuncId.
    pub fn register_function(&mut self, mir: MirFunction) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        self.functions.push(FuncBody::Interpreted {
            mir: Arc::new(mir),
            bytecode: None,
            call_count: 0,
        });
        self.compiling_tier.push(None);
        self.jit_code.push(std::ptr::null());
        self.jit_leaf.push(false);
        self.jit_metadata.push(None);
        self.tier_states.push(TierState::Interpreted);
        self.tier_stats.push(FuncTierStats::default());
        self.baseline_code.push(std::ptr::null());
        self.baseline_leaf.push(false);
        self.baseline_metadata.push(None);
        self.optimized_code.push(std::ptr::null());
        self.optimized_leaf.push(false);
        self.optimized_metadata.push(None);
        self.bc_cache.push(std::ptr::null());
        self.type_profiles.push(None);
        id
    }

    /// Get the MIR for a function by ID.
    /// Returns an Arc clone so the caller can hold it without borrowing the engine.
    pub fn get_mir(&self, id: FuncId) -> Option<Arc<MirFunction>> {
        self.functions
            .get(id.0 as usize)
            .map(|body| Arc::clone(body.mir()))
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

    fn callsite_ic_ptrs_for_compile(&mut self, id: FuncId) -> Option<Vec<usize>> {
        let bc_ptr = self.ensure_bytecode(id)?;
        let bc = unsafe { &mut *(bc_ptr as *mut BytecodeFunction) };
        let ic_table = unsafe { &mut *bc.ic_table.get() };
        Some(
            ic_table
                .iter_mut()
                .map(|ic| ic as *mut CallSiteIC as usize)
                .collect(),
        )
    }

    /// Whether any background compilations are waiting to be installed.
    #[inline(always)]
    pub fn has_pending_compilations(&self) -> bool {
        self.pending_count != 0
    }

    /// Invalidate all monomorphic call-site inline caches.
    ///
    /// This must run after moving GC because cached class/closure pointers
    /// may have been relocated.
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
                self.jit_leaf[idx] = false;
                self.jit_metadata[idx] = None;
            }
            TierState::BaselineNative => {
                self.jit_code[idx] = self.baseline_code[idx];
                self.jit_leaf[idx] = self.baseline_leaf[idx];
                self.jit_metadata[idx] = self.baseline_metadata[idx].clone();
            }
            TierState::OptimizedNative => {
                self.jit_code[idx] = self.optimized_code[idx];
                self.jit_leaf[idx] = self.optimized_leaf[idx];
                self.jit_metadata[idx] = self.optimized_metadata[idx].clone();
            }
        }
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
            match self.tier_states.get(idx).copied().unwrap_or(TierState::Interpreted) {
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

    pub fn dump_tier_stats(&self, interner: &crate::intern::Interner) {
        eprintln!("=== WLIFT tier stats ===");
        for (idx, body) in self.functions.iter().enumerate() {
            let stats = self.tier_stats.get(idx).copied().unwrap_or_default();
            if stats == FuncTierStats::default() {
                continue;
            }
            let name = interner.resolve(body.mir().name);
            eprintln!(
                "FuncId({idx}) {name} tier={:?} interp={} baseline={} opt={} compiles={}/{} ic={}/{} native2native={} deopts={} fallbacks={}",
                self.tier_states.get(idx).copied().unwrap_or(TierState::Interpreted),
                stats.interpreted_entries,
                stats.baseline_entries,
                stats.optimized_entries,
                stats.compile_successes,
                stats.compile_attempts,
                stats.ic_hits,
                stats.ic_misses,
                stats.native_to_native_calls,
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
                // Baseline: range loop only. Profile-guided guards are
                // applied at optimized tier to avoid disrupting leaf classification.
                let mut baseline_mir = (**mir).clone();
                {
                    use crate::mir::opt::{range_loop::RangeLoop, MirPass};
                    let range_loop = RangeLoop { interner };
                    range_loop.run(&mut baseline_mir);
                }
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

        match tier {
            CompileTier::Baseline => {
                let body = match std::mem::replace(
                    &mut self.functions[idx],
                    FuncBody::Interpreted {
                        mir: Arc::new(MirFunction::new(SymbolId::from_raw(0), 0)),
                        bytecode: None,
                        call_count: 0,
                    },
                ) {
                    FuncBody::Interpreted {
                        mir,
                        bytecode,
                        ..
                    } => FuncBody::Native {
                        baseline_executable: executable,
                        optimized_executable: None,
                        mir,
                        bytecode,
                        opt_call_count: 0,
                    },
                    native @ FuncBody::Native { .. } => native,
                };
                self.functions[idx] = body;
                self.baseline_code[idx] = native_ptr;
                self.baseline_leaf[idx] = inline_safe;
                self.baseline_metadata[idx] = native_meta;
                self.tier_states[idx] = TierState::BaselineNative;
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
                self.optimized_leaf[idx] = inline_safe;
                self.optimized_metadata[idx] = native_meta;
                self.tier_states[idx] = TierState::OptimizedNative;
            }
        }

        if let Some(bytecode) = self.peek_bytecode(FuncId(idx as u32)) {
            self.bc_cache[idx] = Arc::as_ptr(&bytecode);
        }
        self.sync_active_tier_cache(idx);
        self.invalidate_inline_caches();
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
            Some(FuncBody::Interpreted { call_count, .. }) => {
                *call_count += 1;
                *call_count >= self.jit_threshold
            }
            Some(FuncBody::Native {
                opt_call_count,
                optimized_executable,
                ..
            }) if optimized_executable.is_none()
                && self.tier_states.get(idx).copied() == Some(TierState::BaselineNative) =>
            {
                *opt_call_count += 1;
                *opt_call_count >= self.opt_threshold
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
        let mir = Arc::clone(body.mir());
        let profile = self.get_type_profile(id).cloned();
        if self.collect_tier_stats {
            if let Some(stats) = self.tier_stats.get_mut(idx) {
                stats.compile_attempts += 1;
            }
        }
        let compile_mir = Self::build_compile_mir(&mir, tier, interner, profile.as_ref());
        let callsite_ic_ptrs = self.callsite_ic_ptrs_for_compile(id);
        if std::env::var("WLIFT_JIT_DUMP").is_ok() {
            eprintln!("=== {:?} compile FuncId({}) ===", tier, id.0);
            eprintln!("{}", compile_mir.pretty_print(interner));
        }
        let target = Self::native_target();
        let compiled = match crate::codegen::compile_function_artifact_with_interner_and_callsite_ics(
            &compile_mir,
            target,
            interner,
            tier,
            callsite_ic_ptrs,
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
        let idx = id.0 as usize;
        let Some(tier) = self.next_compile_tier(idx) else {
            return;
        };
        if idx >= self.compiling_tier.len() || self.compiling_tier[idx].is_some() {
            return;
        }
        if let Some(ref handle) = self.compile_handle {
            if !handle.is_finished() {
                return;
            }
            if let Some(h) = self.compile_handle.take() {
                let _ = h.join();
            }
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
        let callsite_ic_ptrs = self.callsite_ic_ptrs_for_compile(id);

        if tier_trace_enabled() {
            eprintln!("tier-trace: queue {:?} FuncId({}) {}", tier, id.0, trace_name);
        }

        let compile_fn = move || {
            if tier_trace_enabled() {
                eprintln!(
                    "tier-trace: start {:?} FuncId({}) {}",
                    tier, id.0, trace_name_clone
                );
            }
            let compile_mir = Self::build_compile_mir(&mir, tier, &interner_clone, profile.as_ref());
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
            )
            .ok()
            .and_then(|artifact| {
                let native_meta = artifact.native_meta;
                let inline_safe = is_mir_inline_safe(&compile_mir, tier)
                    || !artifact.needs_shadow_frame;
                artifact
                    .code
                    .into_executable()
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
            let _ = tx.send(result.unwrap_or(CompilationResult::Failed { id }));
        };

        self.compile_handle = Some(std::thread::spawn(compile_fn));
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
                }
            }
            if idx < self.compiling_tier.len() {
                self.compiling_tier[idx] = None;
            }
            self.pending_count = self.pending_count.saturating_sub(1);
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
        // Only join a finished compilation thread. Waiting for a live tier-up
        // worker here can stall process exit long after the program has already
        // printed its result, especially if an optimize-tier compile started
        // right before shutdown. Dropping the JoinHandle detaches the worker.
        if let Some(handle) = self.compile_handle.take() {
            if handle.is_finished() {
                if tier_trace_enabled() {
                    eprintln!("tier-trace: drop joining finished compile thread");
                }
                let _ = handle.join();
                if tier_trace_enabled() {
                    eprintln!("tier-trace: drop compile thread joined");
                }
            } else if tier_trace_enabled() {
                eprintln!("tier-trace: drop detaching live compile thread");
            }
        }
        // Drain any pending results so compiled code buffers are freed.
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
                current_func_id: u32::MAX,
                closure: std::ptr::null_mut(),
                defining_class: std::ptr::null_mut(),
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
