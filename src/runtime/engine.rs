/// Tiered execution engine for WrenLift.
///
/// Manages function bodies across three execution tiers:
/// - Interpreter: walk MIR directly
/// - Tiered: start interpreted, JIT-compile hot functions
/// - JIT: compile everything to native before execution
///
/// The engine owns the function registry (MIR + compiled code) and
/// handles tier-up decisions based on call-count profiling.
use std::collections::HashMap;
use std::sync::Arc;

use crate::codegen::ExecutableFunction;
use crate::intern::SymbolId;
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
pub enum FuncBody {
    /// MIR available for interpretation. Not yet compiled to native.
    Interpreted {
        mir: Arc<MirFunction>,
        /// Number of times this function has been called (for tier-up).
        call_count: u32,
    },
    /// JIT-compiled native code. MIR retained for deopt/debugging.
    Compiled {
        executable: ExecutableFunction,
        mir: Arc<MirFunction>,
    },
}

impl FuncBody {
    /// Get a shared reference to the MIR (available in all tiers).
    /// Returns an Arc so callers can hold the MIR independently of the engine.
    pub fn mir(&self) -> &Arc<MirFunction> {
        match self {
            FuncBody::Interpreted { mir, .. } | FuncBody::Compiled { mir, .. } => mir,
        }
    }

    /// Whether this function has been compiled to native code.
    pub fn is_compiled(&self) -> bool {
        matches!(self, FuncBody::Compiled { .. })
    }
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

/// The execution engine: owns all function bodies, dispatches calls,
/// and manages tiered compilation.
pub struct ExecutionEngine {
    /// Current execution mode.
    pub mode: ExecutionMode,
    /// All registered functions, indexed by FuncId.
    pub functions: Vec<FuncBody>,
    /// Profiling threshold: calls before JIT compilation triggers.
    pub jit_threshold: u32,
    /// Module registry: module name → (top_level FuncId, module var storage).
    pub modules: HashMap<String, ModuleEntry>,
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
        Self {
            mode,
            functions: Vec::new(),
            jit_threshold: 100,
            modules: HashMap::new(),
        }
    }

    /// Register a MIR function and return its FuncId.
    pub fn register_function(&mut self, mir: MirFunction) -> FuncId {
        let id = FuncId(self.functions.len() as u32);
        self.functions.push(FuncBody::Interpreted {
            mir: Arc::new(mir),
            call_count: 0,
        });
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

    /// Record a call to a function. Returns true if the function should
    /// be JIT-compiled (threshold exceeded in Tiered mode).
    pub fn record_call(&mut self, id: FuncId) -> bool {
        if self.mode != ExecutionMode::Tiered {
            return false;
        }
        if let Some(FuncBody::Interpreted { call_count, .. }) =
            self.functions.get_mut(id.0 as usize)
        {
            *call_count += 1;
            *call_count >= self.jit_threshold
        } else {
            false
        }
    }

    /// Compile a function to native code and replace its body.
    /// Returns true if compilation succeeded.
    pub fn tier_up(&mut self, id: FuncId, interner: &crate::intern::Interner) -> bool {
        let target = Self::native_target();
        let mir = match self.functions.get(id.0 as usize) {
            Some(FuncBody::Interpreted { mir, .. }) => Arc::clone(mir),
            Some(FuncBody::Compiled { .. }) => return true, // already compiled
            None => return false,
        };

        match crate::codegen::compile_function_with_interner(&mir, target, interner) {
            Ok(compiled) => match compiled.into_executable() {
                Ok(executable) => {
                    self.functions[id.0 as usize] = FuncBody::Compiled { executable, mir };
                    true
                }
                Err(_) => false,
            },
            Err(_) => false,
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
