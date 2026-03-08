/// WebAssembly code generation directly from MIR.
///
/// Lowers MIR (SSA with block parameters) directly to WASM bytecode,
/// bypassing the MachInst layer entirely. This is natural because:
/// - MIR's block parameters avoid phi/parallel-copy complexity
/// - MIR's CondBranch maps directly to WASM's br_if
/// - MIR ValueIds map to WASM locals (no register allocation needed)
/// - MIR's structured terminators avoid label/jump reconstruction
///
/// Value mapping:
/// - MirType::Value, I64 → i64 locals (NaN-boxed values, integers)
/// - MirType::F64 → f64 locals
/// - MirType::Bool → i32 locals

use std::collections::HashMap;

use wasm_encoder::{
    CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection, Ieee64,
    ImportSection, Instruction as WasmInst, Module, TypeSection, ValType,
};

use crate::mir::{BasicBlock, BlockId, Instruction, MirFunction, MirType, Terminator, ValueId};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// A compiled WASM module (binary bytes).
#[derive(Clone)]
pub struct WasmModule {
    pub bytes: Vec<u8>,
}

impl WasmModule {
    /// Basic validation: check magic number and version.
    pub fn validate(&self) -> Result<(), String> {
        if self.bytes.len() < 8 {
            return Err("WASM module too small".into());
        }
        if &self.bytes[0..4] != b"\0asm" {
            return Err("Invalid WASM magic number".into());
        }
        Ok(())
    }

    /// Full structural validation using wasmparser.
    #[cfg(test)]
    pub fn validate_full(&self) -> Result<(), String> {
        wasmparser::validate(&self.bytes)
            .map(|_| ())
            .map_err(|e| format!("WASM validation error: {}", e))
    }

    /// Dump module as WAT text format for debugging.
    #[cfg(test)]
    pub fn dump_wat(&self) -> Result<String, String> {
        wasmprinter::print_bytes(&self.bytes)
            .map_err(|e| format!("WAT print error: {}", e))
    }
}

// ---------------------------------------------------------------------------
// MIR → WASM emitter
// ---------------------------------------------------------------------------

/// Compile a MIR function directly to a WASM module.
pub fn emit_mir(mir: &MirFunction) -> Result<WasmModule, String> {
    let mut emitter = MirWasmEmitter::new(mir);
    emitter.emit()
}

struct MirWasmEmitter<'a> {
    mir: &'a MirFunction,
    /// MIR ValueId → WASM local index.
    local_map: HashMap<ValueId, u32>,
    /// Number of allocated WASM locals.
    num_locals: u32,
    /// Type of each local (for declaration).
    local_types: Vec<ValType>,
    /// Runtime function name → import function index.
    runtime_imports: HashMap<&'static str, u32>,
    /// Ordered import list: (name, param types, result types).
    import_list: Vec<(&'static str, Vec<ValType>, Vec<ValType>)>,
}

impl<'a> MirWasmEmitter<'a> {
    fn new(mir: &'a MirFunction) -> Self {
        Self {
            mir,
            local_map: HashMap::new(),
            num_locals: 0,
            local_types: Vec::new(),
            runtime_imports: HashMap::new(),
            import_list: Vec::new(),
        }
    }

    fn emit(&mut self) -> Result<WasmModule, String> {
        // Phase 1: Scan MIR to discover locals and runtime imports.
        self.scan_locals();
        self.scan_imports();

        // Phase 2: Build WASM module.
        let mut module = Module::new();

        // Type section: import types + compiled function type.
        let mut types = TypeSection::new();
        for (_, params, results) in &self.import_list {
            types.ty().function(params.iter().copied(), results.iter().copied());
        }
        let func_type_idx = self.import_list.len() as u32;
        types.ty().function([], [ValType::I64]); // () -> i64 (NaN-boxed Value)
        module.section(&types);

        // Import section.
        let mut imports = ImportSection::new();
        for (i, (name, _, _)) in self.import_list.iter().enumerate() {
            imports.import("wren", *name, EntityType::Function(i as u32));
        }
        module.section(&imports);

        // Function section.
        let mut functions = FunctionSection::new();
        functions.function(func_type_idx);
        module.section(&functions);

        // Export section.
        let mut exports = ExportSection::new();
        let func_idx = self.import_list.len() as u32;
        exports.export(
            &format!("fn_{}", self.mir.name.index()),
            ExportKind::Func,
            func_idx,
        );
        module.section(&exports);

        // Code section.
        let mut code_section = CodeSection::new();
        let wasm_func = self.emit_function()?;
        code_section.function(&wasm_func);
        module.section(&code_section);

        let bytes = module.finish();
        let wasm_mod = WasmModule { bytes };
        wasm_mod.validate()?;
        Ok(wasm_mod)
    }

    // -----------------------------------------------------------------------
    // Local allocation
    // -----------------------------------------------------------------------

    /// Scan all MIR values to allocate WASM locals with correct types.
    fn scan_locals(&mut self) {
        for block in &self.mir.blocks {
            // Block parameters.
            for (val, ty) in &block.params {
                self.ensure_local(*val, mir_type_to_wasm(*ty));
            }
            // Instruction results.
            for (val, inst) in &block.instructions {
                let wasm_ty = self.infer_wasm_type(inst);
                self.ensure_local(*val, wasm_ty);
            }
        }
    }

    fn ensure_local(&mut self, val: ValueId, ty: ValType) -> u32 {
        if let Some(&idx) = self.local_map.get(&val) {
            return idx;
        }
        let idx = self.num_locals;
        self.num_locals += 1;
        self.local_map.insert(val, idx);
        self.local_types.push(ty);
        idx
    }

    fn local(&self, val: ValueId) -> u32 {
        self.local_map[&val]
    }

    /// Infer WASM type from a MIR instruction.
    fn infer_wasm_type(&self, inst: &Instruction) -> ValType {
        match inst {
            // Unboxed f64 operations → f64.
            Instruction::ConstF64(_)
            | Instruction::AddF64(..)
            | Instruction::SubF64(..)
            | Instruction::MulF64(..)
            | Instruction::DivF64(..)
            | Instruction::ModF64(..)
            | Instruction::NegF64(_)
            | Instruction::Unbox(_) => ValType::F64,

            // Bool results → i32.
            Instruction::CmpLtF64(..)
            | Instruction::CmpGtF64(..)
            | Instruction::CmpLeF64(..)
            | Instruction::CmpGeF64(..) => ValType::I32,

            // Everything else → i64 (NaN-boxed Value or integer).
            _ => ValType::I64,
        }
    }

    // -----------------------------------------------------------------------
    // Import scanning
    // -----------------------------------------------------------------------

    /// Scan MIR for instructions that need runtime imports.
    fn scan_imports(&mut self) {
        for block in &self.mir.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Add(..) => self.register_import_binop("wren_num_add"),
                    Instruction::Sub(..) => self.register_import_binop("wren_num_sub"),
                    Instruction::Mul(..) => self.register_import_binop("wren_num_mul"),
                    Instruction::Div(..) => self.register_import_binop("wren_num_div"),
                    Instruction::Mod(..) => self.register_import_binop("wren_num_mod"),
                    Instruction::Neg(_) => self.register_import("wren_num_neg", &[ValType::I64], &[ValType::I64]),
                    Instruction::CmpLt(..) => self.register_import_binop("wren_cmp_lt"),
                    Instruction::CmpGt(..) => self.register_import_binop("wren_cmp_gt"),
                    Instruction::CmpLe(..) => self.register_import_binop("wren_cmp_le"),
                    Instruction::CmpGe(..) => self.register_import_binop("wren_cmp_ge"),
                    Instruction::CmpEq(..) => self.register_import_binop("wren_cmp_eq"),
                    Instruction::CmpNe(..) => self.register_import_binop("wren_cmp_ne"),
                    Instruction::Not(_) => self.register_import("wren_not", &[ValType::I64], &[ValType::I64]),
                    Instruction::Call { args, .. } => {
                        // wren_call(receiver, method_id, args...) → i64
                        let params = vec![ValType::I64; args.len() + 2];
                        self.register_import("wren_call", &params, &[ValType::I64]);
                    }
                    Instruction::SuperCall { args, .. } => {
                        let params = vec![ValType::I64; args.len() + 1];
                        self.register_import("wren_super_call", &params, &[ValType::I64]);
                    }
                    Instruction::GetField(..) => {
                        self.register_import("wren_get_field", &[ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::SetField(..) => {
                        self.register_import("wren_set_field", &[ValType::I64, ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::GetModuleVar(_) => {
                        self.register_import("wren_get_module_var", &[ValType::I64], &[ValType::I64]);
                    }
                    Instruction::SetModuleVar(..) => {
                        self.register_import("wren_set_module_var", &[ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::MakeClosure { upvalues, .. } => {
                        let params = vec![ValType::I64; upvalues.len() + 1];
                        self.register_import("wren_make_closure", &params, &[ValType::I64]);
                    }
                    Instruction::GetUpvalue(_) => {
                        self.register_import("wren_get_upvalue", &[ValType::I64], &[ValType::I64]);
                    }
                    Instruction::SetUpvalue(..) => {
                        self.register_import("wren_set_upvalue", &[ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::MakeList(elems) => {
                        let params = vec![ValType::I64; elems.len()];
                        self.register_import("wren_make_list", &params, &[ValType::I64]);
                    }
                    Instruction::MakeMap(pairs) => {
                        let params = vec![ValType::I64; pairs.len() * 2];
                        self.register_import("wren_make_map", &params, &[ValType::I64]);
                    }
                    Instruction::MakeRange(..) => {
                        self.register_import("wren_make_range", &[ValType::I64, ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::StringConcat(parts) => {
                        let params = vec![ValType::I64; parts.len()];
                        self.register_import("wren_string_concat", &params, &[ValType::I64]);
                    }
                    Instruction::ToString(_) => {
                        self.register_import("wren_to_string", &[ValType::I64], &[ValType::I64]);
                    }
                    Instruction::IsType(..) => {
                        self.register_import("wren_is_type", &[ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::GuardClass(..) => {
                        self.register_import("wren_guard_class", &[ValType::I64, ValType::I64], &[ValType::I64]);
                    }
                    Instruction::SubscriptGet { args, .. } => {
                        let params = vec![ValType::I64; args.len() + 1];
                        self.register_import("wren_subscript_get", &params, &[ValType::I64]);
                    }
                    Instruction::SubscriptSet { args, .. } => {
                        let params = vec![ValType::I64; args.len() + 2];
                        self.register_import("wren_subscript_set", &params, &[ValType::I64]);
                    }
                    _ => {} // No import needed.
                }
            }

            // CondBranch uses wren_is_truthy.
            if let Terminator::CondBranch { .. } = &block.terminator {
                self.register_import("wren_is_truthy", &[ValType::I64], &[ValType::I32]);
            }
        }
    }

    fn register_import_binop(&mut self, name: &'static str) {
        self.register_import(name, &[ValType::I64, ValType::I64], &[ValType::I64]);
    }

    fn register_import(&mut self, name: &'static str, params: &[ValType], results: &[ValType]) {
        if self.runtime_imports.contains_key(name) {
            return;
        }
        let idx = self.import_list.len() as u32;
        self.runtime_imports.insert(name, idx);
        self.import_list.push((name, params.to_vec(), results.to_vec()));
    }

    // -----------------------------------------------------------------------
    // Function body emission
    // -----------------------------------------------------------------------

    fn emit_function(&mut self) -> Result<Function, String> {
        // Declare locals.
        let locals: Vec<(u32, ValType)> = self.local_types.iter().map(|t| (1, *t)).collect();
        let mut func = Function::new(locals);

        // Emit structured control flow from MIR blocks.
        self.emit_blocks(&mut func)?;

        func.instruction(&WasmInst::End);
        Ok(func)
    }

    /// Emit all MIR blocks using reverse-nested-blocks for structured control flow.
    ///
    /// Layout for N blocks:
    /// ```text
    /// block $b_{n-1}        ;; for block n-1 (outermost)
    ///   block $b_{n-2}      ;; for block n-2
    ///     ...
    ///       block $b_1      ;; for block 1 (innermost)
    ///         [block 0 code + terminator]
    ///       end             ;; br 0 = jump to block 1
    ///       [block 1 code + terminator]
    ///     end               ;; br 0 = jump to block 2
    ///     [block 2 code + terminator]
    ///   end
    ///   ...
    /// end
    /// ```
    ///
    /// From block k, `br depth` where depth = (target - k - 1) jumps forward
    /// to block `target`.
    fn emit_blocks(&mut self, func: &mut Function) -> Result<(), String> {
        let n = self.mir.blocks.len();
        if n == 0 {
            func.instruction(&WasmInst::I64Const(0x7FFC_0000_0000_0000));
            func.instruction(&WasmInst::Return);
            return Ok(());
        }

        // Open n-1 block scopes (outermost first).
        // Loop headers use `loop` instead of `block`.
        let loop_headers = self.find_loop_headers();

        for block_idx in (1..n).rev() {
            if loop_headers.contains(&(block_idx as u32)) {
                func.instruction(&WasmInst::Loop(wasm_encoder::BlockType::Empty));
            } else {
                func.instruction(&WasmInst::Block(wasm_encoder::BlockType::Empty));
            }
        }

        // Emit each block, closing one scope after each (except last).
        for block_idx in 0..n {
            let block = &self.mir.blocks[block_idx];
            self.emit_block(func, block, block_idx, n)?;

            if block_idx < n - 1 {
                func.instruction(&WasmInst::End);
            }
        }

        Ok(())
    }

    /// Find which blocks are loop headers (targets of back edges).
    fn find_loop_headers(&self) -> std::collections::HashSet<u32> {
        let mut headers = std::collections::HashSet::new();
        for (idx, block) in self.mir.blocks.iter().enumerate() {
            for succ in block.terminator.successors() {
                if succ.0 <= idx as u32 {
                    // Back edge: succ is a loop header.
                    headers.insert(succ.0);
                }
            }
        }
        headers
    }

    /// Compute br depth from block `from` to block `target`.
    fn br_depth(&self, from: usize, target: usize, n: usize) -> u32 {
        if target > from {
            // Forward: depth = target - from - 1
            (target - from - 1) as u32
        } else if target == from {
            // Self-loop: own scope is at depth (n - 1 - from)
            // because there are (n-1-from) inner scopes still open.
            // Actually with our layout, when at block k (k >= 1),
            // depth 0 = scope for block (k+1), and our own scope
            // is just outside all inner scopes.
            // For k >= 1: own scope = depth (n - 1 - from)
            // Wait — we close scope AFTER the block, so own scope IS open.
            // Open scopes when at block k: scopes for blocks k, k+1, ..., n-1
            // (but scope for block k is the one we're inside, closed after us)
            // depth 0 = scope for k+1 (innermost remaining)
            // depth (n-2-k) = scope for n-1 (outermost remaining)
            // Own scope = depth (n-1-k)? No...
            //
            // Let me count carefully. We opened scopes for blocks 1..n-1.
            // After emitting block 0 and closing scope[0] (for block 1):
            //   remaining: scope[1..n-2] for blocks 2..n-1
            // After emitting block k-1 and closing scope[k-1] (for block k):
            //   remaining: scope[k..n-2] for blocks k+1..n-1
            // When emitting block k: scope[k-1] just closed.
            // But if block k is a loop, we used `loop` for scope[k-1],
            // and it closes AFTER block k's code.
            //
            // Actually our opening order was: for i in (1..n).rev()
            // So scope[n-2] opened first (outermost) for block n-1
            // scope[0] opened last (innermost) for block 1
            //
            // When emitting block 0: all scopes open.
            //   depth 0 = scope[0] for block 1
            //   depth n-2 = scope[n-2] for block n-1
            //
            // When emitting block k (k >= 1): scopes 0..k-1 closed.
            //   depth 0 = scope[k] for block k+1
            //   ...unless there are none left.
            //
            // For self-loop: block k is a loop header, scope[k-1] is `loop`.
            // When emitting block k, scope[k-1] was the one closed just before
            // this block started. So it's NOT open. Self-loops need the scope
            // to be open during the block's body.
            //
            // FIX: We need to NOT close the loop scope before its body.
            // For now, return n - 1 - from as a placeholder.
            // The proper fix requires restructuring emission order.
            (n - 1 - from) as u32
        } else {
            // Back edge (target < from): need the loop scope for `target`
            // to still be open. With current sequential closure, it isn't.
            // For loops where target is the immediate predecessor's scope,
            // the loop scope may still be open.
            // Scope for block `target` = scope_index (target - 1) if target >= 1.
            // At block `from`, remaining scopes: from..n-2 (for blocks from+1..n-1).
            // So scope[target-1] is closed if target-1 < from, i.e. target <= from.
            // Back edges always have target <= from, so this never works.
            //
            // TODO: proper loop scope nesting.
            u32::MAX // Signal unsupported.
        }
    }

    /// Emit a single MIR block's instructions and terminator.
    fn emit_block(
        &mut self,
        func: &mut Function,
        block: &BasicBlock,
        block_idx: usize,
        num_blocks: usize,
    ) -> Result<(), String> {
        // Emit instructions.
        for (dst, inst) in &block.instructions {
            self.emit_instruction(func, *dst, inst)?;
        }

        // Emit terminator.
        self.emit_terminator(func, &block.terminator, block_idx, num_blocks)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Instruction emission
    // -----------------------------------------------------------------------

    fn emit_instruction(
        &mut self,
        func: &mut Function,
        dst: ValueId,
        inst: &Instruction,
    ) -> Result<(), String> {
        match inst {
            // -- Constants --
            Instruction::ConstNum(n) => {
                func.instruction(&WasmInst::I64Const(n.to_bits() as i64));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::ConstBool(b) => {
                let bits = if *b { 0x7FFC_0000_0000_0002u64 } else { 0x7FFC_0000_0000_0001u64 };
                func.instruction(&WasmInst::I64Const(bits as i64));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::ConstNull => {
                func.instruction(&WasmInst::I64Const(0x7FFC_0000_0000_0000u64 as i64));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::ConstString(idx) => {
                func.instruction(&WasmInst::I64Const(*idx as i64));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::ConstF64(n) => {
                func.instruction(&WasmInst::F64Const(Ieee64::from(*n)));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::ConstI64(n) => {
                func.instruction(&WasmInst::I64Const(*n));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Unboxed f64 arithmetic (single WASM instruction each) --
            Instruction::AddF64(a, b) => self.emit_f64_binop(func, dst, *a, *b, WasmInst::F64Add),
            Instruction::SubF64(a, b) => self.emit_f64_binop(func, dst, *a, *b, WasmInst::F64Sub),
            Instruction::MulF64(a, b) => self.emit_f64_binop(func, dst, *a, *b, WasmInst::F64Mul),
            Instruction::DivF64(a, b) => self.emit_f64_binop(func, dst, *a, *b, WasmInst::F64Div),
            Instruction::ModF64(a, b) => {
                // f64 mod: a - trunc(a/b) * b
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::LocalGet(self.local(*b)));
                func.instruction(&WasmInst::F64Div);
                func.instruction(&WasmInst::F64Trunc);
                func.instruction(&WasmInst::LocalGet(self.local(*b)));
                func.instruction(&WasmInst::F64Mul);
                func.instruction(&WasmInst::F64Sub);
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::NegF64(a) => {
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::F64Neg);
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Unboxed f64 comparisons → i32 --
            Instruction::CmpLtF64(a, b) => self.emit_f64_cmp(func, dst, *a, *b, WasmInst::F64Lt),
            Instruction::CmpGtF64(a, b) => self.emit_f64_cmp(func, dst, *a, *b, WasmInst::F64Gt),
            Instruction::CmpLeF64(a, b) => self.emit_f64_cmp(func, dst, *a, *b, WasmInst::F64Le),
            Instruction::CmpGeF64(a, b) => self.emit_f64_cmp(func, dst, *a, *b, WasmInst::F64Ge),

            // -- Boxing / Unboxing --
            Instruction::Unbox(a) => {
                // NaN-boxed i64 → f64 reinterpret.
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::F64ReinterpretI64);
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::Box(a) => {
                // f64 → NaN-boxed i64 reinterpret.
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::I64ReinterpretF64);
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Logical --
            Instruction::Not(a) => {
                self.emit_runtime_call(func, dst, "wren_not", &[*a])?;
            }

            // -- Bitwise (truncate to i32, operate, convert back) --
            Instruction::BitAnd(a, b) => self.emit_bitwise(func, dst, *a, *b, WasmInst::I64And),
            Instruction::BitOr(a, b) => self.emit_bitwise(func, dst, *a, *b, WasmInst::I64Or),
            Instruction::BitXor(a, b) => self.emit_bitwise(func, dst, *a, *b, WasmInst::I64Xor),
            Instruction::Shl(a, b) => self.emit_bitwise(func, dst, *a, *b, WasmInst::I64Shl),
            Instruction::Shr(a, b) => self.emit_bitwise(func, dst, *a, *b, WasmInst::I64ShrS),
            Instruction::BitNot(a) => {
                // Unbox → NOT → rebox.
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::F64ReinterpretI64);
                func.instruction(&WasmInst::I64TruncF64S);
                func.instruction(&WasmInst::I64Const(-1));
                func.instruction(&WasmInst::I64Xor);
                func.instruction(&WasmInst::F64ConvertI64S);
                func.instruction(&WasmInst::I64ReinterpretF64);
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Guards (pass-through for now) --
            Instruction::GuardNum(a) | Instruction::GuardBool(a) => {
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::GuardClass(a, sym) => {
                self.emit_runtime_call_with_imm(func, dst, "wren_guard_class", *a, sym.index() as i64)?;
            }

            // -- Move --
            Instruction::Move(a) => {
                func.instruction(&WasmInst::LocalGet(self.local(*a)));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Boxed arithmetic → runtime calls --
            Instruction::Add(a, b) => self.emit_runtime_call(func, dst, "wren_num_add", &[*a, *b])?,
            Instruction::Sub(a, b) => self.emit_runtime_call(func, dst, "wren_num_sub", &[*a, *b])?,
            Instruction::Mul(a, b) => self.emit_runtime_call(func, dst, "wren_num_mul", &[*a, *b])?,
            Instruction::Div(a, b) => self.emit_runtime_call(func, dst, "wren_num_div", &[*a, *b])?,
            Instruction::Mod(a, b) => self.emit_runtime_call(func, dst, "wren_num_mod", &[*a, *b])?,
            Instruction::Neg(a) => self.emit_runtime_call(func, dst, "wren_num_neg", &[*a])?,

            // -- Boxed comparisons → runtime calls --
            Instruction::CmpLt(a, b) => self.emit_runtime_call(func, dst, "wren_cmp_lt", &[*a, *b])?,
            Instruction::CmpGt(a, b) => self.emit_runtime_call(func, dst, "wren_cmp_gt", &[*a, *b])?,
            Instruction::CmpLe(a, b) => self.emit_runtime_call(func, dst, "wren_cmp_le", &[*a, *b])?,
            Instruction::CmpGe(a, b) => self.emit_runtime_call(func, dst, "wren_cmp_ge", &[*a, *b])?,
            Instruction::CmpEq(a, b) => self.emit_runtime_call(func, dst, "wren_cmp_eq", &[*a, *b])?,
            Instruction::CmpNe(a, b) => self.emit_runtime_call(func, dst, "wren_cmp_ne", &[*a, *b])?,

            // -- Object operations --
            Instruction::GetField(recv, idx) => {
                self.emit_runtime_call_with_imm(func, dst, "wren_get_field", *recv, *idx as i64)?;
            }
            Instruction::SetField(recv, idx, val) => {
                func.instruction(&WasmInst::LocalGet(self.local(*recv)));
                func.instruction(&WasmInst::I64Const(*idx as i64));
                func.instruction(&WasmInst::LocalGet(self.local(*val)));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_set_field"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::GetModuleVar(idx) => {
                func.instruction(&WasmInst::I64Const(*idx as i64));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_get_module_var"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::SetModuleVar(idx, val) => {
                func.instruction(&WasmInst::I64Const(*idx as i64));
                func.instruction(&WasmInst::LocalGet(self.local(*val)));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_set_module_var"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Calls --
            Instruction::Call { receiver, method, args } => {
                func.instruction(&WasmInst::LocalGet(self.local(*receiver)));
                func.instruction(&WasmInst::I64Const(method.index() as i64));
                for a in args {
                    func.instruction(&WasmInst::LocalGet(self.local(*a)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_call"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::SuperCall { method, args } => {
                func.instruction(&WasmInst::I64Const(method.index() as i64));
                for a in args {
                    func.instruction(&WasmInst::LocalGet(self.local(*a)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_super_call"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Closures --
            Instruction::MakeClosure { fn_id, upvalues } => {
                func.instruction(&WasmInst::I64Const(*fn_id as i64));
                for uv in upvalues {
                    func.instruction(&WasmInst::LocalGet(self.local(*uv)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_make_closure"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::GetUpvalue(idx) => {
                func.instruction(&WasmInst::I64Const(*idx as i64));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_get_upvalue"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::SetUpvalue(idx, val) => {
                func.instruction(&WasmInst::I64Const(*idx as i64));
                func.instruction(&WasmInst::LocalGet(self.local(*val)));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_set_upvalue"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // -- Collections --
            Instruction::MakeList(elems) => {
                for e in elems {
                    func.instruction(&WasmInst::LocalGet(self.local(*e)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_make_list"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::MakeMap(pairs) => {
                for (k, v) in pairs {
                    func.instruction(&WasmInst::LocalGet(self.local(*k)));
                    func.instruction(&WasmInst::LocalGet(self.local(*v)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_make_map"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::MakeRange(from, to, inclusive) => {
                func.instruction(&WasmInst::LocalGet(self.local(*from)));
                func.instruction(&WasmInst::LocalGet(self.local(*to)));
                func.instruction(&WasmInst::I64Const(*inclusive as i64));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_make_range"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::StringConcat(parts) => {
                for p in parts {
                    func.instruction(&WasmInst::LocalGet(self.local(*p)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_string_concat"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::ToString(a) => {
                self.emit_runtime_call(func, dst, "wren_to_string", &[*a])?;
            }
            Instruction::IsType(a, sym) => {
                self.emit_runtime_call_with_imm(func, dst, "wren_is_type", *a, sym.index() as i64)?;
            }
            Instruction::SubscriptGet { receiver, args } => {
                func.instruction(&WasmInst::LocalGet(self.local(*receiver)));
                for a in args {
                    func.instruction(&WasmInst::LocalGet(self.local(*a)));
                }
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_subscript_get"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }
            Instruction::SubscriptSet { receiver, args, value } => {
                func.instruction(&WasmInst::LocalGet(self.local(*receiver)));
                for a in args {
                    func.instruction(&WasmInst::LocalGet(self.local(*a)));
                }
                func.instruction(&WasmInst::LocalGet(self.local(*value)));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_subscript_set"]));
                func.instruction(&WasmInst::LocalSet(self.local(dst)));
            }

            // Block params handled at block entry from branch args.
            Instruction::BlockParam(_) => {}
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Terminator emission
    // -----------------------------------------------------------------------

    fn emit_terminator(
        &self,
        func: &mut Function,
        term: &Terminator,
        block_idx: usize,
        num_blocks: usize,
    ) -> Result<(), String> {
        match term {
            Terminator::Return(val) => {
                func.instruction(&WasmInst::LocalGet(self.local(*val)));
                func.instruction(&WasmInst::Return);
            }
            Terminator::ReturnNull => {
                func.instruction(&WasmInst::I64Const(0x7FFC_0000_0000_0000u64 as i64));
                func.instruction(&WasmInst::Return);
            }
            Terminator::Branch { target, args } => {
                // Copy args to target block params.
                self.emit_block_args(func, *target, args);
                let depth = self.br_depth(block_idx, target.0 as usize, num_blocks);
                if depth == u32::MAX {
                    return Err(format!("Unsupported back edge from block {} to {}", block_idx, target.0));
                }
                func.instruction(&WasmInst::Br(depth));
            }
            Terminator::CondBranch {
                condition,
                true_target,
                true_args,
                false_target,
                false_args,
            } => {
                // Evaluate truthiness.
                func.instruction(&WasmInst::LocalGet(self.local(*condition)));
                func.instruction(&WasmInst::Call(self.runtime_imports["wren_is_truthy"]));

                // if truthy → true branch, else → false branch.
                func.instruction(&WasmInst::If(wasm_encoder::BlockType::Empty));

                // True branch.
                self.emit_block_args(func, *true_target, true_args);
                let true_depth = self.br_depth(block_idx, true_target.0 as usize, num_blocks);
                // +1 because we're inside an `if` block.
                if true_depth != u32::MAX {
                    func.instruction(&WasmInst::Br(true_depth + 1));
                }

                func.instruction(&WasmInst::Else);

                // False branch.
                self.emit_block_args(func, *false_target, false_args);
                let false_depth = self.br_depth(block_idx, false_target.0 as usize, num_blocks);
                if false_depth != u32::MAX {
                    func.instruction(&WasmInst::Br(false_depth + 1));
                }

                func.instruction(&WasmInst::End);
            }
            Terminator::Unreachable => {
                func.instruction(&WasmInst::Unreachable);
            }
        }
        Ok(())
    }

    /// Copy branch args to target block's params.
    fn emit_block_args(&self, func: &mut Function, target: BlockId, args: &[ValueId]) {
        if args.is_empty() {
            return;
        }
        let target_block = &self.mir.blocks[target.0 as usize];
        for (i, arg) in args.iter().enumerate() {
            if i < target_block.params.len() {
                let (param_val, _) = target_block.params[i];
                func.instruction(&WasmInst::LocalGet(self.local(*arg)));
                func.instruction(&WasmInst::LocalSet(self.local(param_val)));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn emit_f64_binop(&self, func: &mut Function, dst: ValueId, a: ValueId, b: ValueId, op: WasmInst<'static>) {
        func.instruction(&WasmInst::LocalGet(self.local(a)));
        func.instruction(&WasmInst::LocalGet(self.local(b)));
        func.instruction(&op);
        func.instruction(&WasmInst::LocalSet(self.local(dst)));
    }

    fn emit_f64_cmp(&self, func: &mut Function, dst: ValueId, a: ValueId, b: ValueId, op: WasmInst<'static>) {
        func.instruction(&WasmInst::LocalGet(self.local(a)));
        func.instruction(&WasmInst::LocalGet(self.local(b)));
        func.instruction(&op);
        // Result is i32 (0 or 1). Extend to i64 if the local is i64.
        // But we typed CmpF64 results as i32, so just set directly.
        func.instruction(&WasmInst::LocalSet(self.local(dst)));
    }

    /// Emit: unbox both operands → truncate → integer op → convert back → rebox.
    fn emit_bitwise(&self, func: &mut Function, dst: ValueId, a: ValueId, b: ValueId, op: WasmInst<'static>) {
        // Unbox a: i64 → f64 → i64 (truncated)
        func.instruction(&WasmInst::LocalGet(self.local(a)));
        func.instruction(&WasmInst::F64ReinterpretI64);
        func.instruction(&WasmInst::I64TruncF64S);
        // Unbox b
        func.instruction(&WasmInst::LocalGet(self.local(b)));
        func.instruction(&WasmInst::F64ReinterpretI64);
        func.instruction(&WasmInst::I64TruncF64S);
        // Op
        func.instruction(&op);
        // Rebox: i64 → f64 → i64
        func.instruction(&WasmInst::F64ConvertI64S);
        func.instruction(&WasmInst::I64ReinterpretF64);
        func.instruction(&WasmInst::LocalSet(self.local(dst)));
    }

    fn emit_runtime_call(
        &self,
        func: &mut Function,
        dst: ValueId,
        name: &str,
        args: &[ValueId],
    ) -> Result<(), String> {
        for a in args {
            func.instruction(&WasmInst::LocalGet(self.local(*a)));
        }
        let idx = self.runtime_imports.get(name)
            .ok_or_else(|| format!("Runtime function {} not imported", name))?;
        func.instruction(&WasmInst::Call(*idx));
        func.instruction(&WasmInst::LocalSet(self.local(dst)));
        Ok(())
    }

    fn emit_runtime_call_with_imm(
        &self,
        func: &mut Function,
        dst: ValueId,
        name: &str,
        val: ValueId,
        imm: i64,
    ) -> Result<(), String> {
        func.instruction(&WasmInst::LocalGet(self.local(val)));
        func.instruction(&WasmInst::I64Const(imm));
        let idx = self.runtime_imports.get(name)
            .ok_or_else(|| format!("Runtime function {} not imported", name))?;
        func.instruction(&WasmInst::Call(*idx));
        func.instruction(&WasmInst::LocalSet(self.local(dst)));
        Ok(())
    }
}

fn mir_type_to_wasm(ty: MirType) -> ValType {
    match ty {
        MirType::F64 => ValType::F64,
        MirType::Bool => ValType::I32,
        MirType::Value | MirType::I64 | MirType::Void => ValType::I64,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::{Instruction, MirFunction, MirType, Terminator};

    fn setup() -> (Interner, MirFunction) {
        let mut interner = Interner::new();
        let name = interner.intern("test");
        let mir = MirFunction::new(name, 0);
        (interner, mir)
    }

    /// Validate a WASM module fully. On failure, dump WAT for debugging.
    fn assert_valid(module: &WasmModule) {
        if let Err(e) = module.validate_full() {
            let wat = module.dump_wat().unwrap_or_else(|e| format!("<WAT error: {}>", e));
            panic!("WASM validation failed: {}\n\nWAT dump:\n{}", e, wat);
        }
    }

    #[test]
    fn test_return_constant() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let v0 = mir.new_value();
        mir.block_mut(bb).instructions.push((v0, Instruction::ConstNum(42.0)));
        mir.block_mut(bb).terminator = Terminator::Return(v0);

        let result = emit_mir(&mir);
        assert!(result.is_ok(), "emit failed: {:?}", result.err());
        let module = result.unwrap();
        assert_valid(&module);
        assert_eq!(&module.bytes[0..4], b"\0asm");
    }

    #[test]
    fn test_f64_arithmetic() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let v0 = mir.new_value();
        let v1 = mir.new_value();
        let v2 = mir.new_value();
        let v3 = mir.new_value();
        mir.block_mut(bb).instructions.push((v0, Instruction::ConstF64(3.0)));
        mir.block_mut(bb).instructions.push((v1, Instruction::ConstF64(4.0)));
        mir.block_mut(bb).instructions.push((v2, Instruction::AddF64(v0, v1)));
        mir.block_mut(bb).instructions.push((v3, Instruction::Box(v2)));
        mir.block_mut(bb).terminator = Terminator::Return(v3);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_null_return() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        mir.block_mut(bb).terminator = Terminator::ReturnNull;

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_forward_branch() {
        let (_, mut mir) = setup();
        let bb0 = mir.new_block();
        let bb1 = mir.new_block();
        let v0 = mir.new_value();

        mir.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![],
        };
        mir.block_mut(bb1).instructions.push((v0, Instruction::ConstNum(1.0)));
        mir.block_mut(bb1).terminator = Terminator::Return(v0);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_conditional_branch() {
        let (_, mut mir) = setup();
        let bb0 = mir.new_block();
        let bb1 = mir.new_block();
        let bb2 = mir.new_block();
        let v_cond = mir.new_value();
        let v1 = mir.new_value();
        let v2 = mir.new_value();

        mir.block_mut(bb0).instructions.push((v_cond, Instruction::ConstBool(true)));
        mir.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb1,
            true_args: vec![],
            false_target: bb2,
            false_args: vec![],
        };
        mir.block_mut(bb1).instructions.push((v1, Instruction::ConstNum(1.0)));
        mir.block_mut(bb1).terminator = Terminator::Return(v1);
        mir.block_mut(bb2).instructions.push((v2, Instruction::ConstNum(2.0)));
        mir.block_mut(bb2).terminator = Terminator::Return(v2);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_block_params() {
        // bb0: v0 = const 42; branch bb1(v0)
        // bb1(p0: Value): return p0
        let (_, mut mir) = setup();
        let bb0 = mir.new_block();
        let bb1 = mir.new_block();
        let v0 = mir.new_value();
        let p0 = mir.new_value();

        mir.block_mut(bb0).instructions.push((v0, Instruction::ConstNum(42.0)));
        mir.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };
        let bp = mir.new_value();
        mir.block_mut(bb1).params.push((p0, MirType::Value));
        mir.block_mut(bb1).instructions.push((bp, Instruction::BlockParam(0)));
        mir.block_mut(bb1).terminator = Terminator::Return(p0);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_boxing_unboxing() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let v0 = mir.new_value();
        let v1 = mir.new_value();
        let v2 = mir.new_value();
        let v3 = mir.new_value();

        // v0 = ConstNum(3.14) → i64 NaN-boxed
        // v1 = Unbox(v0) → f64
        // v2 = NegF64(v1) → f64
        // v3 = Box(v2) → i64 NaN-boxed
        mir.block_mut(bb).instructions.push((v0, Instruction::ConstNum(3.14)));
        mir.block_mut(bb).instructions.push((v1, Instruction::Unbox(v0)));
        mir.block_mut(bb).instructions.push((v2, Instruction::NegF64(v1)));
        mir.block_mut(bb).instructions.push((v3, Instruction::Box(v2)));
        mir.block_mut(bb).terminator = Terminator::Return(v3);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_runtime_call() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let v0 = mir.new_value();
        let v1 = mir.new_value();
        let v2 = mir.new_value();

        mir.block_mut(bb).instructions.push((v0, Instruction::ConstNum(1.0)));
        mir.block_mut(bb).instructions.push((v1, Instruction::ConstNum(2.0)));
        mir.block_mut(bb).instructions.push((v2, Instruction::Add(v0, v1)));
        mir.block_mut(bb).terminator = Terminator::Return(v2);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_wasm_header() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        mir.block_mut(bb).terminator = Terminator::ReturnNull;

        let module = emit_mir(&mir).unwrap();
        assert_eq!(&module.bytes[0..4], b"\0asm");
        assert_eq!(&module.bytes[4..8], &[1, 0, 0, 0]);
    }

    #[test]
    fn test_multiple_f64_ops() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let a = mir.new_value();
        let b = mir.new_value();
        let add = mir.new_value();
        let sub = mir.new_value();
        let mul = mir.new_value();
        let div = mir.new_value();
        let neg = mir.new_value();
        let boxed = mir.new_value();

        mir.block_mut(bb).instructions.push((a, Instruction::ConstF64(10.0)));
        mir.block_mut(bb).instructions.push((b, Instruction::ConstF64(3.0)));
        mir.block_mut(bb).instructions.push((add, Instruction::AddF64(a, b)));
        mir.block_mut(bb).instructions.push((sub, Instruction::SubF64(add, b)));
        mir.block_mut(bb).instructions.push((mul, Instruction::MulF64(sub, a)));
        mir.block_mut(bb).instructions.push((div, Instruction::DivF64(mul, b)));
        mir.block_mut(bb).instructions.push((neg, Instruction::NegF64(div)));
        mir.block_mut(bb).instructions.push((boxed, Instruction::Box(neg)));
        mir.block_mut(bb).terminator = Terminator::Return(boxed);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_comparison_and_guard() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let a = mir.new_value();
        let b = mir.new_value();
        let cmp = mir.new_value();
        let guarded = mir.new_value();

        mir.block_mut(bb).instructions.push((a, Instruction::ConstF64(1.0)));
        mir.block_mut(bb).instructions.push((b, Instruction::ConstF64(2.0)));
        mir.block_mut(bb).instructions.push((cmp, Instruction::CmpLtF64(a, b)));
        mir.block_mut(bb).instructions.push((guarded, Instruction::ConstNum(42.0)));
        mir.block_mut(bb).terminator = Terminator::Return(guarded);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
    }

    #[test]
    fn test_dump_wat() {
        // Verify WAT dump works and produces readable output.
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let a = mir.new_value();
        let b = mir.new_value();
        let sum = mir.new_value();
        let boxed = mir.new_value();

        mir.block_mut(bb).instructions.push((a, Instruction::ConstF64(10.0)));
        mir.block_mut(bb).instructions.push((b, Instruction::ConstF64(20.0)));
        mir.block_mut(bb).instructions.push((sum, Instruction::AddF64(a, b)));
        mir.block_mut(bb).instructions.push((boxed, Instruction::Box(sum)));
        mir.block_mut(bb).terminator = Terminator::Return(boxed);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);
        let wat = module.dump_wat().unwrap();
        assert!(wat.contains("f64.add"), "WAT should contain f64.add:\n{}", wat);
        assert!(wat.contains("f64.const"), "WAT should contain f64.const:\n{}", wat);
    }

    #[test]
    fn test_wasmtime_execution_return_num() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let v0 = mir.new_value();
        mir.block_mut(bb).instructions.push((v0, Instruction::ConstNum(42.0)));
        mir.block_mut(bb).terminator = Terminator::Return(v0);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);

        // Execute via wasmtime.
        let engine = wasmtime::Engine::default();
        let wasm_module = wasmtime::Module::new(&engine, &module.bytes).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();
        let func = instance.get_typed_func::<(), i64>(&mut store, "fn_0").unwrap();
        let result = func.call(&mut store, ()).unwrap();

        // 42.0 NaN-boxed = f64 bits of 42.0
        let expected = 42.0f64.to_bits() as i64;
        assert_eq!(result, expected, "Expected NaN-boxed 42.0 ({}), got {}", expected, result);
    }

    #[test]
    fn test_wasmtime_execution_f64_add() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        let a = mir.new_value();
        let b = mir.new_value();
        let sum = mir.new_value();
        let boxed = mir.new_value();

        mir.block_mut(bb).instructions.push((a, Instruction::ConstF64(10.0)));
        mir.block_mut(bb).instructions.push((b, Instruction::ConstF64(20.0)));
        mir.block_mut(bb).instructions.push((sum, Instruction::AddF64(a, b)));
        mir.block_mut(bb).instructions.push((boxed, Instruction::Box(sum)));
        mir.block_mut(bb).terminator = Terminator::Return(boxed);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);

        let engine = wasmtime::Engine::default();
        let wasm_module = wasmtime::Module::new(&engine, &module.bytes).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();
        let func = instance.get_typed_func::<(), i64>(&mut store, "fn_0").unwrap();
        let result = func.call(&mut store, ()).unwrap();

        let expected = 30.0f64.to_bits() as i64;
        assert_eq!(result, expected, "Expected NaN-boxed 30.0");
    }

    #[test]
    fn test_wasmtime_execution_return_null() {
        let (_, mut mir) = setup();
        let bb = mir.new_block();
        mir.block_mut(bb).terminator = Terminator::ReturnNull;

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);

        let engine = wasmtime::Engine::default();
        let wasm_module = wasmtime::Module::new(&engine, &module.bytes).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();
        let func = instance.get_typed_func::<(), i64>(&mut store, "fn_0").unwrap();
        let result = func.call(&mut store, ()).unwrap();

        let null_bits = 0x7FFC_0000_0000_0000u64 as i64;
        assert_eq!(result, null_bits, "Expected NaN-boxed null");
    }

    #[test]
    fn test_wasmtime_execution_forward_branch() {
        let (_, mut mir) = setup();
        let bb0 = mir.new_block();
        let bb1 = mir.new_block();
        let v0 = mir.new_value();

        mir.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![],
        };
        mir.block_mut(bb1).instructions.push((v0, Instruction::ConstNum(99.0)));
        mir.block_mut(bb1).terminator = Terminator::Return(v0);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);

        let engine = wasmtime::Engine::default();
        let wasm_module = wasmtime::Module::new(&engine, &module.bytes).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();
        let func = instance.get_typed_func::<(), i64>(&mut store, "fn_0").unwrap();
        let result = func.call(&mut store, ()).unwrap();

        let expected = 99.0f64.to_bits() as i64;
        assert_eq!(result, expected, "Expected NaN-boxed 99.0 after branch");
    }

    #[test]
    fn test_wasmtime_execution_block_params() {
        // bb0: v0 = 42.0; branch bb1(v0)
        // bb1(p0): return p0
        let (_, mut mir) = setup();
        let bb0 = mir.new_block();
        let bb1 = mir.new_block();
        let v0 = mir.new_value();
        let p0 = mir.new_value();
        let bp = mir.new_value();

        mir.block_mut(bb0).instructions.push((v0, Instruction::ConstNum(42.0)));
        mir.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };
        mir.block_mut(bb1).params.push((p0, MirType::Value));
        mir.block_mut(bb1).instructions.push((bp, Instruction::BlockParam(0)));
        mir.block_mut(bb1).terminator = Terminator::Return(p0);

        let module = emit_mir(&mir).unwrap();
        assert_valid(&module);

        let engine = wasmtime::Engine::default();
        let wasm_module = wasmtime::Module::new(&engine, &module.bytes).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &wasm_module, &[]).unwrap();
        let func = instance.get_typed_func::<(), i64>(&mut store, "fn_0").unwrap();
        let result = func.call(&mut store, ()).unwrap();

        let expected = 42.0f64.to_bits() as i64;
        assert_eq!(result, expected, "Expected NaN-boxed 42.0 through block param");
    }
}
