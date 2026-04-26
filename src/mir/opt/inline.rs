/// Type specialization (devirtualization).
///
/// When operands of boxed arithmetic are known to be Num (via ConstNum or
/// GuardNum), replaces the boxed operation with Unbox → unboxed op → Box.
/// This eliminates runtime type checks and enables further f64 optimizations.
///
/// Also inlines Num math method calls (abs, sin, sqrt, etc.) when the receiver
/// is known to be Num, replacing the method dispatch with direct f64 intrinsics.
use std::collections::{HashMap, HashSet};

use super::MirPass;
use crate::intern::{Interner, SymbolId};
use crate::mir::{Instruction, MathBinaryOp, MathUnaryOp, MirFunction, ValueId};

pub struct TypeSpecialize {
    /// Maps method SymbolId → unary math intrinsic for known-Num receivers.
    math_unary: HashMap<SymbolId, MathUnaryOp>,
    /// Maps method SymbolId → binary math intrinsic for known-Num receivers.
    math_binary: HashMap<SymbolId, MathBinaryOp>,
}

impl Default for TypeSpecialize {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeSpecialize {
    /// Create without math intrinsic inlining (no interner available).
    pub fn new() -> Self {
        Self {
            math_unary: HashMap::new(),
            math_binary: HashMap::new(),
        }
    }

    /// Create with math intrinsic inlining enabled.
    /// Looks up SymbolIds for all known Num math methods.
    pub fn with_math(interner: &Interner) -> Self {
        let mut math_unary = HashMap::new();
        let mut math_binary = HashMap::new();

        let unary_methods: &[(&str, MathUnaryOp)] = &[
            ("abs", MathUnaryOp::Abs),
            ("acos", MathUnaryOp::Acos),
            ("asin", MathUnaryOp::Asin),
            ("atan", MathUnaryOp::Atan),
            ("cbrt", MathUnaryOp::Cbrt),
            ("ceil", MathUnaryOp::Ceil),
            ("cos", MathUnaryOp::Cos),
            ("floor", MathUnaryOp::Floor),
            ("round", MathUnaryOp::Round),
            ("sin", MathUnaryOp::Sin),
            ("sqrt", MathUnaryOp::Sqrt),
            ("tan", MathUnaryOp::Tan),
            ("log", MathUnaryOp::Log),
            ("log2", MathUnaryOp::Log2),
            ("exp", MathUnaryOp::Exp),
            ("truncate", MathUnaryOp::Trunc),
            ("fraction", MathUnaryOp::Fract),
            ("sign", MathUnaryOp::Sign),
        ];

        let binary_methods: &[(&str, MathBinaryOp)] = &[
            ("atan(_)", MathBinaryOp::Atan2),
            ("min(_)", MathBinaryOp::Min),
            ("max(_)", MathBinaryOp::Max),
            ("pow(_)", MathBinaryOp::Pow),
        ];

        for (name, op) in unary_methods {
            if let Some(id) = interner.lookup(name) {
                math_unary.insert(id, *op);
            }
        }
        for (name, op) in binary_methods {
            if let Some(id) = interner.lookup(name) {
                math_binary.insert(id, *op);
            }
        }

        Self {
            math_unary,
            math_binary,
        }
    }
}

impl MirPass for TypeSpecialize {
    fn name(&self) -> &str {
        "type-specialize"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut known_nums: HashSet<ValueId> = HashSet::new();
        let mut changed = false;

        // Pre-pass: if this function has GuardNum on any parameter,
        // assume CallStaticSelf (recursive calls) also return Num.
        // This enables unboxing the `add` of two recursive results
        // (e.g., fib: calc(n-1) + calc(n-2) → fadd instead of wren_num_add).
        let has_num_guard = func.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|(_, inst)| matches!(inst, Instruction::GuardNum(_)))
        });
        let self_call_returns_num = has_num_guard;

        for block_idx in 0..func.blocks.len() {
            let old_instructions = std::mem::take(&mut func.blocks[block_idx].instructions);
            let mut new_instructions = Vec::new();

            for (val_id, inst) in &old_instructions {
                match inst {
                    Instruction::ConstNum(_) | Instruction::Box(_) => {
                        known_nums.insert(*val_id);
                        new_instructions.push((*val_id, inst.clone()));
                    }
                    Instruction::GuardNum(src) => {
                        // Both the guard output AND the guarded source are known-Num.
                        known_nums.insert(*val_id);
                        known_nums.insert(*src);
                        new_instructions.push((*val_id, inst.clone()));
                    }
                    Instruction::Move(src) if known_nums.contains(src) => {
                        // Propagate known-Num through moves.
                        known_nums.insert(*val_id);
                        new_instructions.push((*val_id, inst.clone()));
                    }

                    Instruction::Add(a, b) if known_nums.contains(a) && known_nums.contains(b) => {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Add);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Sub(a, b) if known_nums.contains(a) && known_nums.contains(b) => {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Sub);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Mul(a, b) if known_nums.contains(a) && known_nums.contains(b) => {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Mul);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Div(a, b) if known_nums.contains(a) && known_nums.contains(b) => {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Div);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Mod(a, b) if known_nums.contains(a) && known_nums.contains(b) => {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Mod);
                        known_nums.insert(*val_id);
                        changed = true;
                    }

                    Instruction::Neg(a) if known_nums.contains(a) => {
                        let ua = func.new_value();
                        let result_f = func.new_value();
                        new_instructions.push((ua, Instruction::Unbox(*a)));
                        new_instructions.push((result_f, Instruction::NegF64(ua)));
                        new_instructions.push((*val_id, Instruction::Box(result_f)));
                        known_nums.insert(*val_id);
                        changed = true;
                    }

                    Instruction::CmpLt(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Lt);
                        changed = true;
                    }
                    Instruction::CmpGt(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Gt);
                        changed = true;
                    }
                    Instruction::CmpLe(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Le);
                        changed = true;
                    }
                    Instruction::CmpGe(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Ge);
                        changed = true;
                    }

                    // Inline unary Num math methods: receiver.abs → Unbox → fabs → Box
                    Instruction::Call {
                        receiver,
                        method,
                        args,
                        pure_call: _,
                    } if args.is_empty()
                        && known_nums.contains(receiver)
                        && self.math_unary.contains_key(method) =>
                    {
                        let op = self.math_unary[method];
                        let ua = func.new_value();
                        let result_f = func.new_value();
                        new_instructions.push((ua, Instruction::Unbox(*receiver)));
                        new_instructions.push((result_f, Instruction::MathUnaryF64(op, ua)));
                        new_instructions.push((*val_id, Instruction::Box(result_f)));
                        known_nums.insert(*val_id);
                        changed = true;
                    }

                    // Inline binary Num math methods: receiver.pow(arg) → Unbox both → fpow → Box
                    Instruction::Call {
                        receiver,
                        method,
                        args,
                        pure_call: _,
                    } if args.len() == 1
                        && known_nums.contains(receiver)
                        && known_nums.contains(&args[0])
                        && self.math_binary.contains_key(method) =>
                    {
                        let op = self.math_binary[method];
                        let arg = args[0];
                        let ua = func.new_value();
                        let ub = func.new_value();
                        let result_f = func.new_value();
                        new_instructions.push((ua, Instruction::Unbox(*receiver)));
                        new_instructions.push((ub, Instruction::Unbox(arg)));
                        new_instructions.push((result_f, Instruction::MathBinaryF64(op, ua, ub)));
                        new_instructions.push((*val_id, Instruction::Box(result_f)));
                        known_nums.insert(*val_id);
                        changed = true;
                    }

                    // CallStaticSelf returns Num if function has Num guards
                    Instruction::CallStaticSelf { .. } if self_call_returns_num => {
                        known_nums.insert(*val_id);
                        new_instructions.push((*val_id, inst.clone()));
                    }

                    _ => {
                        new_instructions.push((*val_id, inst.clone()));
                    }
                }
            }

            func.blocks[block_idx].instructions = new_instructions;
        }

        changed
    }
}

enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
}

fn expand_binop(
    func: &mut MirFunction,
    out: &mut Vec<(ValueId, Instruction)>,
    result_id: ValueId,
    a: ValueId,
    b: ValueId,
    op: BinOp,
) {
    let ua = func.new_value();
    let ub = func.new_value();
    let result_f = func.new_value();

    out.push((ua, Instruction::Unbox(a)));
    out.push((ub, Instruction::Unbox(b)));
    out.push((
        result_f,
        match op {
            BinOp::Add => Instruction::AddF64(ua, ub),
            BinOp::Sub => Instruction::SubF64(ua, ub),
            BinOp::Mul => Instruction::MulF64(ua, ub),
            BinOp::Div => Instruction::DivF64(ua, ub),
            BinOp::Mod => Instruction::ModF64(ua, ub),
        },
    ));
    out.push((result_id, Instruction::Box(result_f)));
}

fn expand_cmp(
    func: &mut MirFunction,
    out: &mut Vec<(ValueId, Instruction)>,
    result_id: ValueId,
    a: ValueId,
    b: ValueId,
    op: CmpOp,
) {
    let ua = func.new_value();
    let ub = func.new_value();

    out.push((ua, Instruction::Unbox(a)));
    out.push((ub, Instruction::Unbox(b)));
    out.push((
        result_id,
        match op {
            CmpOp::Lt => Instruction::CmpLtF64(ua, ub),
            CmpOp::Gt => Instruction::CmpGtF64(ua, ub),
            CmpOp::Le => Instruction::CmpLeF64(ua, ub),
            CmpOp::Ge => Instruction::CmpGeF64(ua, ub),
        },
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::interp::{eval, InterpValue};
    use crate::mir::{Instruction, Terminator};
    use crate::runtime::value::Value;

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_specialize_add() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(20.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let before = eval(&f).unwrap();
        assert!(TypeSpecialize::new().run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(30.0)));
        let last = f.block(bb).instructions.iter().find(|(v, _)| *v == v2);
        assert!(matches!(last, Some((_, Instruction::Box(_)))));
    }

    #[test]
    fn test_non_num_not_specialized() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::GetModuleVar(0)));
            b.instructions.push((v1, Instruction::ConstNum(10.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        assert!(!TypeSpecialize::new().run(&mut f));
        let (_, ref inst) = f.block(bb).instructions[2];
        assert!(matches!(inst, Instruction::Add(..)));
    }

    #[test]
    fn test_specialize_neg() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.instructions.push((v1, Instruction::Neg(v0)));
            b.terminator = Terminator::Return(v1);
        }

        let before = eval(&f).unwrap();
        assert!(TypeSpecialize::new().run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(-42.0)));
    }

    #[test]
    fn test_specialize_guard_num() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::GetModuleVar(0)));
            b.instructions.push((v1, Instruction::GuardNum(v0)));
            b.instructions.push((v2, Instruction::ConstNum(5.0)));
            b.instructions.push((v3, Instruction::Add(v1, v2)));
            b.terminator = Terminator::Return(v3);
        }

        assert!(TypeSpecialize::new().run(&mut f));
        let last = f.block(bb).instructions.iter().find(|(v, _)| *v == v3);
        assert!(matches!(last, Some((_, Instruction::Box(_)))));
    }

    #[test]
    fn test_specialize_comparison() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let bb_t = f.new_block();
        let bb_f = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::ConstNum(2.0)));
            b.instructions.push((v2, Instruction::CmpLt(v0, v1)));
            b.terminator = Terminator::CondBranch {
                condition: v2,
                true_target: bb_t,
                true_args: vec![],
                false_target: bb_f,
                false_args: vec![],
            };
        }
        f.block_mut(bb_t)
            .instructions
            .push((v_yes, Instruction::ConstNum(1.0)));
        f.block_mut(bb_t).terminator = Terminator::Return(v_yes);
        f.block_mut(bb_f)
            .instructions
            .push((v_no, Instruction::ConstNum(0.0)));
        f.block_mut(bb_f).terminator = Terminator::Return(v_no);

        let before = eval(&f).unwrap();
        assert!(TypeSpecialize::new().run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(1.0)));
    }

    // -- Math intrinsic inlining tests --

    /// Helper: build a MIR function that does `Call { receiver: ConstNum(n), method, args }`.
    fn build_unary_call(interner: &mut Interner, n: f64, method_name: &str) -> MirFunction {
        let mut f = make_func(interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v_result = f.new_value();
        let method = interner.intern(method_name);
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(n)));
            b.instructions.push((
                v_result,
                Instruction::Call {
                    receiver: v0,
                    method,
                    args: vec![],
                pure_call: false,
},
            ));
            b.terminator = Terminator::Return(v_result);
        }
        f
    }

    fn build_binary_call(
        interner: &mut Interner,
        a: f64,
        b_val: f64,
        method_name: &str,
    ) -> MirFunction {
        let mut f = make_func(interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v_result = f.new_value();
        let method = interner.intern(method_name);
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(a)));
            b.instructions.push((v1, Instruction::ConstNum(b_val)));
            b.instructions.push((
                v_result,
                Instruction::Call {
                    receiver: v0,
                    method,
                    args: vec![v1],
                pure_call: false,
},
            ));
            b.terminator = Terminator::Return(v_result);
        }
        f
    }

    #[test]
    fn test_inline_abs() {
        let mut interner = Interner::new();
        let mut f = build_unary_call(&mut interner, -42.0, "abs");
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));

        // Should produce Box(MathUnaryF64(Abs, Unbox(v0)))
        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_inline_sqrt() {
        let mut interner = Interner::new();
        let mut f = build_unary_call(&mut interner, 144.0, "sqrt");
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));
        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(12.0)));
    }

    #[test]
    fn test_inline_floor_ceil() {
        let mut interner = Interner::new();
        let mut f = build_unary_call(&mut interner, 3.7, "floor");
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));
        assert_eq!(eval(&f).unwrap(), InterpValue::Boxed(Value::num(3.0)));

        let mut f2 = build_unary_call(&mut interner, 3.2, "ceil");
        let pass2 = TypeSpecialize::with_math(&interner);
        assert!(pass2.run(&mut f2));
        assert_eq!(eval(&f2).unwrap(), InterpValue::Boxed(Value::num(4.0)));
    }

    #[test]
    fn test_inline_sin_cos() {
        let mut interner = Interner::new();
        let mut f = build_unary_call(&mut interner, 0.0, "sin");
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));
        assert_eq!(eval(&f).unwrap(), InterpValue::Boxed(Value::num(0.0)));

        let mut f2 = build_unary_call(&mut interner, 0.0, "cos");
        let pass2 = TypeSpecialize::with_math(&interner);
        assert!(pass2.run(&mut f2));
        assert_eq!(eval(&f2).unwrap(), InterpValue::Boxed(Value::num(1.0)));
    }

    #[test]
    fn test_inline_pow() {
        let mut interner = Interner::new();
        let mut f = build_binary_call(&mut interner, 2.0, 10.0, "pow(_)");
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));
        assert_eq!(eval(&f).unwrap(), InterpValue::Boxed(Value::num(1024.0)));
    }

    #[test]
    fn test_inline_min_max() {
        let mut interner = Interner::new();
        let mut f = build_binary_call(&mut interner, 5.0, 3.0, "min(_)");
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));
        assert_eq!(eval(&f).unwrap(), InterpValue::Boxed(Value::num(3.0)));

        let mut f2 = build_binary_call(&mut interner, 5.0, 3.0, "max(_)");
        let pass2 = TypeSpecialize::with_math(&interner);
        assert!(pass2.run(&mut f2));
        assert_eq!(eval(&f2).unwrap(), InterpValue::Boxed(Value::num(5.0)));
    }

    #[test]
    fn test_no_inline_unknown_receiver() {
        // If receiver is not known-Num, Call should NOT be inlined.
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v_result = f.new_value();
        let method = interner.intern("abs");
        {
            let b = f.block_mut(bb);
            // GetModuleVar — type unknown
            b.instructions.push((v0, Instruction::GetModuleVar(0)));
            b.instructions.push((
                v_result,
                Instruction::Call {
                    receiver: v0,
                    method,
                    args: vec![],
                pure_call: false,
},
            ));
            b.terminator = Terminator::Return(v_result);
        }
        let pass = TypeSpecialize::with_math(&interner);
        assert!(!pass.run(&mut f));
        // Call should still be there
        let (_, ref inst) = f.block(bb).instructions[1];
        assert!(matches!(inst, Instruction::Call { .. }));
    }

    #[test]
    fn test_inline_chained_math() {
        // sqrt(abs(-16)) = 4.0 — chain through multiple inlined calls
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v_abs = f.new_value();
        let v_sqrt = f.new_value();
        let abs_sym = interner.intern("abs");
        let sqrt_sym = interner.intern("sqrt");
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(-16.0)));
            b.instructions.push((
                v_abs,
                Instruction::Call {
                    receiver: v0,
                    method: abs_sym,
                    args: vec![],
                pure_call: false,
},
            ));
            b.instructions.push((
                v_sqrt,
                Instruction::Call {
                    receiver: v_abs,
                    method: sqrt_sym,
                    args: vec![],
                pure_call: false,
},
            ));
            b.terminator = Terminator::Return(v_sqrt);
        }
        let pass = TypeSpecialize::with_math(&interner);
        assert!(pass.run(&mut f));

        // Both calls should be inlined: v_abs and v_sqrt are Box results → known_nums
        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(4.0)));
    }
}
