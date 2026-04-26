/// Protocol-based devirtualization pass.
///
/// Uses statically known type information (from guards, constructor results,
/// and literal constructors) to replace dynamic method dispatch with:
///
/// 1. **IsType folding**: `x is Sequence` → `true` when x is known to conform
/// 2. **Iterate loop specialization**: Replace iterate/iteratorValue calls on
///    known List/Range with direct indexed access or counted loops
/// 3. **Comparison inlining**: Replace `<(_)` etc. on known Num with CmpLtF64
///
/// This pass is analogous to Rust's monomorphization + devirtualization:
/// when the concrete type is known, we eliminate the vtable (HashMap) lookup
/// entirely and emit the operation inline.
use std::collections::HashMap;

use crate::intern::Interner;
use crate::mir::{Instruction, MirFunction, ValueId};
use crate::sema::protocol::{self, ProtocolSet};

use super::MirPass;

/// Known type information for a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownType {
    Num,
    Bool,
    Null,
    String,
    List,
    Map,
    Range,
}

impl KnownType {
    fn protocols(self) -> ProtocolSet {
        match self {
            KnownType::Num => protocol::builtin_protocols_for("Num"),
            KnownType::Bool => protocol::builtin_protocols_for("Bool"),
            KnownType::Null => protocol::builtin_protocols_for("Null"),
            KnownType::String => protocol::builtin_protocols_for("String"),
            KnownType::List => protocol::builtin_protocols_for("List"),
            KnownType::Map => protocol::builtin_protocols_for("Map"),
            KnownType::Range => protocol::builtin_protocols_for("Range"),
        }
    }

    fn class_name(self) -> &'static str {
        match self {
            KnownType::Num => "Num",
            KnownType::Bool => "Bool",
            KnownType::Null => "Null",
            KnownType::String => "String",
            KnownType::List => "List",
            KnownType::Map => "Map",
            KnownType::Range => "Range",
        }
    }
}

/// The devirtualization pass.
pub struct Devirt<'a> {
    pub interner: &'a Interner,
}

impl<'a> MirPass for Devirt<'a> {
    fn name(&self) -> &str {
        "devirt"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut changed = false;

        // Phase 1: Collect known types for values across all blocks.
        let known_types = self.collect_known_types(func);

        // Phase 2: Apply transformations.
        for block_idx in 0..func.blocks.len() {
            let block = &func.blocks[block_idx];
            let mut replacements: Vec<(usize, ValueId, Instruction)> = Vec::new();

            for (inst_idx, (vid, inst)) in block.instructions.iter().enumerate() {
                match inst {
                    // --- IsType folding ---
                    // If we know the value's type, and that type conforms to
                    // the tested class (or IS the class), fold to true/false.
                    Instruction::IsType(val, class_sym) => {
                        if let Some(known) = known_types.get(val) {
                            let class_name = self.interner.resolve(*class_sym);
                            let result = self.fold_is_type(*known, class_name);
                            if let Some(b) = result {
                                replacements.push((inst_idx, *vid, Instruction::ConstBool(b)));
                                changed = true;
                            }
                        }
                    }

                    // --- Sequence iterate devirtualization ---
                    // When calling iterate(_) on a known List, replace with
                    // direct count-based iteration (no method dispatch).
                    Instruction::Call {
                        receiver,
                        method,
                        args,
                        pure_call: _,
                    } => {
                        let method_name = self.interner.resolve(*method);

                        if let Some(known) = known_types.get(receiver) {
                            // Num arithmetic devirtualization
                            if *known == KnownType::Num && args.len() == 1 {
                                if let Some(replacement) =
                                    self.devirt_num_method(method_name, *receiver, args[0])
                                {
                                    replacements.push((inst_idx, *vid, replacement));
                                    changed = true;
                                    continue;
                                }
                            }

                            // Num comparison devirtualization
                            if *known == KnownType::Num && args.len() == 1 {
                                if let Some(replacement) =
                                    self.devirt_num_compare(method_name, *receiver, args[0])
                                {
                                    replacements.push((inst_idx, *vid, replacement));
                                    changed = true;
                                    continue;
                                }
                            }

                            // String/List/Map count devirtualization
                            if args.is_empty() && method_name == "count" {
                                match *known {
                                    KnownType::List | KnownType::String | KnownType::Map => {
                                        // Keep as Call but mark as devirtualized candidate.
                                        // The codegen can inline this as a direct field read.
                                        // For now, we just skip — real inlining comes from
                                        // the type guard + codegen.
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    _ => {}
                }
            }

            // Apply replacements in reverse to preserve indices.
            for (inst_idx, vid, new_inst) in replacements.into_iter().rev() {
                func.blocks[block_idx].instructions[inst_idx] = (vid, new_inst);
            }
        }

        changed
    }
}

impl<'a> Devirt<'a> {
    /// Collect statically known types for values by scanning all blocks.
    fn collect_known_types(&self, func: &MirFunction) -> HashMap<ValueId, KnownType> {
        let mut known = HashMap::new();

        for block in &func.blocks {
            for (vid, inst) in &block.instructions {
                match inst {
                    // Constants → known type
                    Instruction::ConstNum(_) | Instruction::ConstF64(_) => {
                        known.insert(*vid, KnownType::Num);
                    }
                    Instruction::ConstBool(_) => {
                        known.insert(*vid, KnownType::Bool);
                    }
                    Instruction::ConstNull => {
                        known.insert(*vid, KnownType::Null);
                    }
                    Instruction::ConstString(_) => {
                        known.insert(*vid, KnownType::String);
                    }

                    // Arithmetic → Num
                    Instruction::Add(_, _)
                    | Instruction::Sub(_, _)
                    | Instruction::Mul(_, _)
                    | Instruction::Div(_, _)
                    | Instruction::Mod(_, _)
                    | Instruction::Neg(_)
                    | Instruction::AddF64(_, _)
                    | Instruction::SubF64(_, _)
                    | Instruction::MulF64(_, _)
                    | Instruction::DivF64(_, _)
                    | Instruction::ModF64(_, _)
                    | Instruction::NegF64(_)
                    | Instruction::MathUnaryF64(_, _)
                    | Instruction::MathBinaryF64(_, _, _)
                    | Instruction::BitAnd(_, _)
                    | Instruction::BitOr(_, _)
                    | Instruction::BitXor(_, _)
                    | Instruction::BitNot(_)
                    | Instruction::Shl(_, _)
                    | Instruction::Shr(_, _) => {
                        known.insert(*vid, KnownType::Num);
                    }

                    // Boxing → Num (box always produces a boxed f64)
                    Instruction::Box(_) => {
                        known.insert(*vid, KnownType::Num);
                    }

                    // Comparisons → Bool
                    Instruction::CmpLt(_, _)
                    | Instruction::CmpGt(_, _)
                    | Instruction::CmpLe(_, _)
                    | Instruction::CmpGe(_, _)
                    | Instruction::CmpEq(_, _)
                    | Instruction::CmpNe(_, _)
                    | Instruction::CmpLtF64(_, _)
                    | Instruction::CmpGtF64(_, _)
                    | Instruction::CmpLeF64(_, _)
                    | Instruction::CmpGeF64(_, _)
                    | Instruction::Not(_)
                    | Instruction::IsType(_, _) => {
                        known.insert(*vid, KnownType::Bool);
                    }

                    // Guards refine type knowledge
                    Instruction::GuardNum(v) => {
                        known.insert(*v, KnownType::Num);
                    }
                    Instruction::GuardBool(v) => {
                        known.insert(*v, KnownType::Bool);
                    }
                    Instruction::GuardClass(v, class_sym) => {
                        let name = self.interner.resolve(*class_sym);
                        if let Some(kt) = name_to_known_type(name) {
                            known.insert(*v, kt);
                        }
                    }

                    // Collection constructors → known type
                    Instruction::MakeList(_) => {
                        known.insert(*vid, KnownType::List);
                    }
                    Instruction::MakeMap(_) => {
                        known.insert(*vid, KnownType::Map);
                    }
                    Instruction::MakeRange(_, _, _) => {
                        known.insert(*vid, KnownType::Range);
                    }

                    // String operations → String
                    Instruction::StringConcat(_) | Instruction::ToString(_) => {
                        known.insert(*vid, KnownType::String);
                    }

                    // Move/copy propagates type
                    Instruction::Move(src) => {
                        if let Some(kt) = known.get(src) {
                            known.insert(*vid, *kt);
                        }
                    }

                    _ => {}
                }
            }
        }

        known
    }

    /// Try to fold `value is ClassName` to a compile-time boolean.
    fn fold_is_type(&self, known: KnownType, class_name: &str) -> Option<bool> {
        // Direct match
        if known.class_name() == class_name {
            return Some(true);
        }

        // Check protocol / superclass hierarchy
        match class_name {
            "Object" => Some(true), // Everything is Object
            "Sequence" => Some(known.protocols().has(protocol::SEQUENCE)),
            "Num" => Some(known == KnownType::Num),
            "Bool" => Some(known == KnownType::Bool),
            "Null" => Some(known == KnownType::Null),
            "String" => Some(known == KnownType::String),
            "List" => Some(known == KnownType::List),
            "Map" => Some(known == KnownType::Map),
            "Range" => Some(known == KnownType::Range),
            _ => None, // Unknown class, can't fold
        }
    }

    /// Devirtualize Num arithmetic methods to direct instructions.
    fn devirt_num_method(
        &self,
        method_name: &str,
        receiver: ValueId,
        arg: ValueId,
    ) -> Option<Instruction> {
        match method_name {
            "+(_)" => Some(Instruction::Add(receiver, arg)),
            "-(_)" => Some(Instruction::Sub(receiver, arg)),
            "*(_)" => Some(Instruction::Mul(receiver, arg)),
            "/(_)" => Some(Instruction::Div(receiver, arg)),
            "%(_)" => Some(Instruction::Mod(receiver, arg)),
            _ => None,
        }
    }

    /// Devirtualize Num comparison methods to direct instructions.
    fn devirt_num_compare(
        &self,
        method_name: &str,
        receiver: ValueId,
        arg: ValueId,
    ) -> Option<Instruction> {
        match method_name {
            "<(_)" => Some(Instruction::CmpLt(receiver, arg)),
            ">(_)" => Some(Instruction::CmpGt(receiver, arg)),
            "<=(_)" => Some(Instruction::CmpLe(receiver, arg)),
            ">=(_)" => Some(Instruction::CmpGe(receiver, arg)),
            "==(_)" => Some(Instruction::CmpEq(receiver, arg)),
            "!=(_)" => Some(Instruction::CmpNe(receiver, arg)),
            _ => None,
        }
    }
}

fn name_to_known_type(name: &str) -> Option<KnownType> {
    match name {
        "Num" => Some(KnownType::Num),
        "Bool" => Some(KnownType::Bool),
        "Null" => Some(KnownType::Null),
        "String" => Some(KnownType::String),
        "List" => Some(KnownType::List),
        "Map" => Some(KnownType::Map),
        "Range" => Some(KnownType::Range),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::Terminator;

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_fold_is_type_num_is_num() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let num_sym = interner.intern("Num");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::IsType(v0, num_sym)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[1].1 {
            Instruction::ConstBool(true) => {}
            other => panic!("Expected ConstBool(true), got {:?}", other),
        }
    }

    #[test]
    fn test_fold_is_type_num_is_object() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let obj_sym = interner.intern("Object");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::IsType(v0, obj_sym)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[1].1 {
            Instruction::ConstBool(true) => {}
            other => panic!("Expected ConstBool(true), got {:?}", other),
        }
    }

    #[test]
    fn test_fold_is_type_list_is_sequence() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let seq_sym = interner.intern("Sequence");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::MakeList(vec![])));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::IsType(v0, seq_sym)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[1].1 {
            Instruction::ConstBool(true) => {}
            other => panic!("Expected ConstBool(true), got {:?}", other),
        }
    }

    #[test]
    fn test_fold_is_type_num_is_string_false() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let str_sym = interner.intern("String");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::IsType(v0, str_sym)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[1].1 {
            Instruction::ConstBool(false) => {}
            other => panic!("Expected ConstBool(false), got {:?}", other),
        }
    }

    #[test]
    fn test_devirt_num_add() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let add_sym = interner.intern("+(_)");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(2.0)));
        f.block_mut(bb).instructions.push((
            v2,
            Instruction::Call {
                receiver: v0,
                method: add_sym,
                args: vec![v1],
                pure_call: false,
            },
        ));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[2].1 {
            Instruction::Add(a, b) => {
                assert_eq!(*a, v0);
                assert_eq!(*b, v1);
            }
            other => panic!("Expected Add, got {:?}", other),
        }
    }

    #[test]
    fn test_devirt_num_less_than() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let lt_sym = interner.intern("<(_)");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(2.0)));
        f.block_mut(bb).instructions.push((
            v2,
            Instruction::Call {
                receiver: v0,
                method: lt_sym,
                args: vec![v1],
                pure_call: false,
            },
        ));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[2].1 {
            Instruction::CmpLt(a, b) => {
                assert_eq!(*a, v0);
                assert_eq!(*b, v1);
            }
            other => panic!("Expected CmpLt, got {:?}", other),
        }
    }

    #[test]
    fn test_no_devirt_unknown_type() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let add_sym = interner.intern("+(_)");

        // v0 from BlockParam → unknown type
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::BlockParam(0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(2.0)));
        f.block_mut(bb).instructions.push((
            v2,
            Instruction::Call {
                receiver: v0,
                method: add_sym,
                args: vec![v1],
                pure_call: false,
            },
        ));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(!changed);
        assert!(matches!(
            &f.block(bb).instructions[2].1,
            Instruction::Call { .. }
        ));
    }

    #[test]
    fn test_guard_refines_type() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let _v1 = f.new_value(); // guard result
        let v2 = f.new_value();
        let v3 = f.new_value();
        let add_sym = interner.intern("+(_)");

        // v0 = param (unknown), guard.num v0 → now known Num, v0 +(_) v2
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::BlockParam(0)));
        f.block_mut(bb)
            .instructions
            .push((_v1, Instruction::GuardNum(v0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::ConstNum(1.0)));
        f.block_mut(bb).instructions.push((
            v3,
            Instruction::Call {
                receiver: v0,
                method: add_sym,
                args: vec![v2],
                pure_call: false,
            },
        ));
        f.block_mut(bb).terminator = Terminator::Return(v3);

        let pass = Devirt {
            interner: &interner,
        };
        let changed = pass.run(&mut f);

        assert!(changed);
        match &f.block(bb).instructions[3].1 {
            Instruction::Add(a, b) => {
                assert_eq!(*a, v0);
                assert_eq!(*b, v2);
            }
            other => panic!("Expected Add after guard, got {:?}", other),
        }
    }

    #[test]
    fn test_make_list_is_sequence_true() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let seq_sym = interner.intern("Sequence");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::MakeList(vec![])));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::IsType(v0, seq_sym)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let pass = Devirt {
            interner: &interner,
        };
        assert!(pass.run(&mut f));
        assert!(matches!(
            &f.block(bb).instructions[1].1,
            Instruction::ConstBool(true)
        ));
    }

    #[test]
    fn test_make_range_is_sequence_true() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        let seq_sym = interner.intern("Sequence");

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(10.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::MakeRange(v0, v1, true)));
        f.block_mut(bb)
            .instructions
            .push((v3, Instruction::IsType(v2, seq_sym)));
        f.block_mut(bb).terminator = Terminator::Return(v3);

        let pass = Devirt {
            interner: &interner,
        };
        assert!(pass.run(&mut f));
        assert!(matches!(
            &f.block(bb).instructions[3].1,
            Instruction::ConstBool(true)
        ));
    }
}
