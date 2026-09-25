//! Num guards on field reads that a class's field-kind bytes answer.
//!
//! A class keeps a byte per field recording the kinds of value its
//! instances have stored there; `FIELD_NUM` alone says every instance
//! has only ever held a Num in that field. A Num guard on such a field's
//! read, from a receiver a class guard pinned to that class, can check
//! the class's bytes instead of the value, and one check covers every
//! such field of the class. A check holds until something runs that
//! could store another kind: a call, an operator that may dispatch, or
//! a store of a value not known to be a Num. The plan says, for each
//! such guard, whether it makes the check or a check on every path to
//! it already covers it. As in `redundant_guards`, a block a back edge
//! enters starts with nothing checked, since an OSR entry arrives there
//! from the interpreter.

use std::collections::{HashMap, HashSet};

use super::licm::compute_rpo;
use crate::mir::{BlockId, Instruction, MirFunction, ValueId};

/// What the lowering does in place of one Num guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindGuard {
    /// Check that the bytes of the class at `class` for `fields`, a bit
    /// per field index, all say Num.
    Check { class: usize, fields: u64 },
    /// A check on every path here already covers the field.
    Covered,
}

/// What an instruction can do to the kinds fields hold.
enum Effect {
    None,
    /// May store a value of another kind into this field of any object.
    Field(u16),
    /// May run code that stores anything anywhere.
    All,
}

/// The plan for `func`'s Num guards on field reads. `num_fields(class)`
/// gives the fields of the class at that address whose byte says only
/// Num today, a bit per index.
pub fn plan(func: &MirFunction, num_fields: &dyn Fn(usize) -> u64) -> HashMap<ValueId, KindGuard> {
    let mut out = HashMap::new();
    if func.blocks.is_empty() {
        return out;
    }
    let defs: HashMap<ValueId, &Instruction> = func
        .blocks
        .iter()
        .flat_map(|b| b.instructions.iter().map(|(v, i)| (*v, i)))
        .collect();
    let root = |mut v: ValueId| {
        while let Some(Instruction::Move(a)) = defs.get(&v) {
            v = *a;
        }
        v
    };
    let is_num = |v: ValueId| {
        matches!(
            defs.get(&root(v)),
            Some(Instruction::Box(_) | Instruction::ConstNum(_))
        )
    };

    let n = func.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, block) in func.blocks.iter().enumerate() {
        for s in block.terminator.successors() {
            if let Some(p) = preds.get_mut(s.0 as usize) {
                p.push(b);
            }
        }
    }
    let rpo: Vec<BlockId> = compute_rpo(func);
    let mut order = vec![usize::MAX; n];
    for (i, b) in rpo.iter().enumerate() {
        order[b.0 as usize] = i;
    }
    let back_edge = |b: usize| preds[b].iter().any(|&p| order[p] >= order[b]);

    // The class each guarded field read's receiver is pinned to.
    let mut candidates: HashMap<ValueId, (usize, u16)> = HashMap::new();
    let mut classes_out: Vec<Option<HashSet<(ValueId, usize)>>> = vec![None; n];
    for bid in &rpo {
        let bi = bid.0 as usize;
        let mut facts = if back_edge(bi) {
            HashSet::new()
        } else {
            meet(preds[bi].iter().filter_map(|&p| classes_out[p].as_ref()))
        };
        for (v, inst) in &func.blocks[bi].instructions {
            match inst {
                Instruction::GuardClassAt { value, class, .. } => {
                    facts.insert((root(*value), *class));
                }
                Instruction::NewInstance { class, .. } => {
                    facts.insert((*v, *class));
                }
                Instruction::GuardNumAt { value, .. } => {
                    let Some(Instruction::GetField(recv, idx)) = defs.get(&root(*value)) else {
                        continue;
                    };
                    let recv = root(*recv);
                    let Some(&(_, class)) = facts.iter().find(|(r, _)| *r == recv) else {
                        continue;
                    };
                    if *idx < 64 && num_fields(class) & (1 << idx) != 0 {
                        candidates.insert(*v, (class, *idx));
                    }
                }
                _ => {}
            }
        }
        classes_out[bi] = Some(facts);
    }
    if candidates.is_empty() {
        return out;
    }
    let mut sets: HashMap<usize, u64> = HashMap::new();
    for &(class, idx) in candidates.values() {
        *sets.entry(class).or_default() |= 1 << idx;
    }

    // Checked fields per class, on every path to each point.
    let mut checked_out: Vec<Option<HashMap<usize, u64>>> = vec![None; n];
    for bid in &rpo {
        let bi = bid.0 as usize;
        let mut checked: HashMap<usize, u64> = HashMap::new();
        if !back_edge(bi) {
            let mut incoming = preds[bi].iter().filter_map(|&p| checked_out[p].as_ref());
            if let Some(first) = incoming.next() {
                checked = first.clone();
                for other in incoming {
                    checked.retain(|class, mask| {
                        *mask &= other.get(class).copied().unwrap_or(0);
                        *mask != 0
                    });
                }
            }
        }
        for (v, inst) in &func.blocks[bi].instructions {
            if let Some(&(class, idx)) = candidates.get(v) {
                let mask = checked.entry(class).or_default();
                if *mask & (1 << idx) != 0 {
                    out.insert(*v, KindGuard::Covered);
                } else {
                    let fields = sets[&class];
                    out.insert(*v, KindGuard::Check { class, fields });
                    *mask |= fields;
                }
                continue;
            }
            match effect(inst, &is_num) {
                Effect::None => {}
                Effect::Field(idx) if idx < 64 => {
                    checked.retain(|_, mask| {
                        *mask &= !(1 << idx);
                        *mask != 0
                    });
                }
                Effect::Field(_) => {}
                Effect::All => checked.clear(),
            }
        }
        checked_out[bi] = Some(checked);
    }
    out
}

/// The loads that read the bytes of `fields`, a bit per index, from an
/// array of `len` bytes: `(offset, width)` pairs, each width 1, 2, 4 or
/// 8 and each load inside the array. A load may overlap the one before.
/// Fields past the array are left out.
pub fn kind_chunks(fields: u64, len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let fields = if len < 64 {
        fields & ((1 << len) - 1)
    } else {
        fields
    };
    if fields == 0 {
        return out;
    }
    let hi = 63 - fields.leading_zeros() as usize;
    let mut widest = 8;
    while widest > len {
        widest /= 2;
    }
    let mut pos = fields.trailing_zeros() as usize;
    while pos <= hi {
        let mut size = widest;
        while size > 1 && size / 2 > hi - pos {
            size /= 2;
        }
        let start = pos.min(len - size);
        out.push((start, size));
        let next = start + size;
        if next > hi {
            break;
        }
        pos = next + (fields >> next).trailing_zeros() as usize;
    }
    out
}

/// The facts every set holds.
fn meet<'a>(
    mut sets: impl Iterator<Item = &'a HashSet<(ValueId, usize)>>,
) -> HashSet<(ValueId, usize)> {
    let Some(first) = sets.next() else {
        return HashSet::new();
    };
    let mut facts = first.clone();
    for other in sets {
        facts.retain(|f| other.contains(f));
    }
    facts
}

fn effect(inst: &Instruction, is_num: &dyn Fn(ValueId) -> bool) -> Effect {
    use Instruction::*;
    match inst {
        ConstNum(_)
        | ConstBool(_)
        | ConstNull
        | ConstString(_)
        | ConstF64(_)
        | ConstI64(_)
        | Move(_)
        | BlockParam(_)
        | Box(_)
        | Unbox(_)
        | AddF64(..)
        | SubF64(..)
        | MulF64(..)
        | DivF64(..)
        | ModF64(..)
        | NegF64(_)
        | CmpLtF64(..)
        | CmpGtF64(..)
        | CmpLeF64(..)
        | CmpGeF64(..)
        | AddI64(..)
        | SubI64(..)
        | MulI64(..)
        | RemI64(..)
        | BandI64(..)
        | NegI64(_)
        | CmpLtI64(..)
        | CmpGtI64(..)
        | CmpLeI64(..)
        | CmpGeI64(..)
        | I64ToF64(_)
        | F64ToI64(_)
        | MathUnaryF64(..)
        | MathBinaryF64(..)
        | GuardNum(_)
        | GuardBool(_)
        | GuardClass(..)
        | GuardProtocol(..)
        | GuardNumAt { .. }
        | GuardClassAt { .. }
        | IsNum(_)
        | IsType(..)
        | ClassIs(..)
        | ObjectIs(..)
        | ClosureFnIs(..)
        | GetField(..)
        | GetModuleVar(_)
        | SetModuleVar(..)
        | GetUpvalue(_)
        | SetUpvalue(..)
        | GetStaticField(_)
        | SetStaticField(..)
        | ListCount(_)
        | SlowPathExit { .. }
        | MakeClosure { .. }
        | MakeList(_)
        | NewInstance { .. } => Effect::None,
        SetField(_, idx, value) => {
            if is_num(*value) {
                Effect::None
            } else {
                Effect::Field(*idx)
            }
        }
        _ => Effect::All,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::Terminator;

    const CLASS: usize = 0x1000;

    fn num_guard(value: ValueId) -> Instruction {
        Instruction::GuardNumAt {
            value,
            pc: 0,
            live: vec![],
            call_pc: 0,
            call_live: vec![],
        }
    }

    /// Reads of fields 0 and 1 of a pinned receiver, each guarded, with
    /// `between` placed after the first guard.
    fn two_reads(between: Option<Instruction>) -> (MirFunction, ValueId, ValueId) {
        let mut interner = Interner::new();
        let mut f = MirFunction::new(interner.intern("test"), 1);
        let bb = f.new_block();
        let obj = f.new_value();
        let pin = f.new_value();
        let x = f.new_value();
        let gx = f.new_value();
        let y = f.new_value();
        let gy = f.new_value();
        let mut insts = vec![
            (obj, Instruction::BlockParam(0)),
            (
                pin,
                Instruction::GuardClassAt {
                    value: obj,
                    class: CLASS,
                    pc: 0,
                    live: vec![],
                },
            ),
            (x, Instruction::GetField(obj, 0)),
            (gx, num_guard(x)),
        ];
        if let Some(inst) = between {
            let v = f.new_value();
            insts.push((v, inst));
        }
        insts.push((y, Instruction::GetField(obj, 1)));
        insts.push((gy, num_guard(y)));
        f.block_mut(bb).instructions = insts;
        f.block_mut(bb).terminator = Terminator::Return(y);
        (f, gx, gy)
    }

    #[test]
    fn the_loads_cover_every_field_inside_the_array() {
        assert_eq!(kind_chunks(0b111_1111, 7), vec![(0, 4), (3, 4)]);
        assert_eq!(kind_chunks(0b100_0001, 7), vec![(0, 4), (6, 1)]);
        assert_eq!(kind_chunks(0b1, 1), vec![(0, 1)]);
        assert_eq!(kind_chunks(0xF_FFFF, 20), vec![(0, 8), (8, 8), (16, 4)]);
        for len in 1..=20usize {
            for fields in [1u64, 0b101, (1 << len) - 1, 1 << (len - 1)] {
                let chunks = kind_chunks(fields, len);
                for (start, size) in &chunks {
                    assert!(start + size <= len);
                }
                for b in 0..len {
                    if fields & (1 << b) != 0 {
                        assert!(chunks.iter().any(|(s, w)| (*s..s + w).contains(&b)));
                    }
                }
            }
        }
    }

    #[test]
    fn one_check_covers_the_fields_of_a_class() {
        let (f, gx, gy) = two_reads(None);
        let plan = plan(&f, &|c| if c == CLASS { 0b11 } else { 0 });
        assert_eq!(
            plan.get(&gx),
            Some(&KindGuard::Check {
                class: CLASS,
                fields: 0b11
            })
        );
        assert_eq!(plan.get(&gy), Some(&KindGuard::Covered));
    }

    #[test]
    fn a_field_that_held_another_kind_keeps_its_guard() {
        let (f, gx, gy) = two_reads(None);
        let plan = plan(&f, &|c| if c == CLASS { 0b01 } else { 0 });
        assert!(matches!(
            plan.get(&gx),
            Some(KindGuard::Check { fields: 0b01, .. })
        ));
        assert_eq!(plan.get(&gy), None);
    }

    #[test]
    fn a_call_ends_the_check() {
        let mut interner = Interner::new();
        let method = interner.intern("f()");
        let (mut f, _, gy) = two_reads(None);
        let recv = ValueId(0);
        let bb = f.entry_block();
        let pos = f.block(bb).instructions.len() - 2;
        let call = f.new_value();
        f.block_mut(bb).instructions.insert(
            pos,
            (
                call,
                Instruction::Call {
                    receiver: recv,
                    method,
                    args: vec![],
                    pure_call: false,
                },
            ),
        );
        let plan = plan(&f, &|c| if c == CLASS { 0b11 } else { 0 });
        assert!(matches!(plan.get(&gy), Some(KindGuard::Check { .. })));
    }

    #[test]
    fn a_store_of_an_unknown_value_ends_the_check_on_its_field() {
        let (f, _, gy) = two_reads(Some(Instruction::SetField(ValueId(0), 1, ValueId(0))));
        let plan = plan(&f, &|c| if c == CLASS { 0b11 } else { 0 });
        assert!(matches!(plan.get(&gy), Some(KindGuard::Check { .. })));
    }
}
