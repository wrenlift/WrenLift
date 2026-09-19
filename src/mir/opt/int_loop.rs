//! Carry provably-integral f64 values as i64.
//!
//! Wren has only doubles, but a loop counter or an accumulator that is
//! built from integer constants by addition, subtraction, multiplication
//! and remainder by a constant stays an exact integer as long as it never
//! leaves ±2^53. A range analysis over the f64 values proves that bound,
//! using the loop's own exit comparison to bound a counter; every value
//! that passes is then defined by the integer instruction instead, block
//! parameters included, and converted back to f64 only where a float
//! consumer reads it. Runs last, after the f64 parameter pass.

use std::collections::{HashMap, HashSet};

use super::MirPass;
use super::licm::{compute_dominators, compute_rpo};
use crate::mir::{BlockId, Instruction, MirFunction, MirType, Terminator, ValueId};

/// Every value this pass carries as an integer stays within this bound,
/// where f64 arithmetic on integers is exact.
const LIMIT: i64 = 1 << 53;
const WIDEN_AFTER: usize = 8;

#[derive(Clone, Copy, PartialEq, Debug)]
struct Range {
    lo: i64,
    hi: i64,
}

impl Range {
    fn hull(self, o: Range) -> Range {
        Range {
            lo: self.lo.min(o.lo),
            hi: self.hi.max(o.hi),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Lat {
    Bottom,
    Int(Range),
    Top,
}

fn bounded(lo: i128, hi: i128) -> Lat {
    if lo < -(LIMIT as i128) || hi > LIMIT as i128 {
        Lat::Top
    } else {
        Lat::Int(Range {
            lo: lo as i64,
            hi: hi as i64,
        })
    }
}

fn combine(a: Lat, b: Lat, f: impl Fn(Range, Range) -> Lat) -> Lat {
    match (a, b) {
        (Lat::Top, _) | (_, Lat::Top) => Lat::Top,
        (Lat::Bottom, _) | (_, Lat::Bottom) => Lat::Bottom,
        (Lat::Int(x), Lat::Int(y)) => f(x, y),
    }
}

/// A comparison a block is only reached through.
#[derive(Clone, Copy)]
struct Fact {
    scope: BlockId,
    x: ValueId,
    y: ValueId,
    op: Cmp,
}

#[derive(Clone, Copy)]
enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn negate(self) -> Cmp {
        match self {
            Cmp::Lt => Cmp::Ge,
            Cmp::Le => Cmp::Gt,
            Cmp::Gt => Cmp::Le,
            Cmp::Ge => Cmp::Lt,
        }
    }
    fn flip(self) -> Cmp {
        match self {
            Cmp::Lt => Cmp::Gt,
            Cmp::Le => Cmp::Ge,
            Cmp::Gt => Cmp::Lt,
            Cmp::Ge => Cmp::Le,
        }
    }
}

pub struct IntSpecialize;

impl MirPass for IntSpecialize {
    fn name(&self) -> &str {
        "int-specialize"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        func.compute_predecessors();
        let rpo = compute_rpo(func);
        let idom = compute_dominators(func, &rpo);
        let defs: HashMap<ValueId, Instruction> = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter().map(|(v, i)| (*v, i.clone())))
            .collect();
        let facts = collect_facts(func);
        let counts = list_counts(func, &defs);
        let f64_params: HashSet<ValueId> = func
            .blocks
            .iter()
            .skip(1)
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| *t == MirType::F64)
            .map(|(p, _)| *p)
            .collect();
        let entry_params: HashSet<ValueId> =
            func.blocks[0].params.iter().map(|(p, _)| *p).collect();

        // Forward analysis to a fixed point. A parameter that keeps
        // growing is widened to the full bound, after which the facts in
        // scope narrow it back (a counter to its exit bound); if that
        // never settles, everything becomes Top.
        let mut lat: HashMap<ValueId, Lat> = HashMap::new();
        let mut changes: HashMap<ValueId, usize> = HashMap::new();
        let mut rounds = 0usize;
        loop {
            rounds += 1;
            if rounds > 64 {
                for p in &f64_params {
                    lat.insert(*p, Lat::Top);
                }
                for block in &func.blocks {
                    for (v, _) in &block.instructions {
                        lat.insert(*v, Lat::Top);
                    }
                }
                break;
            }
            let mut changed = false;
            for &bid in &rpo {
                let block = &func.blocks[bid.0 as usize];
                for &(p, _) in &block.params {
                    if !f64_params.contains(&p) || entry_params.contains(&p) {
                        continue;
                    }
                    let mut acc = Lat::Bottom;
                    for &pred in &block.predecessors {
                        let pb = &func.blocks[pred.0 as usize];
                        for (target, args) in edges(&pb.terminator) {
                            if target != bid {
                                continue;
                            }
                            let idx = block.params.iter().position(|(q, _)| *q == p).unwrap();
                            let Some(arg) = args.get(idx) else { continue };
                            let v = refined(*arg, pred, &lat, &facts, &idom);
                            acc = match (acc, v) {
                                (Lat::Bottom, v) => v,
                                (a, Lat::Bottom) => a,
                                (Lat::Top, _) | (_, Lat::Top) => Lat::Top,
                                (Lat::Int(a), Lat::Int(b)) => {
                                    let h = a.hull(b);
                                    bounded(h.lo as i128, h.hi as i128)
                                }
                            };
                        }
                    }
                    let old = lat.get(&p).copied().unwrap_or(Lat::Bottom);
                    if acc != old {
                        let grew = match (old, acc) {
                            (Lat::Int(o), Lat::Int(n)) => n.lo < o.lo || n.hi > o.hi,
                            (Lat::Bottom, _) => false,
                            _ => true,
                        };
                        let acc = if grew {
                            let count = changes.entry(p).or_insert(0);
                            *count += 1;
                            match (old, acc) {
                                (Lat::Int(o), Lat::Int(n)) if *count > WIDEN_AFTER => {
                                    Lat::Int(Range {
                                        lo: if n.lo < o.lo { -LIMIT } else { o.lo },
                                        hi: if n.hi > o.hi { LIMIT } else { o.hi },
                                    })
                                }
                                _ => acc,
                            }
                        } else {
                            acc
                        };
                        if acc != old {
                            lat.insert(p, acc);
                            changed = true;
                        }
                    }
                }
                for (v, inst) in &block.instructions {
                    let l = transfer(inst, bid, &lat, &facts, &idom, &defs, &counts);
                    let old = lat.get(v).copied().unwrap_or(Lat::Bottom);
                    if l != old {
                        lat.insert(*v, l);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Which values become integers: those with a proven range whose
        // definition the rewrite knows.
        let mut int_vals: HashSet<ValueId> = HashSet::new();
        for (v, l) in &lat {
            if let Lat::Int(_) = l {
                let ok = if f64_params.contains(v) {
                    true
                } else {
                    matches!(
                        defs.get(v),
                        Some(
                            Instruction::ConstF64(_)
                                | Instruction::AddF64(..)
                                | Instruction::SubF64(..)
                                | Instruction::MulF64(..)
                                | Instruction::NegF64(_)
                                | Instruction::ModF64(..)
                                | Instruction::Move(_)
                                | Instruction::Unbox(_)
                        )
                    )
                };
                if ok {
                    int_vals.insert(*v);
                }
            }
        }
        // A parameter is an integer only if every incoming argument is.
        loop {
            let mut dropped = false;
            for block in &func.blocks {
                for (target, args) in edges(&block.terminator) {
                    let params = &func.blocks[target.0 as usize].params;
                    for (i, a) in args.iter().enumerate() {
                        if let Some((p, _)) = params.get(i)
                            && int_vals.contains(p)
                            && !int_vals.contains(a)
                        {
                            int_vals.remove(p);
                            dropped = true;
                        }
                    }
                }
            }
            // An instruction is an integer only if its operands are; a
            // proven unbox reads a boxed value and converts.
            for block in &func.blocks {
                for (v, inst) in &block.instructions {
                    if int_vals.contains(v)
                        && !matches!(inst, Instruction::Unbox(_))
                        && inst.operands().iter().any(|o| !int_vals.contains(o))
                    {
                        int_vals.remove(v);
                        dropped = true;
                    }
                }
            }
            if !dropped {
                break;
            }
        }
        if int_vals.is_empty() {
            return false;
        }

        // Rewrite definitions.
        for block in &mut func.blocks {
            for (p, t) in &mut block.params {
                if int_vals.contains(p) {
                    *t = MirType::I64;
                }
            }
        }
        for bi in 0..func.blocks.len() {
            let old = std::mem::take(&mut func.blocks[bi].instructions);
            let mut out = Vec::with_capacity(old.len());
            for (v, inst) in old {
                if !int_vals.contains(&v) {
                    out.push((v, inst));
                    continue;
                }
                let new = match inst {
                    Instruction::ConstF64(c) => Instruction::ConstI64(c as i64),
                    Instruction::AddF64(a, b) => Instruction::AddI64(a, b),
                    Instruction::SubF64(a, b) => Instruction::SubI64(a, b),
                    Instruction::MulF64(a, b) => Instruction::MulI64(a, b),
                    Instruction::NegF64(a) => Instruction::NegI64(a),
                    Instruction::Move(a) => Instruction::Move(a),
                    Instruction::Unbox(a) => {
                        let f = func.new_value();
                        out.push((f, Instruction::Unbox(a)));
                        Instruction::F64ToI64(f)
                    }
                    Instruction::ModF64(a, b) => {
                        let c = match defs.get(&b) {
                            Some(Instruction::ConstF64(c)) => *c as i64,
                            _ => unreachable!("integral remainder needs a constant divisor"),
                        };
                        let c = c.abs();
                        if (c & (c - 1)) == 0 {
                            let mask = func.new_value();
                            out.push((mask, Instruction::ConstI64(c - 1)));
                            Instruction::BandI64(a, mask)
                        } else {
                            let d = func.new_value();
                            out.push((d, Instruction::ConstI64(c)));
                            Instruction::RemI64(a, d)
                        }
                    }
                    other => unreachable!("not an integer definition: {:?}", other),
                };
                out.push((v, new));
            }
            func.blocks[bi].instructions = out;
        }

        // Comparisons between integers stay integer; every other float
        // consumer of an integer reads a conversion planted before it.
        let chosen_params: Vec<Vec<bool>> = func
            .blocks
            .iter()
            .map(|b| b.params.iter().map(|(p, _)| int_vals.contains(p)).collect())
            .collect();
        for bi in 0..func.blocks.len() {
            let old = std::mem::take(&mut func.blocks[bi].instructions);
            let mut out = Vec::with_capacity(old.len());
            let mut conv: HashMap<ValueId, ValueId> = HashMap::new();
            let mut next_value = func.next_value;
            let mut as_f64 = |v: ValueId, out: &mut Vec<(ValueId, Instruction)>| -> ValueId {
                *conv.entry(v).or_insert_with(|| {
                    let c = ValueId(next_value);
                    next_value += 1;
                    out.push((c, Instruction::I64ToF64(v)));
                    c
                })
            };
            for (v, mut inst) in old {
                if int_vals.contains(&v) {
                    out.push((v, inst));
                    continue;
                }
                let both_int =
                    |a: &ValueId, b: &ValueId| int_vals.contains(a) && int_vals.contains(b);
                inst = match inst {
                    Instruction::CmpLtF64(a, b) if both_int(&a, &b) => Instruction::CmpLtI64(a, b),
                    Instruction::CmpGtF64(a, b) if both_int(&a, &b) => Instruction::CmpGtI64(a, b),
                    Instruction::CmpLeF64(a, b) if both_int(&a, &b) => Instruction::CmpLeI64(a, b),
                    Instruction::CmpGeF64(a, b) if both_int(&a, &b) => Instruction::CmpGeI64(a, b),
                    other => other,
                };
                if consumes_f64(&inst) {
                    let needs: Vec<ValueId> = inst
                        .operands()
                        .into_iter()
                        .filter(|o| int_vals.contains(o))
                        .collect();
                    if !needs.is_empty() {
                        let map: HashMap<ValueId, ValueId> =
                            needs.iter().map(|o| (*o, as_f64(*o, &mut out))).collect();
                        super::remap_inst(&mut inst, &map);
                    }
                }
                out.push((v, inst));
            }
            // Edges into parameters that stayed f64.
            let mut term =
                std::mem::replace(&mut func.blocks[bi].terminator, Terminator::Unreachable);
            let mut needs: Vec<ValueId> = Vec::new();
            for (target, args) in edges(&term) {
                for (i, a) in args.iter().enumerate() {
                    let int_param = chosen_params[target.0 as usize]
                        .get(i)
                        .copied()
                        .unwrap_or(false);
                    if !int_param && int_vals.contains(a) {
                        needs.push(*a);
                    }
                }
            }
            if !needs.is_empty() {
                let map: HashMap<ValueId, ValueId> =
                    needs.iter().map(|o| (*o, as_f64(*o, &mut out))).collect();
                let targets: Vec<&[bool]> = edges(&term)
                    .into_iter()
                    .map(|(t, _)| chosen_params[t.0 as usize].as_slice())
                    .collect();
                remap_edges_into_f64(&mut term, &map, &targets);
            }
            func.next_value = next_value;
            func.blocks[bi].instructions = out;
            func.blocks[bi].terminator = term;
        }
        true
    }
}

/// The boxed values that are the count of a List: the counts, their
/// copies, and the parameters every incoming edge passes one to.
fn list_counts(func: &MirFunction, defs: &HashMap<ValueId, Instruction>) -> HashSet<ValueId> {
    let mut incoming: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for b in &func.blocks {
        for (target, args) in edges(&b.terminator) {
            let params = &func.blocks[target.0 as usize].params;
            for (i, a) in args.iter().enumerate() {
                if let Some((p, _)) = params.get(i) {
                    incoming.entry(*p).or_default().push(*a);
                }
            }
        }
    }
    // Optimistically every copy and parameter is a count; one whose
    // source or any incoming argument is not drops out, to a fixed
    // point, so a count that cycles through parameters stays in.
    let mut counts: HashSet<ValueId> = defs
        .iter()
        .filter(|(_, i)| matches!(i, Instruction::ListCount(_) | Instruction::Move(_)))
        .map(|(v, _)| *v)
        .chain(incoming.keys().copied())
        .collect();
    loop {
        let mut shrank = false;
        for (v, inst) in defs {
            if let Instruction::Move(a) = inst
                && counts.contains(v)
                && !counts.contains(a)
            {
                counts.remove(v);
                shrank = true;
            }
        }
        for (p, args) in &incoming {
            if counts.contains(p) && args.iter().any(|a| !counts.contains(a)) {
                counts.remove(p);
                shrank = true;
            }
        }
        if !shrank {
            break;
        }
    }
    counts
}

fn consumes_f64(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::AddF64(..)
            | Instruction::SubF64(..)
            | Instruction::MulF64(..)
            | Instruction::DivF64(..)
            | Instruction::ModF64(..)
            | Instruction::NegF64(_)
            | Instruction::CmpLtF64(..)
            | Instruction::CmpGtF64(..)
            | Instruction::CmpLeF64(..)
            | Instruction::CmpGeF64(..)
            | Instruction::MathUnaryF64(..)
            | Instruction::MathBinaryF64(..)
            | Instruction::Box(_)
    )
}

fn edges(term: &Terminator) -> Vec<(BlockId, &[ValueId])> {
    match term {
        Terminator::Branch { target, args } => vec![(*target, args.as_slice())],
        Terminator::CondBranch {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => vec![
            (*true_target, true_args.as_slice()),
            (*false_target, false_args.as_slice()),
        ],
        _ => Vec::new(),
    }
}

fn remap_edges_into_f64(
    term: &mut Terminator,
    map: &HashMap<ValueId, ValueId>,
    targets: &[&[bool]],
) {
    let fix = |args: &mut Vec<ValueId>, int_params: &[bool]| {
        for (i, a) in args.iter_mut().enumerate() {
            if !int_params.get(i).copied().unwrap_or(false)
                && let Some(c) = map.get(a)
            {
                *a = *c;
            }
        }
    };
    match term {
        Terminator::Branch { args, .. } => fix(args, targets[0]),
        Terminator::CondBranch {
            true_args,
            false_args,
            ..
        } => {
            fix(true_args, targets[0]);
            fix(false_args, targets[1]);
        }
        _ => {}
    }
}

/// Comparisons a block is only entered through: a block with one
/// predecessor whose conditional branch reaches it on exactly one edge.
fn collect_facts(func: &MirFunction) -> Vec<Fact> {
    let mut facts = Vec::new();
    for block in &func.blocks {
        if block.predecessors.len() != 1 {
            continue;
        }
        let pred = &func.blocks[block.predecessors[0].0 as usize];
        let Terminator::CondBranch {
            condition,
            true_target,
            false_target,
            ..
        } = &pred.terminator
        else {
            continue;
        };
        if true_target == false_target {
            continue;
        }
        let Some((_, cond)) = pred.instructions.iter().find(|(v, _)| v == condition) else {
            continue;
        };
        let (x, y, op) = match cond {
            Instruction::CmpLtF64(a, b) => (*a, *b, Cmp::Lt),
            Instruction::CmpLeF64(a, b) => (*a, *b, Cmp::Le),
            Instruction::CmpGtF64(a, b) => (*a, *b, Cmp::Gt),
            Instruction::CmpGeF64(a, b) => (*a, *b, Cmp::Ge),
            _ => continue,
        };
        let op = if *true_target == block.id {
            op
        } else {
            op.negate()
        };
        facts.push(Fact {
            scope: block.id,
            x,
            y,
            op,
        });
        facts.push(Fact {
            scope: block.id,
            x: y,
            y: x,
            op: op.flip(),
        });
    }
    facts
}

fn dominated(idom: &[usize], a: usize, b: usize) -> bool {
    let mut cur = b;
    loop {
        if cur == a {
            return true;
        }
        let next = idom[cur];
        if next == usize::MAX || next == cur {
            return false;
        }
        cur = next;
    }
}

/// The lattice value of `v` as seen from `at`, narrowed by the facts in
/// scope there.
fn refined(
    v: ValueId,
    at: BlockId,
    lat: &HashMap<ValueId, Lat>,
    facts: &[Fact],
    idom: &[usize],
) -> Lat {
    let base = lat.get(&v).copied().unwrap_or(Lat::Bottom);
    let Lat::Int(mut r) = base else {
        return base;
    };
    for f in facts {
        if f.x != v || !dominated(idom, f.scope.0 as usize, at.0 as usize) {
            continue;
        }
        let Some(Lat::Int(y)) = lat.get(&f.y).copied() else {
            continue;
        };
        let (lo, hi) = match f.op {
            Cmp::Lt => (r.lo, r.hi.min(y.hi.saturating_sub(1))),
            Cmp::Le => (r.lo, r.hi.min(y.hi)),
            Cmp::Gt => (r.lo.max(y.lo.saturating_add(1)), r.hi),
            Cmp::Ge => (r.lo.max(y.lo), r.hi),
        };
        if lo <= hi {
            r = Range { lo, hi };
        }
    }
    Lat::Int(r)
}

fn transfer(
    inst: &Instruction,
    at: BlockId,
    lat: &HashMap<ValueId, Lat>,
    facts: &[Fact],
    idom: &[usize],
    defs: &HashMap<ValueId, Instruction>,
    counts: &HashSet<ValueId>,
) -> Lat {
    let get = |v: &ValueId| refined(*v, at, lat, facts, idom);
    match inst {
        // Integer arithmetic never yields a negative zero, so any f64
        // operation that could is left alone: a product with a zero and a
        // negative factor, the negation of zero, a remainder of a
        // negative dividend, and the constant -0 itself.
        Instruction::ConstF64(c) => {
            if c.is_finite()
                && *c == c.trunc()
                && c.abs() <= LIMIT as f64
                && !(*c == 0.0 && c.is_sign_negative())
            {
                bounded(*c as i128, *c as i128)
            } else {
                Lat::Top
            }
        }
        Instruction::Move(a) => get(a),
        // A List's count is a u32.
        Instruction::Unbox(a) if counts.contains(a) => bounded(0, u32::MAX as i128),
        Instruction::AddF64(a, b) => combine(get(a), get(b), |x, y| {
            bounded(x.lo as i128 + y.lo as i128, x.hi as i128 + y.hi as i128)
        }),
        Instruction::SubF64(a, b) => combine(get(a), get(b), |x, y| {
            bounded(x.lo as i128 - y.hi as i128, x.hi as i128 - y.lo as i128)
        }),
        Instruction::MulF64(a, b) => combine(get(a), get(b), |x, y| {
            let has_zero = |r: Range| r.lo <= 0 && r.hi >= 0;
            if (has_zero(x) && y.lo < 0) || (has_zero(y) && x.lo < 0) {
                return Lat::Top;
            }
            let c = [
                x.lo as i128 * y.lo as i128,
                x.lo as i128 * y.hi as i128,
                x.hi as i128 * y.lo as i128,
                x.hi as i128 * y.hi as i128,
            ];
            bounded(*c.iter().min().unwrap(), *c.iter().max().unwrap())
        }),
        Instruction::NegF64(a) => match get(a) {
            Lat::Int(x) if x.lo <= 0 && x.hi >= 0 => Lat::Top,
            Lat::Int(x) => bounded(-(x.hi as i128), -(x.lo as i128)),
            other => other,
        },
        Instruction::ModF64(a, b) => {
            let c = match defs.get(b) {
                Some(Instruction::ConstF64(c))
                    if c.is_finite() && *c == c.trunc() && *c != 0.0 && c.abs() <= LIMIT as f64 =>
                {
                    (*c as i64).abs()
                }
                _ => return Lat::Top,
            };
            match get(a) {
                Lat::Int(x) if x.lo >= 0 => Lat::Int(Range { lo: 0, hi: c - 1 }),
                Lat::Int(_) => Lat::Top,
                other => other,
            }
        }
        _ => Lat::Top,
    }
}
