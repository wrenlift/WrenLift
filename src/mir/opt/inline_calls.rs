//! Inline guarded known calls before type specialisation.
//!
//! A call site the inline cache resolved to one callee becomes a guard on
//! the receiver (its class, or a closure's function) with the callee's
//! body spliced in behind it. A site inside a loop first versions the
//! loop: the guard's failure edge performs the generic call and then
//! continues in a copy of the loop that keeps every call generic, so the
//! fast copy's block parameters are only ever fed by the inlined body and
//! the type specialiser can unbox them. A site outside any loop merges
//! the generic result back into the same continuation.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::licm::{compute_dominators, compute_rpo, detect_loops, merge_loops_by_header};
use super::{remap_inst, remap_term};
use crate::mir::{
    BlockId, DeoptReg, Instruction, MirFunction, MirType, Terminator, ValueId, live_in_sets,
};

/// What the receiver is checked against before the inlined body runs.
#[derive(Clone, Copy, Debug)]
pub enum CalleeGuard {
    /// The receiver's class pointer.
    Class(usize),
    /// The receiver is this object (a class receiving a static call).
    Object(usize),
    /// The receiver is a closure of this `ObjFn` pointer.
    ClosureFn(usize),
}

pub struct KnownCallee {
    pub guard: CalleeGuard,
    pub body: Arc<MirFunction>,
    /// The site constructs an instance of this class: `body` is the
    /// initialiser, run on a fresh instance, and the call's value is
    /// that instance.
    pub constructor: Option<usize>,
    /// A failed guard resumes the interpreter at this offset with
    /// these registers instead of making the call.
    pub exit: Option<(u32, Vec<DeoptReg>)>,
}

impl KnownCallee {
    /// A method body takes the receiver as its first parameter; a
    /// closure body takes only the call's arguments.
    fn takes_receiver(&self) -> bool {
        !matches!(self.guard, CalleeGuard::ClosureFn(_))
    }
}

/// Fields an initialiser stores into its instance before anything
/// could observe it: the leading `SetField`s on `this` in the entry
/// block, up to the first instruction that reads the instance, lets
/// it escape, or could run other code.
fn fields_assigned_first(body: &MirFunction) -> u64 {
    let Some(entry) = body.blocks.first() else {
        return 0;
    };
    let this = entry.instructions.iter().find_map(|(v, inst)| match inst {
        Instruction::BlockParam(0) => Some(*v),
        _ => None,
    });
    let Some(this) = this else {
        return 0;
    };
    let mut mask = 0u64;
    for (_, inst) in &entry.instructions {
        match inst {
            Instruction::BlockParam(_) => {}
            Instruction::SetField(r, idx, val) if *r == this && *val != this => {
                if *idx < 64 {
                    mask |= 1 << idx;
                }
            }
            _ if inst.has_side_effects() || inst.operands().contains(&this) => break,
            _ => {}
        }
    }
    mask
}

const MAX_BODY_INSTRUCTIONS: usize = 48;
const MAX_BODY_BLOCKS: usize = 8;

/// Whether a callee body can be spliced into another function of the
/// same module. Bodies that reach their own frame (upvalues, static
/// fields, the defining class, self recursion) cannot, and a body with
/// a loop of its own keeps its own tier-up path instead.
pub fn inlinable_body(mir: &MirFunction) -> bool {
    if mir.blocks.is_empty() || mir.blocks.len() > MAX_BODY_BLOCKS {
        return false;
    }
    let mut count = 0usize;
    for (i, block) in mir.blocks.iter().enumerate() {
        count += block.instructions.len();
        if count > MAX_BODY_INSTRUCTIONS {
            return false;
        }
        if block
            .terminator
            .successors()
            .iter()
            .any(|s| s.0 as usize <= i)
        {
            return false;
        }
        for (_, inst) in &block.instructions {
            match inst {
                Instruction::BlockParam(_) if i != 0 => return false,
                Instruction::CallStaticSelf { .. }
                | Instruction::SuperCall { .. }
                | Instruction::GetUpvalue(_)
                | Instruction::SetUpvalue(..)
                | Instruction::MakeClosure { .. }
                | Instruction::GetStaticField(_)
                | Instruction::SetStaticField(..)
                | Instruction::CallKnownFunc { .. }
                | Instruction::ClassIs(..)
                | Instruction::ObjectIs(..)
                | Instruction::ClosureFnIs(..) => return false,
                _ => {}
            }
        }
    }
    true
}

/// Whether inlining the body can let the caller's types reach any
/// arithmetic; a body of loads, stores and calls gains nothing over the
/// backend's own guarded splice.
pub fn body_has_arithmetic(mir: &MirFunction) -> bool {
    mir.blocks.iter().any(|b| {
        b.instructions.iter().any(|(_, i)| {
            matches!(
                i,
                Instruction::Add(..)
                    | Instruction::Sub(..)
                    | Instruction::Mul(..)
                    | Instruction::Div(..)
                    | Instruction::Mod(..)
                    | Instruction::Neg(_)
                    | Instruction::CmpLt(..)
                    | Instruction::CmpGt(..)
                    | Instruction::CmpLe(..)
                    | Instruction::CmpGe(..)
                    | Instruction::BitAnd(..)
                    | Instruction::BitOr(..)
                    | Instruction::BitXor(..)
                    | Instruction::Shl(..)
                    | Instruction::Shr(..)
            )
        })
    })
}

struct Site {
    dst: ValueId,
}

/// Inline every call whose destination is a key of `sites`. Returns
/// whether the function changed.
pub fn inline_known_calls(func: &mut MirFunction, sites: &HashMap<ValueId, KnownCallee>) -> bool {
    let mut pending: Vec<Site> = Vec::new();
    let mut in_place: Vec<Site> = Vec::new();
    for block in &func.blocks {
        for (dst, inst) in &block.instructions {
            if let Instruction::Call { args, .. } = inst
                && let Some(callee) = sites.get(dst)
            {
                let expected = args.len() + usize::from(callee.takes_receiver());
                if callee.body.arity as usize == expected && inlinable_body(&callee.body) {
                    if splices_in_place(callee) {
                        in_place.push(Site { dst: *dst });
                    } else {
                        pending.push(Site { dst: *dst });
                    }
                }
            }
        }
    }
    if pending.is_empty() && in_place.is_empty() {
        return false;
    }
    for site in in_place {
        inline_in_place(func, site.dst, sites);
    }

    while let Some(site) = pending.pop() {
        func.compute_predecessors();
        let Some((block, _)) = locate(func, site.dst) else {
            continue;
        };
        let rpo = compute_rpo(func);
        let idom = compute_dominators(func, &rpo);
        let loops = merge_loops_by_header(&detect_loops(func, &idom));
        let innermost = loops
            .iter()
            .filter(|l| l.body.contains(&block))
            .min_by_key(|l| l.body.len());
        match innermost {
            Some(lp) => {
                let body: HashSet<BlockId> = lp.body.iter().copied().collect();
                // Every remaining site in this loop shares the slow copy.
                let mut in_loop = vec![site];
                let mut i = 0;
                while i < pending.len() {
                    match locate(func, pending[i].dst) {
                        Some((b, _)) if body.contains(&b) => in_loop.push(pending.swap_remove(i)),
                        _ => i += 1,
                    }
                }
                let mut slow = version_loop(func, lp.header, &body, &idom);
                for s in in_loop {
                    inline_site(func, s.dst, sites, slow.as_mut());
                }
                if let Some(copy) = slow {
                    repair_copy_ssa(func, &copy);
                }
            }
            None => inline_site(func, site.dst, sites, None),
        }
    }
    func.compute_predecessors();
    true
}

/// A body of one block behind a class guard, with an exit to resume
/// the interpreter at the call, goes in where the call was: no split,
/// no loop copy.
fn splices_in_place(callee: &KnownCallee) -> bool {
    matches!(callee.guard, CalleeGuard::Class(_))
        && callee.constructor.is_none()
        && callee.exit.is_some()
        && callee.body.blocks.len() == 1
        && matches!(
            callee.body.blocks[0].terminator,
            Terminator::Return(_) | Terminator::ReturnNull
        )
}

/// Replace the call with a guard on the receiver's class and the
/// body's instructions, its parameters reading the call's operands
/// and its return defining the call's value.
fn inline_in_place(func: &mut MirFunction, dst: ValueId, sites: &HashMap<ValueId, KnownCallee>) {
    let Some((block, k)) = locate(func, dst) else {
        return;
    };
    let Instruction::Call { receiver, args, .. } = func.block(block).instructions[k].1.clone()
    else {
        return;
    };
    let callee = &sites[&dst];
    let CalleeGuard::Class(class) = callee.guard else {
        return;
    };
    let Some((pc, live)) = callee.exit.clone() else {
        return;
    };
    let body = &callee.body.blocks[0];
    let mut operands = Vec::with_capacity(1 + args.len());
    operands.push(receiver);
    operands.extend_from_slice(&args);
    let mut out: Vec<(ValueId, Instruction)> = Vec::with_capacity(body.instructions.len() + 2);
    // An object's class never changes: a guard earlier in the block on
    // the same receiver covers this site.
    let guarded = func.block(block).instructions[..k].iter().any(|(_, inst)| {
        matches!(inst, Instruction::GuardClassAt { value, class: c, .. }
                if *value == receiver && *c == class)
    });
    if !guarded {
        let guard = func.new_value();
        out.push((
            guard,
            Instruction::GuardClassAt {
                value: receiver,
                class,
                pc,
                live,
            },
        ));
    }
    // The body's instructions are the call's, for a trace.
    let site_span = func.span_map.get(&dst).cloned();
    let mut vmap: HashMap<ValueId, ValueId> = HashMap::new();
    for (v, inst) in &body.instructions {
        if let Instruction::BlockParam(idx) = inst {
            vmap.insert(*v, operands[*idx as usize]);
            continue;
        }
        let nv = func.new_value();
        vmap.insert(*v, nv);
        if let Some(span) = site_span.clone() {
            func.span_map.insert(nv, span);
        }
        let mut inst = inst.clone();
        remap_inst(&mut inst, &vmap);
        out.push((nv, inst));
    }
    match &body.terminator {
        Terminator::Return(v) => out.push((dst, Instruction::Move(vmap[v]))),
        _ => out.push((dst, Instruction::ConstNull)),
    }
    func.block_mut(block).instructions.splice(k..=k, out);
}

fn locate(func: &MirFunction, dst: ValueId) -> Option<(BlockId, usize)> {
    for block in &func.blocks {
        if let Some(k) = block.instructions.iter().position(|(d, _)| *d == dst) {
            return Some((block.id, k));
        }
    }
    None
}

/// The generic copy of a versioned loop.
struct SlowCopy {
    /// The copy's loop header.
    header: BlockId,
    /// Blocks of the copy, extended as sites split them.
    blocks: HashSet<BlockId>,
    /// Fast-world value → its clone in the copy.
    fast_to_slow: HashMap<ValueId, ValueId>,
    /// Copy value → the fast-world value it stands for.
    slow_to_fast: HashMap<ValueId, ValueId>,
    /// `(fast-world block, copy block, (copy value, generic result))`:
    /// a guard's failure path enters the copy there, after its generic
    /// call, whose result stands for the site's value on that edge
    /// alone; every other edge hands over the fast continuation's
    /// parameter.
    entries: Vec<(BlockId, BlockId, (ValueId, ValueId))>,
}

/// Give every copy block a parameter for each value the copy defines
/// elsewhere and the block reads, and pass it along every edge. The
/// copy is then entered anywhere with only what the entry names: the
/// failure edges hand over the fast world's own values.
fn repair_copy_ssa(func: &mut MirFunction, copy: &SlowCopy) {
    func.compute_predecessors();
    let mut copy_defs: HashSet<ValueId> = HashSet::new();
    for &b in &copy.blocks {
        copy_defs.extend(func.block(b).defined_values());
    }
    let live_in = live_in_sets(func);
    let types = crate::mir::infer_value_types(func);
    let mut ordered: Vec<BlockId> = copy.blocks.iter().copied().collect();
    ordered.sort_by_key(|b| b.0);
    // Per block: the values it needs, and the parameter standing for
    // each inside it.
    let mut needed: HashMap<BlockId, Vec<ValueId>> = HashMap::new();
    let mut names: HashMap<BlockId, HashMap<ValueId, ValueId>> = HashMap::new();
    for &b in &ordered {
        if b == copy.header {
            continue;
        }
        let mut need: Vec<ValueId> = live_in
            .iter(b.0 as usize)
            .filter(|v| copy_defs.contains(v))
            .collect();
        need.sort_by_key(|v| v.0);
        if need.is_empty() {
            continue;
        }
        let mut rename: HashMap<ValueId, ValueId> = HashMap::new();
        for &v in &need {
            let p = func.new_value();
            func.block_mut(b).params.push((p, param_type(&types, v)));
            rename.insert(v, p);
        }
        let block = func.block_mut(b);
        for (_, inst) in &mut block.instructions {
            remap_inst(inst, &rename);
        }
        remap_term(&mut block.terminator, &rename);
        needed.insert(b, need);
        names.insert(b, rename);
    }
    // Edges inside the copy pass the predecessor's name for each value.
    for &p in &ordered {
        let succs = func.block(p).terminator.successors();
        for s in succs {
            let Some(need) = needed.get(&s) else {
                continue;
            };
            let empty = HashMap::new();
            let mine = names.get(&p).unwrap_or(&empty);
            let args: Vec<ValueId> = need
                .iter()
                .map(|v| mine.get(v).copied().unwrap_or(*v))
                .collect();
            append_edge_args(&mut func.block_mut(p).terminator, s, &args);
        }
    }
    // Failure edges pass the fast world's values.
    for &(from, post, (site, result)) in &copy.entries {
        let Some(need) = needed.get(&post) else {
            continue;
        };
        let args: Vec<ValueId> = need
            .iter()
            .map(|v| {
                if *v == site {
                    result
                } else {
                    copy.slow_to_fast[v]
                }
            })
            .collect();
        func.block_mut(from).terminator = Terminator::Branch { target: post, args };
    }
}

/// `a` dominates `b`; false for blocks the entry does not reach.
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

/// Clone the loop so a guard can fail into a copy that keeps every call
/// generic. Returns `None` when an exit block has a predecessor outside
/// the loop, which the exit-parameter rewrite does not handle.
fn version_loop(
    func: &mut MirFunction,
    header: BlockId,
    body: &HashSet<BlockId>,
    idom: &[usize],
) -> Option<SlowCopy> {
    // WLIFT_DISABLE_LOOP_VERSIONING merges the generic result back into
    // the fast loop instead; safe to run with.
    if std::env::var_os("WLIFT_DISABLE_LOOP_VERSIONING").is_some() {
        return None;
    }
    // Exit targets and their predecessors.
    let mut exits: Vec<BlockId> = Vec::new();
    for &b in body {
        for succ in func.block(b).terminator.successors() {
            if !body.contains(&succ) && !exits.contains(&succ) {
                exits.push(succ);
            }
        }
    }
    exits.sort_by_key(|b| b.0);
    for &e in &exits {
        if func.block(e).predecessors.iter().any(|p| !body.contains(p)) {
            return None;
        }
    }

    // Loop-defined values live into an exit become parameters of the
    // exit block, so both copies can feed them.
    let mut loop_defs: HashSet<ValueId> = HashSet::new();
    for &b in body {
        loop_defs.extend(func.block(b).defined_values());
    }
    let live_in = live_in_sets(func);
    for &e in &exits {
        let mut needed: Vec<ValueId> = live_in
            .iter(e.0 as usize)
            .filter(|v| loop_defs.contains(v))
            .collect();
        needed.sort_by_key(|v| v.0);
        if needed.is_empty() {
            continue;
        }
        let types = crate::mir::infer_value_types(func);
        let mut rename: HashMap<ValueId, ValueId> = HashMap::new();
        for &v in &needed {
            let p = func.new_value();
            func.block_mut(e).params.push((p, param_type(&types, v)));
            rename.insert(v, p);
        }
        for bi in 0..func.blocks.len() {
            if dominated(idom, e.0 as usize, bi) {
                let block = &mut func.blocks[bi];
                for (_, inst) in &mut block.instructions {
                    remap_inst(inst, &rename);
                }
                remap_term(&mut block.terminator, &rename);
            }
        }
        for &b in body {
            append_edge_args(&mut func.block_mut(b).terminator, e, &needed);
        }
    }

    // Clone the loop blocks.
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
    let mut ordered: Vec<BlockId> = body.iter().copied().collect();
    ordered.sort_by_key(|b| b.0);
    for &b in &ordered {
        let nb = func.new_block();
        block_map.insert(b, nb);
    }
    let mut fast_to_slow: HashMap<ValueId, ValueId> = HashMap::new();
    for &b in &ordered {
        for v in func.block(b).defined_values() {
            let nv = func.new_value();
            fast_to_slow.insert(v, nv);
            if let Some(span) = func.span_map.get(&v).cloned() {
                func.span_map.insert(nv, span);
            }
        }
    }
    for &b in &ordered {
        let src = func.block(b).clone();
        let nb = block_map[&b];
        let params: Vec<(ValueId, MirType)> = src
            .params
            .iter()
            .map(|(v, t)| (fast_to_slow[v], *t))
            .collect();
        let mut instructions = Vec::with_capacity(src.instructions.len());
        for (dst, inst) in &src.instructions {
            let mut inst = inst.clone();
            remap_inst(&mut inst, &fast_to_slow);
            instructions.push((fast_to_slow[dst], inst));
        }
        let mut terminator = src.terminator.clone();
        remap_term(&mut terminator, &fast_to_slow);
        retarget(&mut terminator, &block_map);
        let dstb = func.block_mut(nb);
        dstb.params = params;
        dstb.instructions = instructions;
        dstb.terminator = terminator;
    }
    let slow_to_fast: HashMap<ValueId, ValueId> =
        fast_to_slow.iter().map(|(f, s)| (*s, *f)).collect();
    // The copy's calls keep their inline-cache entries.
    let cloned_sites: Vec<(ValueId, u32)> = fast_to_slow
        .iter()
        .filter_map(|(f, s)| func.ic_sites.get(f).map(|i| (*s, *i)))
        .collect();
    func.ic_sites.extend(cloned_sites);
    func.osr_excluded.extend(block_map.values().copied());
    Some(SlowCopy {
        header: block_map[&header],
        blocks: block_map.values().copied().collect(),
        fast_to_slow,
        slow_to_fast,
        entries: Vec::new(),
    })
}

/// A parameter carrying `v` keeps `v`'s representation; comparison
/// results travel boxed.
fn param_type(types: &[MirType], v: ValueId) -> MirType {
    match types.get(v.0 as usize) {
        Some(MirType::F64) => MirType::F64,
        Some(MirType::I64) => MirType::I64,
        _ => MirType::Value,
    }
}

fn append_edge_args(term: &mut Terminator, target: BlockId, extra: &[ValueId]) {
    match term {
        Terminator::Branch { target: t, args } if *t == target => args.extend_from_slice(extra),
        Terminator::CondBranch {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            if *true_target == target {
                true_args.extend_from_slice(extra);
            }
            if *false_target == target {
                false_args.extend_from_slice(extra);
            }
        }
        _ => {}
    }
}

fn retarget(term: &mut Terminator, map: &HashMap<BlockId, BlockId>) {
    match term {
        Terminator::Branch { target, .. } => {
            if let Some(n) = map.get(target) {
                *target = *n;
            }
        }
        Terminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            if let Some(n) = map.get(true_target) {
                *true_target = *n;
            }
            if let Some(n) = map.get(false_target) {
                *false_target = *n;
            }
        }
        _ => {}
    }
}

/// Split `block` after instruction `k`: the tail and the terminator move
/// to a fresh block, which is returned. The head keeps an `Unreachable`
/// terminator for the caller to replace.
pub(crate) fn split_after(func: &mut MirFunction, block: BlockId, k: usize) -> BlockId {
    let post = func.new_block();
    let head = func.block_mut(block);
    let tail: Vec<(ValueId, Instruction)> = head.instructions.drain(k + 1..).collect();
    let term = std::mem::replace(&mut head.terminator, Terminator::Unreachable);
    let post_block = func.block_mut(post);
    post_block.instructions = tail;
    post_block.terminator = term;
    post
}

fn inline_site(
    func: &mut MirFunction,
    dst: ValueId,
    sites: &HashMap<ValueId, KnownCallee>,
    slow: Option<&mut SlowCopy>,
) {
    let Some((block, k)) = locate(func, dst) else {
        return;
    };
    let call = func.block(block).instructions[k].1.clone();
    let Instruction::Call { receiver, args, .. } = &call else {
        return;
    };
    let receiver = *receiver;
    let args = args.clone();
    let callee = &sites[&dst];

    // Fast continuation: everything after the call, with the result as
    // its parameter.
    let post = split_after(func, block, k);
    func.block_mut(block).instructions.pop();
    func.block_mut(post).params.insert(0, (dst, MirType::Value));

    let guard = func.new_value();
    let guard_inst = match callee.guard {
        CalleeGuard::Class(p) => Instruction::ClassIs(receiver, p),
        CalleeGuard::Object(p) => Instruction::ObjectIs(receiver, p),
        CalleeGuard::ClosureFn(p) => Instruction::ClosureFnIs(receiver, p),
    };
    func.block_mut(block).instructions.push((guard, guard_inst));

    let constructor = callee.constructor.map(|class| Instruction::NewInstance {
        class,
        assigned: fields_assigned_first(&callee.body),
    });
    // The body reads the call's operands directly: its entry has the
    // guard's true edge as its only predecessor. An initialiser's
    // instance is allocated by the body itself.
    let mut operands = Vec::with_capacity(1 + args.len());
    if callee.takes_receiver() && callee.constructor.is_none() {
        operands.push(receiver);
    }
    operands.extend_from_slice(&args);
    let site_span = func.span_map.get(&dst).cloned();
    let entry = splice_body(func, &callee.body, post, constructor, &operands, site_span);

    if let Some((pc, live)) = callee.exit.clone() {
        let exit_block = func.new_block();
        let exit = func.new_value();
        func.block_mut(exit_block)
            .instructions
            .push((exit, Instruction::SlowPathExit { pc, live }));
        func.block_mut(exit_block).terminator = Terminator::Unreachable;
        func.block_mut(block).terminator = Terminator::CondBranch {
            condition: guard,
            true_target: entry,
            true_args: Vec::new(),
            false_target: exit_block,
            false_args: Vec::new(),
        };
        return;
    }

    // Generic path: the original call, then either the fast
    // continuation or the slow copy's continuation.
    let slow_block = func.new_block();
    let slow_result = func.new_value();
    if let Some(site) = func.ic_sites.get(&dst).copied() {
        func.ic_sites.insert(slow_result, site);
    }
    func.block_mut(slow_block)
        .instructions
        .push((slow_result, call.clone()));
    func.block_mut(block).terminator = Terminator::CondBranch {
        condition: guard,
        true_target: entry,
        true_args: Vec::new(),
        false_target: slow_block,
        false_args: Vec::new(),
    };

    let slow_term = match slow {
        None => Terminator::Branch {
            target: post,
            args: vec![slow_result],
        },
        Some(copy) => {
            let slow_dst = copy.fast_to_slow[&dst];
            let target = slow_continuation(func, copy, slow_dst);
            copy.entries
                .push((slow_block, target, (slow_dst, slow_result)));
            // `repair_copy_ssa` supplies the arguments once every site
            // in the loop is done.
            Terminator::Branch {
                target,
                args: Vec::new(),
            }
        }
    };
    func.block_mut(slow_block).terminator = slow_term;
}

/// Split the slow copy after its own copy of the call; the tail is
/// where the fast world's guard failure enters once `repair_copy_ssa`
/// has given it parameters.
fn slow_continuation(func: &mut MirFunction, copy: &mut SlowCopy, slow_dst: ValueId) -> BlockId {
    let (block, k) = locate(func, slow_dst).expect("slow copy holds the cloned call");
    let post = split_after(func, block, k);
    func.block_mut(block).terminator = Terminator::Branch {
        target: post,
        args: Vec::new(),
    };
    copy.blocks.insert(post);
    post
}

/// Copy the callee's blocks into `func`. The body's parameters read
/// `operands` (the receiver, then the arguments) in place; every
/// return jumps to `post` with the returned value. With `constructor`,
/// the receiver is instead that allocation, made first in the entry
/// block and absent from `operands`, and every return hands it to
/// `post`.
/// `site_span` is the call's, given to every value of the body so a
/// trace places them at the call.
fn splice_body(
    func: &mut MirFunction,
    body: &MirFunction,
    post: BlockId,
    constructor: Option<Instruction>,
    operands: &[ValueId],
    site_span: Option<crate::ast::Span>,
) -> BlockId {
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();
    for b in &body.blocks {
        block_map.insert(b.id, func.new_block());
    }
    let mut vmap: HashMap<ValueId, ValueId> = HashMap::new();
    for b in &body.blocks {
        for v in b.defined_values() {
            let nv = func.new_value();
            vmap.insert(v, nv);
            if let Some(span) = site_span.clone() {
                func.span_map.insert(nv, span);
            }
        }
    }
    let entry = block_map[&body.blocks[0].id];
    // What each `BlockParam` of the body reads.
    let mut params_in: Vec<ValueId> = Vec::with_capacity(body.arity as usize);
    if constructor.is_some() {
        params_in.push(func.new_value());
    }
    params_in.extend_from_slice(operands);
    let receiver = params_in[0];
    for b in &body.blocks {
        let nb = block_map[&b.id];
        let mut params: Vec<(ValueId, MirType)> =
            b.params.iter().map(|(v, t)| (vmap[v], *t)).collect();
        let mut instructions = Vec::with_capacity(b.instructions.len());
        if b.id == body.blocks[0].id {
            params.clear();
            if let Some(alloc) = constructor.clone() {
                instructions.push((receiver, alloc));
            }
        }
        for (dst, inst) in &b.instructions {
            if let Instruction::BlockParam(idx) = inst {
                vmap.insert(*dst, params_in[*idx as usize]);
                continue;
            }
            let mut inst = inst.clone();
            remap_inst(&mut inst, &vmap);
            instructions.push((vmap[dst], inst));
        }
        let terminator = match &b.terminator {
            Terminator::Return(_) | Terminator::ReturnNull if constructor.is_some() => {
                Terminator::Branch {
                    target: post,
                    args: vec![receiver],
                }
            }
            Terminator::Return(v) => Terminator::Branch {
                target: post,
                args: vec![vmap[v]],
            },
            Terminator::ReturnNull => {
                let n = func.new_value();
                instructions.push((n, Instruction::ConstNull));
                Terminator::Branch {
                    target: post,
                    args: vec![n],
                }
            }
            other => {
                let mut t = other.clone();
                remap_term(&mut t, &vmap);
                retarget(&mut t, &block_map);
                t
            }
        };
        let dstb = func.block_mut(nb);
        dstb.params = params;
        dstb.instructions = instructions;
        dstb.terminator = terminator;
    }
    entry
}
