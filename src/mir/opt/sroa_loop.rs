//! Scalar replacement of loop-carried objects.
//!
//! A loop that rebinds `val = Complex.new(...)` every iteration
//! allocates an object whose only purpose is to carry two numbers to
//! the next trip. When the class's constructor only stores its
//! arguments into fields and every use of the object is a trivial
//! getter or a branch argument into a block parameter fed the same
//! way, the object never needs to exist: the allocation goes away,
//! each block parameter splits into one parameter per field, and
//! every getter reads the field's value directly. Loop-carried type
//! inference then unboxes the arithmetic.
//!
//! Any other use is an escape and the whole flow group keeps its
//! allocations, so the pass is a pure win or a no-op.

use std::collections::{HashMap, HashSet};

use crate::intern::SymbolId;
use crate::mir::{BlockId, Instruction, MirFunction, MirType, Terminator, ValueId};

/// What the pass needs to know about a class whose instances can be
/// replaced by scalars.
#[derive(Debug, Clone)]
pub struct ScalarClass {
    pub num_fields: usize,
    /// Constructor signature -> for each argument (receiver excluded),
    /// the field it initialises, if any.
    pub ctors: HashMap<SymbolId, Vec<Option<usize>>>,
    /// Trivial getter signature -> field index.
    pub getters: HashMap<SymbolId, usize>,
}

/// Resolves a module variable slot to a scalar-replaceable class.
pub type ClassResolver<'a> = dyn Fn(u32) -> Option<std::sync::Arc<ScalarClass>> + 'a;

/// Whether the MIR is a constructor that only stores arguments into
/// fields. Returns the argument index for each field it initialises.
pub fn trivial_ctor_field_map(mir: &MirFunction) -> Option<HashMap<usize, usize>> {
    if mir.blocks.len() != 1 {
        return None;
    }
    let block = &mir.blocks[0];
    let mut params: HashMap<ValueId, u16> = HashMap::new();
    let mut aliases: HashMap<ValueId, ValueId> = HashMap::new();
    let mut this: Option<ValueId> = None;
    let mut fields: HashMap<usize, usize> = HashMap::new();
    let resolve = |v: ValueId, aliases: &HashMap<ValueId, ValueId>| -> ValueId {
        let mut cur = v;
        while let Some(&src) = aliases.get(&cur) {
            cur = src;
        }
        cur
    };
    for (vid, inst) in &block.instructions {
        match inst {
            Instruction::BlockParam(idx) => {
                params.insert(*vid, *idx);
                if *idx == 0 {
                    this = Some(*vid);
                }
            }
            Instruction::Move(src) => {
                aliases.insert(*vid, *src);
            }
            Instruction::SetField(obj, field, val) => {
                let obj = resolve(*obj, &aliases);
                if Some(obj) != this {
                    return None;
                }
                let val = resolve(*val, &aliases);
                let arg = *params.get(&val)?;
                if arg == 0 || fields.contains_key(&(*field as usize)) {
                    return None;
                }
                fields.insert(*field as usize, arg as usize - 1);
            }
            _ => return None,
        }
    }
    match &block.terminator {
        Terminator::Return(v) if Some(resolve(*v, &aliases)) == this => Some(fields),
        _ => None,
    }
}

/// Whether the MIR is a getter that returns one field of `this`.
pub fn trivial_getter_field(mir: &MirFunction) -> Option<usize> {
    if mir.blocks.len() != 1 {
        return None;
    }
    let block = &mir.blocks[0];
    let mut this: Option<ValueId> = None;
    let mut loaded: Option<(ValueId, usize)> = None;
    let mut aliases: HashMap<ValueId, ValueId> = HashMap::new();
    for (vid, inst) in &block.instructions {
        match inst {
            Instruction::BlockParam(0) => this = Some(*vid),
            Instruction::BlockParam(_) => return None,
            Instruction::Move(src) => {
                aliases.insert(*vid, *src);
            }
            Instruction::GetField(obj, field) => {
                let mut o = *obj;
                while let Some(&s) = aliases.get(&o) {
                    o = s;
                }
                if Some(o) != this || loaded.is_some() {
                    return None;
                }
                loaded = Some((*vid, *field as usize));
            }
            _ => return None,
        }
    }
    let (lv, field) = loaded?;
    match &block.terminator {
        Terminator::Return(v) => {
            let mut r = *v;
            while let Some(&s) = aliases.get(&r) {
                r = s;
            }
            if r == lv {
                Some(field)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[derive(Clone)]
enum Scalar {
    /// Result of a constructor call: field values in field order.
    Alloc {
        class: std::sync::Arc<ScalarClass>,
        fields: Vec<Option<ValueId>>,
    },
    /// Block parameter fed only by scalars of one class.
    Param { class: std::sync::Arc<ScalarClass> },
    /// Move of another scalar.
    Alias,
}

/// Run the pass. Returns true if anything changed.
pub fn scalar_replace_loop_objects(func: &mut MirFunction, resolve: &ClassResolver) -> bool {
    // Module variables that hold candidate classes, by value id.
    let mut class_of_value: HashMap<ValueId, std::sync::Arc<ScalarClass>> = HashMap::new();
    let mut moves: HashMap<ValueId, ValueId> = HashMap::new();
    // One Arc per slot so allocations of the same class compare equal
    // by pointer.
    let mut by_slot: HashMap<u32, Option<std::sync::Arc<ScalarClass>>> = HashMap::new();
    for block in &func.blocks {
        for (vid, inst) in &block.instructions {
            match inst {
                Instruction::GetModuleVar(idx) => {
                    let c = by_slot
                        .entry(*idx as u32)
                        .or_insert_with(|| resolve(*idx as u32))
                        .clone();
                    if let Some(c) = c {
                        class_of_value.insert(*vid, c);
                    }
                }
                Instruction::Move(src) => {
                    moves.insert(*vid, *src);
                }
                _ => {}
            }
        }
    }
    if class_of_value.is_empty() {
        return false;
    }
    let root = |v: ValueId, moves: &HashMap<ValueId, ValueId>| {
        let mut cur = v;
        while let Some(&s) = moves.get(&cur) {
            cur = s;
        }
        cur
    };

    // Candidate scalars: constructor calls on those classes, every
    // non-entry block parameter (optimistically), moves of either.
    let mut scalars: HashMap<ValueId, Scalar> = HashMap::new();
    for block in func.blocks.iter() {
        for (vid, inst) in block.instructions.iter() {
            if let Instruction::Call {
                receiver,
                method,
                args,
                ..
            } = inst
            {
                let recv = root(*receiver, &moves);
                let Some(class) = class_of_value.get(&recv) else {
                    continue;
                };
                if std::env::var_os("WLIFT_SROA_TRACE").is_some() {
                    eprintln!(
                        "sroa-trace: call on class value {:?} method sym {} ctor syms {:?}",
                        vid,
                        method.index(),
                        class.ctors.keys().map(|k| k.index()).collect::<Vec<_>>()
                    );
                }
                let Some(arg_fields) = class.ctors.get(method) else {
                    continue;
                };
                if arg_fields.len() != args.len() {
                    continue;
                }
                let mut fields: Vec<Option<ValueId>> = vec![None; class.num_fields];
                for (arg, slot) in args.iter().zip(arg_fields.iter()) {
                    if let Some(f) = slot {
                        fields[*f] = Some(*arg);
                    }
                }
                scalars.insert(
                    *vid,
                    Scalar::Alloc {
                        class: class.clone(),
                        fields,
                    },
                );
            }
        }
    }
    if scalars.is_empty() {
        return false;
    }
    // Params start as candidates without a class; the class is fixed
    // by their incoming values during the fixed point.
    let mut param_candidates: HashSet<ValueId> = HashSet::new();
    for block in func.blocks.iter().skip(1) {
        for &(p, _) in &block.params {
            param_candidates.insert(p);
        }
    }

    // Fixed point: assign classes to params from incoming values, then
    // drop anything with a disallowed use or an inconsistent feed.
    loop {
        let mut param_class: HashMap<ValueId, Option<std::sync::Arc<ScalarClass>>> = HashMap::new();
        let class_of = |v: ValueId,
                        scalars: &HashMap<ValueId, Scalar>,
                        param_class: &HashMap<ValueId, Option<std::sync::Arc<ScalarClass>>>|
         -> Option<std::sync::Arc<ScalarClass>> {
            let r = root(v, &moves);
            match scalars.get(&r) {
                Some(Scalar::Alloc { class, .. }) => Some(class.clone()),
                Some(Scalar::Param { class, .. }) => Some(class.clone()),
                _ => match param_class.get(&r) {
                    Some(Some(c)) => Some(c.clone()),
                    _ => None,
                },
            }
        };
        // Seed param classes from any scalar incoming value; verify all
        // incoming values agree.
        let mut dropped_params: HashSet<ValueId> = HashSet::new();
        for block in &func.blocks {
            let mut visit = |target: BlockId, args: &[ValueId]| {
                let params = &func.blocks[target.0 as usize].params;
                for (i, arg) in args.iter().enumerate() {
                    let Some(&(p, _)) = params.get(i) else { continue };
                    if !param_candidates.contains(&p) {
                        continue;
                    }
                    let arg_root = root(*arg, &moves);
                    let arg_class = match scalars.get(&arg_root) {
                        Some(Scalar::Alloc { class, .. }) | Some(Scalar::Param { class, .. }) => {
                            Some(class.clone())
                        }
                        _ => None,
                    };
                    match arg_class {
                        Some(c) => match param_class.get(&p) {
                            None => {
                                param_class.insert(p, Some(c));
                            }
                            Some(Some(existing)) => {
                                if !std::sync::Arc::ptr_eq(existing, &c) {
                                    dropped_params.insert(p);
                                }
                            }
                            Some(None) => {}
                        },
                        None => {
                            // A param fed by itself around a loop is
                            // consistent with whatever else feeds it.
                            if arg_root == p {
                                continue;
                            }
                            // Fed by a non-scalar; unless that feed is
                            // itself a candidate param resolved later,
                            // this param cannot be scalar.
                            if !param_candidates.contains(&arg_root) {
                                dropped_params.insert(p);
                            } else {
                                param_class.entry(p).or_insert(None);
                            }
                        }
                    }
                }
            };
            match &block.terminator {
                Terminator::Branch { target, args } => visit(*target, args),
                Terminator::CondBranch {
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                    ..
                } => {
                    visit(*true_target, true_args);
                    visit(*false_target, false_args);
                }
                _ => {}
            }
        }
        // Params fed only by other candidate params (no class yet) are
        // resolved through those params; if none has a class, drop.
        let mut progressed = true;
        while progressed {
            progressed = false;
            for block in &func.blocks {
                let mut visit = |target: BlockId, args: &[ValueId]| {
                    let params = &func.blocks[target.0 as usize].params;
                    for (i, arg) in args.iter().enumerate() {
                        let Some(&(p, _)) = params.get(i) else { continue };
                        if !param_candidates.contains(&p) || dropped_params.contains(&p) {
                            continue;
                        }
                        let arg_root = root(*arg, &moves);
                        if let Some(Some(c)) = param_class.get(&arg_root).cloned() {
                            match param_class.get(&p) {
                                Some(None) | None => {
                                    param_class.insert(p, Some(c));
                                    progressed = true;
                                }
                                Some(Some(existing)) => {
                                    if !std::sync::Arc::ptr_eq(existing, &c) {
                                        dropped_params.insert(p);
                                    }
                                }
                            }
                        }
                    }
                };
                match &block.terminator {
                    Terminator::Branch { target, args } => visit(*target, args),
                    Terminator::CondBranch {
                        true_target,
                        true_args,
                        false_target,
                        false_args,
                        ..
                    } => {
                        visit(*true_target, true_args);
                        visit(*false_target, false_args);
                    }
                    _ => {}
                }
            }
        }
        for (p, c) in &param_class {
            if c.is_none() {
                dropped_params.insert(*p);
            }
        }
        for p in param_candidates.iter() {
            if !param_class.contains_key(p) {
                dropped_params.insert(*p);
            }
        }
        // Install params as scalars for this round.
        let mut round: HashMap<ValueId, Scalar> = scalars
            .iter()
            .filter(|(_, s)| matches!(s, Scalar::Alloc { .. }))
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        for block in func.blocks.iter() {
            for &(p, _) in block.params.iter() {
                if param_candidates.contains(&p) && !dropped_params.contains(&p) {
                    if let Some(Some(c)) = param_class.get(&p) {
                        round.insert(p, Scalar::Param { class: c.clone() });
                    }
                }
            }
        }
        for (m, src) in &moves {
            if round.contains_key(&root(*src, &moves)) {
                round.insert(*m, Scalar::Alias);
            }
        }
        // Check every use of every scalar.
        let mut escaped: HashSet<ValueId> = HashSet::new();
        for block in &func.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::Move(_) => {}
                    Instruction::Call {
                        receiver,
                        method,
                        args,
                        ..
                    } => {
                        let recv_root = root(*receiver, &moves);
                        if round.contains_key(&recv_root) {
                            let ok = args.is_empty()
                                && class_of(recv_root, &round, &param_class)
                                    .map(|c| c.getters.contains_key(method))
                                    .unwrap_or(false);
                            if !ok {
                                escaped.insert(recv_root);
                            }
                        }
                        for a in args {
                            let r = root(*a, &moves);
                            if round.contains_key(&r) {
                                // Constructor arguments feeding a
                                // candidate allocation are fine only if
                                // that allocation is itself scalar; treat
                                // as escape (the field would hold an
                                // object).
                                escaped.insert(r);
                            }
                        }
                    }
                    other => {
                        for op in other.operands() {
                            let r = root(op, &moves);
                            if round.contains_key(&r) {
                                escaped.insert(r);
                            }
                        }
                    }
                }
            }
            match &block.terminator {
                Terminator::Return(v) => {
                    let r = root(*v, &moves);
                    if round.contains_key(&r) {
                        escaped.insert(r);
                    }
                }
                Terminator::CondBranch { condition, .. } => {
                    let r = root(*condition, &moves);
                    if round.contains_key(&r) {
                        escaped.insert(r);
                    }
                }
                _ => {}
            }
            // Branch args into a non-scalar param are escapes.
            let mut visit = |target: BlockId, args: &[ValueId]| {
                let params = &func.blocks[target.0 as usize].params;
                for (i, arg) in args.iter().enumerate() {
                    let r = root(*arg, &moves);
                    if !round.contains_key(&r) {
                        continue;
                    }
                    match params.get(i) {
                        Some(&(p, _)) if round.contains_key(&p) => {}
                        _ => {
                            escaped.insert(r);
                        }
                    }
                }
            };
            match &block.terminator {
                Terminator::Branch { target, args } => visit(*target, args),
                Terminator::CondBranch {
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                    ..
                } => {
                    visit(*true_target, true_args);
                    visit(*false_target, false_args);
                }
                _ => {}
            }
        }
        if std::env::var_os("WLIFT_SROA_TRACE").is_some() {
            eprintln!(
                "sroa-trace: round: allocs={} params={} escaped={:?} dropped={:?}",
                scalars.len(),
                round.values().filter(|s| matches!(s, Scalar::Param { .. })).count(),
                escaped,
                dropped_params
            );
        }
        if escaped.is_empty() {
            scalars = round;
            break;
        }
        for e in &escaped {
            scalars.remove(e);
            param_candidates.remove(e);
        }
        scalars.retain(|_, s| matches!(s, Scalar::Alloc { .. }));
        if scalars.is_empty() {
            return false;
        }
    }
    if scalars.is_empty() {
        return false;
    }

    // ---- Rewrite ----
    // Field values of a scalar, materialising nulls where the
    // constructor left a field unset.
    let mut split_params: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for (v, s) in &scalars {
        if let Scalar::Param { class, .. } = s {
            let fresh: Vec<ValueId> = (0..class.num_fields).map(|_| func.new_value()).collect();
            for (f, nv) in fresh.iter().enumerate() {
                func.scalar_param_sources.insert(*nv, (*v, f as u16));
            }
            split_params.insert(*v, fresh);
        }
    }
    // Field values of a scalar; `None` is a field the constructor left
    // null.
    let fields_of = |v: ValueId,
                     scalars: &HashMap<ValueId, Scalar>,
                     split_params: &HashMap<ValueId, Vec<ValueId>>|
     -> Vec<Option<ValueId>> {
        let r = root(v, &moves);
        match scalars.get(&r) {
            Some(Scalar::Alloc { fields, .. }) => fields.clone(),
            Some(Scalar::Param { .. }) => split_params[&r].iter().map(|v| Some(*v)).collect(),
            _ => unreachable!("scalar root without a definition"),
        }
    };
    // Null constants needed as branch arguments, inserted before the
    // block's terminator.
    let mut null_consts: Vec<(usize, ValueId)> = Vec::new();

    // Getters -> field moves; allocations and aliases deleted.
    let nblocks = func.blocks.len();
    #[allow(clippy::needless_range_loop)] // the body also allocates values on `func`
    for bi in 0..nblocks {
        let mut new_insts = Vec::with_capacity(func.blocks[bi].instructions.len());
        let old = std::mem::take(&mut func.blocks[bi].instructions);
        for (vid, inst) in old {
            if scalars.contains_key(&vid) {
                // Allocation or alias: gone.
                continue;
            }
            if let Instruction::Call {
                receiver, method, ..
            } = &inst
            {
                let r = root(*receiver, &moves);
                if let Some(class) = match scalars.get(&r) {
                    Some(Scalar::Alloc { class, .. }) | Some(Scalar::Param { class, .. }) => {
                        Some(class.clone())
                    }
                    _ => None,
                } {
                    let field = class.getters[method];
                    let fields = fields_of(r, &scalars, &split_params);
                    match fields[field] {
                        Some(src) => new_insts.push((vid, Instruction::Move(src))),
                        None => new_insts.push((vid, Instruction::ConstNull)),
                    }
                    continue;
                }
            }
            new_insts.push((vid, inst));
        }
        func.blocks[bi].instructions = new_insts;
    }
    // Branch args -> field lists; params -> split params.
    for bi in 0..nblocks {
        let succ_params: Vec<(BlockId, Vec<ValueId>)> = match &func.blocks[bi].terminator {
            Terminator::Branch { target, args } => vec![(*target, args.clone())],
            Terminator::CondBranch {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => vec![
                (*true_target, true_args.clone()),
                (*false_target, false_args.clone()),
            ],
            _ => vec![],
        };
        let mut rewritten: Vec<Vec<ValueId>> = Vec::new();
        for (target, args) in &succ_params {
            let params: Vec<ValueId> = func.blocks[target.0 as usize]
                .params
                .iter()
                .map(|(p, _)| *p)
                .collect();
            let mut out = Vec::new();
            for (i, arg) in args.iter().enumerate() {
                match params.get(i) {
                    Some(p) if split_params.contains_key(p) => {
                        for f in fields_of(*arg, &scalars, &split_params) {
                            match f {
                                Some(v) => out.push(v),
                                None => {
                                    let nv = func.new_value();
                                    null_consts.push((bi, nv));
                                    out.push(nv);
                                }
                            }
                        }
                    }
                    _ => out.push(*arg),
                }
            }
            rewritten.push(out);
        }
        match &mut func.blocks[bi].terminator {
            Terminator::Branch { args, .. } => *args = rewritten.remove(0),
            Terminator::CondBranch {
                true_args,
                false_args,
                ..
            } => {
                *true_args = rewritten.remove(0);
                *false_args = rewritten.remove(0);
            }
            _ => {}
        }
    }
    for block in func.blocks.iter_mut() {
        let mut new_params = Vec::new();
        for (p, ty) in std::mem::take(&mut block.params) {
            match split_params.get(&p) {
                Some(fresh) => {
                    for nv in fresh {
                        new_params.push((*nv, MirType::Value));
                    }
                }
                None => new_params.push((p, ty)),
            }
        }
        block.params = new_params;
    }
    for (bi, nv) in null_consts {
        func.blocks[bi]
            .instructions
            .push((nv, Instruction::ConstNull));
    }
    true
}
