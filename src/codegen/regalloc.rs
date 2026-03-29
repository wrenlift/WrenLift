/// Linear-scan register allocator.
///
/// Transforms a `MachFunc` with virtual registers into one with physical registers.
/// Algorithm:
/// 1. Compute live intervals [first_def, last_use] for each VReg
/// 2. Sort intervals by start point
/// 3. Walk intervals, assigning physical registers from a free pool
/// 4. When no register is free, spill the interval ending latest
/// 5. Rewrite all VRegs to assigned PhysRegs, inserting spill/reload as needed
use super::{
    MachFunc, MachInst, Mem, PhysReg, RegClass, VReg, ABI_RET_SENTINEL, CALL_SCRATCH_SENTINEL,
    COPY_SCRATCH_SENTINEL, CTX_PTR_SENTINEL, FRAME_PTR_SENTINEL, SPILL_SCRATCH_SENTINEL,
};
use std::collections::{BTreeSet, HashMap};

// ---------------------------------------------------------------------------
// Live interval
// ---------------------------------------------------------------------------

/// Half-open live range [start, end) for a virtual register.
#[derive(Debug, Clone)]
pub struct LiveInterval {
    pub vreg: VReg,
    /// Instruction index of first definition.
    pub start: u32,
    /// Instruction index just past last use (exclusive).
    pub end: u32,
}

// ---------------------------------------------------------------------------
// Allocation result
// ---------------------------------------------------------------------------

/// Where a virtual register was placed.
#[derive(Debug, Clone, Copy)]
pub enum Location {
    /// In a physical register.
    Reg(PhysReg),
    /// Spilled to a stack slot (offset from frame pointer).
    /// AArch64: positive (above FP/LR pair). x86_64: negative (below RBP).
    Spill(i32),
}

/// Full allocation output.
pub struct RegAllocResult {
    /// VReg → assigned location.
    pub assignments: HashMap<VReg, Location>,
    /// Number of spill slots used (each 8 bytes).
    pub num_spill_slots: u32,
    /// Computed live intervals (useful for debugging).
    pub intervals: Vec<LiveInterval>,
}

// ---------------------------------------------------------------------------
// Compute live intervals
// ---------------------------------------------------------------------------

/// Build live intervals by scanning instructions forward.
pub fn compute_live_intervals(func: &MachFunc) -> Vec<LiveInterval> {
    // Track first def, first use, and last use for each vreg.
    let mut first_def: HashMap<VReg, u32> = HashMap::new();
    let mut first_use: HashMap<VReg, u32> = HashMap::new();
    let mut last_use: HashMap<VReg, u32> = HashMap::new();

    for (i, inst) in func.insts.iter().enumerate() {
        let idx = i as u32;

        // Record definitions.
        for d in inst.defs() {
            if is_fixed_sentinel(d) {
                continue;
            }
            first_def.entry(d).or_insert(idx);
            // A def also extends the interval (needed for single-point intervals).
            last_use.entry(d).or_insert(idx);
        }

        // Record uses.
        for u in inst.uses() {
            if is_fixed_sentinel(u) {
                continue;
            }
            first_use.entry(u).or_insert(idx);
            last_use.insert(u, idx);
        }
    }

    let func_end = func.insts.len() as u32;

    let mut intervals: Vec<LiveInterval> = first_def
        .into_iter()
        .map(|(vreg, def_pos)| {
            let use_end = last_use.get(&vreg).copied().unwrap_or(def_pos) + 1;
            let use_start = first_use.get(&vreg).copied().unwrap_or(def_pos);

            // If a vreg is used before its first definition (in linear order),
            // it's a loop-carried value (e.g., a block parameter in a loop
            // header whose definition is at a back-edge ParallelCopy placed
            // later in the instruction stream). Extend the interval to cover
            // the full function so the regalloc keeps it live across the loop.
            if use_start < def_pos {
                LiveInterval {
                    vreg,
                    start: use_start,
                    end: func_end.max(use_end),
                }
            } else {
                LiveInterval {
                    vreg,
                    start: def_pos,
                    end: use_end,
                }
            }
        })
        .collect();

    // Sort by start point (primary), then by vreg index (stable).
    intervals.sort_by_key(|iv| (iv.start, iv.vreg.index));
    intervals
}

#[inline]
fn is_fixed_sentinel(vreg: VReg) -> bool {
    match vreg.class {
        RegClass::Gp => vreg.index >= CTX_PTR_SENTINEL,
        RegClass::Fp | RegClass::Vec => vreg.index == SPILL_SCRATCH_SENTINEL,
    }
}

// ---------------------------------------------------------------------------
// Register pool
// ---------------------------------------------------------------------------

/// A pool of physical registers available for allocation.
struct RegPool {
    /// Free registers (ordered for deterministic allocation).
    free: BTreeSet<u8>,
    /// All allocatable registers in this class.
    all: Vec<u8>,
}

impl RegPool {
    fn new(regs: &[PhysReg]) -> Self {
        let hw: Vec<u8> = regs.iter().map(|r| r.hw_enc).collect();
        Self {
            free: hw.iter().copied().collect(),
            all: hw,
        }
    }

    fn alloc(&mut self) -> Option<u8> {
        let r = self.free.iter().next().copied()?;
        self.free.remove(&r);
        Some(r)
    }

    fn free(&mut self, r: u8) {
        debug_assert!(self.all.contains(&r), "freeing non-pool register {}", r);
        self.free.insert(r);
    }
}

// ---------------------------------------------------------------------------
// Target register sets
// ---------------------------------------------------------------------------

/// Describes allocatable registers for a target, excluding reserved registers.
pub struct TargetRegs {
    /// GP registers available for allocation (excludes SP, FP, scratch).
    pub gp_allocatable: Vec<PhysReg>,
    /// FP/Vec registers available for allocation (excludes scratch).
    pub fp_allocatable: Vec<PhysReg>,
    /// Caller-saved registers (clobbered by any function call).
    /// Values in these registers that are live across a call must be spilled.
    pub caller_saved: Vec<PhysReg>,
    /// Bytes to reserve in the frame for target-specific saved registers.
    /// AArch64: 16 (stp x29/x30 saved at bottom of frame).
    /// x86_64: 0 (push rbp is handled outside frame_size).
    pub frame_reserved: u32,
    /// Offset of the first spill slot from the frame pointer.
    /// AArch64: 16 (spills above saved FP/LR pair at [x29]).
    /// x86_64: -8 (spills below saved RBP at [rbp]).
    pub spill_first_offset: i32,
    /// Offset increment per additional spill slot.
    /// AArch64: 8 (positive, growing upward from FP).
    /// x86_64: -8 (negative, growing downward from FP).
    pub spill_stride: i32,
}

/// Build the default aarch64 allocatable set.
pub fn aarch64_target_regs() -> TargetRegs {
    // x0-x15 plus x19-x28. x16/x17 are scratch, x18 is platform-reserved,
    // x29 is FP, x30 is LR, x31 is SP.
    //
    // GP callee-saved regs are now handled by insert_callee_saves(), which
    // makes the larger set worthwhile for spill-heavy non-leaf module bodies.
    // x16/x17 are scratch, x18 is platform-reserved, x29 is FP, x30 is LR, x31 is SP.
    // x20 reserved for JitContext pointer (set at interpreter→JIT entry,
    // preserved across all calls since it's callee-saved in AAPCS64).
    // x19 is reserved by LLVM on this platform.
    let gp: Vec<PhysReg> = (0u8..16)
        .chain(19u8..20)
        .chain(21u8..29)
        .map(PhysReg::gp)
        .collect();
    // d0-d15 allocatable (d16-d31 exist but we keep it simple; d16/d17 scratch)
    let fp: Vec<PhysReg> = (0..16).map(PhysReg::fp).collect();
    // On AArch64: x0-x17 are caller-saved; we allocate x0-x15 so all are caller-saved.
    // d0-d7 are caller-saved; d8-d15 are callee-saved.
    let cs_gp: Vec<PhysReg> = (0u8..16).map(PhysReg::gp).collect();
    let cs_fp: Vec<PhysReg> = (0u8..8).map(PhysReg::fp).collect();
    TargetRegs {
        gp_allocatable: gp,
        fp_allocatable: fp,
        caller_saved: cs_gp.into_iter().chain(cs_fp).collect(),
        // AArch64 prologue saves x29/x30 at [sp], then sets x29 = sp.
        // Spill slots start at [x29+16] (after the 16-byte FP/LR pair).
        frame_reserved: 16,
        spill_first_offset: 16,
        spill_stride: 8,
    }
}

/// Build the default x86_64 allocatable set.
pub fn x86_64_target_regs() -> TargetRegs {
    // All GP except RSP(4), RBP(5), R10(10=spill scratch), R11(11=call scratch/SCRATCH_GP),
    // R12(12=copy scratch — second spill reload register, avoids R11/SCRATCH_GP conflict)
    let gp: Vec<PhysReg> = (0..16)
        .filter(|&r| r != 4 && r != 5 && r != 10 && r != 11 && r != 12)
        .map(PhysReg::gp)
        .collect();
    // XMM0-XMM14 (XMM15 is scratch)
    let fp: Vec<PhysReg> = (0..15).map(PhysReg::fp).collect();
    // x86_64 System V caller-saved GP: rax(0), rcx(1), rdx(2), rsi(6), rdi(7), r8(8), r9(9), r10(10)
    // All XMM registers are caller-saved in System V ABI.
    let cs_gp: Vec<PhysReg> = [0u8, 1, 2, 6, 7, 8, 9, 10]
        .iter()
        .map(|&e| PhysReg::gp(e))
        .collect();
    let cs_fp: Vec<PhysReg> = (0u8..15).map(PhysReg::fp).collect();
    TargetRegs {
        gp_allocatable: gp,
        fp_allocatable: fp,
        caller_saved: cs_gp.into_iter().chain(cs_fp).collect(),
        frame_reserved: 0, // x86_64: push rbp is outside frame_size
        spill_first_offset: -8,
        spill_stride: -8,
    }
}

// ---------------------------------------------------------------------------
// Linear scan allocator
// ---------------------------------------------------------------------------

/// Run linear-scan register allocation.
///
/// Takes a `MachFunc` with virtual registers and a target register description.
/// Returns an assignment mapping each VReg to a physical register or spill slot.
pub fn linear_scan(func: &MachFunc, target: &TargetRegs) -> RegAllocResult {
    let intervals = compute_live_intervals(func);

    let mut gp_pool = RegPool::new(&target.gp_allocatable);
    let mut fp_pool = RegPool::new(&target.fp_allocatable);

    let mut assignments: HashMap<VReg, Location> = HashMap::new();

    // Active intervals, sorted by end point. (end, vreg, hw_enc).
    let mut active: Vec<(u32, VReg, u8)> = Vec::new();

    let mut next_spill_slot: u32 = 0;

    for iv in &intervals {
        if func.force_boxed_gp_spills() && func.is_boxed_gp(iv.vreg) {
            let slot_offset =
                target.spill_first_offset + (next_spill_slot as i32) * target.spill_stride;
            next_spill_slot += 1;
            assignments.insert(iv.vreg, Location::Spill(slot_offset));
            continue;
        }

        // Expire old intervals that ended before this one starts.
        let start = iv.start;
        let mut expired = Vec::new();
        for (idx, &(end, ref vreg, hw)) in active.iter().enumerate() {
            if end <= start {
                expired.push(idx);
                // Return register to pool.
                match vreg.class {
                    RegClass::Gp => gp_pool.free(hw),
                    RegClass::Fp | RegClass::Vec => fp_pool.free(hw),
                }
            }
        }
        // Remove expired (reverse order to preserve indices).
        for idx in expired.into_iter().rev() {
            active.remove(idx);
        }

        // Try to allocate a register.
        let pool = match iv.vreg.class {
            RegClass::Gp => &mut gp_pool,
            RegClass::Fp | RegClass::Vec => &mut fp_pool,
        };

        if let Some(hw) = pool.alloc() {
            let preg = PhysReg {
                class: iv.vreg.class,
                hw_enc: hw,
            };
            assignments.insert(iv.vreg, Location::Reg(preg));
            active.push((iv.end, iv.vreg, hw));
            // Keep active sorted by end point for efficient spill selection.
            active.sort_by_key(|&(end, _, _)| end);
        } else {
            // Spill: either spill this interval or the one ending latest.
            // Heuristic: spill the one with the longest remaining range.
            let last_active_idx = active.len().checked_sub(1);
            let spill_active = last_active_idx.filter(|&idx| active[idx].0 > iv.end);

            if let Some(idx) = spill_active {
                // Spill the active interval that ends latest; give its register to us.
                let (_, spilled_vreg, hw) = active.remove(idx);

                // Reassign: spilled vreg goes to stack, we get its register.
                let slot_offset =
                    target.spill_first_offset + (next_spill_slot as i32) * target.spill_stride;
                next_spill_slot += 1;
                assignments.insert(spilled_vreg, Location::Spill(slot_offset));

                let preg = PhysReg {
                    class: iv.vreg.class,
                    hw_enc: hw,
                };
                assignments.insert(iv.vreg, Location::Reg(preg));
                active.push((iv.end, iv.vreg, hw));
                active.sort_by_key(|&(end, _, _)| end);
            } else {
                // Spill this interval directly.
                let slot_offset =
                    target.spill_first_offset + (next_spill_slot as i32) * target.spill_stride;
                next_spill_slot += 1;
                assignments.insert(iv.vreg, Location::Spill(slot_offset));
            }
        }
    }

    RegAllocResult {
        assignments,
        num_spill_slots: next_spill_slot,
        intervals,
    }
}

// ---------------------------------------------------------------------------
// Rewrite pass: VRegs → PhysRegs
// ---------------------------------------------------------------------------

/// Rewrite a `MachFunc` in-place, replacing virtual register indices with
/// physical register encodings. Inserts spill stores and reload loads as needed.
pub fn apply_allocation(func: &mut MachFunc, result: &RegAllocResult, frame_reserved: u32) {
    let mut new_insts: Vec<MachInst> = Vec::with_capacity(func.insts.len() * 2);

    // Update frame size to include spill slots and any target-reserved bytes.
    // `frame_reserved` is 16 on AArch64 (space for stp x29/x30 at frame bottom)
    // so that spill offsets from x29 land within the frame.
    let spill_bytes = result.num_spill_slots * 8;
    let aligned = (func.frame_size + frame_reserved + spill_bytes + 15) & !15;
    func.frame_size = aligned;

    for inst in &func.insts {
        if matches!(
            inst,
            MachInst::CallRuntime { .. }
                | MachInst::CallLocal { .. }
                | MachInst::CallIndirectAbi { .. }
        ) {
            new_insts.push(inst.clone());
            continue;
        }

        // For spilled uses: insert reload before the instruction.
        let uses = inst.uses();
        let defs = inst.defs();
        let mut spill_overrides: HashMap<VReg, VReg> = HashMap::new();
        let mut gp_scratch_uses = 0usize;
        for u in &uses {
            if let Some(&Location::Spill(offset)) = result.assignments.get(u) {
                let scratch = *spill_overrides.entry(*u).or_insert_with(|| {
                    let slot = if u.class == RegClass::Gp {
                        let slot = gp_scratch_uses.min(1);
                        gp_scratch_uses += 1;
                        slot
                    } else {
                        0
                    };
                    spill_scratch(*u, slot)
                });
                if u.class == RegClass::Fp || u.class == RegClass::Vec {
                    new_insts.push(MachInst::FLdr {
                        dst: scratch,
                        mem: Mem::new(frame_ptr_vreg(), offset),
                    });
                } else {
                    new_insts.push(MachInst::Ldr {
                        dst: scratch,
                        mem: Mem::new(frame_ptr_vreg(), offset),
                    });
                }
            }
        }

        for d in &defs {
            if matches!(result.assignments.get(d), Some(Location::Spill(_))) {
                spill_overrides
                    .entry(*d)
                    .or_insert_with(|| spill_scratch(*d, 0));
            }
        }

        // Rewrite the instruction's registers.
        let rewritten = rewrite_inst(inst, &result.assignments, &spill_overrides);
        new_insts.push(rewritten);

        // For spilled defs: insert store after the instruction.
        for d in &defs {
            if let Some(&Location::Spill(offset)) = result.assignments.get(d) {
                let scratch = *spill_overrides.get(d).unwrap_or(&spill_scratch(*d, 0));
                if d.class == RegClass::Fp || d.class == RegClass::Vec {
                    new_insts.push(MachInst::FStr {
                        src: scratch,
                        mem: Mem::new(frame_ptr_vreg(), offset),
                    });
                } else {
                    new_insts.push(MachInst::Str {
                        src: scratch,
                        mem: Mem::new(frame_ptr_vreg(), offset),
                    });
                }
            }
        }
    }

    // Update prologue/epilogue frame sizes.
    for inst in &mut new_insts {
        match inst {
            MachInst::Prologue { frame_size } | MachInst::Epilogue { frame_size } => {
                *frame_size = func.frame_size;
            }
            _ => {}
        }
    }

    func.insts = new_insts;
}

/// The frame pointer as a VReg (already physical — index = hw encoding).
fn frame_ptr_vreg() -> VReg {
    // RBP on x86_64 (encoding 5), X29 on aarch64 (encoding 29).
    // We use a sentinel GP vreg. The actual encoding is set during rewrite.
    // Convention: frame_ptr uses the reserved GP index u32::MAX.
    VReg::gp(FRAME_PTR_SENTINEL)
}

/// Scratch register for spill/reload. Uses the reserved scratch slot per class.
fn spill_scratch(vreg: VReg, slot: usize) -> VReg {
    match vreg.class {
        RegClass::Gp => {
            if slot == 0 {
                VReg::gp(SPILL_SCRATCH_SENTINEL)
            } else {
                VReg::gp(COPY_SCRATCH_SENTINEL)
            }
        }
        RegClass::Fp | RegClass::Vec => VReg::fp(SPILL_SCRATCH_SENTINEL),
    }
}

/// Map a single VReg through the allocation, returning a VReg whose `index`
/// is the physical register encoding.
fn map_vreg(
    vreg: VReg,
    assignments: &HashMap<VReg, Location>,
    spill_overrides: &HashMap<VReg, VReg>,
) -> VReg {
    if let Some(&override_vreg) = spill_overrides.get(&vreg) {
        return override_vreg;
    }
    // Sentinel: frame pointer
    if vreg.index == FRAME_PTR_SENTINEL && vreg.class == RegClass::Gp {
        return vreg; // stays as-is, patched by target-specific fixup
    }
    // Sentinel: spill scratch
    if vreg.index == SPILL_SCRATCH_SENTINEL {
        return vreg; // stays as-is, patched by target-specific fixup
    }
    // Sentinel: ABI return register
    if vreg.index == ABI_RET_SENTINEL && vreg.class == RegClass::Gp {
        return vreg; // stays as-is, patched by target-specific fixup
    }
    if vreg.class == RegClass::Gp
        && (vreg.index == CALL_SCRATCH_SENTINEL || vreg.index == COPY_SCRATCH_SENTINEL)
    {
        return vreg; // stays as-is, patched by target-specific fixup
    }
    if vreg.class == RegClass::Gp && vreg.index >= COPY_SCRATCH_SENTINEL {
        return vreg; // fixed ABI arg/call sentinel
    }
    match assignments.get(&vreg) {
        Some(&Location::Reg(phys)) => VReg {
            index: phys.hw_enc as u32,
            class: vreg.class,
        },
        Some(&Location::Spill(_)) => {
            // Spilled registers use scratch for the actual instruction.
            spill_scratch(vreg, 0)
        }
        None => vreg, // Already physical or not tracked (e.g., label-only pseudo-insts)
    }
}

/// Rewrite all VRegs in a single instruction.
fn rewrite_inst(
    inst: &MachInst,
    assignments: &HashMap<VReg, Location>,
    spill_overrides: &HashMap<VReg, VReg>,
) -> MachInst {
    use MachInst::*;
    let m = |v: VReg| map_vreg(v, assignments, spill_overrides);
    let mm = |mem: &Mem| Mem::new(m(mem.base), mem.offset);

    match inst {
        // No registers
        Prologue { frame_size } => Prologue {
            frame_size: *frame_size,
        },
        Epilogue { frame_size } => Epilogue {
            frame_size: *frame_size,
        },
        Jmp { target } => Jmp { target: *target },
        DefLabel(l) => DefLabel(*l),
        Nop => Nop,
        Trap => Trap,
        Ret => Ret,
        StackAlloc { bytes } => StackAlloc { bytes: *bytes },
        StackFree { bytes } => StackFree { bytes: *bytes },

        // Single dst
        LoadImm { dst, bits } => LoadImm {
            dst: m(*dst),
            bits: *bits,
        },
        LoadFpImm { dst, value } => LoadFpImm {
            dst: m(*dst),
            value: *value,
        },
        Pop { dst } => Pop { dst: m(*dst) },
        FuncArg { dst, index } => FuncArg {
            dst: m(*dst),
            index: *index,
        },

        // Single src
        Push { src } => Push { src: m(*src) },
        CallInd { target } => CallInd { target: m(*target) },
        CallLabel { target } => CallLabel { target: *target },

        // dst, src
        Mov { dst, src } => Mov {
            dst: m(*dst),
            src: m(*src),
        },
        FMov { dst, src } => FMov {
            dst: m(*dst),
            src: m(*src),
        },
        BitcastGpToFp { dst, src } => BitcastGpToFp {
            dst: m(*dst),
            src: m(*src),
        },
        BitcastFpToGp { dst, src } => BitcastFpToGp {
            dst: m(*dst),
            src: m(*src),
        },
        INeg { dst, src } => INeg {
            dst: m(*dst),
            src: m(*src),
        },
        Not { dst, src } => Not {
            dst: m(*dst),
            src: m(*src),
        },
        FNeg { dst, src } => FNeg {
            dst: m(*dst),
            src: m(*src),
        },
        FAbs { dst, src } => FAbs {
            dst: m(*dst),
            src: m(*src),
        },
        FSqrt { dst, src } => FSqrt {
            dst: m(*dst),
            src: m(*src),
        },
        FFloor { dst, src } => FFloor {
            dst: m(*dst),
            src: m(*src),
        },
        FCeil { dst, src } => FCeil {
            dst: m(*dst),
            src: m(*src),
        },
        FRound { dst, src } => FRound {
            dst: m(*dst),
            src: m(*src),
        },
        FTrunc { dst, src } => FTrunc {
            dst: m(*dst),
            src: m(*src),
        },
        FCvtToI64 { dst, src } => FCvtToI64 {
            dst: m(*dst),
            src: m(*src),
        },
        I64CvtToF { dst, src } => I64CvtToF {
            dst: m(*dst),
            src: m(*src),
        },

        // dst, src, imm
        IAddImm { dst, src, imm } => IAddImm {
            dst: m(*dst),
            src: m(*src),
            imm: *imm,
        },
        AndImm { dst, src, imm } => AndImm {
            dst: m(*dst),
            src: m(*src),
            imm: *imm,
        },
        OrImm { dst, src, imm } => OrImm {
            dst: m(*dst),
            src: m(*src),
            imm: *imm,
        },

        // dst, lhs, rhs (3-address)
        IAdd { dst, lhs, rhs } => IAdd {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        ISub { dst, lhs, rhs } => ISub {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        IMul { dst, lhs, rhs } => IMul {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        IDiv { dst, lhs, rhs } => IDiv {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        And { dst, lhs, rhs } => And {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        Or { dst, lhs, rhs } => Or {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        Xor { dst, lhs, rhs } => Xor {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        Shl { dst, lhs, rhs } => Shl {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        Sar { dst, lhs, rhs } => Sar {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        Shr { dst, lhs, rhs } => Shr {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        FAdd { dst, lhs, rhs } => FAdd {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        FSub { dst, lhs, rhs } => FSub {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        FMul { dst, lhs, rhs } => FMul {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        FDiv { dst, lhs, rhs } => FDiv {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        FMin { dst, lhs, rhs } => FMin {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        FMax { dst, lhs, rhs } => FMax {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
        },

        // 4-operand FMA
        IMulSub { dst, lhs, rhs, acc } => IMulSub {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
            acc: m(*acc),
        },
        FMAdd { dst, a, b, c } => FMAdd {
            dst: m(*dst),
            a: m(*a),
            b: m(*b),
            c: m(*c),
        },
        FMSub { dst, a, b, c } => FMSub {
            dst: m(*dst),
            a: m(*a),
            b: m(*b),
            c: m(*c),
        },
        FNMAdd { dst, a, b, c } => FNMAdd {
            dst: m(*dst),
            a: m(*a),
            b: m(*b),
            c: m(*c),
        },
        FNMSub { dst, a, b, c } => FNMSub {
            dst: m(*dst),
            a: m(*a),
            b: m(*b),
            c: m(*c),
        },

        // Comparisons
        ICmp { lhs, rhs } => ICmp {
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        ICmpImm { lhs, imm } => ICmpImm {
            lhs: m(*lhs),
            imm: *imm,
        },
        FCmp { lhs, rhs } => FCmp {
            lhs: m(*lhs),
            rhs: m(*rhs),
        },
        CSet { dst, cond } => CSet {
            dst: m(*dst),
            cond: *cond,
        },

        // Branches with register operands
        JmpIf { cond, target } => JmpIf {
            cond: *cond,
            target: *target,
        },
        JmpZero { src, target } => JmpZero {
            src: m(*src),
            target: *target,
        },
        JmpNonZero { src, target } => JmpNonZero {
            src: m(*src),
            target: *target,
        },
        TestBitJmpZero { src, bit, target } => TestBitJmpZero {
            src: m(*src),
            bit: *bit,
            target: *target,
        },
        TestBitJmpNonZero { src, bit, target } => TestBitJmpNonZero {
            src: m(*src),
            bit: *bit,
            target: *target,
        },

        // Memory
        Ldr { dst, mem } => Ldr {
            dst: m(*dst),
            mem: mm(mem),
        },
        Str { src, mem } => Str {
            src: m(*src),
            mem: mm(mem),
        },
        FLdr { dst, mem } => FLdr {
            dst: m(*dst),
            mem: mm(mem),
        },
        FStr { src, mem } => FStr {
            src: m(*src),
            mem: mm(mem),
        },

        // Vector
        VLoad { dst, mem, width } => VLoad {
            dst: m(*dst),
            mem: mm(mem),
            width: *width,
        },
        VStore { src, mem, width } => VStore {
            src: m(*src),
            mem: mm(mem),
            width: *width,
        },
        VFAdd {
            dst,
            lhs,
            rhs,
            width,
        } => VFAdd {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
            width: *width,
        },
        VFSub {
            dst,
            lhs,
            rhs,
            width,
        } => VFSub {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
            width: *width,
        },
        VFMul {
            dst,
            lhs,
            rhs,
            width,
        } => VFMul {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
            width: *width,
        },
        VFDiv {
            dst,
            lhs,
            rhs,
            width,
        } => VFDiv {
            dst: m(*dst),
            lhs: m(*lhs),
            rhs: m(*rhs),
            width: *width,
        },
        VFMAdd {
            dst,
            a,
            b,
            c,
            width,
        } => VFMAdd {
            dst: m(*dst),
            a: m(*a),
            b: m(*b),
            c: m(*c),
            width: *width,
        },
        VBroadcast { dst, src, width } => VBroadcast {
            dst: m(*dst),
            src: m(*src),
            width: *width,
        },
        VExtractLane { dst, src, lane } => VExtractLane {
            dst: m(*dst),
            src: m(*src),
            lane: *lane,
        },
        VInsertLane {
            dst,
            src,
            lane,
            val,
        } => VInsertLane {
            dst: m(*dst),
            src: m(*src),
            lane: *lane,
            val: m(*val),
        },
        VFNeg { dst, src, width } => VFNeg {
            dst: m(*dst),
            src: m(*src),
            width: *width,
        },
        VReduceAdd { dst, src, width } => VReduceAdd {
            dst: m(*dst),
            src: m(*src),
            width: *width,
        },

        // Runtime calls
        CallLocal { target, args, ret } => CallLocal {
            target: *target,
            args: args.iter().map(|a| m(*a)).collect(),
            ret: ret.map(&m),
        },
        CallIndirectAbi { target, args, ret } => CallIndirectAbi {
            target: m(*target),
            args: args.iter().map(|a| m(*a)).collect(),
            ret: ret.map(&m),
        },
        CallRuntime { name, args, ret } => CallRuntime {
            name,
            args: args.iter().map(|a| m(*a)).collect(),
            ret: ret.map(&m),
        },

        // Parallel copies should be resolved before regalloc, but handle gracefully
        ParallelCopy { copies } => ParallelCopy {
            copies: copies.iter().map(|(d, s)| (m(*d), m(*s))).collect(),
        },
    }
}

// ---------------------------------------------------------------------------
// Call-boundary spilling
// ---------------------------------------------------------------------------

/// Post-allocation pass: any vreg whose live interval spans a call instruction
/// AND is assigned to a caller-saved physical register must be spilled.
/// Without this, the call would clobber the register and corrupt the live value.
///
/// A vreg at interval [s, e) with `e = last_use + 1` is spilled when:
///   - It is defined BEFORE the call: `s < call_pos`
///   - It is used AFTER the call: `e > call_pos + 1` (equivalently, last_use > call_pos)
///
/// Values whose last use IS the call (e.g., call arguments) must NOT be spilled
/// here — they are consumed by the call itself and `link_runtime_calls` handles
/// placing them in the correct ABI registers. Spilling them would cause multiple
/// args to share the same scratch register (x16), corrupting all but the last.
pub fn respill_caller_saved_across_calls(
    func: &MachFunc,
    result: &mut RegAllocResult,
    caller_saved: &[PhysReg],
    spill_first_offset: i32,
    spill_stride: i32,
) {
    // Collect call instruction positions.
    let call_positions: Vec<u32> = func
        .insts
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| {
            if matches!(
                inst,
                MachInst::CallInd { .. }
                    | MachInst::CallLabel { .. }
                    | MachInst::CallLocal { .. }
                    | MachInst::CallIndirectAbi { .. }
                    | MachInst::CallRuntime { .. }
            ) {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect();

    if call_positions.is_empty() {
        return;
    }

    // Build a quick-lookup set: (class, hw_enc) for caller-saved registers.
    let cs_set: std::collections::HashSet<(RegClass, u8)> =
        caller_saved.iter().map(|r| (r.class, r.hw_enc)).collect();

    // For each interval assigned to a caller-saved register, check if any
    // call falls strictly inside the interval.
    let intervals = result.intervals.clone();
    for iv in &intervals {
        if let Some(&Location::Reg(phys)) = result.assignments.get(&iv.vreg) {
            if !cs_set.contains(&(phys.class, phys.hw_enc)) {
                continue;
            }
            // Value must be defined BEFORE the call AND used AFTER it.
            // - cp > iv.start: call is after the vreg's definition
            // - iv.end > cp + 1: last_use (= iv.end - 1) > cp, i.e. used after call
            // Values whose last use IS the call (args) have iv.end == cp+1 → not spilled.
            // Values defined AT the call (return value) have iv.start == cp → not spilled.
            // Values that are the RETURN VALUE of the call are re-defined by the call
            // itself, so even though their interval spans it, they don't need preservation.
            let spans = call_positions.iter().any(|&cp| {
                if cp <= iv.start || iv.end <= cp + 1 {
                    return false;
                }
                // Don't spill if the call at `cp` is what defines this vreg (return value).
                !func.insts[cp as usize].defs().contains(&iv.vreg)
            });
            if spans {
                let slot = spill_first_offset + (result.num_spill_slots as i32) * spill_stride;
                result.num_spill_slots += 1;
                result.assignments.insert(iv.vreg, Location::Spill(slot));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: full pipeline
// ---------------------------------------------------------------------------

/// Run the full register allocation pipeline on a MachFunc.
///
/// 1. Resolve parallel copies
/// 2. Compute live intervals + linear scan allocation
/// 3. Respill caller-saved registers that are live across calls
/// 4. Rewrite VRegs to PhysRegs with spill/reload insertion
pub fn allocate_registers_result(func: &MachFunc, target: &TargetRegs) -> RegAllocResult {
    let mut result = linear_scan(func, target);
    respill_caller_saved_across_calls(
        func,
        &mut result,
        &target.caller_saved,
        target.spill_first_offset,
        target.spill_stride,
    );
    result
}

pub fn allocate_registers(func: &mut MachFunc, target: &TargetRegs) -> RegAllocResult {
    let result = allocate_registers_result(func, target);
    apply_allocation(func, &result, target.frame_reserved);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{MachFunc, MachInst, VReg};

    /// Helper: build a MachFunc from a closure.
    fn build(f: impl FnOnce(&mut MachFunc)) -> MachFunc {
        let mut mf = MachFunc::new("test".into());
        f(&mut mf);
        mf
    }

    // -- Live interval tests --

    #[test]
    fn test_single_def_use() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 42,
            });
            mf.emit(MachInst::Mov {
                dst: VReg::gp(1),
                src: VReg::gp(0),
            });
            mf.emit(MachInst::Ret);
        });
        let ivs = compute_live_intervals(&mf);
        let r0 = ivs.iter().find(|iv| iv.vreg == VReg::gp(0)).unwrap();
        assert_eq!(r0.start, 0);
        assert_eq!(r0.end, 2); // used at instruction 1, end = 1+1
    }

    #[test]
    fn test_interval_extends_to_last_use() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 1,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 2,
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(2),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(3),
                lhs: VReg::gp(2),
                rhs: VReg::gp(0),
            }); // r0 used again
            mf.emit(MachInst::Ret);
        });
        let ivs = compute_live_intervals(&mf);
        let r0 = ivs.iter().find(|iv| iv.vreg == VReg::gp(0)).unwrap();
        assert_eq!(r0.start, 0);
        assert_eq!(r0.end, 4); // last use at inst 3
    }

    #[test]
    fn test_fp_intervals() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 1.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 2.0,
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(2),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::Ret);
        });
        let ivs = compute_live_intervals(&mf);
        assert_eq!(ivs.iter().filter(|iv| iv.vreg.is_fp()).count(), 3);
    }

    #[test]
    fn test_intervals_sorted_by_start() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 1,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 2,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(2),
                bits: 3,
            });
            mf.emit(MachInst::Ret);
        });
        let ivs = compute_live_intervals(&mf);
        for w in ivs.windows(2) {
            assert!(w[0].start <= w[1].start);
        }
    }

    // -- Linear scan tests --

    #[test]
    fn test_simple_allocation() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 10,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 20,
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(2),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::Ret);
        });
        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);

        // All 3 vregs should get registers (plenty available).
        for vreg in [VReg::gp(0), VReg::gp(1), VReg::gp(2)] {
            match result.assignments.get(&vreg) {
                Some(Location::Reg(_)) => {}
                other => panic!("expected Reg for {:?}, got {:?}", vreg, other),
            }
        }
        assert_eq!(result.num_spill_slots, 0);
    }

    #[test]
    fn test_no_conflicts_distinct_regs() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 1,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 2,
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(2),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::Ret);
        });
        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);

        // All simultaneously live vregs must get distinct physical registers.
        let r0 = result.assignments[&VReg::gp(0)];
        let r1 = result.assignments[&VReg::gp(1)];
        if let (Location::Reg(p0), Location::Reg(p1)) = (r0, r1) {
            assert_ne!(
                p0.hw_enc, p1.hw_enc,
                "simultaneously live vregs got same phys reg"
            );
        }
    }

    #[test]
    fn test_reuse_after_dead() {
        // r0 is defined and used, then r1 is defined after r0 is dead.
        // They can share the same physical register.
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 1,
            });
            mf.emit(MachInst::Mov {
                dst: VReg::gp(1),
                src: VReg::gp(0),
            });
            // r0 is dead after inst 1. r2 defined at inst 2.
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(2),
                bits: 2,
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(3),
                lhs: VReg::gp(1),
                rhs: VReg::gp(2),
            });
            mf.emit(MachInst::Ret);
        });
        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);

        // r0 and r2 are not simultaneously live, so they may share a register.
        // We don't assert they *must* share (allocator doesn't guarantee it),
        // but we assert no spills (enough registers).
        assert_eq!(result.num_spill_slots, 0);
    }

    #[test]
    fn test_mixed_gp_fp() {
        let mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 42,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 1.234,
            });
            mf.emit(MachInst::BitcastGpToFp {
                dst: VReg::fp(1),
                src: VReg::gp(0),
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(2),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::Ret);
        });
        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);

        // GP and FP are allocated from separate pools.
        let gp0 = result.assignments[&VReg::gp(0)];
        let fp0 = result.assignments[&VReg::fp(0)];
        if let (Location::Reg(pg), Location::Reg(pf)) = (gp0, fp0) {
            assert_eq!(pg.class, RegClass::Gp);
            assert_eq!(pf.class, RegClass::Fp);
        }
        assert_eq!(result.num_spill_slots, 0);
    }

    #[test]
    fn test_spill_under_pressure() {
        // Create more simultaneously-live GP vregs than allocatable registers.
        // x86_64 has 13 allocatable GP regs (16 - RSP - RBP - R11).
        let mut mf = MachFunc::new("spill_test".into());
        let n = 15; // more than 13 available
        for i in 0..n {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(i),
                bits: i as u64,
            });
        }
        // Use all of them simultaneously.
        for i in 0..(n - 1) {
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(n + i),
                lhs: VReg::gp(i),
                rhs: VReg::gp(i + 1),
            });
        }
        mf.emit(MachInst::Ret);

        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);

        // Some vregs must be spilled.
        assert!(
            result.num_spill_slots > 0,
            "expected spills under register pressure"
        );

        // Every vreg must have an assignment (register or spill).
        for i in 0..n {
            assert!(
                result.assignments.contains_key(&VReg::gp(i)),
                "vreg gp({}) missing assignment",
                i
            );
        }
    }

    // -- Apply allocation tests --

    #[test]
    fn test_apply_rewrites_registers() {
        let mut mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 42,
            });
            mf.emit(MachInst::Mov {
                dst: VReg::gp(1),
                src: VReg::gp(0),
            });
            mf.emit(MachInst::Ret);
        });
        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);
        apply_allocation(&mut mf, &result, 0);

        // After rewrite, the first inst's dst should be a physical reg encoding.
        if let MachInst::LoadImm { dst, .. } = &mf.insts[0] {
            // Index should be a valid x86_64 GP encoding (0-15, not 4/5/10/11).
            assert!(dst.index < 16, "expected physical register, got {:?}", dst);
            assert_ne!(dst.index, 4, "should not use RSP");
            assert_ne!(dst.index, 5, "should not use RBP");
            assert_ne!(dst.index, 10, "should not use R10 scratch");
            assert_ne!(dst.index, 11, "should not use R11 scratch");
        }
    }

    #[test]
    fn test_apply_inserts_spill_reload() {
        // Force a spill and verify store/load instructions are inserted.
        let mut mf = MachFunc::new("spill_apply".into());
        let n = 15u32;
        for i in 0..n {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(i),
                bits: i as u64,
            });
        }
        // Use all at once.
        for i in 0..(n - 1) {
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(n + i),
                lhs: VReg::gp(i),
                rhs: VReg::gp(i + 1),
            });
        }
        mf.emit(MachInst::Ret);

        let target = x86_64_target_regs();
        let result = linear_scan(&mf, &target);
        apply_allocation(&mut mf, &result, 0);

        // Should have spill stores (Str) or reloads (Ldr) inserted.
        let has_spill_ops = mf
            .insts
            .iter()
            .any(|inst| matches!(inst, MachInst::Str { .. } | MachInst::Ldr { .. }));
        assert!(
            has_spill_ops,
            "expected spill/reload instructions after apply"
        );
    }

    // -- Full pipeline test --

    #[test]
    fn test_full_pipeline() {
        let mut mf = build(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 1.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 2.0,
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(2),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::BitcastFpToGp {
                dst: VReg::gp(0),
                src: VReg::fp(2),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let target = x86_64_target_regs();
        let result = allocate_registers(&mut mf, &target);

        assert_eq!(result.num_spill_slots, 0);
        // Verify all FP vregs mapped to distinct physical XMM registers.
        let fp0 = result.assignments[&VReg::fp(0)];
        let fp1 = result.assignments[&VReg::fp(1)];
        let fp2 = result.assignments[&VReg::fp(2)];
        if let (Location::Reg(p0), Location::Reg(p1), Location::Reg(p2)) = (fp0, fp1, fp2) {
            assert_ne!(p0.hw_enc, p1.hw_enc);
            // fp0 may be reused for fp2 since fp0 is dead after the FAdd.
            // But fp1 and fp2 must be distinct (fp1 alive during FAdd).
            assert_ne!(p1.hw_enc, p2.hw_enc);
        }
    }

    #[test]
    fn test_allocate_registers_respills_values_live_across_calls() {
        let mut mf = build(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 42,
            });
            mf.emit(MachInst::CallRuntime {
                name: "dummy_call",
                args: vec![],
                ret: Some(VReg::gp(1)),
            });
            mf.emit(MachInst::Mov {
                dst: VReg::gp(2),
                src: VReg::gp(0),
            });
            mf.emit(MachInst::Ret);
        });
        let target = aarch64_target_regs();
        let result = allocate_registers(&mut mf, &target);

        assert!(
            matches!(
                result.assignments.get(&VReg::gp(0)),
                Some(Location::Spill(_))
            ),
            "value live across a call should be respilled on aarch64 caller-saved regs"
        );

        let has_spill_ops = mf
            .insts
            .iter()
            .any(|inst| matches!(inst, MachInst::Str { .. } | MachInst::Ldr { .. }));
        assert!(
            has_spill_ops,
            "respilling should insert spill/reload instructions around the call"
        );
    }

    #[test]
    fn test_aarch64_target_regs() {
        let target = aarch64_target_regs();
        assert_eq!(target.gp_allocatable.len(), 25); // x0-x15 + x19 + x21-x28 (x20 reserved for JitContext)
        assert_eq!(target.fp_allocatable.len(), 16); // d0-d15
    }

    #[test]
    fn test_x86_64_target_regs() {
        let target = x86_64_target_regs();
        assert_eq!(target.gp_allocatable.len(), 11); // 16 - RSP(4) - RBP(5) - R10(10) - R11(11) - R12(12)
        assert_eq!(target.fp_allocatable.len(), 15); // XMM0-XMM14
                                                     // Verify excluded registers.
        assert!(!target.gp_allocatable.iter().any(|r| r.hw_enc == 4)); // no RSP
        assert!(!target.gp_allocatable.iter().any(|r| r.hw_enc == 5)); // no RBP
        assert!(!target.gp_allocatable.iter().any(|r| r.hw_enc == 11)); // no R11
        assert!(!target.gp_allocatable.iter().any(|r| r.hw_enc == 12)); // no R12 (copy scratch)
    }

    // -- Integration with MIR lowering --

    #[test]
    fn test_lower_and_allocate() {
        use crate::intern::Interner;
        use crate::mir::{Instruction, MirFunction, Terminator};

        let mut interner = Interner::new();
        let name = interner.intern("add_test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(10.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(20.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Add(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        // Lower MIR → MachIR
        let mut mach = super::super::lower_mir(&f);
        let target = x86_64_target_regs();
        let result = allocate_registers(&mut mach, &target);

        assert_eq!(result.num_spill_slots, 0);

        // Verify the function still has a Ret.
        assert!(mach.insts.iter().any(|i| matches!(i, MachInst::Ret)));
    }

    #[test]
    fn test_lower_allocate_f64_ops() {
        use crate::intern::Interner;
        use crate::mir::{Instruction, MirFunction, Terminator};

        let mut interner = Interner::new();
        let name = interner.intern("f64_test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstF64(3.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstF64(4.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::AddF64(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let mut mach = super::super::lower_mir(&f);
        let target = aarch64_target_regs();
        let result = allocate_registers(&mut mach, &target);

        assert_eq!(result.num_spill_slots, 0);

        // FP vregs should all be allocated to FP physical registers.
        for (vreg, loc) in &result.assignments {
            if vreg.is_fp() {
                if let Location::Reg(p) = loc {
                    assert_eq!(p.class, RegClass::Fp);
                }
            }
        }
    }
}
