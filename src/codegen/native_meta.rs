use std::collections::HashMap;

use super::{regalloc::Location, MachFunc, MachInst, VReg};

/// Where a GC-visible boxed value lives at a native safepoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootLocation {
    Reg(u8),
    Spill(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRootMetadata {
    pub slot: u16,
    pub location: RootLocation,
}

/// Kind of native safepoint currently recognized by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafepointKind {
    CallRuntime,
    CallIndirect,
    CallLocal,
}

/// Metadata for a single safepoint in a compiled function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafepointMetadata {
    /// Ordinal among GC safepoints in the virtual MachInst stream.
    pub ordinal: u32,
    /// Instruction index in the pre-rewrite MachInst stream.
    pub inst_index: u32,
    pub kind: SafepointKind,
    /// All boxed MIR values live across this safepoint after regalloc.
    pub live_roots: Vec<LiveRootMetadata>,
}

/// Per-function native-frame metadata retained alongside compiled code.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeFrameMetadata {
    /// All boxed GP vregs the lowering pipeline identified for this function.
    pub boxed_values: Vec<VReg>,
    /// All recognized safepoints and their live boxed roots.
    pub safepoints: Vec<SafepointMetadata>,
    /// Whether the current post-regalloc spill rewriting can execute this
    /// non-leaf function safely without aliasing multiple spilled operands
    /// through a single scratch register.
    pub spill_safe_nonleaf: bool,
}

pub fn build_native_frame_metadata(
    mach: &MachFunc,
    alloc: &crate::codegen::regalloc::RegAllocResult,
) -> NativeFrameMetadata {
    let intervals: HashMap<VReg, &crate::codegen::regalloc::LiveInterval> =
        alloc.intervals.iter().map(|iv| (iv.vreg, iv)).collect();

    let mut boxed_values: Vec<VReg> = mach.boxed_gp_vregs().collect();
    boxed_values.sort_by_key(|vreg| vreg.index);

    let mut safepoints = Vec::new();
    let mut ordinal = 0u32;

    for (inst_index, inst) in mach.insts.iter().enumerate() {
        let Some(kind) = safepoint_kind(inst) else {
            continue;
        };
        let inst_defs = inst.defs();

        let live_roots = boxed_values
            .iter()
            .enumerate()
            .filter_map(|(slot, vreg)| {
                let interval = intervals.get(vreg)?;
                let idx = inst_index as u32;
                if interval.start <= idx
                    && idx < interval.end
                    && !inst_defs.contains(vreg)
                {
                    let location = match alloc.assignments.get(vreg)? {
                        Location::Reg(phys) => RootLocation::Reg(phys.hw_enc),
                        Location::Spill(offset) => RootLocation::Spill(*offset),
                    };
                    Some(LiveRootMetadata {
                        slot: slot as u16,
                        location,
                    })
                } else {
                    None
                }
            })
            .collect();

        safepoints.push(SafepointMetadata {
            ordinal,
            inst_index: inst_index as u32,
            kind,
            live_roots,
        });
        ordinal += 1;
    }

    NativeFrameMetadata {
        boxed_values,
        safepoints,
        spill_safe_nonleaf: is_spill_safe_nonleaf(mach, alloc),
    }
}

fn is_spill_safe_nonleaf(
    mach: &MachFunc,
    alloc: &crate::codegen::regalloc::RegAllocResult,
) -> bool {
    for inst in &mach.insts {
        if matches!(inst, MachInst::CallRuntime { .. }) {
            continue;
        }

        let spilled_gp_uses = inst
            .uses()
            .into_iter()
            .filter(|vreg| {
                vreg.class == crate::codegen::RegClass::Gp
                    && matches!(alloc.assignments.get(vreg), Some(Location::Spill(_)))
            })
            .count();
        let spilled_gp_defs = inst
            .defs()
            .into_iter()
            .filter(|vreg| {
                vreg.class == crate::codegen::RegClass::Gp
                    && matches!(alloc.assignments.get(vreg), Some(Location::Spill(_)))
            })
            .count();
        if spilled_gp_uses + spilled_gp_defs > 2 {
            return false;
        }

        let spilled_fp_uses = inst
            .uses()
            .into_iter()
            .filter(|vreg| {
                vreg.class != crate::codegen::RegClass::Gp
                    && matches!(alloc.assignments.get(vreg), Some(Location::Spill(_)))
            })
            .count();
        let spilled_fp_defs = inst
            .defs()
            .into_iter()
            .filter(|vreg| {
                vreg.class != crate::codegen::RegClass::Gp
                    && matches!(alloc.assignments.get(vreg), Some(Location::Spill(_)))
            })
            .count();
        if spilled_fp_uses + spilled_fp_defs > 1 {
            return false;
        }
    }

    true
}

fn safepoint_kind(inst: &MachInst) -> Option<SafepointKind> {
    match inst {
        MachInst::CallRuntime { name, .. } if *name != "wren_shadow_store" && *name != "wren_shadow_load" => {
            Some(SafepointKind::CallRuntime)
        }
        MachInst::CallInd { .. } => Some(SafepointKind::CallIndirect),
        MachInst::CallLocal { .. } => Some(SafepointKind::CallLocal),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::regalloc;

    #[test]
    fn metadata_tracks_only_boxed_values_at_calls() {
        let mut mf = MachFunc::new("meta_test".into());
        let boxed = mf.new_gp();
        mf.mark_boxed_gp(boxed);
        let raw = mf.new_gp();
        let booly = mf.new_gp();
        let call_ret = mf.new_gp();
        mf.mark_boxed_gp(call_ret);

        mf.emit(MachInst::Prologue { frame_size: 0 });
        mf.emit(MachInst::LoadImm {
            dst: raw,
            bits: 123,
        });
        mf.emit(MachInst::LoadImm {
            dst: booly,
            bits: 1,
        });
        mf.emit(MachInst::CallRuntime {
            name: "wren_not",
            args: vec![boxed],
            ret: Some(call_ret),
        });
        mf.emit(MachInst::Mov {
            dst: raw,
            src: booly,
        });
        mf.emit(MachInst::Ret);

        let target = regalloc::x86_64_target_regs();
        let alloc = regalloc::allocate_registers_result(&mf, &target);
        let meta = build_native_frame_metadata(&mf, &alloc);

        assert_eq!(meta.boxed_values, vec![boxed, call_ret]);
        assert_eq!(meta.safepoints.len(), 1);
        let safepoint = &meta.safepoints[0];
        assert_eq!(safepoint.kind, SafepointKind::CallRuntime);
        assert_eq!(safepoint.ordinal, 0);
        assert_eq!(safepoint.inst_index, 3);
        assert!(safepoint.live_roots.is_empty());
    }
}
