/// AArch64 code emission via dynasmrt.
///
/// Translates `MachInst` (with physical registers) into ARM64 machine code.
/// The result is an `ExecutableBuffer` that can be called as a function pointer.
use dynasmrt::aarch64::Assembler;
use dynasmrt::{dynasm, AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer};

use std::collections::HashMap;

use super::{Cond, Label, MachFunc, MachInst, Mem, VReg, VecWidth};

/// Compiled native code ready to execute.
pub struct CompiledCode {
    buf: ExecutableBuffer,
    start: AssemblyOffset,
}

impl CompiledCode {
    /// Get a callable function pointer.
    ///
    /// # Safety
    /// The caller must ensure the function signature matches the compiled code.
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        let ptr = self.buf.ptr(self.start);
        std::mem::transmute_copy(&ptr)
    }
}

/// Emit ARM64 machine code from a MachFunc.
///
/// Assumes registers are already allocated to physical registers.
/// VReg indices map directly to ARM64 register encodings:
///   GP: VReg::gp(n) → Xn (x0-x30)
///   FP: VReg::fp(n) → Dn (d0-d31)
pub fn emit(func: &MachFunc) -> Result<CompiledCode, String> {
    let mut asm = Assembler::new().map_err(|e| format!("assembler init: {}", e))?;

    // Pre-allocate dynamic labels for each Label in the function.
    let mut labels: HashMap<Label, DynamicLabel> = HashMap::new();
    for inst in &func.insts {
        match inst {
            MachInst::DefLabel(l) => {
                labels.entry(*l).or_insert_with(|| asm.new_dynamic_label());
            }
            MachInst::Jmp { target }
            | MachInst::JmpIf { target, .. }
            | MachInst::JmpZero { target, .. }
            | MachInst::JmpNonZero { target, .. }
            | MachInst::TestBitJmpZero { target, .. }
            | MachInst::TestBitJmpNonZero { target, .. } => {
                labels
                    .entry(*target)
                    .or_insert_with(|| asm.new_dynamic_label());
            }
            _ => {}
        }
    }

    let start = asm.offset();

    for inst in &func.insts {
        emit_inst(&mut asm, inst, &labels)?;
    }

    let buf = asm
        .finalize()
        .map_err(|_| "assembler finalize failed".to_string())?;
    Ok(CompiledCode { buf, start })
}

fn gp(r: VReg) -> u32 {
    debug_assert!(r.is_gp(), "expected GP register, got {:?}", r);
    r.index
}

fn fp(r: VReg) -> u32 {
    debug_assert!(
        r.is_fp() || r.is_vec(),
        "expected FP/Vec register, got {:?}",
        r
    );
    r.index
}

fn get_label(labels: &HashMap<Label, DynamicLabel>, l: &Label) -> DynamicLabel {
    *labels.get(l).expect("unresolved label")
}

/// Compute effective address for a Mem operand.
/// If offset is 0, returns the base register index directly.
/// Otherwise, loads base + offset into x17 (IP1 scratch) and returns 17.
fn emit_addr(asm: &mut Assembler, mem: &Mem) -> u32 {
    let b = gp(mem.base);
    if mem.offset == 0 {
        b
    } else {
        // Use x17 (IP1) as address scratch to avoid conflict with x16 (IP0).
        emit_load_imm64(asm, 17, mem.offset as i64 as u64);
        dynasm!(asm ; add X(17), X(b), X(17));
        17
    }
}

fn emit_inst(
    asm: &mut Assembler,
    inst: &MachInst,
    labels: &HashMap<Label, DynamicLabel>,
) -> Result<(), String> {
    use MachInst::*;
    match inst {
        // =================================================================
        // Data Movement
        // =================================================================
        LoadImm { dst, bits } => {
            let d = gp(*dst);
            // Use movz + movk sequence for arbitrary 64-bit immediate.
            emit_load_imm64(asm, d, *bits);
        }

        LoadFpImm { dst, value } => {
            let d = fp(*dst);
            let bits = value.to_bits();
            // Load via GP scratch (x16/IP0) then transfer to FP.
            emit_load_imm64(asm, 16, bits); // x16 = IP0 scratch
            dynasm!(asm ; fmov D(d), X(16));
        }

        Mov { dst, src } => {
            let d = gp(*dst);
            let s = gp(*src);
            if d != s {
                dynasm!(asm ; mov X(d), X(s));
            }
        }

        FMov { dst, src } => {
            let d = fp(*dst);
            let s = fp(*src);
            if d != s {
                dynasm!(asm ; fmov D(d), D(s));
            }
        }

        BitcastGpToFp { dst, src } => {
            dynasm!(asm ; fmov D(fp(*dst)), X(gp(*src)));
        }

        BitcastFpToGp { dst, src } => {
            dynasm!(asm ; fmov X(gp(*dst)), D(fp(*src)));
        }

        // =================================================================
        // Integer Arithmetic
        // =================================================================
        IAdd { dst, lhs, rhs } => {
            dynasm!(asm ; add X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        IAddImm { dst, src, imm } => {
            // Load immediate into scratch x16, use register form.
            emit_load_imm64(asm, 16, *imm as i64 as u64);
            dynasm!(asm ; add X(gp(*dst)), X(gp(*src)), X(16));
        }

        ISub { dst, lhs, rhs } => {
            dynasm!(asm ; sub X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        IMul { dst, lhs, rhs } => {
            dynasm!(asm ; mul X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        IDiv { dst, lhs, rhs } => {
            dynasm!(asm ; sdiv X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        IMulSub { dst, lhs, rhs, acc } => {
            dynasm!(asm ; msub X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)), X(gp(*acc)));
        }

        INeg { dst, src } => {
            dynasm!(asm ; neg X(gp(*dst)), X(gp(*src)));
        }

        // =================================================================
        // Bitwise
        // =================================================================
        And { dst, lhs, rhs } => {
            dynasm!(asm ; and X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        AndImm { dst, src, imm } => {
            // ARM64 logical immediates are encoded specially; fall back to
            // loading the immediate into a scratch register.
            emit_load_imm64(asm, 16, *imm);
            dynasm!(asm ; and X(gp(*dst)), X(gp(*src)), X(16));
        }

        Or { dst, lhs, rhs } => {
            dynasm!(asm ; orr X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        OrImm { dst, src, imm } => {
            emit_load_imm64(asm, 16, *imm);
            dynasm!(asm ; orr X(gp(*dst)), X(gp(*src)), X(16));
        }

        Xor { dst, lhs, rhs } => {
            dynasm!(asm ; eor X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        Not { dst, src } => {
            dynasm!(asm ; mvn X(gp(*dst)), X(gp(*src)));
        }

        Shl { dst, lhs, rhs } => {
            dynasm!(asm ; lsl X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        Sar { dst, lhs, rhs } => {
            dynasm!(asm ; asr X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        Shr { dst, lhs, rhs } => {
            dynasm!(asm ; lsr X(gp(*dst)), X(gp(*lhs)), X(gp(*rhs)));
        }

        // =================================================================
        // FP Arithmetic
        // =================================================================
        FAdd { dst, lhs, rhs } => {
            dynasm!(asm ; fadd D(fp(*dst)), D(fp(*lhs)), D(fp(*rhs)));
        }

        FSub { dst, lhs, rhs } => {
            dynasm!(asm ; fsub D(fp(*dst)), D(fp(*lhs)), D(fp(*rhs)));
        }

        FMul { dst, lhs, rhs } => {
            dynasm!(asm ; fmul D(fp(*dst)), D(fp(*lhs)), D(fp(*rhs)));
        }

        FDiv { dst, lhs, rhs } => {
            dynasm!(asm ; fdiv D(fp(*dst)), D(fp(*lhs)), D(fp(*rhs)));
        }

        FNeg { dst, src } => {
            dynasm!(asm ; fneg D(fp(*dst)), D(fp(*src)));
        }

        FAbs { dst, src } => {
            dynasm!(asm ; fabs D(fp(*dst)), D(fp(*src)));
        }

        FSqrt { dst, src } => {
            dynasm!(asm ; fsqrt D(fp(*dst)), D(fp(*src)));
        }

        FFloor { dst, src } => {
            dynasm!(asm ; frintm D(fp(*dst)), D(fp(*src)));
        }

        FCeil { dst, src } => {
            dynasm!(asm ; frintp D(fp(*dst)), D(fp(*src)));
        }

        FRound { dst, src } => {
            dynasm!(asm ; frintn D(fp(*dst)), D(fp(*src)));
        }

        FTrunc { dst, src } => {
            dynasm!(asm ; frintz D(fp(*dst)), D(fp(*src)));
        }

        FMin { dst, lhs, rhs } => {
            dynasm!(asm ; fmin D(fp(*dst)), D(fp(*lhs)), D(fp(*rhs)));
        }

        FMax { dst, lhs, rhs } => {
            dynasm!(asm ; fmax D(fp(*dst)), D(fp(*lhs)), D(fp(*rhs)));
        }

        // =================================================================
        // FMA
        // =================================================================
        FMAdd { dst, a, b, c } => {
            dynasm!(asm ; fmadd D(fp(*dst)), D(fp(*a)), D(fp(*b)), D(fp(*c)));
        }

        FMSub { dst, a, b, c } => {
            dynasm!(asm ; fmsub D(fp(*dst)), D(fp(*a)), D(fp(*b)), D(fp(*c)));
        }

        FNMAdd { dst, a, b, c } => {
            dynasm!(asm ; fnmadd D(fp(*dst)), D(fp(*a)), D(fp(*b)), D(fp(*c)));
        }

        FNMSub { dst, a, b, c } => {
            dynasm!(asm ; fnmsub D(fp(*dst)), D(fp(*a)), D(fp(*b)), D(fp(*c)));
        }

        // =================================================================
        // Conversions
        // =================================================================
        FCvtToI64 { dst, src } => {
            dynasm!(asm ; fcvtzs X(gp(*dst)), D(fp(*src)));
        }

        I64CvtToF { dst, src } => {
            dynasm!(asm ; scvtf D(fp(*dst)), X(gp(*src)));
        }

        // =================================================================
        // Comparison
        // =================================================================
        ICmp { lhs, rhs } => {
            dynasm!(asm ; cmp X(gp(*lhs)), X(gp(*rhs)));
        }

        ICmpImm { lhs, imm } => {
            emit_load_imm64(asm, 16, *imm);
            dynasm!(asm ; cmp X(gp(*lhs)), X(16));
        }

        FCmp { lhs, rhs } => {
            dynasm!(asm ; fcmp D(fp(*lhs)), D(fp(*rhs)));
        }

        CSet { dst, cond } => {
            let d = gp(*dst);
            match cond {
                Cond::Eq => dynasm!(asm ; cset X(d), eq),
                Cond::Ne => dynasm!(asm ; cset X(d), ne),
                Cond::Lt => dynasm!(asm ; cset X(d), lt),
                Cond::Le => dynasm!(asm ; cset X(d), le),
                Cond::Gt => dynasm!(asm ; cset X(d), gt),
                Cond::Ge => dynasm!(asm ; cset X(d), ge),
                Cond::Below => dynasm!(asm ; cset X(d), lo),
                Cond::BelowEq => dynasm!(asm ; cset X(d), ls),
                Cond::Above => dynasm!(asm ; cset X(d), hi),
                Cond::AboveEq => dynasm!(asm ; cset X(d), hs),
            }
        }

        // =================================================================
        // Memory
        // =================================================================
        Ldr { dst, mem } => {
            let d = gp(*dst);
            let b = emit_addr(asm, mem);
            dynasm!(asm ; ldr X(d), [X(b)]);
        }

        Str { src, mem } => {
            let s = gp(*src);
            let b = emit_addr(asm, mem);
            dynasm!(asm ; str X(s), [X(b)]);
        }

        FLdr { dst, mem } => {
            let d = fp(*dst);
            let b = emit_addr(asm, mem);
            dynasm!(asm ; ldr D(d), [X(b)]);
        }

        FStr { src, mem } => {
            let s = fp(*src);
            let b = emit_addr(asm, mem);
            dynasm!(asm ; str D(s), [X(b)]);
        }

        // =================================================================
        // Control Flow
        // =================================================================
        Jmp { target } => {
            let lbl = get_label(labels, target);
            dynasm!(asm ; b =>lbl);
        }

        JmpIf { cond, target } => {
            let lbl = get_label(labels, target);
            match cond {
                Cond::Eq => dynasm!(asm ; b.eq =>lbl),
                Cond::Ne => dynasm!(asm ; b.ne =>lbl),
                Cond::Lt => dynasm!(asm ; b.lt =>lbl),
                Cond::Le => dynasm!(asm ; b.le =>lbl),
                Cond::Gt => dynasm!(asm ; b.gt =>lbl),
                Cond::Ge => dynasm!(asm ; b.ge =>lbl),
                Cond::Below => dynasm!(asm ; b.lo =>lbl),
                Cond::BelowEq => dynasm!(asm ; b.ls =>lbl),
                Cond::Above => dynasm!(asm ; b.hi =>lbl),
                Cond::AboveEq => dynasm!(asm ; b.hs =>lbl),
            }
        }

        JmpZero { src, target } => {
            let lbl = get_label(labels, target);
            dynasm!(asm ; cbz X(gp(*src)), =>lbl);
        }

        JmpNonZero { src, target } => {
            let lbl = get_label(labels, target);
            dynasm!(asm ; cbnz X(gp(*src)), =>lbl);
        }

        TestBitJmpZero { src, bit, target } => {
            let lbl = get_label(labels, target);
            dynasm!(asm ; tbz X(gp(*src)), *bit as u32, =>lbl);
        }

        TestBitJmpNonZero { src, bit, target } => {
            let lbl = get_label(labels, target);
            dynasm!(asm ; tbnz X(gp(*src)), *bit as u32, =>lbl);
        }

        // =================================================================
        // Calls & Returns
        // =================================================================
        CallInd { target } => {
            dynasm!(asm ; blr X(gp(*target)));
        }

        CallRuntime { .. } => {
            // Runtime calls require the address to be loaded into a register
            // and called via blr. The actual address patching happens at link time.
            // For now, emit a placeholder blr x16.
            return Err(
                "CallRuntime not yet linked — use CallInd with resolved address".to_string(),
            );
        }

        Ret => {
            dynasm!(asm ; ret);
        }

        // =================================================================
        // Stack Frame
        // =================================================================
        Prologue { frame_size } => {
            // Save frame pointer and link register, set up frame.
            let fs = *frame_size as i32;
            let total: u32 = (if fs < 16 { 16 } else { (fs + 15) & !15 }) as u32;
            dynasm!(asm
                ; stp x29, x30, [sp, -(total as i32)]!
                ; mov x29, sp
            );
        }

        Epilogue { frame_size } => {
            let fs = *frame_size as i32;
            let total: u32 = (if fs < 16 { 16 } else { (fs + 15) & !15 }) as u32;
            dynasm!(asm
                ; ldp x29, x30, [sp], total as i32
            );
        }

        Push { src } => {
            dynasm!(asm ; str X(gp(*src)), [sp, -16]!);
        }

        Pop { dst } => {
            dynasm!(asm ; ldr X(gp(*dst)), [sp], 16);
        }

        // =================================================================
        // Pseudo-instructions
        // =================================================================
        DefLabel(l) => {
            let lbl = get_label(labels, l);
            dynasm!(asm ; =>lbl);
        }

        Nop => {
            dynasm!(asm ; nop);
        }

        Trap => {
            dynasm!(asm ; brk 1);
        }

        ParallelCopy { .. } => {
            return Err("ParallelCopy must be resolved before emission".to_string());
        }

        // =================================================================
        // SIMD (V128 = NEON 2×f64)
        // =================================================================
        VLoad { dst, mem, width } => match width {
            VecWidth::V128 => {
                let b = emit_addr(asm, mem);
                dynasm!(asm ; ldr Q(fp(*dst)), [X(b)]);
            }
            VecWidth::V256 => {
                return Err("V256 not supported on aarch64 (no AVX)".to_string());
            }
        },

        VStore { src, mem, width } => match width {
            VecWidth::V128 => {
                let b = emit_addr(asm, mem);
                dynasm!(asm ; str Q(fp(*src)), [X(b)]);
            }
            VecWidth::V256 => {
                return Err("V256 not supported on aarch64".to_string());
            }
        },

        // Vector arithmetic — emit 2d NEON variants for V128.
        VFAdd {
            dst,
            lhs,
            rhs,
            width: VecWidth::V128,
        } => {
            dynasm!(asm ; fadd V(fp(*dst)).d2, V(fp(*lhs)).d2, V(fp(*rhs)).d2);
        }
        VFSub {
            dst,
            lhs,
            rhs,
            width: VecWidth::V128,
        } => {
            dynasm!(asm ; fsub V(fp(*dst)).d2, V(fp(*lhs)).d2, V(fp(*rhs)).d2);
        }
        VFMul {
            dst,
            lhs,
            rhs,
            width: VecWidth::V128,
        } => {
            dynasm!(asm ; fmul V(fp(*dst)).d2, V(fp(*lhs)).d2, V(fp(*rhs)).d2);
        }
        VFDiv {
            dst,
            lhs,
            rhs,
            width: VecWidth::V128,
        } => {
            dynasm!(asm ; fdiv V(fp(*dst)).d2, V(fp(*lhs)).d2, V(fp(*rhs)).d2);
        }
        VFNeg {
            dst,
            src,
            width: VecWidth::V128,
        } => {
            dynasm!(asm ; fneg V(fp(*dst)).d2, V(fp(*src)).d2);
        }

        VBroadcast {
            dst,
            src,
            width: VecWidth::V128,
        } => {
            // dup Vd.2D, Vn.D[0] — broadcast scalar d into both lanes
            dynasm!(asm ; dup V(fp(*dst)).d2, V(fp(*src)).d[0]);
        }

        VFMAdd {
            dst,
            a,
            b,
            c,
            width: VecWidth::V128,
        } => {
            // NEON fmla is destructive: Vd = Vd + Va * Vb.
            // If dst != c, we need to move c into dst first.
            let d = fp(*dst);
            let ca = fp(*c);
            if d != ca {
                // Move accumulator to dst first.
                dynasm!(asm ; mov V(d).b16, V(ca).b16);
            }
            dynasm!(asm ; fmla V(d).d2, V(fp(*a)).d2, V(fp(*b)).d2);
        }

        VExtractLane { dst, src, lane } => {
            // mov Dd, Vn.d[lane]
            // For lane 0, this is just a fmov; for lane 1, use `mov` from element.
            let d = fp(*dst);
            let s = fp(*src);
            match lane {
                0 => dynasm!(asm ; fmov D(d), D(s)),
                // dynasm may not directly support `mov d, v.d[1]`, so use umov+fmov.
                _ => {
                    // Use x16 as scratch: umov x16, Vn.d[lane]; fmov Dd, x16
                    dynasm!(asm
                        ; mov X(16), V(s).d[*lane as u32]
                        ; fmov D(d), X(16)
                    );
                }
            }
        }

        VInsertLane {
            dst,
            src,
            lane,
            val,
        } => {
            let d = fp(*dst);
            let s = fp(*src);
            let v = fp(*val);
            if d != s {
                dynasm!(asm ; mov V(d).b16, V(s).b16);
            }
            // ins Vd.d[lane], Dn
            dynasm!(asm
                ; fmov X(16), D(v)
                ; mov V(d).d[*lane as u32], X(16)
            );
        }

        VReduceAdd {
            dst,
            src,
            width: VecWidth::V128,
        } => {
            // Horizontal add of 2 f64 lanes: faddp Dd, Vn.2d
            dynasm!(asm ; faddp D(fp(*dst)), V(fp(*src)).d2);
        }

        // V256 fallback — not supported on aarch64 NEON.
        VFAdd {
            width: VecWidth::V256,
            ..
        }
        | VFSub {
            width: VecWidth::V256,
            ..
        }
        | VFMul {
            width: VecWidth::V256,
            ..
        }
        | VFDiv {
            width: VecWidth::V256,
            ..
        }
        | VFNeg {
            width: VecWidth::V256,
            ..
        }
        | VBroadcast {
            width: VecWidth::V256,
            ..
        }
        | VFMAdd {
            width: VecWidth::V256,
            ..
        }
        | VReduceAdd {
            width: VecWidth::V256,
            ..
        } => {
            return Err("V256 not supported on aarch64 (use V128)".to_string());
        }
    }
    Ok(())
}

/// Emit a 64-bit immediate load into GP register `rd` using movz/movk.
fn emit_load_imm64(asm: &mut Assembler, rd: u32, imm: u64) {
    if imm == 0 {
        dynasm!(asm ; mov X(rd), xzr);
        return;
    }

    let hw0 = (imm & 0xFFFF) as u32;
    let hw1 = ((imm >> 16) & 0xFFFF) as u32;
    let hw2 = ((imm >> 32) & 0xFFFF) as u32;
    let hw3 = ((imm >> 48) & 0xFFFF) as u32;

    // Find first non-zero halfword for movz, then movk the rest.
    let halfwords = [(hw0, 0u32), (hw1, 16), (hw2, 32), (hw3, 48)];
    let mut first = true;

    for &(hw, shift) in &halfwords {
        if hw != 0 || (first && shift == 48) {
            // We need at least one instruction even if all zeros (handled above).
            if first {
                dynasm!(asm ; movz X(rd), hw, LSL shift);
                first = false;
            } else {
                dynasm!(asm ; movk X(rd), hw, LSL shift);
            }
        }
    }

    // If the entire value was zero except we already handled that above.
    // Edge case: if first is still true, value was 0 (handled at top).
    if first {
        dynasm!(asm ; movz X(rd), 0, LSL 0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{Cond, MachFunc, MachInst, VReg};

    /// Helper: build a MachFunc, emit, and return CompiledCode.
    fn compile(build: impl FnOnce(&mut MachFunc)) -> CompiledCode {
        let mut mf = MachFunc::new("test".to_string());
        build(&mut mf);
        emit(&mf).expect("emission failed")
    }

    #[test]
    fn test_return_constant() {
        // return 42 (as u64)
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 42,
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 42);
    }

    #[test]
    fn test_integer_add() {
        // x0 = 10, x1 = 32, return x0 + x1
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 10,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 32,
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(0),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 42);
    }

    #[test]
    fn test_f64_add() {
        // d0 = 1.5, d1 = 2.5, d0 = d0 + d1, return via fmov x0, d0
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 1.5,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 2.5,
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(0),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            // Return f64 in d0 (AAPCS64 returns f64 in d0).
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: f64 = unsafe {
            let f: extern "C" fn() -> f64 = code.as_fn();
            f()
        };
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_f64_mul_sub_div() {
        // (10.0 * 3.0) - 5.0 = 25.0, then 25.0 / 5.0 = 5.0
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 10.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 3.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(2),
                value: 5.0,
            });
            mf.emit(MachInst::FMul {
                dst: VReg::fp(3),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::FSub {
                dst: VReg::fp(3),
                lhs: VReg::fp(3),
                rhs: VReg::fp(2),
            });
            mf.emit(MachInst::FDiv {
                dst: VReg::fp(0),
                lhs: VReg::fp(3),
                rhs: VReg::fp(2),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: f64 = unsafe {
            let f: extern "C" fn() -> f64 = code.as_fn();
            f()
        };
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_fma() {
        // fmadd: d0 = 2.0 * 3.0 + 1.0 = 7.0
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 2.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(2),
                value: 3.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(3),
                value: 1.0,
            });
            mf.emit(MachInst::FMAdd {
                dst: VReg::fp(0),
                a: VReg::fp(1),
                b: VReg::fp(2),
                c: VReg::fp(3),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: f64 = unsafe {
            let f: extern "C" fn() -> f64 = code.as_fn();
            f()
        };
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_branch_and_label() {
        // if (true) return 1 else return 0
        let code = compile(|mf| {
            let l_true = mf.new_label();
            let l_end = mf.new_label();

            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 1,
            });
            mf.emit(MachInst::JmpNonZero {
                src: VReg::gp(0),
                target: l_true,
            });
            // False path:
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 0,
            });
            mf.emit(MachInst::Jmp { target: l_end });
            // True path:
            mf.emit(MachInst::DefLabel(l_true));
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 99,
            });
            // End:
            mf.emit(MachInst::DefLabel(l_end));
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 99);
    }

    #[test]
    fn test_cmp_and_cset() {
        // x0 = (10 < 20) ? 1 : 0
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 10,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(2),
                bits: 20,
            });
            mf.emit(MachInst::ICmp {
                lhs: VReg::gp(1),
                rhs: VReg::gp(2),
            });
            mf.emit(MachInst::CSet {
                dst: VReg::gp(0),
                cond: Cond::Lt,
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_fcmp_and_cset() {
        // d0=1.0, d1=2.0; x0 = (d0 < d1) ? 1 : 0
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 1.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 2.0,
            });
            mf.emit(MachInst::FCmp {
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::CSet {
                dst: VReg::gp(0),
                cond: Cond::Lt,
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_box_unbox_roundtrip() {
        // Load 3.14 as NaN-boxed value, unbox to f64, return it.
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            // Load 3.14 bits into x1 (NaN-boxed num = raw f64 bits)
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 1.234_f64.to_bits(),
            });
            // Unbox: x1 → d0 (bitwise transfer)
            mf.emit(MachInst::BitcastGpToFp {
                dst: VReg::fp(0),
                src: VReg::gp(1),
            });
            // Return d0 as f64
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: f64 = unsafe {
            let f: extern "C" fn() -> f64 = code.as_fn();
            f()
        };
        assert_eq!(result, 1.234);
    }

    #[test]
    fn test_loop_sum_1_to_10() {
        // sum = 0; i = 1; while (i <= 10) { sum += i; i++; } return sum
        let code = compile(|mf| {
            let l_loop = mf.new_label();
            let l_end = mf.new_label();

            mf.emit(MachInst::Prologue { frame_size: 0 });
            // x0 = sum = 0, x1 = i = 1, x2 = 10
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 0,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 1,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(2),
                bits: 10,
            });

            // loop:
            mf.emit(MachInst::DefLabel(l_loop));
            mf.emit(MachInst::ICmp {
                lhs: VReg::gp(1),
                rhs: VReg::gp(2),
            });
            mf.emit(MachInst::JmpIf {
                cond: Cond::Gt,
                target: l_end,
            });
            // sum += i
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(0),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            // i++
            mf.emit(MachInst::IAddImm {
                dst: VReg::gp(1),
                src: VReg::gp(1),
                imm: 1,
            });
            mf.emit(MachInst::Jmp { target: l_loop });

            // end:
            mf.emit(MachInst::DefLabel(l_end));
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 55); // 1+2+...+10
    }

    #[test]
    fn test_f64_negation() {
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 42.0,
            });
            mf.emit(MachInst::FNeg {
                dst: VReg::fp(0),
                src: VReg::fp(1),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: f64 = unsafe {
            let f: extern "C" fn() -> f64 = code.as_fn();
            f()
        };
        assert_eq!(result, -42.0);
    }

    #[test]
    fn test_bitwise_and_or_xor() {
        // (0xFF00 & 0x0FF0) | 0x000F = 0x0F0F
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 0xFF00,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(2),
                bits: 0x0FF0,
            });
            mf.emit(MachInst::And {
                dst: VReg::gp(0),
                lhs: VReg::gp(1),
                rhs: VReg::gp(2),
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(3),
                bits: 0x000F,
            });
            mf.emit(MachInst::Or {
                dst: VReg::gp(0),
                lhs: VReg::gp(0),
                rhs: VReg::gp(3),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 0x0F0F);
    }

    #[test]
    fn test_conversion_f64_to_i64() {
        // convert 3.7 → 3 (truncate)
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 3.7,
            });
            mf.emit(MachInst::FCvtToI64 {
                dst: VReg::gp(0),
                src: VReg::fp(0),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn() -> u64 = code.as_fn();
            f()
        };
        assert_eq!(result, 3);
    }

    #[test]
    fn test_function_with_args() {
        // extern "C" fn(a: u64, b: u64) -> u64 { a + b }
        // Args arrive in x0, x1 per AAPCS64.
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 0 });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(0),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::Epilogue { frame_size: 0 });
            mf.emit(MachInst::Ret);
        });
        let result: u64 = unsafe {
            let f: extern "C" fn(u64, u64) -> u64 = code.as_fn();
            f(17, 25)
        };
        assert_eq!(result, 42);
    }
}
