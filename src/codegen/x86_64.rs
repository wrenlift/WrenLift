/// x86_64 code emission via manual byte encoding.
///
/// Translates `MachInst` (with physical registers) into x86_64 machine code.
/// Unlike the aarch64 backend, this does NOT use dynasmrt — it encodes
/// instruction bytes directly, making it cross-platform (assemblable on ARM).
/// Tests use capstone for disassembly validation and cross-check against
/// the MIR interpreter for semantic correctness.
///
/// # Register conventions
/// - GP: VReg::gp(n) maps to x86_64 encoding n (0=RAX..7=RDI, 8-15=R8-R15)
/// - FP: VReg::fp(n) maps to XMM(n) (0=XMM0..15=XMM15)
/// - Scratch: R11 (GP 11) and XMM15 (FP 15) reserved for emitter use
use super::{Cond, Label, MachFunc, MachInst, VReg, VecWidth};
use std::collections::HashMap;

// Scratch register indices — register allocator must not assign these.
const SCRATCH_GP: u32 = 11; // R11
const SCRATCH_XMM: u32 = 15; // XMM15

/// Assembled x86_64 machine code (byte buffer).
///
/// On x86_64 hosts, call `make_executable()` to get a callable function pointer.
pub struct EmittedCode {
    code: Vec<u8>,
}

impl EmittedCode {
    pub fn bytes(&self) -> &[u8] {
        &self.code
    }
    pub fn len(&self) -> usize {
        self.code.len()
    }
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Copy the code into executable memory via mmap.
    ///
    /// Returns an `ExecutableCode` that keeps the mapping alive and provides
    /// a function pointer. Only useful on x86_64 hosts.
    pub fn make_executable(&self) -> Result<ExecutableCode, String> {
        use dynasmrt::mmap::MutableBuffer;
        let mut buf =
            MutableBuffer::new(self.code.len()).map_err(|e| format!("mmap alloc: {}", e))?;
        buf.set_len(self.code.len());
        buf[..self.code.len()].copy_from_slice(&self.code);
        let exec = buf.make_exec().map_err(|e| format!("mprotect: {}", e))?;
        Ok(ExecutableCode { buf: exec })
    }
}

/// Executable x86_64 code backed by mmap'd memory.
pub struct ExecutableCode {
    buf: dynasmrt::ExecutableBuffer,
}

impl ExecutableCode {
    /// Get a callable function pointer.
    ///
    /// # Safety
    /// The caller must ensure the type `F` matches the compiled function's ABI.
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        let ptr = self.buf.ptr(dynasmrt::AssemblyOffset(0));
        std::mem::transmute_copy(&ptr)
    }
}

struct Fixup {
    offset: usize, // byte offset of the rel32 placeholder
    target: Label,
}

struct X64Emitter {
    code: Vec<u8>,
    labels: HashMap<Label, usize>,
    fixups: Vec<Fixup>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level encoding helpers
// ─────────────────────────────────────────────────────────────────────────────

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

/// x86_64 condition code nibble (used in JCC = 0F 8x, SETcc = 0F 9x).
fn cond_code(cond: &Cond) -> u8 {
    match cond {
        Cond::Eq => 0x4,
        Cond::Ne => 0x5,
        Cond::Lt => 0xC,
        Cond::Le => 0xE,
        Cond::Gt => 0xF,
        Cond::Ge => 0xD,
        Cond::Below => 0x2,
        Cond::BelowEq => 0x6,
        Cond::Above => 0x7,
        Cond::AboveEq => 0x3,
    }
}

impl X64Emitter {
    fn new() -> Self {
        Self {
            code: Vec::with_capacity(256),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    // ── Byte emission ──

    fn push(&mut self, b: u8) {
        self.code.push(b);
    }
    fn push_i32_le(&mut self, v: i32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }
    fn push_u64_le(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    // ── REX prefix ──
    // REX = 0100 W R X B

    fn rex(w: bool, reg: u32, idx: u32, rm: u32) -> u8 {
        0x40 | if w { 8 } else { 0 }
            | (((reg >> 3) & 1) << 2) as u8
            | (((idx >> 3) & 1) << 1) as u8
            | ((rm >> 3) & 1) as u8
    }

    fn need_rex(w: bool, reg: u32, idx: u32, rm: u32) -> bool {
        w || reg >= 8 || idx >= 8 || rm >= 8
    }

    fn emit_rex_if_needed(&mut self, w: bool, reg: u32, idx: u32, rm: u32) {
        if Self::need_rex(w, reg, idx, rm) {
            self.push(Self::rex(w, reg, idx, rm));
        }
    }

    // ── ModRM ──

    fn modrm(mode: u8, reg: u32, rm: u32) -> u8 {
        (mode << 6) | (((reg & 7) as u8) << 3) | ((rm & 7) as u8)
    }

    /// Emit ModRM + optional SIB + displacement for memory [base + disp].
    fn emit_modrm_mem(&mut self, reg: u32, base: u32, disp: i32) {
        let base_lo = (base & 7) as u8;
        let reg_lo = (reg & 7) as u8;
        let need_sib = base_lo == 4; // RSP/R12 always need SIB

        if disp == 0 && base_lo != 5 {
            // mod=00, no displacement (except RBP/R13 which encodes [RIP+disp32])
            self.push((reg_lo << 3) | if need_sib { 4 } else { base_lo });
            if need_sib {
                self.push(0x24); // SIB: scale=0, index=RSP(none), base=RSP
            }
        } else if (-128..=127).contains(&disp) {
            // mod=01, disp8
            self.push(0x40 | (reg_lo << 3) | if need_sib { 4 } else { base_lo });
            if need_sib {
                self.push(0x24);
            }
            self.push(disp as u8);
        } else {
            // mod=10, disp32
            self.push(0x80 | (reg_lo << 3) | if need_sib { 4 } else { base_lo });
            if need_sib {
                self.push(0x24);
            }
            self.push_i32_le(disp);
        }
    }

    // ── VEX prefix (3-byte form for FMA3/AVX) ──

    #[allow(clippy::too_many_arguments)]
    fn emit_vex3(
        &mut self,
        reg: u32,
        idx: u32,
        rm: u32,
        mmmmm: u8,
        w: bool,
        vvvv: u32,
        l: bool,
        pp: u8,
    ) {
        self.push(0xC4);
        let byte1 = (if reg < 8 { 0x80 } else { 0 })
            | (if idx < 8 { 0x40 } else { 0 })
            | (if rm < 8 { 0x20 } else { 0 })
            | (mmmmm & 0x1F);
        self.push(byte1);
        let byte2 = (if w { 0x80 } else { 0 })
            | (((!vvvv) & 0xF) << 3) as u8
            | (if l { 0x04 } else { 0 })
            | (pp & 0x03);
        self.push(byte2);
    }

    // ── Label / fixup helpers ──

    fn define_label(&mut self, label: Label) {
        self.labels.insert(label, self.code.len());
    }

    /// Emit a rel32 placeholder for a forward/backward branch.
    fn emit_rel32_fixup(&mut self, target: Label) {
        self.fixups.push(Fixup {
            offset: self.code.len(),
            target,
        });
        self.push_i32_le(0); // placeholder
    }

    fn resolve_fixups(&mut self) -> Result<(), String> {
        for fixup in &self.fixups {
            let target_offset = self
                .labels
                .get(&fixup.target)
                .ok_or_else(|| format!("unresolved label {:?}", fixup.target))?;
            // rel32 = target - (fixup_offset + 4)
            let rel = (*target_offset as i64) - ((fixup.offset + 4) as i64);
            if rel < i32::MIN as i64 || rel > i32::MAX as i64 {
                return Err(format!("branch offset out of range: {}", rel));
            }
            let bytes = (rel as i32).to_le_bytes();
            self.code[fixup.offset..fixup.offset + 4].copy_from_slice(&bytes);
        }
        Ok(())
    }

    // ═════════════════════════════════════════════════════════════════════════
    // GP instruction emitters
    // ═════════════════════════════════════════════════════════════════════════

    /// mov r64, r64
    fn emit_mov_rr(&mut self, dst: u32, src: u32) {
        if dst == src {
            return;
        }
        // 89 /r : MOV r/m64, r64 (reg=src, rm=dst)
        self.push(Self::rex(true, src, 0, dst));
        self.push(0x89);
        self.push(Self::modrm(3, src, dst));
    }

    /// mov r64, imm64
    fn emit_mov_imm64(&mut self, dst: u32, imm: u64) {
        // REX.W + B8+rd + imm64
        self.push(Self::rex(true, 0, 0, dst));
        self.push(0xB8 + (dst & 7) as u8);
        self.push_u64_le(imm);
    }

    /// Generic 2-operand ALU: op r/m64, r64 (opcode with reg=src, rm=dst).
    /// Used for ADD(01), SUB(29), AND(21), OR(09), XOR(31), CMP(39), TEST(85).
    fn emit_alu_rr(&mut self, opcode: u8, dst: u32, src: u32) {
        self.push(Self::rex(true, src, 0, dst));
        self.push(opcode);
        self.push(Self::modrm(3, src, dst));
    }

    /// Generic unary/ext ALU: op r/m64 (opcode + /ext).
    /// Used for NEG(F7/3), NOT(F7/2), IDIV(F7/7).
    fn emit_alu_ext(&mut self, opcode: u8, ext: u8, rm: u32) {
        self.push(Self::rex(true, ext as u32, 0, rm));
        self.push(opcode);
        self.push(Self::modrm(3, ext as u32, rm));
    }

    /// add r/m64, imm32
    fn emit_add_imm32(&mut self, dst: u32, imm: i32) {
        self.push(Self::rex(true, 0, 0, dst));
        self.push(0x81);
        self.push(Self::modrm(3, 0, dst)); // /0 = ADD
        self.push_i32_le(imm);
    }

    /// sub r/m64, imm32
    fn emit_sub_imm32(&mut self, dst: u32, imm: i32) {
        self.push(Self::rex(true, 5, 0, dst));
        self.push(0x81);
        self.push(Self::modrm(3, 5, dst)); // /5 = SUB
        self.push_i32_le(imm);
    }

    /// imul r64, r/m64 (2-operand: dst *= src)
    fn emit_imul_rr(&mut self, dst: u32, src: u32) {
        self.push(Self::rex(true, dst, 0, src));
        self.push(0x0F);
        self.push(0xAF);
        self.push(Self::modrm(3, dst, src));
    }

    /// cqo: sign-extend RAX into RDX:RAX
    fn emit_cqo(&mut self) {
        self.push(0x48); // REX.W
        self.push(0x99);
    }

    /// test r/m64, r64
    fn emit_test_rr(&mut self, r1: u32, r2: u32) {
        self.emit_alu_rr(0x85, r1, r2);
    }

    /// shl/sar/shr r/m64, cl
    fn emit_shift_cl(&mut self, ext: u8, dst: u32) {
        self.push(Self::rex(true, ext as u32, 0, dst));
        self.push(0xD3);
        self.push(Self::modrm(3, ext as u32, dst));
    }

    /// bt r/m64, imm8
    fn emit_bt_imm(&mut self, src: u32, bit: u8) {
        self.push(Self::rex(true, 4, 0, src));
        self.push(0x0F);
        self.push(0xBA);
        self.push(Self::modrm(3, 4, src)); // /4 = BT
        self.push(bit);
    }

    /// setcc r8 + zero-extend → materialize condition into full 64-bit register.
    fn emit_setcc_full(&mut self, cc: u8, dst: u32) {
        // setcc r8 — need REX for uniform byte registers (SPL+)
        if dst >= 4 {
            self.push(0x40 | if dst >= 8 { 0x01 } else { 0 });
        }
        self.push(0x0F);
        self.push(0x90 | cc);
        self.push(Self::modrm(3, 0, dst));

        // movzx r64, r8 — zero-extend byte to 64-bit (preserves flags, unlike xor)
        self.push(Self::rex(true, dst, 0, dst));
        self.push(0x0F);
        self.push(0xB6);
        self.push(Self::modrm(3, dst, dst));
    }

    // ── Memory (GP) ──

    /// mov r64, [base + disp]
    fn emit_load_mem(&mut self, dst: u32, base: u32, disp: i32) {
        self.push(Self::rex(true, dst, 0, base));
        self.push(0x8B);
        self.emit_modrm_mem(dst, base, disp);
    }

    /// mov [base + disp], r64
    fn emit_store_mem(&mut self, base: u32, disp: i32, src: u32) {
        self.push(Self::rex(true, src, 0, base));
        self.push(0x89);
        self.emit_modrm_mem(src, base, disp);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // SSE2 floating-point instruction emitters
    // ═════════════════════════════════════════════════════════════════════════

    /// SSE2 reg-reg with mandatory prefix: prefix [REX] 0F opcode ModRM
    fn emit_sse_rr(&mut self, prefix: u8, opcode: u8, reg: u32, rm: u32) {
        self.push(prefix);
        self.emit_rex_if_needed(false, reg, 0, rm);
        self.push(0x0F);
        self.push(opcode);
        self.push(Self::modrm(3, reg, rm));
    }

    /// SSE2 memory load: prefix [REX] 0F opcode ModRM_mem
    fn emit_sse_load(&mut self, prefix: u8, opcode: u8, xmm: u32, base: u32, disp: i32) {
        self.push(prefix);
        self.emit_rex_if_needed(false, xmm, 0, base);
        self.push(0x0F);
        self.push(opcode);
        self.emit_modrm_mem(xmm, base, disp);
    }

    /// SSE2 memory store: prefix [REX] 0F opcode ModRM_mem
    fn emit_sse_store(&mut self, prefix: u8, opcode: u8, base: u32, disp: i32, xmm: u32) {
        self.push(prefix);
        self.emit_rex_if_needed(false, xmm, 0, base);
        self.push(0x0F);
        self.push(opcode);
        self.emit_modrm_mem(xmm, base, disp);
    }

    /// roundsd dst, src, imm8 (SSE4.1): 66 0F 3A 0B /r ib
    fn emit_roundsd(&mut self, dst: u32, src: u32, imm: u8) {
        self.push(0x66);
        self.emit_rex_if_needed(false, dst, 0, src);
        self.push(0x0F);
        self.push(0x3A);
        self.push(0x0B);
        self.push(Self::modrm(3, dst, src));
        self.push(imm);
    }

    /// movsd xmm, xmm (reg-reg)
    fn emit_movsd_rr(&mut self, dst: u32, src: u32) {
        if dst == src {
            return;
        }
        self.emit_sse_rr(0xF2, 0x10, dst, src);
    }

    /// movapd xmm, xmm (reg-reg, for packed moves)
    fn emit_movapd_rr(&mut self, dst: u32, src: u32) {
        if dst == src {
            return;
        }
        self.emit_sse_rr(0x66, 0x28, dst, src);
    }

    /// movq xmm, r64: 66 REX.W 0F 6E /r (reg=xmm, rm=gp)
    fn emit_movq_gp_to_xmm(&mut self, xmm: u32, gp_reg: u32) {
        self.push(0x66);
        self.push(Self::rex(true, xmm, 0, gp_reg));
        self.push(0x0F);
        self.push(0x6E);
        self.push(Self::modrm(3, xmm, gp_reg));
    }

    /// movq r64, xmm: 66 REX.W 0F 7E /r (reg=xmm, rm=gp)
    fn emit_movq_xmm_to_gp(&mut self, gp_reg: u32, xmm: u32) {
        self.push(0x66);
        self.push(Self::rex(true, xmm, 0, gp_reg));
        self.push(0x0F);
        self.push(0x7E);
        self.push(Self::modrm(3, xmm, gp_reg));
    }

    /// cvttsd2si r64, xmm: F2 REX.W 0F 2C /r
    fn emit_cvttsd2si(&mut self, gp_dst: u32, xmm_src: u32) {
        self.push(0xF2);
        self.push(Self::rex(true, gp_dst, 0, xmm_src));
        self.push(0x0F);
        self.push(0x2C);
        self.push(Self::modrm(3, gp_dst, xmm_src));
    }

    /// cvtsi2sd xmm, r64: F2 REX.W 0F 2A /r
    fn emit_cvtsi2sd(&mut self, xmm_dst: u32, gp_src: u32) {
        self.push(0xF2);
        self.push(Self::rex(true, xmm_dst, 0, gp_src));
        self.push(0x0F);
        self.push(0x2A);
        self.push(Self::modrm(3, xmm_dst, gp_src));
    }

    /// ucomisd xmm, xmm: 66 [REX] 0F 2E /r
    fn emit_ucomisd(&mut self, lhs: u32, rhs: u32) {
        self.emit_sse_rr(0x66, 0x2E, lhs, rhs);
    }

    /// movddup xmm, xmm (SSE3): F2 [REX] 0F 12 /r
    fn emit_movddup_rr(&mut self, dst: u32, src: u32) {
        self.emit_sse_rr(0xF2, 0x12, dst, src);
    }

    /// haddpd xmm, xmm (SSE3): 66 [REX] 0F 7C /r
    fn emit_haddpd_rr(&mut self, dst: u32, src: u32) {
        self.emit_sse_rr(0x66, 0x7C, dst, src);
    }

    /// movhlps xmm, xmm: [REX] 0F 12 /r
    fn emit_movhlps_rr(&mut self, dst: u32, src: u32) {
        self.emit_rex_if_needed(false, dst, 0, src);
        self.push(0x0F);
        self.push(0x12);
        self.push(Self::modrm(3, dst, src));
    }

    /// shufpd xmm, xmm, imm8: 66 [REX] 0F C6 /r ib
    fn emit_shufpd(&mut self, dst: u32, src: u32, imm: u8) {
        self.push(0x66);
        self.emit_rex_if_needed(false, dst, 0, src);
        self.push(0x0F);
        self.push(0xC6);
        self.push(Self::modrm(3, dst, src));
        self.push(imm);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // FMA3 (VEX-encoded)
    // ═════════════════════════════════════════════════════════════════════════

    /// VEX-encoded FMA scalar double: vf{op}213sd dst, src2, src3
    fn emit_fma213sd(&mut self, opcode: u8, dst: u32, src2: u32, src3: u32) {
        // VEX.LIG.66.0F38.W1: mmmmm=00010, pp=01, W=1, L=0
        self.emit_vex3(dst, 0, src3, 0b00010, true, src2, false, 0b01);
        self.push(opcode);
        self.push(Self::modrm(3, dst, src3));
    }

    /// VEX-encoded FMA packed double: vf{op}213pd dst, src2, src3
    fn emit_fma213pd(&mut self, opcode: u8, dst: u32, src2: u32, src3: u32, v256: bool) {
        self.emit_vex3(dst, 0, src3, 0b00010, true, src2, v256, 0b01);
        self.push(opcode);
        self.push(Self::modrm(3, dst, src3));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // AVX (VEX-encoded, for V256)
    // ═════════════════════════════════════════════════════════════════════════

    /// VEX 3-operand packed double: vop dst, lhs, rhs
    fn emit_avx_pd(&mut self, opcode: u8, dst: u32, lhs: u32, rhs: u32, v256: bool) {
        // VEX.{128|256}.66.0F.WIG: mmmmm=00001, pp=01, W=0
        self.emit_vex3(dst, 0, rhs, 0b00001, false, lhs, v256, 0b01);
        self.push(opcode);
        self.push(Self::modrm(3, dst, rhs));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Control flow
    // ═════════════════════════════════════════════════════════════════════════

    fn emit_jmp(&mut self, target: Label) {
        self.push(0xE9);
        self.emit_rel32_fixup(target);
    }

    fn emit_jcc(&mut self, cc: u8, target: Label) {
        self.push(0x0F);
        self.push(0x80 | cc);
        self.emit_rel32_fixup(target);
    }

    fn emit_call_ind(&mut self, reg: u32) {
        self.emit_rex_if_needed(false, 2, 0, reg);
        self.push(0xFF);
        self.push(Self::modrm(3, 2, reg));
    }

    fn emit_ret(&mut self) {
        self.push(0xC3);
    }

    fn emit_push(&mut self, reg: u32) {
        if reg >= 8 {
            self.push(0x41); // REX.B
        }
        self.push(0x50 + (reg & 7) as u8);
    }

    fn emit_pop(&mut self, reg: u32) {
        if reg >= 8 {
            self.push(0x41);
        }
        self.push(0x58 + (reg & 7) as u8);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // MachInst dispatch
    // ═════════════════════════════════════════════════════════════════════════

    fn emit_inst(&mut self, inst: &MachInst) -> Result<(), String> {
        use MachInst::*;
        match inst {
            // ── Data Movement ──
            LoadImm { dst, bits } => {
                self.emit_mov_imm64(gp(*dst), *bits);
            }

            LoadFpImm { dst, value } => {
                let bits = value.to_bits();
                self.emit_mov_imm64(SCRATCH_GP, bits);
                self.emit_movq_gp_to_xmm(fp(*dst), SCRATCH_GP);
            }

            Mov { dst, src } => {
                self.emit_mov_rr(gp(*dst), gp(*src));
            }

            FMov { dst, src } => {
                self.emit_movsd_rr(fp(*dst), fp(*src));
            }

            FuncArg { dst, index } => {
                // Move from ABI argument register to allocated dst register.
                // System V AMD64: rdi(7), rsi(6), rdx(2), rcx(1), r8(8), r9(9)
                const ARG_REGS: [u8; 6] = [7, 6, 2, 1, 8, 9];
                let s = ARG_REGS[*index as usize] as u32;
                self.emit_mov_rr(gp(*dst), s);
            }

            BitcastGpToFp { dst, src } => {
                self.emit_movq_gp_to_xmm(fp(*dst), gp(*src));
            }

            BitcastFpToGp { dst, src } => {
                self.emit_movq_xmm_to_gp(gp(*dst), fp(*src));
            }

            // ── Integer Arithmetic ──
            // x86_64 is 2-address: mov dst, lhs; op dst, rhs.
            IAdd { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_alu_rr(0x01, d, gp(*rhs));
            }

            IAddImm { dst, src, imm } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*src));
                self.emit_add_imm32(d, *imm);
            }

            ISub { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_alu_rr(0x29, d, gp(*rhs));
            }

            IMul { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_imul_rr(d, gp(*rhs));
            }

            IDiv { dst, lhs, rhs } => {
                let d = gp(*dst);
                let r = gp(*rhs);
                self.emit_mov_rr(0, gp(*lhs));
                self.emit_cqo();
                self.emit_alu_ext(0xF7, 7, r);
                self.emit_mov_rr(d, 0);
            }

            IMulSub { dst, lhs, rhs, acc } => {
                // dst = acc - lhs * rhs
                let d = gp(*dst);
                self.emit_mov_rr(SCRATCH_GP, gp(*lhs));
                self.emit_imul_rr(SCRATCH_GP, gp(*rhs));
                self.emit_mov_rr(d, gp(*acc));
                self.emit_alu_rr(0x29, d, SCRATCH_GP);
            }

            INeg { dst, src } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*src));
                self.emit_alu_ext(0xF7, 3, d);
            }

            // ── Bitwise ──
            And { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_alu_rr(0x21, d, gp(*rhs));
            }

            AndImm { dst, src, imm } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*src));
                self.emit_mov_imm64(SCRATCH_GP, *imm);
                self.emit_alu_rr(0x21, d, SCRATCH_GP);
            }

            Or { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_alu_rr(0x09, d, gp(*rhs));
            }

            OrImm { dst, src, imm } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*src));
                self.emit_mov_imm64(SCRATCH_GP, *imm);
                self.emit_alu_rr(0x09, d, SCRATCH_GP);
            }

            Xor { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_alu_rr(0x31, d, gp(*rhs));
            }

            Not { dst, src } => {
                let d = gp(*dst);
                self.emit_mov_rr(d, gp(*src));
                self.emit_alu_ext(0xF7, 2, d);
            }

            Shl { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(1, gp(*rhs)); // mov rcx, shift_amount
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_shift_cl(4, d);
            }

            Sar { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(1, gp(*rhs));
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_shift_cl(7, d);
            }

            Shr { dst, lhs, rhs } => {
                let d = gp(*dst);
                self.emit_mov_rr(1, gp(*rhs));
                self.emit_mov_rr(d, gp(*lhs));
                self.emit_shift_cl(5, d);
            }

            // ── FP Arithmetic (SSE2 scalar double) ──
            FAdd { dst, lhs, rhs } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*lhs));
                self.emit_sse_rr(0xF2, 0x58, d, fp(*rhs));
            }

            FSub { dst, lhs, rhs } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*lhs));
                self.emit_sse_rr(0xF2, 0x5C, d, fp(*rhs));
            }

            FMul { dst, lhs, rhs } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*lhs));
                self.emit_sse_rr(0xF2, 0x59, d, fp(*rhs));
            }

            FDiv { dst, lhs, rhs } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*lhs));
                self.emit_sse_rr(0xF2, 0x5E, d, fp(*rhs));
            }

            FNeg { dst, src } => {
                let d = fp(*dst);
                let s = fp(*src);
                self.emit_mov_imm64(SCRATCH_GP, 0x8000_0000_0000_0000);
                self.emit_movq_gp_to_xmm(SCRATCH_XMM, SCRATCH_GP);
                self.emit_movsd_rr(d, s);
                self.emit_sse_rr(0x66, 0x57, d, SCRATCH_XMM); // xorpd
            }

            FAbs { dst, src } => {
                // Clear sign bit: AND with 0x7FFFFFFFFFFFFFFF
                let d = fp(*dst);
                let s = fp(*src);
                self.emit_mov_imm64(SCRATCH_GP, 0x7FFF_FFFF_FFFF_FFFF);
                self.emit_movq_gp_to_xmm(SCRATCH_XMM, SCRATCH_GP);
                self.emit_movsd_rr(d, s);
                self.emit_sse_rr(0x66, 0x54, d, SCRATCH_XMM); // andpd
            }

            FSqrt { dst, src } => {
                // sqrtsd dst, src: F2 0F 51 /r
                let d = fp(*dst);
                self.emit_sse_rr(0xF2, 0x51, d, fp(*src));
            }

            FFloor { dst, src } => {
                // roundsd dst, src, 0x09 (floor + suppress precision)
                self.emit_roundsd(fp(*dst), fp(*src), 0x09);
            }

            FCeil { dst, src } => {
                // roundsd dst, src, 0x0A (ceil + suppress precision)
                self.emit_roundsd(fp(*dst), fp(*src), 0x0A);
            }

            FRound { dst, src } => {
                // roundsd dst, src, 0x08 (nearest even + suppress precision)
                self.emit_roundsd(fp(*dst), fp(*src), 0x08);
            }

            FTrunc { dst, src } => {
                // roundsd dst, src, 0x0B (truncate + suppress precision)
                self.emit_roundsd(fp(*dst), fp(*src), 0x0B);
            }

            FMin { dst, lhs, rhs } => {
                // minsd dst, rhs: F2 0F 5D /r
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*lhs));
                self.emit_sse_rr(0xF2, 0x5D, d, fp(*rhs));
            }

            FMax { dst, lhs, rhs } => {
                // maxsd dst, rhs: F2 0F 5F /r
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*lhs));
                self.emit_sse_rr(0xF2, 0x5F, d, fp(*rhs));
            }

            // ── FMA3 ──
            FMAdd { dst, a, b, c } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*a));
                self.emit_fma213sd(0xA9, d, fp(*b), fp(*c));
            }

            FMSub { dst, a, b, c } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*a));
                self.emit_fma213sd(0xAB, d, fp(*b), fp(*c));
            }

            FNMAdd { dst, a, b, c } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*a));
                self.emit_fma213sd(0xAD, d, fp(*b), fp(*c));
            }

            FNMSub { dst, a, b, c } => {
                let d = fp(*dst);
                self.emit_movsd_rr(d, fp(*a));
                self.emit_fma213sd(0xAF, d, fp(*b), fp(*c));
            }

            // ── Conversions ──
            FCvtToI64 { dst, src } => {
                self.emit_cvttsd2si(gp(*dst), fp(*src));
            }

            I64CvtToF { dst, src } => {
                self.emit_cvtsi2sd(fp(*dst), gp(*src));
            }

            // ── Comparison ──
            ICmp { lhs, rhs } => {
                self.emit_alu_rr(0x39, gp(*lhs), gp(*rhs));
            }

            ICmpImm { lhs, imm } => {
                self.emit_mov_imm64(SCRATCH_GP, *imm);
                self.emit_alu_rr(0x39, gp(*lhs), SCRATCH_GP);
            }

            FCmp { lhs, rhs } => {
                self.emit_ucomisd(fp(*lhs), fp(*rhs));
            }

            CSet { dst, cond } => {
                self.emit_setcc_full(cond_code(cond), gp(*dst));
            }

            // ── Memory ──
            Ldr { dst, mem } => {
                self.emit_load_mem(gp(*dst), gp(mem.base), mem.offset);
            }

            Str { src, mem } => {
                self.emit_store_mem(gp(mem.base), mem.offset, gp(*src));
            }

            FLdr { dst, mem } => {
                self.emit_sse_load(0xF2, 0x10, fp(*dst), gp(mem.base), mem.offset);
            }

            FStr { src, mem } => {
                self.emit_sse_store(0xF2, 0x11, gp(mem.base), mem.offset, fp(*src));
            }

            // ── Control Flow ──
            Jmp { target } => {
                self.emit_jmp(*target);
            }

            JmpIf { cond, target } => {
                self.emit_jcc(cond_code(cond), *target);
            }

            JmpZero { src, target } => {
                self.emit_test_rr(gp(*src), gp(*src));
                self.emit_jcc(cond_code(&Cond::Eq), *target);
            }

            JmpNonZero { src, target } => {
                self.emit_test_rr(gp(*src), gp(*src));
                self.emit_jcc(cond_code(&Cond::Ne), *target);
            }

            TestBitJmpZero { src, bit, target } => {
                self.emit_bt_imm(gp(*src), *bit);
                self.emit_jcc(cond_code(&Cond::AboveEq), *target); // jnc
            }

            TestBitJmpNonZero { src, bit, target } => {
                self.emit_bt_imm(gp(*src), *bit);
                self.emit_jcc(cond_code(&Cond::Below), *target); // jc
            }

            // ── Calls & Returns ──
            CallInd { target } => {
                self.emit_call_ind(gp(*target));
            }

            CallLabel { target } => {
                self.push(0xE8);
                self.emit_rel32_fixup(*target);
            }

            CallLocal { .. } => {
                return Err(
                    "CallLocal not yet linked — use CallLabel with ABI setup".to_string(),
                );
            }

            CallRuntime { .. } => {
                return Err(
                    "CallRuntime not yet linked — use CallInd with resolved address".to_string(),
                );
            }

            Ret => {
                self.emit_ret();
            }

            // ── Stack Frame ──
            Prologue { frame_size } => {
                self.emit_push(5); // push rbp
                self.emit_mov_rr(5, 4); // mov rbp, rsp
                if *frame_size > 0 {
                    let aligned = ((*frame_size + 15) & !15) as i32;
                    self.emit_sub_imm32(4, aligned);
                }
            }

            Epilogue { frame_size: _ } => {
                self.emit_mov_rr(4, 5); // mov rsp, rbp
                self.emit_pop(5); // pop rbp
            }

            Push { src } => {
                self.emit_push(gp(*src));
            }

            Pop { dst } => {
                self.emit_pop(gp(*dst));
            }

            StackAlloc { bytes } => {
                self.emit_sub_imm32(4, *bytes as i32); // sub rsp, bytes
            }

            StackFree { bytes } => {
                self.emit_add_imm32(4, *bytes as i32); // add rsp, bytes
            }

            // ── Pseudo-instructions ──
            DefLabel(l) => {
                self.define_label(*l);
            }

            Nop => {
                self.push(0x90);
            }

            Trap => {
                self.push(0xCC);
            }

            ParallelCopy { .. } => {
                return Err("ParallelCopy must be resolved before emission".to_string());
            }

            // ── SIMD V128 (SSE2 packed double, 2×f64) ──
            VLoad {
                dst,
                mem,
                width: VecWidth::V128,
            } => {
                self.emit_sse_load(0x66, 0x10, fp(*dst), gp(mem.base), mem.offset);
            }

            VStore {
                src,
                mem,
                width: VecWidth::V128,
            } => {
                self.emit_sse_store(0x66, 0x11, gp(mem.base), mem.offset, fp(*src));
            }

            VFAdd {
                dst,
                lhs,
                rhs,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*lhs));
                self.emit_sse_rr(0x66, 0x58, d, fp(*rhs));
            }

            VFSub {
                dst,
                lhs,
                rhs,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*lhs));
                self.emit_sse_rr(0x66, 0x5C, d, fp(*rhs));
            }

            VFMul {
                dst,
                lhs,
                rhs,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*lhs));
                self.emit_sse_rr(0x66, 0x59, d, fp(*rhs));
            }

            VFDiv {
                dst,
                lhs,
                rhs,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*lhs));
                self.emit_sse_rr(0x66, 0x5E, d, fp(*rhs));
            }

            VFNeg {
                dst,
                src,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_sse_rr(0x66, 0x57, SCRATCH_XMM, SCRATCH_XMM); // xorpd zero
                self.emit_sse_rr(0x66, 0x5C, SCRATCH_XMM, fp(*src)); // subpd
                self.emit_movapd_rr(d, SCRATCH_XMM);
            }

            VBroadcast {
                dst,
                src,
                width: VecWidth::V128,
            } => {
                self.emit_movddup_rr(fp(*dst), fp(*src));
            }

            VFMAdd {
                dst,
                a,
                b,
                c,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*a));
                self.emit_fma213pd(0xA8, d, fp(*b), fp(*c), false);
            }

            VExtractLane { dst, src, lane } => {
                let d = fp(*dst);
                let s = fp(*src);
                match lane {
                    0 => self.emit_movsd_rr(d, s),
                    1 => self.emit_movhlps_rr(d, s),
                    _ => return Err(format!("invalid lane {} for V128", lane)),
                }
            }

            VInsertLane {
                dst,
                src,
                lane,
                val,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*src));
                match lane {
                    0 => self.emit_movsd_rr(d, fp(*val)),
                    1 => self.emit_shufpd(d, fp(*val), 0),
                    _ => return Err(format!("invalid lane {} for V128", lane)),
                }
            }

            VReduceAdd {
                dst,
                src,
                width: VecWidth::V128,
            } => {
                let d = fp(*dst);
                self.emit_movapd_rr(d, fp(*src));
                self.emit_haddpd_rr(d, d);
            }

            // ── SIMD V256 (AVX packed double, 4×f64) ──
            VFAdd {
                dst,
                lhs,
                rhs,
                width: VecWidth::V256,
            } => {
                self.emit_avx_pd(0x58, fp(*dst), fp(*lhs), fp(*rhs), true);
            }

            VFSub {
                dst,
                lhs,
                rhs,
                width: VecWidth::V256,
            } => {
                self.emit_avx_pd(0x5C, fp(*dst), fp(*lhs), fp(*rhs), true);
            }

            VFMul {
                dst,
                lhs,
                rhs,
                width: VecWidth::V256,
            } => {
                self.emit_avx_pd(0x59, fp(*dst), fp(*lhs), fp(*rhs), true);
            }

            VFDiv {
                dst,
                lhs,
                rhs,
                width: VecWidth::V256,
            } => {
                self.emit_avx_pd(0x5E, fp(*dst), fp(*lhs), fp(*rhs), true);
            }

            VFMAdd {
                dst,
                a,
                b,
                c,
                width: VecWidth::V256,
            } => {
                self.emit_avx_pd(0x28, fp(*dst), fp(*a), fp(*a), true); // vmovapd
                self.emit_fma213pd(0xA8, fp(*dst), fp(*b), fp(*c), true);
            }

            VLoad {
                dst,
                mem,
                width: VecWidth::V256,
            } => {
                self.emit_vex3(fp(*dst), 0, gp(mem.base), 0b00001, false, 0, true, 0b01);
                self.push(0x10);
                self.emit_modrm_mem(fp(*dst), gp(mem.base), mem.offset);
            }

            VStore {
                src,
                mem,
                width: VecWidth::V256,
            } => {
                self.emit_vex3(fp(*src), 0, gp(mem.base), 0b00001, false, 0, true, 0b01);
                self.push(0x11);
                self.emit_modrm_mem(fp(*src), gp(mem.base), mem.offset);
            }

            VBroadcast {
                dst,
                src,
                width: VecWidth::V256,
            } => {
                // vbroadcastsd ymm, xmm: VEX.256.66.0F38.W0 19 /r
                self.emit_vex3(fp(*dst), 0, fp(*src), 0b00010, false, 0, true, 0b01);
                self.push(0x19);
                self.push(Self::modrm(3, fp(*dst), fp(*src)));
            }

            VFNeg {
                dst,
                src,
                width: VecWidth::V256,
            } => {
                self.emit_avx_pd(0x57, SCRATCH_XMM, SCRATCH_XMM, SCRATCH_XMM, true);
                self.emit_avx_pd(0x5C, fp(*dst), SCRATCH_XMM, fp(*src), true);
            }

            VReduceAdd {
                dst,
                src,
                width: VecWidth::V256,
            } => {
                let d = fp(*dst);
                let s = fp(*src);
                // vextractf128 xmm_scratch, ymm_src, 1
                self.emit_vex3(s, 0, SCRATCH_XMM, 0b00011, false, 0, true, 0b01);
                self.push(0x19);
                self.push(Self::modrm(3, s, SCRATCH_XMM));
                self.push(1);
                // addpd xmm_dst, xmm_scratch
                self.emit_movapd_rr(d, s);
                self.emit_sse_rr(0x66, 0x58, d, SCRATCH_XMM);
                // haddpd xmm_dst, xmm_dst
                self.emit_haddpd_rr(d, d);
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Emit x86_64 machine code from a MachFunc.
pub fn emit(func: &MachFunc) -> Result<EmittedCode, String> {
    let mut e = X64Emitter::new();
    for inst in &func.insts {
        e.emit_inst(inst)?;
    }
    e.resolve_fixups()?;
    Ok(EmittedCode { code: e.code })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{Cond, MachFunc, MachInst, Mem, VReg};
    use capstone::prelude::*;

    /// Disassemble x86_64 bytes into (mnemonic, operands) pairs.
    fn disasm(code: &[u8]) -> Vec<(String, String)> {
        let cs = Capstone::new()
            .x86()
            .mode(arch::x86::ArchMode::Mode64)
            .build()
            .expect("capstone init");
        let insns = cs.disasm_all(code, 0).expect("disassembly");
        insns
            .iter()
            .map(|i| {
                (
                    i.mnemonic().unwrap_or("?").to_string(),
                    i.op_str().unwrap_or("").to_string(),
                )
            })
            .collect()
    }

    /// Helper: build MachFunc with emitted instructions.
    fn compile(build: impl FnOnce(&mut MachFunc)) -> EmittedCode {
        let mut mf = MachFunc::new("test".to_string());
        build(&mut mf);
        emit(&mf).expect("emission failed")
    }

    /// Assert the nth instruction has given mnemonic and operand substring.
    fn assert_inst(asm: &[(String, String)], idx: usize, mnemonic: &str, op_contains: &str) {
        let (ref m, ref o) = asm[idx];
        assert_eq!(
            m, mnemonic,
            "instruction {}: expected mnemonic '{}', got '{}'",
            idx, mnemonic, m
        );
        assert!(
            o.contains(op_contains),
            "instruction {}: expected ops containing '{}', got '{}'",
            idx,
            op_contains,
            o
        );
    }

    // ── Individual encoding tests ──

    #[test]
    fn test_mov_imm64() {
        let mut e = X64Emitter::new();
        e.emit_mov_imm64(0, 42);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "movabs");
        assert!(asm[0].1.contains("rax"));
    }

    #[test]
    fn test_mov_rr() {
        let mut e = X64Emitter::new();
        e.emit_mov_rr(0, 3); // mov rax, rbx
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "mov");
        assert!(asm[0].1.contains("rax") && asm[0].1.contains("rbx"));
    }

    #[test]
    fn test_mov_rr_high_regs() {
        let mut e = X64Emitter::new();
        e.emit_mov_rr(8, 15); // mov r8, r15
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "mov");
        assert!(asm[0].1.contains("r8") && asm[0].1.contains("r15"));
    }

    #[test]
    fn test_add_rr() {
        let mut e = X64Emitter::new();
        e.emit_alu_rr(0x01, 0, 3);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "add");
        assert!(asm[0].1.contains("rax") && asm[0].1.contains("rbx"));
    }

    #[test]
    fn test_sub_rr() {
        let mut e = X64Emitter::new();
        e.emit_alu_rr(0x29, 6, 7);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "sub");
        assert!(asm[0].1.contains("rsi") && asm[0].1.contains("rdi"));
    }

    #[test]
    fn test_imul_rr() {
        let mut e = X64Emitter::new();
        e.emit_imul_rr(0, 1);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "imul");
        assert!(asm[0].1.contains("rax") && asm[0].1.contains("rcx"));
    }

    #[test]
    fn test_neg() {
        let mut e = X64Emitter::new();
        e.emit_alu_ext(0xF7, 3, 2);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "neg");
        assert!(asm[0].1.contains("rdx"));
    }

    #[test]
    fn test_and_or_xor() {
        let mut e = X64Emitter::new();
        e.emit_alu_rr(0x21, 0, 1);
        e.emit_alu_rr(0x09, 0, 2);
        e.emit_alu_rr(0x31, 0, 3);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "and");
        assert_eq!(asm[1].0, "or");
        assert_eq!(asm[2].0, "xor");
    }

    #[test]
    fn test_shift_cl() {
        let mut e = X64Emitter::new();
        e.emit_shift_cl(4, 0);
        e.emit_shift_cl(7, 0);
        e.emit_shift_cl(5, 0);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "shl");
        assert_eq!(asm[1].0, "sar");
        assert_eq!(asm[2].0, "shr");
    }

    #[test]
    fn test_cmp_and_setcc() {
        let mut e = X64Emitter::new();
        e.emit_alu_rr(0x39, 0, 1);
        e.emit_setcc_full(cond_code(&Cond::Lt), 0);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "cmp");
        assert_eq!(asm[1].0, "setl");
        assert_eq!(asm[2].0, "movzx");
    }

    #[test]
    fn test_movsd_rr() {
        let mut e = X64Emitter::new();
        e.emit_movsd_rr(0, 1);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "movsd");
        assert!(asm[0].1.contains("xmm0") && asm[0].1.contains("xmm1"));
    }

    #[test]
    fn test_addsd() {
        let mut e = X64Emitter::new();
        e.emit_sse_rr(0xF2, 0x58, 0, 1);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "addsd");
        assert!(asm[0].1.contains("xmm0") && asm[0].1.contains("xmm1"));
    }

    #[test]
    fn test_subsd_mulsd_divsd() {
        let mut e = X64Emitter::new();
        e.emit_sse_rr(0xF2, 0x5C, 2, 3);
        e.emit_sse_rr(0xF2, 0x59, 4, 5);
        e.emit_sse_rr(0xF2, 0x5E, 6, 7);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "subsd");
        assert_eq!(asm[1].0, "mulsd");
        assert_eq!(asm[2].0, "divsd");
    }

    #[test]
    fn test_movq_gp_xmm() {
        let mut e = X64Emitter::new();
        e.emit_movq_gp_to_xmm(0, 0);
        e.emit_movq_xmm_to_gp(0, 0);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "movq");
        assert_eq!(asm[1].0, "movq");
    }

    #[test]
    fn test_cvt_instructions() {
        let mut e = X64Emitter::new();
        e.emit_cvttsd2si(0, 0);
        e.emit_cvtsi2sd(0, 0);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "cvttsd2si");
        assert_eq!(asm[1].0, "cvtsi2sd");
    }

    #[test]
    fn test_ucomisd() {
        let mut e = X64Emitter::new();
        e.emit_ucomisd(0, 1);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "ucomisd");
    }

    #[test]
    fn test_jmp_and_label() {
        let mut e = X64Emitter::new();
        let lbl = Label(0);
        e.emit_jmp(lbl);
        e.push(0x90);
        e.define_label(lbl);
        e.emit_ret();
        e.resolve_fixups().unwrap();
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "jmp");
        assert_eq!(asm[1].0, "nop");
        assert_eq!(asm[2].0, "ret");
    }

    #[test]
    fn test_jcc() {
        let mut e = X64Emitter::new();
        let lbl = Label(0);
        e.emit_alu_rr(0x39, 0, 1);
        e.emit_jcc(cond_code(&Cond::Lt), lbl);
        e.emit_ret();
        e.define_label(lbl);
        e.emit_ret();
        e.resolve_fixups().unwrap();
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "cmp");
        assert_eq!(asm[1].0, "jl");
    }

    #[test]
    fn test_push_pop() {
        let mut e = X64Emitter::new();
        e.emit_push(5);
        e.emit_pop(5);
        e.emit_push(12);
        e.emit_pop(12);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "push");
        assert!(asm[0].1.contains("rbp"));
        assert_eq!(asm[1].0, "pop");
        assert!(asm[1].1.contains("rbp"));
        assert_eq!(asm[2].0, "push");
        assert!(asm[2].1.contains("r12"));
        assert_eq!(asm[3].0, "pop");
        assert!(asm[3].1.contains("r12"));
    }

    #[test]
    fn test_call_ind() {
        let mut e = X64Emitter::new();
        e.emit_call_ind(0);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "call");
        assert!(asm[0].1.contains("rax"));
    }

    #[test]
    fn test_ret() {
        let mut e = X64Emitter::new();
        e.emit_ret();
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "ret");
    }

    #[test]
    fn test_bt_imm() {
        let mut e = X64Emitter::new();
        e.emit_bt_imm(0, 63);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "bt");
    }

    #[test]
    fn test_nop_trap() {
        let code = compile(|mf| {
            mf.emit(MachInst::Nop);
            mf.emit(MachInst::Trap);
        });
        let asm = disasm(code.bytes());
        assert_eq!(asm[0].0, "nop");
        assert_eq!(asm[1].0, "int3");
    }

    // ── SSE2 packed / SIMD ──

    #[test]
    fn test_addpd() {
        let mut e = X64Emitter::new();
        e.emit_sse_rr(0x66, 0x58, 0, 1);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "addpd");
    }

    #[test]
    fn test_movddup() {
        let mut e = X64Emitter::new();
        e.emit_movddup_rr(0, 1);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "movddup");
    }

    #[test]
    fn test_haddpd() {
        let mut e = X64Emitter::new();
        e.emit_haddpd_rr(0, 0);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "haddpd");
    }

    // ── FMA3 ──

    #[test]
    fn test_vfmadd213sd() {
        let mut e = X64Emitter::new();
        e.emit_fma213sd(0xA9, 0, 1, 2);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "vfmadd213sd");
        assert!(asm[0].1.contains("xmm0"));
    }

    #[test]
    fn test_vfmsub213sd() {
        let mut e = X64Emitter::new();
        e.emit_fma213sd(0xAB, 0, 1, 2);
        let asm = disasm(&e.code);
        assert_eq!(asm[0].0, "vfmsub213sd");
    }

    // ── Full MachInst sequence tests ──

    #[test]
    fn test_return_constant_sequence() {
        let code = compile(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 42,
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert_inst(&asm, 0, "movabs", "rax");
        assert_eq!(asm.last().unwrap().0, "ret");
    }

    #[test]
    fn test_integer_add_sequence() {
        let code = compile(|mf| {
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
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert_inst(&asm, 0, "movabs", "rax");
        assert_inst(&asm, 1, "movabs", "rcx");
        let add_idx = asm.iter().position(|(m, _)| m == "add").expect("no add");
        assert!(asm[add_idx].1.contains("rax") && asm[add_idx].1.contains("rcx"));
        assert_eq!(asm.last().unwrap().0, "ret");
    }

    #[test]
    fn test_f64_add_sequence() {
        let code = compile(|mf| {
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
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        let has_addsd = asm.iter().any(|(m, _)| m == "addsd");
        assert!(has_addsd, "expected addsd in: {:?}", asm);
        assert_eq!(asm.last().unwrap().0, "ret");
    }

    #[test]
    fn test_prologue_epilogue() {
        let code = compile(|mf| {
            mf.emit(MachInst::Prologue { frame_size: 32 });
            mf.emit(MachInst::Epilogue { frame_size: 32 });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert_inst(&asm, 0, "push", "rbp");
        assert_inst(&asm, 1, "mov", "rbp");
        assert_inst(&asm, 2, "sub", "rsp");
        assert_eq!(asm.last().unwrap().0, "ret");
    }

    #[test]
    fn test_branch_loop_sequence() {
        let code = compile(|mf| {
            let l_loop = mf.new_label();
            let l_end = mf.new_label();

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
            mf.emit(MachInst::DefLabel(l_loop));
            mf.emit(MachInst::ICmp {
                lhs: VReg::gp(1),
                rhs: VReg::gp(2),
            });
            mf.emit(MachInst::JmpIf {
                cond: Cond::Gt,
                target: l_end,
            });
            mf.emit(MachInst::IAdd {
                dst: VReg::gp(0),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::IAddImm {
                dst: VReg::gp(1),
                src: VReg::gp(1),
                imm: 1,
            });
            mf.emit(MachInst::Jmp { target: l_loop });
            mf.emit(MachInst::DefLabel(l_end));
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert!(asm.iter().any(|(m, _)| m == "cmp"), "expected cmp");
        assert!(asm.iter().any(|(m, _)| m == "jg"), "expected jg");
        assert!(asm.iter().any(|(m, _)| m == "jmp"), "expected jmp");
        assert_eq!(asm.last().unwrap().0, "ret");
    }

    #[test]
    fn test_fneg_sequence() {
        let code = compile(|mf| {
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 42.0,
            });
            mf.emit(MachInst::FNeg {
                dst: VReg::fp(0),
                src: VReg::fp(0),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        let has_xorpd = asm.iter().any(|(m, _)| m == "xorpd");
        assert!(has_xorpd, "expected xorpd for FNeg: {:?}", asm);
    }

    #[test]
    fn test_bitwise_sequence() {
        let code = compile(|mf| {
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: 0xFF00,
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(1),
                bits: 0x0FF0,
            });
            mf.emit(MachInst::And {
                dst: VReg::gp(0),
                lhs: VReg::gp(0),
                rhs: VReg::gp(1),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert!(
            asm.iter().any(|(m, _)| m == "and"),
            "expected and: {:?}",
            asm
        );
    }

    #[test]
    fn test_fma_sequence() {
        let code = compile(|mf| {
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
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        let has_fma = asm.iter().any(|(m, _)| m == "vfmadd213sd");
        assert!(has_fma, "expected vfmadd213sd: {:?}", asm);
    }

    #[test]
    fn test_cset_all_conditions() {
        for cond in &[
            Cond::Eq,
            Cond::Ne,
            Cond::Lt,
            Cond::Le,
            Cond::Gt,
            Cond::Ge,
            Cond::Below,
            Cond::BelowEq,
            Cond::Above,
            Cond::AboveEq,
        ] {
            let code = compile(|mf| {
                mf.emit(MachInst::ICmp {
                    lhs: VReg::gp(0),
                    rhs: VReg::gp(1),
                });
                mf.emit(MachInst::CSet {
                    dst: VReg::gp(0),
                    cond: *cond,
                });
                mf.emit(MachInst::Ret);
            });
            let asm = disasm(code.bytes());
            let has_set = asm.iter().any(|(m, _)| m.starts_with("set"));
            assert!(has_set, "expected setcc for {:?}: {:?}", cond, asm);
        }
    }

    #[test]
    fn test_load_store_mem() {
        let code = compile(|mf| {
            mf.emit(MachInst::Ldr {
                dst: VReg::gp(0),
                mem: Mem::new(VReg::gp(1), 8),
            });
            mf.emit(MachInst::Str {
                src: VReg::gp(0),
                mem: Mem::new(VReg::gp(1), 16),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert_eq!(asm[0].0, "mov");
        assert!(asm[0].1.contains("rax") && asm[0].1.contains("rcx"));
        assert_eq!(asm[1].0, "mov");
    }

    #[test]
    fn test_conversion_sequence() {
        let code = compile(|mf| {
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 3.7,
            });
            mf.emit(MachInst::FCvtToI64 {
                dst: VReg::gp(0),
                src: VReg::fp(0),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        let has_cvt = asm.iter().any(|(m, _)| m == "cvttsd2si");
        assert!(has_cvt, "expected cvttsd2si: {:?}", asm);
    }

    // ── Interpreter cross-check tests ──

    #[test]
    fn test_interp_crosscheck_add_num() {
        use crate::intern::Interner;
        use crate::mir::interp::{eval, InterpValue};
        use crate::mir::{Instruction, MirFunction, Terminator};
        use crate::runtime::value::Value;

        // MIR: return 10 + 32 (boxed num add)
        let mut interner = Interner::new();
        let name = interner.intern("test_add");
        let mut func = MirFunction::new(name, 0);
        let bb = func.new_block();
        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();
        func.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(10.0)));
        func.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(32.0)));
        func.block_mut(bb)
            .instructions
            .push((v2, Instruction::Add(v0, v1)));
        func.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&func).unwrap();
        match result {
            InterpValue::Boxed(v) => {
                assert_eq!(v.as_num(), Some(42.0), "interpreter: 10+32 should be 42");
            }
            _ => panic!("expected boxed value from interpreter"),
        }

        // Equivalent x86_64 (unbox, addsd, rebox)
        let code = compile(|mf| {
            let v10_bits = Value::num(10.0).to_bits();
            let v32_bits = Value::num(32.0).to_bits();
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: v10_bits,
            });
            mf.emit(MachInst::BitcastGpToFp {
                dst: VReg::fp(0),
                src: VReg::gp(0),
            });
            mf.emit(MachInst::LoadImm {
                dst: VReg::gp(0),
                bits: v32_bits,
            });
            mf.emit(MachInst::BitcastGpToFp {
                dst: VReg::fp(1),
                src: VReg::gp(0),
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(0),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::BitcastFpToGp {
                dst: VReg::gp(0),
                src: VReg::fp(0),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        let has_addsd = asm.iter().any(|(m, _)| m == "addsd");
        let movq_count = asm.iter().filter(|(m, _)| m == "movq").count();
        assert!(has_addsd, "expected addsd: {:?}", asm);
        assert!(movq_count >= 3, "expected at least 3 movq: {:?}", asm);
    }

    #[test]
    fn test_interp_crosscheck_mul_sub() {
        use crate::intern::Interner;
        use crate::mir::interp::{eval, InterpValue};
        use crate::mir::{Instruction, MirFunction, Terminator};

        // MIR: (10 * 3) - 5 = 25
        let mut interner = Interner::new();
        let name = interner.intern("test_mulsub");
        let mut func = MirFunction::new(name, 0);
        let bb = func.new_block();
        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();
        let v3 = func.new_value();
        let v4 = func.new_value();
        func.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(10.0)));
        func.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(3.0)));
        func.block_mut(bb)
            .instructions
            .push((v2, Instruction::ConstNum(5.0)));
        func.block_mut(bb)
            .instructions
            .push((v3, Instruction::Mul(v0, v1)));
        func.block_mut(bb)
            .instructions
            .push((v4, Instruction::Sub(v3, v2)));
        func.block_mut(bb).terminator = Terminator::Return(v4);

        let result = eval(&func).unwrap();
        match result {
            InterpValue::Boxed(v) => assert_eq!(v.as_num(), Some(25.0)),
            _ => panic!("expected boxed"),
        }

        // x86_64: unboxed f64 path
        let code = compile(|mf| {
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
                dst: VReg::fp(0),
                lhs: VReg::fp(3),
                rhs: VReg::fp(2),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert!(
            asm.iter().any(|(m, _)| m == "mulsd"),
            "expected mulsd: {:?}",
            asm
        );
        assert!(
            asm.iter().any(|(m, _)| m == "subsd"),
            "expected subsd: {:?}",
            asm
        );
    }

    #[test]
    fn test_interp_crosscheck_comparison() {
        use crate::intern::Interner;
        use crate::mir::interp::{eval, InterpValue};
        use crate::mir::{Instruction, MirFunction, Terminator};

        // MIR: 10 < 20 → true
        let mut interner = Interner::new();
        let name = interner.intern("test_cmp");
        let mut func = MirFunction::new(name, 0);
        let bb = func.new_block();
        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();
        func.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(10.0)));
        func.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(20.0)));
        func.block_mut(bb)
            .instructions
            .push((v2, Instruction::CmpLt(v0, v1)));
        func.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&func).unwrap();
        match result {
            InterpValue::Boxed(v) => assert_eq!(v.as_bool(), Some(true)),
            _ => panic!("expected boxed bool"),
        }

        // x86_64: ucomisd + setb (unsigned below for FP <)
        let code = compile(|mf| {
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 10.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 20.0,
            });
            mf.emit(MachInst::FCmp {
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::CSet {
                dst: VReg::gp(0),
                cond: Cond::Below,
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert!(asm.iter().any(|(m, _)| m == "ucomisd"), "expected ucomisd");
        assert!(asm.iter().any(|(m, _)| m == "setb"), "expected setb");
    }

    #[test]
    fn test_interp_crosscheck_unboxed_f64() {
        use crate::intern::Interner;
        use crate::mir::interp::{eval, InterpValue};
        use crate::mir::{Instruction, MirFunction, Terminator};

        // MIR: ConstF64(2.0) + ConstF64(3.0) = 5.0
        let mut interner = Interner::new();
        let name = interner.intern("test_f64");
        let mut func = MirFunction::new(name, 0);
        let bb = func.new_block();
        let v0 = func.new_value();
        let v1 = func.new_value();
        let v2 = func.new_value();
        func.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstF64(2.0)));
        func.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstF64(3.0)));
        func.block_mut(bb)
            .instructions
            .push((v2, Instruction::AddF64(v0, v1)));
        func.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&func).unwrap();
        match result {
            InterpValue::F64(v) => assert_eq!(v, 5.0),
            _ => panic!("expected F64"),
        }

        // x86_64: addsd directly
        let code = compile(|mf| {
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 2.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 3.0,
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(0),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert!(asm.iter().any(|(m, _)| m == "addsd"), "expected addsd");
    }

    #[test]
    fn test_interp_crosscheck_loop_sum() {
        use crate::intern::Interner;
        use crate::mir::interp::{eval, InterpValue};
        use crate::mir::{Instruction, MirFunction, MirType, Terminator};

        // MIR: sum 1..5 via loop with block params = 15
        let mut interner = Interner::new();
        let name = interner.intern("sum_loop");
        let mut func = MirFunction::new(name, 0);
        let entry = func.new_block();
        let loop_block = func.new_block();
        let exit_block = func.new_block();

        // entry: goto loop(0.0, 1.0)
        let v_zero = func.new_value();
        let v_one = func.new_value();
        func.block_mut(entry)
            .instructions
            .push((v_zero, Instruction::ConstF64(0.0)));
        func.block_mut(entry)
            .instructions
            .push((v_one, Instruction::ConstF64(1.0)));
        func.block_mut(entry).terminator = Terminator::Branch {
            target: loop_block,
            args: vec![v_zero, v_one],
        };

        // loop(sum, i): if i > 5 goto exit(sum) else loop(sum+i, i+1)
        let bp_sum = func.new_value();
        let bp_i = func.new_value();
        func.block_mut(loop_block)
            .params
            .push((bp_sum, MirType::F64));
        func.block_mut(loop_block).params.push((bp_i, MirType::F64));
        let v_five = func.new_value();
        let v_cmp = func.new_value();
        let v_sum2 = func.new_value();
        let v_inc = func.new_value();
        let v_i2 = func.new_value();
        func.block_mut(loop_block)
            .instructions
            .push((v_five, Instruction::ConstF64(5.0)));
        func.block_mut(loop_block)
            .instructions
            .push((v_cmp, Instruction::CmpGtF64(bp_i, v_five)));
        func.block_mut(loop_block)
            .instructions
            .push((v_sum2, Instruction::AddF64(bp_sum, bp_i)));
        func.block_mut(loop_block)
            .instructions
            .push((v_inc, Instruction::ConstF64(1.0)));
        func.block_mut(loop_block)
            .instructions
            .push((v_i2, Instruction::AddF64(bp_i, v_inc)));
        func.block_mut(loop_block).terminator = Terminator::CondBranch {
            condition: v_cmp,
            true_target: exit_block,
            true_args: vec![bp_sum],
            false_target: loop_block,
            false_args: vec![v_sum2, v_i2],
        };

        // exit(result): return
        let bp_result = func.new_value();
        func.block_mut(exit_block)
            .params
            .push((bp_result, MirType::F64));
        func.block_mut(exit_block).terminator = Terminator::Return(bp_result);

        let result = eval(&func).unwrap();
        match result {
            InterpValue::F64(v) => assert_eq!(v, 15.0, "sum 1..5 = 15"),
            _ => panic!("expected F64, got {:?}", result),
        }

        // Verify x86_64 loop structure
        let code = compile(|mf| {
            let l_loop = mf.new_label();
            let l_end = mf.new_label();
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(0),
                value: 0.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(1),
                value: 1.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(2),
                value: 5.0,
            });
            mf.emit(MachInst::LoadFpImm {
                dst: VReg::fp(3),
                value: 1.0,
            });
            mf.emit(MachInst::DefLabel(l_loop));
            mf.emit(MachInst::FCmp {
                lhs: VReg::fp(1),
                rhs: VReg::fp(2),
            });
            mf.emit(MachInst::JmpIf {
                cond: Cond::Above,
                target: l_end,
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(0),
                lhs: VReg::fp(0),
                rhs: VReg::fp(1),
            });
            mf.emit(MachInst::FAdd {
                dst: VReg::fp(1),
                lhs: VReg::fp(1),
                rhs: VReg::fp(3),
            });
            mf.emit(MachInst::Jmp { target: l_loop });
            mf.emit(MachInst::DefLabel(l_end));
            mf.emit(MachInst::Ret);
        });
        let asm = disasm(code.bytes());
        assert!(asm.iter().any(|(m, _)| m == "ucomisd"), "expected ucomisd");
        assert!(
            asm.iter().filter(|(m, _)| m == "addsd").count() >= 2,
            "expected >=2 addsd"
        );
        assert!(asm.iter().any(|(m, _)| m == "jmp"), "expected jmp");
    }
}
