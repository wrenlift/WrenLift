# Multi-Target Code Generation

This module lowers MIR (mid-level IR) to executable code for three targets: x86_64, AArch64, and WebAssembly. Native targets share a common platform-agnostic machine IR (`MachInst`) and linear-scan register allocator; the WASM target bypasses both and emits directly from MIR.

## Platform-Agnostic Machine IR

Defines the core IR and compilation pipeline.

**Key types:**
- `VReg` -- Virtual register with a `RegClass` (Gp, Fp, Vec) and unique index. Resolved to physical registers by the allocator.
- `MachInst` -- RISC-style 3-address instruction enum. Load/store architecture; explicit flag materialization via `CSet`; no implicit flags register. Covers data movement, integer/FP arithmetic, FMA, SIMD (V128/V256 packed f64), comparisons, branches, calls, stack frame, and pseudo-instructions (`DefLabel`, `ParallelCopy`).
- `MachFunc` -- A function body: flat `Vec<MachInst>`, frame size, and VReg/label counters.
- `PhysReg` -- Hardware register encoding. Physical register maps for both targets defined in `phys_aarch64` and `phys_x86_64` submodules.
- `AbiInfo` -- Calling convention descriptor (argument registers, callee/caller-saved sets, frame/stack pointer, link register).
- `Target` -- Enum: `X86_64`, `Aarch64`, `Wasm`.

**Key functions:**
- `lower_mir(mir) -> MachFunc` -- Walks MIR blocks and instructions, maps `ValueId` to `VReg`, emits `MachInst` sequences including parallel copies for block parameter binding (SSA phi resolution).
- `compile_function(mir, target) -> CompiledFunction` -- Full pipeline entry point. For native targets: `lower_mir` -> `resolve_parallel_copies` -> `linear_scan` regalloc -> `fixup_sentinels` -> target-specific `emit`. For WASM: calls `wasm::emit_mir` directly.
- `resolve_parallel_copies(mf)` -- Expands `ParallelCopy` pseudo-instructions into sequential `Mov`/`FMov` with cycle-breaking temporaries.
- `fixup_sentinels(mach, target)` -- Rewrites sentinel VReg indices (frame pointer = `u32::MAX`, spill scratch = `u32::MAX - 1`) to actual hardware encodings.

## Linear-Scan Register Allocator

Transforms a `MachFunc` from virtual to physical registers.

**Algorithm:**
1. `compute_live_intervals` -- Forward scan of instructions, recording `[first_def, last_use)` half-open intervals per VReg, sorted by start point.
2. `linear_scan` -- Walks intervals in order. Maintains a free-register pool (`BTreeSet` for deterministic allocation) and an active-intervals list sorted by end point. When no register is free, spills the interval with the longest remaining range.
3. `apply_allocation` -- Rewrites all VRegs to PhysRegs. Inserts `Ldr`/`FLdr` reloads before uses and `Str`/`FStr` spills after defs for spilled intervals. Updates prologue/epilogue frame sizes to include spill slots (8-byte aligned).

**Target register sets:**
- `aarch64_target_regs` -- x0-x15 GP, d0-d15 FP (x16/x17 scratch, x29 FP, x30 LR, x31 SP excluded).
- `x86_64_target_regs` -- All GP except RSP(4), RBP(5), R11(11=scratch); XMM0-XMM14 (XMM15 scratch).

`allocate_registers` combines parallel copy resolution, linear scan, and rewriting into a single call.

## x86_64 Code Emission

Manual byte-level encoding -- no external assembler. This allows cross-compilation from ARM hosts.

**Encoding infrastructure:**
- `X64Emitter` -- Internal builder that accumulates a `Vec<u8>` byte buffer.
- REX prefix generation (`rex`, `emit_rex_if_needed`) for 64-bit operands and extended registers (R8-R15, XMM8-XMM15).
- `ModR/M` + optional SIB + displacement encoding (`emit_modrm_mem`) with special handling for RSP/R12 (SIB required) and RBP/R13 (no zero-displacement mode).
- VEX 3-byte prefix (`emit_vex3`) for FMA3/AVX instructions.
- Label/fixup system: rel32 placeholders resolved after full emission.

**Instruction categories:** GP ALU (`emit_alu_rr`, `emit_alu_ext`), shifts via CL, SSE2 scalar/packed FP (`emit_sse_rr`, `movsd`, `ucomisd`, `cvttsd2si`, `cvtsi2sd`), FMA3 scalar (`emit_fma213sd`), AVX packed operations.

**JIT execution:** `EmittedCode::make_executable()` copies the byte buffer into an `mmap`'d `MutableBuffer` (via dynasmrt), flips to executable via `mprotect`, and returns an `ExecutableCode` handle with `unsafe fn as_fn<F>()`.

Scratch registers: R11 (GP), XMM15 (FP) -- reserved, never assigned by the allocator.

## AArch64 Code Emission

Uses `dynasmrt::aarch64::Assembler` for instruction encoding via the `dynasm!` macro.

**Structure:**
- Pre-allocates `DynamicLabel`s by scanning for all `DefLabel`, `Jmp*`, and `TestBit*` instructions.
- `emit_inst` maps each `MachInst` variant to the corresponding ARM64 instruction: `add`/`sub`/`mul`/`sdiv`/`msub` for integer; `fadd`/`fsub`/`fmul`/`fdiv`/`fmadd`/`fmsub` for FP; `cmp`/`cset`/`cbz`/`cbnz`/`tbz`/`tbnz` for comparisons and branches; `ldr`/`str` via `ld1`/`st1` for NEON vectors.
- Address computation uses x17 (IP1) as scratch when a memory operand has a non-zero offset.
- FP immediates loaded via GP scratch x16 (IP0) then `fmov` transfer.
- 64-bit immediates use a `movz`/`movk` sequence (up to 4 halfwords).

**Output:** `CompiledCode` wrapping a dynasmrt `ExecutableBuffer` with `unsafe fn as_fn<F>()`.

## Control Flow Graph Analysis

Extracts CFG structure from a `MachFunc`'s flat instruction stream.

**CFG construction (`Cfg::build`):**
1. Identifies block boundaries at `DefLabel` markers and post-terminator fall-through points.
2. Computes successor edges from branch targets and implicit fall-through.
3. Derives predecessor edges by inverting the successor relation.

**Analysis passes:**
- `rpo()` -- Reverse post-order via iterative DFS. Used by dominator computation and forward dataflow.
- `dominators()` -- Immediate dominator tree using the Cooper-Harvey-Kennedy iterative algorithm. Returns `idom[i]` for each block; entry block dominates itself; unreachable blocks get `usize::MAX`.
- `dominates(idom, a, b)` -- Walk-up check on the dominator tree.
- `dominator_tree(idom)` -- Children lists for tree traversal.
- `detect_loops()` -- Finds natural loops by identifying back edges (successor where target dominates source). Computes loop body via reverse reachability from latch to header.
- `loop_depths()` / `loop_header_set()` -- Per-block nesting depth and header identification.

**Consumers:** WASM emitter (structured control flow conversion), register allocator (liveness), optimization scheduling.

## Direct MIR-to-WASM Emission

Compiles MIR directly to a WASM binary module, bypassing `MachInst` and register allocation entirely. This is natural because MIR's ValueIds map to WASM locals, block parameters avoid phi nodes, and structured terminators map to WASM's block/loop/br.

**Emitter phases (`MirWasmEmitter::emit`):**
1. `scan_locals` -- Allocates a WASM local for each MIR ValueId and block parameter. Types: `i64` for NaN-boxed Values/integers, `f64` for unboxed doubles, `i32` for booleans.
2. `scan_imports` -- Discovers runtime functions needed by MIR instructions (arithmetic, comparisons, field access, closures, etc.) and registers them as imports from the `"wren"` host module.
3. Module assembly via `wasm_encoder`: type section (import + function types), import section (`"wren"` module), function/export sections, code section.

**Structured control flow (reverse-nested-blocks):**
For N MIR blocks, opens N-1 `block` (or `loop` for loop headers) scopes in reverse order. Block k's code sits inside scopes for blocks k+1 through N-1. Forward branches use `br depth` where depth = target - source - 1. Loop headers emit `loop` instead of `block` to enable back-edge branching.

**Instruction mapping:**
- Unboxed f64 operations -> single WASM f64 instructions (add, sub, mul, div, neg, comparisons).
- NaN-boxed Value operations (Add, Sub, CmpLt, etc.) -> `call` to imported runtime functions (`wren_num_add`, `wren_cmp_lt`, etc.).
- Box/Unbox -> `i64.reinterpret_f64` / `f64.reinterpret_i64`.
- `CondBranch` -> `call wren_is_truthy` + `br_if`.
- All compiled functions export as `fn_{index}` and return `i64`.

## Compilation Pipelines

### Native (x86_64 / AArch64)

```
MIR -> lower_mir() -> MachFunc (VRegs)
    -> resolve_parallel_copies()
    -> linear_scan() -> RegAllocResult
    -> apply_allocation() -> MachFunc (PhysRegs + spill/reload)
    -> fixup_sentinels()
    -> x86_64::emit() or aarch64::emit()
    -> EmittedCode / CompiledCode
    -> make_executable() -> callable function pointer
```

### WASM

```
MIR -> wasm::emit_mir() -> WasmModule (binary bytes)
```

Bypasses MachInst, register allocation, and the entire native pipeline. MIR ValueIds become WASM locals; runtime operations become imports from the `"wren"` host module.
