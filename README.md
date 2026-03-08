<p align="center">
<img style="display: block;" src="wrenlift_logo.png" alt="Wren Lift Logo" width="250"/>
</p>

<h1 align="center">WrenLift</h1>

<p align="center">
Lightning fast JIT runtime for the <a href="https://wren.io">Wren</a> programming language.
</p>

<p align="center">
<img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust"/>
<img src="https://img.shields.io/badge/edition-2021-blue" alt="Rust 2021"/>
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
<img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version 0.1.0"/>
<img src="https://img.shields.io/badge/tests-556_passing-brightgreen" alt="556 tests passing"/>
<img src="https://img.shields.io/badge/targets-x86__64_%7C_aarch64_%7C_WASM-purple" alt="x86_64 | aarch64 | WASM"/>
</p>

---

WrenLift replaces Wren's stack-based bytecode interpreter with a modern compilation pipeline. Source code is tokenized by a logos-based lexer (~60 token variants), then parsed by a recursive-descent parser into a fully spanned AST. Semantic analysis performs name/scope resolution and speculative type inference over the AST. The analyzed program is lowered into an SSA-based mid-level IR (MIR) using block parameters (Cranelift/MLIR-style, not phi nodes). Seven optimization passes run over the MIR before code generation:

- **Constant folding** -- evaluates constant expressions at compile time, folds branches on known conditions, propagates constants through moves and box/unbox boundaries
- **Dead code elimination** -- removes unused pure instructions and unreachable blocks, preserves side-effectful operations
- **Common subexpression elimination** -- global value numbering that deduplicates identical pure computations, respects commutativity
- **Type specialization** -- devirtualizes boxed arithmetic when operand types are known (e.g. two Nums), converting `Add` into `GuardNum` + `Unbox` + `AddF64` + `Box` for native-speed floating point
- **Escape analysis** -- identifies heap allocations that never leave the function (not returned, not stored to fields, not captured by escaping closures)
- **Scalar replacement of aggregates** -- replaces non-escaping fixed-size lists with individual SSA values when all accesses use constant indices
- **Loop-invariant code motion** -- detects natural loops via dominance-based back-edge analysis, inserts preheader blocks, and hoists loop-invariant computations out of the loop body using fixpoint iteration

The optimized MIR is then compiled to native machine code (x86_64 via manual byte-level encoding, aarch64 via dynasmrt) or WebAssembly (direct MIR-to-WASM emission, bypassing the machine IR layer entirely). A linear-scan register allocator handles register assignment and spilling for native targets.

The runtime uses NaN-boxed 64-bit values, a generational semi-space garbage collector with bump-pointer nursery allocation, and an object model with O(1) method dispatch via interned symbol IDs. A planned tiered execution model will use the MIR interpreter for fast cold starts, with invocation counters promoting hot functions to optimized JIT compilation -- eliminating upfront compilation latency while still reaching peak performance on critical paths. Hot module reload will allow recompiling changed source files at runtime, patching the code cache and re-resolving module-level bindings without restarting the VM.

## Getting Started

### Build

```sh
cargo build --release
```

### Run a script

```sh
wlift script.wren
```

### Start the REPL

```sh
wlift
```

### Compile to WebAssembly

```sh
wlift --target=wasm script.wren -o output.wasm
```

### Debug dumps

```sh
wlift --dump-tokens script.wren    # Show lexer output
wlift --dump-ast script.wren       # Show parsed AST
wlift --dump-mir script.wren       # Show MIR before optimization
wlift --dump-opt script.wren       # Show MIR after optimization
wlift --dump-asm script.wren       # Show generated machine code
wlift --no-opt script.wren         # Run without optimization passes
wlift --gc-stats script.wren       # Print GC statistics after execution
```

## Verification & Debugging

WrenLift uses a layered approach to correctness across the compiler and runtime.

**MIR interpreter as optimization oracle.** A dedicated interpreter executes MIR functions directly, stepping through blocks and following terminators with a configurable step limit to catch infinite loops. Every optimization pass is tested by asserting `eval(f) == eval(optimize(f))` -- if constant folding, DCE, CSE, type specialization, escape analysis, SRA, or LICM changes a program's result, the test fails. This catches miscompilations without needing the full Wren VM.

**WASM structural validation.** Emitted WebAssembly modules are validated through wasmparser (full structural and type validation) before any execution. When validation fails, the module is automatically disassembled to WAT text via wasmprinter and included in the error output, making it straightforward to locate the malformed instruction. Integration tests go further and execute the validated WASM through wasmtime, confirming the emitted code actually computes correct results.

**Native code disassembly verification.** x86_64 JIT output can be disassembled via capstone in tests, allowing inspection of the generated instruction stream against expected encodings. The manual byte-level encoder (no external assembler dependency) is tested against known instruction patterns for REX prefixes, ModR/M encoding, SIB bytes, and VEX3 prefixes.

**GC safety.** The generational garbage collector is tested for pointer integrity after promotion (nursery to old generation), write barrier correctness (old-to-young references tracked in the remembered set), forwarding table accuracy, and self-referential pointer fixup (closed upvalues whose `location` field points into their own struct). String interning is tested for deduplication, collection of unreachable interned strings, and pointer equality after interning.

**Diagnostic-driven error reporting.** The lexer and parser never panic on malformed input. Invalid tokens produce `Token::Error` with a diagnostic, and parser failures trigger error recovery (skip to next statement boundary) while accumulating all errors. Semantic analysis reports undefined variables, duplicate declarations, scope violations (`this`/`super` outside methods, `break`/`continue` outside loops), and field access outside methods -- all with source-mapped spans rendered by ariadne.

**Debug dump pipeline.** Every stage of compilation can be inspected independently via `--dump-tokens`, `--dump-ast`, `--dump-mir`, `--dump-opt`, and `--dump-asm` flags. The MIR pretty printer emits a CLIF-style text format (`function %name(arity) { bb0(...): ... }`) that shows block parameters, instruction types, and terminator targets, making it possible to trace a value through the entire optimization pipeline.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `logos` | Lexer generation |
| `ariadne` | Error reporting with source context |
| `dynasmrt` / `dynasm` | AArch64 JIT assembly |
| `wasm-encoder` | WASM binary encoding |
| `clap` | CLI argument parsing |

Dev-only: `capstone` (disassembly verification), `wasmtime` (WASM execution tests), `wasmparser` (WASM validation), `wasmprinter` (WAT dump).

## Status

| Component | Status |
|-----------|--------|
| Lexer | Complete |
| Parser | Complete |
| AST | Complete |
| Semantic Analysis | Complete |
| MIR + SSA | Complete |
| MIR Interpreter | Complete |
| Optimization Passes | Complete (7 passes) |
| Generational GC | Complete |
| x86_64 JIT | Complete |
| aarch64 JIT | Complete |
| WASM Codegen | Complete (structured loops via stackifier) |
| Tiered Runtime | Planned (MIR interp cold start → JIT hot promotion) |
| Hot Module Reload | Planned (recompile + patch code cache at runtime) |
| Core Library | Complete (Object, Class, Bool, Null, Num, String, List, Map, Range, Fn, Fiber, System) |
| Fiber Runtime | Not started |
| End-to-End CLI | Complete (lex/parse/sema/MIR/opt/codegen pipeline, REPL, debug dumps) |
