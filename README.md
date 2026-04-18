<p align="center">
<img style="display: block;" src="wrenlift_logo.png" alt="Wren Lift Logo" width="250"/>
</p>

<h1 align="center">WrenLift</h1>

<p align="center">
Fast JIT runtime for the <a href="https://wren.io">Wren</a> programming language.
</p>

<p align="center">
<a href="https://github.com/wrenlift/WrenLift/actions/workflows/ci.yml"><img src="https://github.com/wrenlift/WrenLift/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
<img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust"/>
<img src="https://img.shields.io/badge/edition-2021-blue" alt="Rust 2021"/>
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
<img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version 0.1.0"/>
<img src="https://img.shields.io/badge/tests-836_passing-brightgreen" alt="836 tests passing"/>
<img src="https://img.shields.io/badge/targets-x86__64_%7C_aarch64_%7C_WASM-purple" alt="x86_64 | aarch64 | WASM"/>
<a href="https://github.com/wrenlift/hatch"><img src="https://img.shields.io/badge/ecosystem-Hatch-f97316" alt="hatch ecosystem"/></a>
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

Optimized MIR is compiled to native code via **Cranelift** (both x86_64 and aarch64) as the primary backend, producing verified machine code with register allocation, instruction selection, and encoding handled by the Cranelift IR pipeline. A legacy hand-rolled emitter (x86_64 byte-level / aarch64 via dynasmrt) plus a linear-scan register allocator is retained behind the non-default `--no-default-features` build and exercised by a dedicated test subset. The WASM target compiles directly from MIR to WebAssembly via a stackifier, bypassing machine IR entirely.

The runtime uses NaN-boxed 64-bit values, a generational semi-space garbage collector with bump-pointer nursery allocation, and an object model with O(1) method dispatch via interned symbol IDs. Tiered execution runs cold code through a bytecode / pre-decoded threaded interpreter, then promotes hot functions to Cranelift-compiled native code. **On-stack replacement (OSR)** transfers hot loops into native mid-iteration without re-running function entry code — the back-edge safepoint walks live interpreter registers into a compiled loop-header entry that Cranelift emits alongside the main function. Method and closure frames, nested native callers, and both unconditional and conditional back-edges are all OSR-eligible; the MIR frame stays parked so the garbage collector still sees every live value in flight. Inline caches, speculative devirtualization (`CallKnownFunc`), and leaf fast paths shrink the method-dispatch tax on hot protocol-heavy code. Background compilation runs on a worker thread; hot module reload is planned.

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

### Execution modes

```sh
wlift --mode=tiered script.wren        # Default: interpret → Cranelift JIT with OSR
wlift --mode=interpreter script.wren   # MIR/bytecode/threaded interpreter only, no JIT
wlift --mode=jit script.wren           # Eagerly compile everything to native
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

### Runtime diagnostics

The runtime also honours a handful of env-gated traces and kill switches — useful for profiling, debugging, or reproducing issues without a rebuild.

| Variable | Effect |
|----------|--------|
| `WLIFT_JIT_DUMP=1` | Print the MIR handed to each Cranelift compile |
| `WLIFT_CL_IR=1` | Print the lowered Cranelift IR for each compiled function |
| `WLIFT_CL_VERIFY=1` | Run `verify_function` before `define_function` and print the offending IR on failure |
| `WLIFT_TIER_TRACE=1` | Trace tier-up queue / install events and IC-snapshot populations |
| `WLIFT_OSR_TRACE=1` | Trace OSR back-edge counts and which entries fire |
| `WLIFT_TIER_STATS=1` | Print per-function tier statistics at shutdown (interp / baseline / opt / OSR / IC hits / native-to-native) |
| `WLIFT_DISABLE_THREADED=1` | Force the bytecode interpreter instead of the pre-decoded threaded interpreter |
| `WLIFT_DISABLE_METHOD_OSR=1` | Disable OSR transfer for method/closure frames (keep top-level only) |
| `WLIFT_DISABLE_NESTED_OSR=1` | Disable OSR transfer when already nested under a native caller |

## Verification & Debugging

WrenLift uses a layered approach to correctness across the compiler and runtime.

**MIR interpreter as optimization oracle.** A dedicated interpreter executes MIR functions directly, stepping through blocks and following terminators with a configurable step limit to catch infinite loops. Every optimization pass is tested by asserting `eval(f) == eval(optimize(f))` -- if constant folding, DCE, CSE, type specialization, escape analysis, SRA, or LICM changes a program's result, the test fails. This catches miscompilations without needing the full Wren VM.

**Tiered cross-checking.** Every benchmark and almost every e2e test runs through `--mode=tiered` by default, but the test runner additionally verifies that the tiered path produces byte-identical output to `--mode=interpreter` on the same source. A correctness gap between the bytecode interpreter and the Cranelift JIT / OSR path therefore shows up as an e2e failure.

**Cranelift verification.** The Cranelift pipeline validates emitted IR before `define_function` runs. When `WLIFT_CL_VERIFY=1` is set, WrenLift calls `cranelift_codegen::verify_function` explicitly and prints both the verifier diagnostic and the failing function's IR to stderr — this is how the `inst … uses value v … from non-dominating inst …` and `return … must match function signature` class of codegen bugs get surfaced. Failed compiles never corrupt the runtime: the function falls back to the bytecode path.

**OSR safepoint regression tests.** Dedicated e2e tests cover every OSR branch: unconditional and conditional back-edges (`e2e_tiered_backedge_enters_osr_entry`, `test_tiered_cond_branch_backedge_enters_osr`), hot method loops (`e2e_tiered_backedge_enters_osr_entry_in_method`), inner methods whose caller is already native (`e2e_tiered_backedge_osr_nested_inside_native_caller`), allocation-heavy loops that force GC mid-OSR (`e2e_tiered_backedge_osr_survives_gc_pressure`), and the module-setup-must-not-rerun invariant (`e2e_tiered_backedge_does_not_restart_module_entry`).

**WASM structural validation.** Emitted WebAssembly modules are validated through wasmparser (full structural and type validation) before any execution. When validation fails, the module is automatically disassembled to WAT text via wasmprinter and included in the error output, making it straightforward to locate the malformed instruction. Integration tests go further and execute the validated WASM through wasmtime, confirming the emitted code actually computes correct results.

**Legacy backend disassembly tests.** The hand-rolled x86_64 / aarch64 emitter — retained for `--no-default-features` builds and as a reference implementation — is tested via capstone against known byte patterns (REX prefixes, ModR/M, SIB, VEX3) and through JIT-execute smoke tests. These tests are feature-gated to `#[cfg(not(feature = "cranelift"))]`.

**GC safety.** The generational garbage collector is tested for pointer integrity after promotion (nursery to old generation), write barrier correctness (old-to-young references tracked in the remembered set), forwarding table accuracy, and self-referential pointer fixup (closed upvalues whose `location` field points into their own struct). String interning is tested for deduplication, collection of unreachable interned strings, and pointer equality after interning. For native execution, a thread-local JIT frame stack registers each active native call so stack walking can root live boxed values — this was the fix for the binary_trees SIGABRT under heavy allocation pressure.

**Diagnostic-driven error reporting.** The lexer and parser never panic on malformed input. Invalid tokens produce `Token::Error` with a diagnostic, and parser failures trigger error recovery (skip to next statement boundary) while accumulating all errors. Semantic analysis reports undefined variables, duplicate declarations, scope violations (`this`/`super` outside methods, `break`/`continue` outside loops), and field access outside methods -- all with source-mapped spans rendered by ariadne.

**Debug dump pipeline.** Every stage of compilation can be inspected independently via `--dump-tokens`, `--dump-ast`, `--dump-mir`, `--dump-opt`, and `--dump-asm` flags. The MIR pretty printer emits a CLIF-style text format (`function %name(arity) { bb0(...): ... }`) that shows block parameters, instruction types, and terminator targets, making it possible to trace a value through the entire optimization pipeline.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `cranelift-codegen` / `cranelift-frontend` / `cranelift-module` / `cranelift-jit` / `cranelift-native` | Primary JIT backend (feature-gated behind `cranelift`, enabled by default) |
| `logos` | Lexer generation |
| `ariadne` | Error reporting with source context |
| `dynasmrt` / `dynasm` | Legacy aarch64 JIT assembly (used when `cranelift` feature is off) |
| `wasm-encoder` | WASM binary encoding |
| `smallvec` | Small-vector args for hot dispatch paths |
| `clap` | CLI argument parsing |

Dev-only: `capstone` (legacy disassembly verification), `wasmtime` (WASM execution tests), `wasmparser` (WASM validation), `wasmprinter` (WAT dump).

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
| Bytecode VM | Complete (compact lowering + pre-decoded threaded interpreter) |
| Generational GC | Complete (thread-local JIT root stack for native frames) |
| Cranelift JIT (x86_64 + aarch64) | Complete (primary backend, `cranelift` feature = default on) |
| Legacy x86_64 / aarch64 JIT | Retained for `--no-default-features` builds |
| WASM Codegen | Complete (structured loops via stackifier) |
| Inline Caching | Complete (monomorphic kind=1 leaf, kind=5 getter inline, devirt via `CallKnownFunc`) |
| Tiered Runtime | Complete (interpret → JIT with background worker thread) |
| On-Stack Replacement | Complete (method / closure / nested native caller / conditional back-edge) |
| Hot Module Reload | Planned (recompile + patch code cache at runtime) |
| Core Library | Complete (Object, Class, Bool, Null, Num, String, List, Map, Range, Fn, Fiber, System) |
| Fiber Runtime | Complete (create, call, yield, resume, isDone, transfer) |
| End-to-End CLI | Complete (lex/parse/sema/MIR/opt/codegen pipeline, REPL, debug dumps) |
| Optional Modules | Complete (Meta, Random — on-demand via import) |
| C Embedding API | Complete (wren.h-compatible, header + example) |
| Lib Test Suite | 747 tests (lexer, parser, sema, MIR, opt passes, GC, codegen, runtime) |
| E2E Test Suite | 89 tests (classes, closures, fibers, inheritance, iterators, GC pressure, JIT tiering, OSR safepoints, benchmarks) |
