# Verification & debugging

WrenLift uses a layered approach to correctness across the
compiler and runtime.

## MIR interpreter as optimization oracle

A dedicated interpreter executes MIR functions directly, stepping
through blocks and following terminators with a configurable step
limit to catch infinite loops. Every optimization pass is tested
by asserting `eval(f) == eval(optimize(f))` — if constant folding,
DCE, CSE, type specialization, escape analysis, SRA, or LICM
changes a program's result, the test fails. This catches
miscompilations without needing the full Wren VM.

## Tiered cross-checking

Every benchmark and almost every e2e test runs through
`--mode=tiered` by default, but the test runner additionally
verifies that the tiered path produces byte-identical output to
`--mode=interpreter` on the same source. A correctness gap
between the bytecode interpreter and the Cranelift JIT / OSR path
therefore shows up as an e2e failure.

## Cranelift verification

The Cranelift pipeline validates emitted IR before
`define_function` runs. When `WLIFT_CL_VERIFY=1` is set, WrenLift
calls `cranelift_codegen::verify_function` explicitly and prints
both the verifier diagnostic and the failing function's IR to
stderr. That is how the
`inst … uses value v … from non-dominating inst …` and
`return … must match function signature` class of codegen bugs
get surfaced. Failed compiles never corrupt the runtime: the
function falls back to the bytecode path.

## OSR safepoint regression tests

Dedicated e2e tests cover every OSR branch: unconditional and
conditional back-edges (`e2e_tiered_backedge_enters_osr_entry`,
`test_tiered_cond_branch_backedge_enters_osr`), hot method loops
(`e2e_tiered_backedge_enters_osr_entry_in_method`), inner methods
whose caller is already native
(`e2e_tiered_backedge_osr_nested_inside_native_caller`),
allocation-heavy loops that force GC mid-OSR
(`e2e_tiered_backedge_osr_survives_gc_pressure`), and the
module-setup-must-not-rerun invariant
(`e2e_tiered_backedge_does_not_restart_module_entry`).

## WASM structural validation

Emitted WebAssembly modules are validated through
[wasmparser](https://crates.io/crates/wasmparser) (full
structural and type validation) before any execution. When
validation fails, the module is automatically disassembled to WAT
text via [wasmprinter](https://crates.io/crates/wasmprinter) and
included in the error output, making it straightforward to locate
the malformed instruction. Integration tests go further and
execute the validated WASM through
[wasmtime](https://crates.io/crates/wasmtime), confirming the
emitted code actually computes correct results.

## Legacy backend disassembly tests

The hand-rolled x86_64 / aarch64 emitter — retained for
`--no-default-features` builds and as a reference implementation
— is tested via [capstone](https://crates.io/crates/capstone)
against known byte patterns (REX prefixes, ModR/M, SIB, VEX3) and
through JIT-execute smoke tests. These tests are feature-gated to
`#[cfg(not(feature = "cranelift"))]`.

## GC safety

The generational garbage collector is tested for pointer
integrity after promotion (nursery to old generation), write
barrier correctness (old-to-young references tracked in the
remembered set), forwarding table accuracy, and self-referential
pointer fixup (closed upvalues whose `location` field points into
their own struct). String interning is tested for deduplication,
collection of unreachable interned strings, and pointer equality
after interning.

For native execution, a thread-local JIT frame stack registers
each active native call so stack walking can root live boxed
values. That registration is what fixed the binary-trees SIGABRT
under heavy allocation pressure.

## Diagnostic-driven error reporting

The lexer and parser never panic on malformed input. Invalid
tokens produce `Token::Error` with a diagnostic, and parser
failures trigger error recovery (skip to next statement boundary)
while accumulating all errors. Semantic analysis reports
undefined variables, duplicate declarations, scope violations
(`this` / `super` outside methods, `break` / `continue` outside
loops), and field access outside methods, all with source-mapped
spans rendered by
[ariadne](https://crates.io/crates/ariadne).
