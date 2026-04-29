# Architecture

WrenLift replaces Wren's stack-based bytecode interpreter with a
modern compilation pipeline. This document walks through each
stage from source bytes to native code.

## Pipeline

Source code is tokenized by a [logos](https://github.com/maciejhirsz/logos)-based
lexer (~60 token variants), then parsed by a recursive-descent
parser into a fully spanned AST. Semantic analysis performs
name/scope resolution and speculative type inference over the
AST. The analyzed program is lowered into an SSA-based mid-level
IR (MIR) using block parameters (Cranelift / MLIR-style, not phi
nodes).

## Optimization passes

Seven passes run over the MIR before code generation:

- **Constant folding** — evaluates constant expressions at compile
  time, folds branches on known conditions, propagates constants
  through moves and box / unbox boundaries.
- **Dead code elimination** — removes unused pure instructions and
  unreachable blocks; preserves side-effectful operations.
- **Common subexpression elimination** — global value numbering
  that deduplicates identical pure computations and respects
  commutativity.
- **Type specialization** — devirtualizes boxed arithmetic when
  operand types are known (e.g. two `Num`s), converting `Add` into
  `GuardNum` + `Unbox` + `AddF64` + `Box` for native-speed floating
  point.
- **Escape analysis** — identifies heap allocations that never
  leave the function (not returned, not stored to fields, not
  captured by escaping closures).
- **Scalar replacement of aggregates** — replaces non-escaping
  fixed-size lists with individual SSA values when all accesses
  use constant indices.
- **Loop-invariant code motion** — detects natural loops via
  dominance-based back-edge analysis, inserts preheader blocks,
  and hoists loop-invariant computations out of the loop body
  using fixpoint iteration.

## Code generation

Optimized MIR is compiled to native code via **Cranelift** (both
`x86_64` and `aarch64`) as the primary backend, producing verified
machine code with register allocation, instruction selection, and
encoding handled by the Cranelift IR pipeline.

A legacy hand-rolled emitter (x86_64 byte-level / aarch64 via
[dynasmrt](https://crates.io/crates/dynasmrt)) plus a linear-scan
register allocator is retained behind the non-default
`--no-default-features` build and exercised by a dedicated test
subset.

The WASM target compiles directly from MIR to WebAssembly via a
stackifier, bypassing machine IR entirely.

## Runtime

The runtime uses NaN-boxed 64-bit values, a generational
semi-space garbage collector with bump-pointer nursery allocation,
and an object model with O(1) method dispatch via interned symbol
IDs.

### Tiered execution

Cold code runs through a bytecode / pre-decoded threaded
interpreter. Hot functions get promoted to Cranelift-compiled
native code on a background worker thread.

### On-stack replacement

OSR transfers hot loops into native mid-iteration without
re-running function entry code. The back-edge safepoint walks live
interpreter registers into a compiled loop-header entry that
Cranelift emits alongside the main function. Method and closure
frames, nested native callers, and both unconditional and
conditional back-edges are all OSR-eligible; the MIR frame stays
parked so the garbage collector still sees every live value in
flight.

### Dispatch fast paths

Inline caches, speculative devirtualization (`CallKnownFunc`), and
leaf fast paths shrink the method-dispatch tax on hot
protocol-heavy code.

### Roadmap

Hot module reload (recompile + patch the code cache at runtime) is
planned but not yet shipped.
