# CLI reference

## Execution modes

```sh
wlift --mode=tiered script.wren        # default — interpret then Cranelift JIT with OSR
wlift --mode=interpreter script.wren   # MIR / bytecode / threaded interpreter only, no JIT
wlift --mode=jit script.wren           # eagerly compile everything to native
```

## Targets

```sh
wlift script.wren                              # native, host arch
wlift --target=wasm script.wren -o output.wasm # WASM target (stackifier path)
```

## Debug dumps

Every stage of compilation can be inspected from the command line:

```sh
wlift --dump-tokens script.wren    # lexer output
wlift --dump-ast    script.wren    # parsed AST
wlift --dump-mir    script.wren    # MIR before optimization
wlift --dump-opt    script.wren    # MIR after optimization
wlift --dump-asm    script.wren    # generated machine code
wlift --no-opt      script.wren    # run without optimization passes
wlift --gc-stats    script.wren    # print GC statistics after execution
```

The MIR pretty-printer emits a CLIF-style text format
(`function %name(arity) { bb0(...): ... }`) that shows block
parameters, instruction types, and terminator targets, so a value
can be traced through the entire optimization pipeline.

## Runtime env vars

Useful for profiling, debugging, or reproducing issues without a
rebuild.

| Variable | Effect |
|----------|--------|
| `WLIFT_JIT_DUMP=1` | Print the MIR handed to each Cranelift compile |
| `WLIFT_CL_IR=1` | Print the lowered Cranelift IR for each compiled function |
| `WLIFT_CL_VERIFY=1` | Run `verify_function` before `define_function` and print the offending IR on failure |
| `WLIFT_TIER_TRACE=1` | Trace tier-up queue / install events and IC-snapshot populations |
| `WLIFT_OSR_TRACE=1` | Trace OSR back-edge counts and which entries fire |
| `WLIFT_TIER_STATS=1` | Print per-function tier statistics at shutdown (interp / baseline / opt / OSR / IC hits / native-to-native) |
| `WLIFT_DISABLE_THREADED=1` | Force the bytecode interpreter instead of the pre-decoded threaded interpreter |
| `WLIFT_DISABLE_METHOD_OSR=1` | Disable OSR transfer for method / closure frames (top-level only) |
| `WLIFT_DISABLE_NESTED_OSR=1` | Disable OSR transfer when already nested under a native caller |
