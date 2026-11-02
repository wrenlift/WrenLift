# Wren, WrenJIT and WrenLift on the same scripts

`compare.py` runs every kernel in this repository's `bench/`, its
larger variants, and the two scripts WrenJIT publishes its own numbers from,
on upstream Wren, WrenJIT and WrenLift. It is not part of CI.

## What is timed

Each kernel times itself with `System.clock` and prints
`elapsed: <seconds>`. That measured region includes the kernel's warm-up
and excludes process start-up and compilation of the script, the same
region WrenJIT's README reports. Each engine runs each kernel seven
times and the table shows the median.

Process start-up and compile latency are not measured. They are a
separate benchmark, and these runs do not report it.

Every engine's output, less its `elapsed:` lines, is compared with
upstream Wren's. Output that holds the same values printed with a
different number of digits is timed and marked; any other difference, a
crash or an error is reported and not timed.

## Kernels

- `bench/*.wren`: the Benchmarks Game set and the classic Wren
  benchmarks this repository tracks in CI.
- `name (large)`: call-, dispatch- and arithmetic-heavy kernels at
  larger sizes, kept in a subdirectory of `bench/`.
- `bench/compare/wrenjit/`: `gc.wren` and `jit.wren` from WrenJIT's
  `scripts/` directory, unchanged apart from a header naming their source
  (MIT). WrenJIT's `mandelbrot (+ others)` row is `jit.wren`. Its
  `classes` script is not in its repository, so it is not included.

All kernels stay within the language and core library upstream Wren
supports.

## Building the engines

Each engine in its fastest release configuration:

- **Wren**: `wren-lang/wren-cli`, `make -C projects/make`.
- **WrenJIT**: `cargo build --release` (its profile sets LTO and
  `opt-level = 3`).
- **WrenLift**: `cargo build --release --features llvm --bin wlift`, the
  configuration releases ship. On a Linux distribution whose LLVM has no
  static Polly, `--features llvm-shared` is the same build linked against
  `libLLVM.so`.

## Running

```
python3 bench/compare/compare.py \
    --wren path/to/wren_cli \
    --wrenjit path/to/WrenJIT \
    --wrenlift path/to/wlift \
    --runs 7 --out results.md
```

On Linux, `--pin 4-7` runs every engine under `taskset` on the same
cores, which keeps a hybrid CPU's efficiency cores out of the
comparison; pick its physical performance cores. `--only fib
'nbody (large)'` runs a subset. The report ends with the machine, OS and
engine paths; record the engines' revisions and compilers beside it,
as `results.md` does.

Run on an otherwise idle machine. The three engines run back to back for
each kernel, so a change in load affects all three.
