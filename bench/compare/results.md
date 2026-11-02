# Results: 2026-11-02

Measured with `compare.py` as the README describes, on an idle machine.

- Machine: Intel Core i5-1250P (4 performance + 8 efficiency cores),
  Ubuntu, Linux 7.0, glibc 2.43. Every run pinned to performance cores
  4-7.
- Wren: wren-lang/wren-cli `1855363`, built with `make -C projects/make`,
  GCC 15.2.
- WrenJIT: Brugarolas/WrenJIT `10c9421`, `cargo build --release`,
  rustc 1.101.0-nightly (c36f14571 2026-10-01).
- WrenLift: `12424e5`, `cargo build --release --features llvm-shared
  --bin wlift` against LLVM 21, same rustc.
- Command:

  ```
  python3 bench/compare/compare.py --wren wren_cli --wrenjit WrenJIT \
      --wrenlift wlift --runs 7 --pin 4-7
  ```

Process start-up and script compilation are outside the measured region
and not reported here.

### Absolute times

Median of 7 runs of each kernel's measured region (`System.clock`). Lower is better.

| Benchmark | Wren | WrenJIT | WrenLift |
|---|---:|---:|---:|
| `binary_trees` | 696 ms | fails ‡ | 164 ms |
| `delta_blue` | 70.3 ms | fails ‡ | 87.9 ms |
| `fannkuch_redux` | 7558 ms | fails ‡ | 604 ms |
| `fasta` | 87.0 ms | fails ‡ | 21.7 ms |
| `fib` | 141 ms | 228 ms | 7.3 ms |
| `k_nucleotide` | 205 ms | fails ‡ | 85.0 ms |
| `mandelbrot` | 770 ms | fails ‡ | 19.7 ms |
| `method_call` | 67.5 ms | fails ‡ | 15.4 ms |
| `nbody` | 2592 ms | fails ‡ | 67.1 ms |
| `pidigits` | 1258 ms | 1548 ms | 176 ms |
| `regex_redux` | 825 ms | fails ‡ | 121 ms |
| `reverse_complement` | 76.7 ms | 127 ms | 25.0 ms |
| `spectral_norm` | 816 ms | 1157 ms † | 33.3 ms |
| `binary_trees (large)` | 4044 ms | fails ‡ | 569 ms |
| `closure_call (large)` | 5767 ms | fails ‡ | 82.9 ms |
| `dyn_dispatch (large)` | 550 ms | fails ‡ | 120 ms |
| `fib (large)` | 9035 ms | 13660 ms | 425 ms |
| `free_call (large)` | 5710 ms | 3762 ms | 125 ms |
| `inlined_call (large)` | 4586 ms | 3703 ms | 75.3 ms |
| `mandelbrot (large)` | 20186 ms | fails ‡ | 265 ms |
| `method_call (large)` | 5645 ms | fails ‡ | 81.1 ms |
| `nbody (large)` | 52758 ms | fails ‡ | 688 ms |
| `wrenjit/gc` | 61.8 ms | 176 ms | 24.4 ms |
| `wrenjit/jit` | 8743 ms | 318 ms | 339 ms |

† Same values as Wren, printed with a different number of digits.
‡ The engine did not finish the kernel; reasons below.

### Relative speed

Higher is better: the other engine's time over WrenLift's.

| Benchmark | WrenLift vs Wren | WrenLift vs WrenJIT |
|---|---:|---:|
| `binary_trees` | 4.25x | — |
| `delta_blue` | 0.80x | — |
| `fannkuch_redux` | 12.51x | — |
| `fasta` | 4.00x | — |
| `fib` | 19.16x | 31.08x |
| `k_nucleotide` | 2.41x | — |
| `mandelbrot` | 39.07x | — |
| `method_call` | 4.39x | — |
| `nbody` | 38.64x | — |
| `pidigits` | 7.14x | 8.78x |
| `regex_redux` | 6.82x | — |
| `reverse_complement` | 3.07x | 5.09x |
| `spectral_norm` | 24.50x | 34.73x |
| `binary_trees (large)` | 7.10x | — |
| `closure_call (large)` | 69.54x | — |
| `dyn_dispatch (large)` | 4.59x | — |
| `fib (large)` | 21.28x | 32.17x |
| `free_call (large)` | 45.65x | 30.08x |
| `inlined_call (large)` | 60.92x | 49.18x |
| `mandelbrot (large)` | 76.20x | — |
| `method_call (large)` | 69.62x | — |
| `nbody (large)` | 76.73x | — |
| `wrenjit/gc` | 2.54x | 7.22x |
| `wrenjit/jit` | 25.83x | 0.94x |

### Runs that did not finish

- `binary_trees` on wrenjit: fail: runtime error: method MethodSymbol#0 is not defined for receiver at FuncId#3:BytecodePc(3)
- `delta_blue` on wrenjit: fail: runtime error: method MethodSymbol#1 is not defined for receiver at FuncId#91:BytecodePc(34)
- `fannkuch_redux` on wrenjit: fail: runtime error: method MethodSymbol#187 is not defined for receiver at FuncId#1:BytecodePc(5)
- `fasta` on wrenjit: fail: runtime error: method MethodSymbol#174 is not defined for receiver at FuncId#1:BytecodePc(2)
- `k_nucleotide` on wrenjit: fail: runtime error: method MethodSymbol#166 is not defined for receiver at FuncId#2:BytecodePc(8)
- `mandelbrot` on wrenjit: fail: runtime error: method MethodSymbol#179 is not defined for receiver at FuncId#10:BytecodePc(5)
- `method_call` on wrenjit: fail: runtime error: entered fused-away instruction at FuncId#3:BytecodePc(2)
- `nbody` on wrenjit: fail: runtime error: entered fused-away instruction at FuncId#9:BytecodePc(2)
- `regex_redux` on wrenjit: fail: panic in cranelift-frontend's SSA builder
- `binary_trees (large)` on wrenjit: fail: runtime error: method MethodSymbol#0 is not defined for receiver at FuncId#3:BytecodePc(3)
- `closure_call (large)` on wrenjit: fail: runtime error: method MethodSymbol#99 is not defined for receiver at FuncId#3:BytecodePc(34)
- `dyn_dispatch (large)` on wrenjit: fail: panic in cranelift-frontend's SSA builder
- `mandelbrot (large)` on wrenjit: fail: runtime error: method MethodSymbol#184 is not defined for receiver at FuncId#10:BytecodePc(5)
- `method_call (large)` on wrenjit: fail: runtime error: method MethodSymbol#169 is not defined for receiver at FuncId#7:BytecodePc(33)
- `nbody (large)` on wrenjit: fail: runtime error: entered fused-away instruction at FuncId#9:BytecodePc(2)
