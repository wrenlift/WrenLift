# AOT distribution

Compile a hatch project (or a standalone `.wren` file) to a
self-contained native executable. End user runs the binary
directly — no `wlift` runtime install, no `.hatch` bundle,
no interpreter-on-the-target-machine.

## Goals

- `wlift build hello.wren -o hello` — single-file standalone.
  Same parser + sema + MIR pipeline as the JIT, but Cranelift
  emits an object file instead of memory pages, and the
  linker pairs it with a statically-linked runtime.
- `hatch build --aot` — for hatch projects: walk the
  manifest, lower every imported module + dep, link the
  whole graph into one binary.
- Cross-compile out of the box for Linux / macOS / Windows ×
  x86_64 / aarch64. Same target-triple shape as `cargo build
  --target …`, surfaced via `--target` on `wlift build` /
  `hatch build`.
- Ship the binaries small enough that "compile once, deploy"
  is realistic for CLI tools and game engines — single-file
  binaries in the 8–15 MB range matching the size of the
  current `wlift` ELF, without the JIT machinery.

## Non-goals (initial release)

- AOT'd code can't tier-up at runtime — the JIT machinery is
  intentionally absent from the AOT binary. Hot loops live or
  die by what the AOT optimiser produced; no profile-guided
  re-jitting on the deployed machine.
- Plugin hot-reload. Plugins (today loaded via `libloading`
  at runtime) get statically linked at AOT time; swapping a
  plugin without rebuilding the host binary is out of scope.
- Incremental AOT compilation. Each `wlift build` is a clean
  build of the full module graph. Salsa-style memoisation
  lands later if compile times bite.
- Reflection / runtime class introspection beyond what the
  existing prelude offers. `Object.is(_)`, `instanceOf`, etc.
  keep working; runtime code-loading (eval-style) doesn't.

## Architecture

```
parse + sema + MIR  ──►  [AOT mode]
                              │
                              ├── CLIF emit (object code path)
                              │     ├── one Cranelift `Module` per Wren module
                              │     ├── declared funcs link by mangled symbol
                              │     └── output: `<crate>.o`
                              │
                              ├── runtime as `staticlib`
                              │     ├── GC, fibers, value boxing
                              │     ├── prelude (Object/Num/String/List/…)
                              │     ├── interpreter (used for code that
                              │     │   couldn't be AOT'd — closures over
                              │     │   non-resolved imports, eval-shaped
                              │     │   constructs, defensive fallback)
                              │     └── plugins linked statically
                              │
                              └── linker (lld / system ld)
                                    └── output: native executable
```

Reuses every layer above MIR unchanged. The split is at the
codegen boundary: Cranelift's `JITModule` becomes
`ObjectModule`. The runtime is already a workspace
member with `crate-type = ["lib", "cdylib", "staticlib"]`, so
the static-link side is mostly a packaging exercise.

## Compared to today

|                         | `wlift hello.wren`      | `hatch build` (`.hatch`) | `wlift build hello.wren -o hello` (planned) |
|-------------------------|-------------------------|--------------------------|----------------------------------------------|
| Runtime install needed  | yes, on every machine   | yes                      | **no** — binary is self-contained            |
| Compile time            | none (parse on launch)  | seconds                  | seconds                                      |
| Startup time            | parser + sema cold      | bytecode load            | ~immediate                                   |
| Binary size             | n/a                     | ~30–500 KB bundle        | 8–15 MB single executable                    |
| Hot tier-up             | yes                     | yes                      | **no** (planned tradeoff)                    |
| Plugins                 | dlopen at runtime       | dlopen                   | static-linked at compile time                |

## Phasing

### Phase 1 — Cranelift object emit

Switch the existing JIT backend behind a feature gate:
`feature = "aot"` selects `ObjectModule`, default stays
`JITModule`. Most of the codegen surface
(`codegen::cranelift_backend`) is shared — the difference is
which `Module` trait impl gets handed to the emitter.

Deliverable: a unit test that compiles a trivial Wren
function (e.g. `class App { static run() { return 42 } }`) to
a `.o` file and runs `nm` to assert the mangled symbol is
present.

### Phase 2 — Single-file driver

`wlift build <file.wren> -o <out>`:

1. Parse + sema the file.
2. Lower the resulting AST through MIR + the AOT codegen
   feature.
3. Drive the system linker (`cc`, falling back to `lld`) to
   pair the object with the runtime's `staticlib`.
4. Write the output binary; `chmod +x` on POSIX.

Deliverable: `wlift build examples/hello.wren -o /tmp/hello`
produces a binary that `/tmp/hello` runs and prints `Hello,
world!` without touching `~/.local/bin/wlift`.

### Phase 3 — Multi-module AOT

The single-file driver only sees one module. Real projects
import siblings + `@hatch:*` deps. Walk the import graph
from the entry, lower every reachable `.wren` source, and
hand the linker every resulting `.o`. Cross-module symbol
references go through Cranelift's external-symbol mechanism
(same shape the JIT already uses for runtime helper calls).

Hatch deps come pre-built as `.hatch` bundles today. AOT
needs the bundle's `Source` section (already present —
that's what the LSP reads for goto-def). Lower the source
into MIR at link time, link the resulting objects in.

Deliverable: `hatch build --aot` on a project that imports
`@hatch:fmt` produces a binary that runs the project's
entry without resolving deps at runtime.

### Phase 4 — Cross-compilation

Add a `--target <triple>` flag on `wlift build` /
`hatch build`. Cranelift already supports cross-target
codegen — the missing piece is ensuring the runtime
`staticlib` is available for the target triple. Lean on the
existing GitHub Actions release matrix
(linux-x86_64, linux-aarch64, macos-aarch64, macos-x86_64)
to publish a `wren_lift_runtime-<triple>.a` artifact
alongside each runtime release.

Deliverable: `wlift build hello.wren --target aarch64-unknown-linux-gnu`
on a macOS host produces a Linux ARM64 ELF that runs on
the target.

### Phase 5 — Plugins

Plugins (`@hatch:gpu`, `@hatch:sqlite`, etc.) currently load
via `libloading` at runtime against the host's plugin ABI.
For AOT we need the link step to bundle the plugin's
`staticlib` instead. Each plugin's `Cargo.toml` already
declares both `cdylib` (for runtime dlopen) and `rlib`/
`staticlib` artifacts; the AOT linker just needs to know
which to pick.

Risk: plugins that depend on system libraries (Vulkan, Metal,
audio frameworks) need the AOT linker to resolve those too.
Acceptable to push that onto the plugin's build script for
v1 — same model `cargo build` already uses.

Deliverable: an AOT'd hatch project that imports
`@hatch:sqlite` runs without `libwlift_sqlite.so` on the
target.

### Phase 6 — Code-size optimisation

AOT binary size starts ~15 MB (runtime + prelude + user
code + Cranelift output). Pass-level work to bring that
down:

- Dead-code elimination across the linked module graph.
  LLVM/lld already does function-level DCE; surface it via
  `--gc-sections` in the linker invocation.
- Strip unused prelude classes. Today every prelude class
  ships in every binary. A reachability analysis driven by
  the user's MIR can prune the prelude graph.
- `panic = "abort"` on the runtime's release profile
  (already on for the JIT path; matters more for AOT
  binary size).

Target: 8–10 MB for "hello world", scaling sub-linearly
with project size.

## Open questions

- Single-binary distribution vs. binary + sidecar plugins?
  Lean: single binary is the explicit goal of AOT; sidecar
  plugins are a separate "ship a hatch project as a
  directory tree of files" mode.
- AOT'd binary picks up `wlift --version`-style metadata so
  the user can ask "which runtime built this?". Lean yes,
  print under `--wrenlift-version` to avoid clashing with
  the user's own `--version` flag.
- Source maps / debug info. AOT'd backtraces should still
  point at the user's `.wren` source. Cranelift can emit
  DWARF; we'd need to thread the existing span info through
  the codegen path.
- AOT-only diagnostics. Some MIR shapes (e.g. closures over
  unresolved imports) can't be statically lowered. Surface
  these as "this function will fall back to the interpreter
  at runtime" warnings rather than hard errors, so the user
  can decide whether to refactor or ship as-is.

## Dependencies

- Cranelift backend (`feature = "cranelift"`) is the only
  codegen path AOT depends on. Phase 1 lifts the AOT switch
  out of `JITModule` vs `ObjectModule` — same crate, same
  IR, same passes.
- Hatch dep bundle's `Source` section (already present, used
  by the LSP) is what Phase 3 consumes for cross-module AOT.
- Runtime workspace `crate-type` already includes
  `staticlib`, so Phase 1's object-emit + static-link
  pipeline doesn't need a Cargo manifest change.
