<p align="center">
<img style="display: block;" src="wrenlift_logo.png" alt="WrenLift Logo" width="250"/>
</p>

<h1 align="center">WrenLift</h1>

<p align="center">
A fast tiered JIT runtime for the <a href="https://wren.io">Wren</a> programming language.
</p>

<p align="center">
<a href="https://github.com/wrenlift/WrenLift/actions/workflows/ci.yml"><img src="https://github.com/wrenlift/WrenLift/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
<img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust"/>
<img src="https://img.shields.io/badge/edition-2021-blue" alt="Rust 2021"/>
<img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version 0.1.0"/>
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
<img src="https://img.shields.io/badge/targets-x86__64_%7C_aarch64_%7C_WASM-purple" alt="x86_64 | aarch64 | WASM"/>
<a href="https://github.com/wrenlift/hatch"><img src="https://img.shields.io/badge/ecosystem-Hatch-f97316" alt="hatch ecosystem"/></a>
</p>

---

[Wren](https://wren.io) was designed to be embedded: small, fast,
and a great fit for game engines and editors. **WrenLift** flips
that around. It runs `.wren` files directly as standalone scripts
and apps, swapping Wren's stack interpreter for a tiered
Cranelift-backed JIT. One static binary (`wlift`), zero runtime
deps, native code on hot paths.

Pair with [Hatch](https://github.com/wrenlift/hatch) when you
want a package manager and library ecosystem to go with it.

## Install

```sh
curl -fsSL wrenlift.com/install.sh | sh
```

Drops `wlift` (the runtime) and `hatch` (the package + build tool)
into `~/.local/bin`, SHA256-verified, pulled from the latest
GitHub Release.

Knobs: `WLIFT_VERSION=v0.1.0` to pin a tag, `INSTALL_DIR=…` to
retarget. macOS (arm64, x86_64) and Linux (x86_64, aarch64) are
supported. Windows users grab binaries from
[Releases](https://github.com/wrenlift/WrenLift/releases).

### From source

```sh
git clone https://github.com/wrenlift/WrenLift
cd WrenLift
cargo build --release
# binaries land in target/release/{wlift, hatch}
```

## Getting started

Try it without installing → [wrenlift.com/playground](https://wrenlift.com/playground/web/).

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
wlift --mode=tiered script.wren        # default — interpret then Cranelift JIT with OSR
wlift --mode=interpreter script.wren   # MIR / bytecode / threaded interpreter only, no JIT
wlift --mode=jit script.wren           # eagerly compile everything to native
```

### Debug dumps

```sh
wlift --dump-tokens script.wren    # lexer output
wlift --dump-ast    script.wren    # parsed AST
wlift --dump-mir    script.wren    # MIR before optimization
wlift --dump-opt    script.wren    # MIR after optimization
wlift --dump-asm    script.wren    # generated machine code
wlift --no-opt      script.wren    # run without optimization passes
wlift --gc-stats    script.wren    # print GC statistics after execution
```

For runtime env-var traces (tier-up tracing, OSR tracing, Cranelift IR dumps, kill switches) see the [CLI reference](docs/cli-reference.md#runtime-env-vars).

## Docs

- [**Architecture**](docs/architecture.md) — the compilation pipeline, optimization passes, tiered runtime, OSR.
- [**CLI reference**](docs/cli-reference.md) — every `wlift` flag, execution mode, debug dump, and runtime env var.
- [**Verification & debugging**](docs/verification.md) — how the runtime is tested for correctness across the compiler and GC.
- [**Benchmarks**](https://wrenlift.com/benchmark/) — head-to-head numbers, refreshed on every commit.

## License

MIT. See [LICENSE](LICENSE).
