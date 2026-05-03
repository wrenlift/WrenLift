# Roadmap

Top-level view of in-flight + near-term initiatives. Each
entry is self-contained; this is the public-facing surface.

## Active initiatives

### LSP server

Language server for Wren projects compiled through WrenLift,
with first-class hatch-package awareness. Diagnostics, hover,
goto-definition (locals + scoped imports + cross-file +
bundled `@hatch:*` source), member completion (literal +
sema-typed receivers), run-action codelens (file-level on
`main.wren` / `*.spec.wren`, per-block on
`Test.describe` / `Test.it`), and an Activity Bar sidebar
with project scaffolder + spec runner all ship today.
Pending: workspace-wide rename, references, document
symbols (outline view), inlay hints, and a manual
`reloadDeps` command.

### Docs generator

JSON doc model walked from `///` comments, consumed by the
LSP hover handler and shipped inside every `.hatch` bundle's
source section so downstream tools (LSP across dep
boundaries, future docs website renderer) see the same
shape. Today's coverage is the model + LSP integration; the
docs-website renderer is the next addition.

### AOT distribution

Compile a hatch project (or a standalone `.wren` file) to a
self-contained deployment artifact: native executable for
desktop / server / edge, or `.wasm` module for browser /
WASI / serverless. Reuses parser → sema → MIR; the only
structural change is at the codegen split — Cranelift's
object-file mode for native, the existing wasm codegen
emitter driven whole-program for wasm. Two host profiles on
the wasm side: `--profile=browser` (links the existing
browser bridges + emits ES-module JS glue) and
`--profile=wasi` (WASI shim for stdio / args / env, single
`.wasm` for wasmtime / wasmer / Cloudflare Workers / Deno
Deploy). Component Model output sits behind the wasi
profile once the ecosystem tooling stabilises.

### First-party `hatch test` integration

`hatch test` today walks `*.spec.wren` files and spawns
`wlift <spec>` per file, parsing the `ok: N/M passed` line
from stdout. The next pass wires `@hatch:test` into the CLI
as a built-in: `--filter "<group> > <name>"` runs only
matching cases, `--json` emits one structured event per
case-start / pass / fail / summary so editors can render
live results, and the VS Code spec-runner sidebar's per-row
▶ Run button drops the "runs the whole file" caveat.

## Smaller open items

These don't carry a dedicated treatment but track real gaps.

- **`textDocument/references` + `documentSymbol` + the
  `wlift-lsp.reloadDeps` `executeCommand`** — the three
  remaining LSP handlers that haven't shipped yet.
- **Implicit-this typo detection** — bare identifier inside
  a method that's neither a local nor a known class member
  is rewritten as `this.<name>` rather than diagnosed. See
  [QUIRKS.md](../QUIRKS.md#implicit-this-swallows-typos-inside-method-bodies).
- **Resolver semver ranges** — the version field in
  hatchfiles is parsed as a literal pin today; needs a
  registry index file plus a real semver matcher. See
  [QUIRKS.md](../QUIRKS.md#resolver-doesnt-accept-version-ranges-or-wildcards).
- **Path drift between `~/.local/bin` and `~/.cargo/bin`** —
  the extension's binary resolver picks the older one when
  both exist. See [QUIRKS.md](../QUIRKS.md#cargo-install-and-installsh-ship-to-different-bin-paths-and-can-drift).

## Recently shipped

- **runtime v0.1.16 + vscode-wrenlift 0.1.5** (2026-05-03).
  Sema-on-partial-AST so undefined-name diagnostics fire
  even when the parser bailed elsewhere; literal + var-typed
  receiver completion (`(10).`, `"foo".`, `[1,2].`,
  `var x = 10; x.<...>`); per-block `▶ Run` codelenses on
  every `Test.describe` / `Test.it`; goto-def for bare
  locals, `for` bindings, method/closure params; `hatch
  init` scaffold with real `@hatch:test` + `@hatch:assert`
  deps; Activity Bar sidebar with project scaffolder + spec
  runner.
- **Editor section on wrenlift.com** — install + setup +
  marketplace / LSP-config tabbed surface.
- **Hatch site at hatch.wrenlift.com** — landing + package
  catalog + readme renderer + blog.
