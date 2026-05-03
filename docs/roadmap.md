# Roadmap

Top-level index of in-flight + near-term initiatives. Each
entry links to the dedicated plan doc when one exists; bare
entries are smaller bodies of work that don't need their own
multi-phase plan yet.

## Active plans

- **LSP server** — [lsp-plan.md](lsp-plan.md). v0–v3 mostly
  shipped (diagnostics, hover, definition incl. local vars,
  scoped imports, prelude docs, member/literal/var-typed
  completion, codelens with per-block `Test.describe`/
  `Test.it` lenses, Activity Bar sidebar). Pending: rename,
  inlay hints, references, documentSymbol, `reloadDeps`
  command. v6 covers performance work.
- **Docs generator** — [hatch-docs-generator-plan.md](hatch-docs-generator-plan.md).
- **AOT distribution** — [aot-plan.md](aot-plan.md). Compile
  a hatch project / standalone `.wren` file to a self-
  contained native executable, no runtime dep at the
  consumption site. Phase 1 (Cranelift `.o` emit) starts
  next; rest of the phases are sketched.
- **First-party `hatch test` integration** —
  [hatch-test-plan.md](hatch-test-plan.md). Wire `@hatch:test`
  into the `hatch` CLI as a built-in verb (vs. today's
  spawn-`wlift`-and-parse-stdout shape) so per-case filtering,
  JSON-streamed results, and the VS Code spec runner's per-
  test ▶ Run button all become real.

## Smaller open items

These don't carry a dedicated plan but track real gaps. Each
should land as a small PR or pair of commits when picked up.

- **`textDocument/references`**, `documentSymbol`, and the
  `wlift-lsp.reloadDeps` `executeCommand` (LSP plan items 9,
  10, 15).
- **Per-class implicit-this typo detection** —
  [QUIRKS.md](../QUIRKS.md#implicit-this-swallows-typos-inside-method-bodies).
  Track each class's declared members so a bare `sed` inside
  `static run()` flags as a soft "unknown member" diagnostic.
- **Resolver semver ranges** — [QUIRKS.md](../QUIRKS.md#resolver-doesnt-accept-version-ranges-or-wildcards).
  Today `Dependency::Version(String)` is a literal pin; needs
  a registry index file + Cargo's `semver` crate.
- **Path drift between `~/.local/bin` and `~/.cargo/bin`** —
  [QUIRKS.md](../QUIRKS.md#cargo-install-and-installsh-ship-to-different-bin-paths-and-can-drift).
  Extension's `resolveBinary` should compare mtimes (or
  surface drift in `Show Toolchain Versions`).

## Recently shipped

- **runtime v0.1.16 + vscode-wrenlift 0.1.5** (2026-05-03).
  Sema-on-partial-AST, literal + var-typed receiver
  completion, per-block Test.describe/it codelens, local-var
  goto-def, `hatch init` scaffold with real `@hatch:test` +
  `@hatch:assert` deps, Activity Bar sidebar with project
  scaffolder + spec runner.
- **Editor section on wrenlift.com** — install + setup +
  marketplace/LSP-config tabbed surface ([site/index.html](../site/index.html)).
- **Hatch site** at hatch.wrenlift.com — landing + package
  catalog + readme renderer + blog.
