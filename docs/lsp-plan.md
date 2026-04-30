# wlift-lsp — plan

A Language Server Protocol implementation for Wren projects compiled
through WrenLift, with first-class hatch-package awareness. Reuses the
existing parser + semantic analyzer; produces diagnostics + hover +
goto-def + completion + rename.

## Goals

- Compile-as-you-type: parse + sema diagnostics on every keystroke,
  same checks the runtime would apply.
- Hatch-aware: `@hatch:foo` imports resolve through the same fetcher
  the runtime uses; symbols across package boundaries hover, complete,
  and goto-def correctly.
- Doc-aware: hover content is the rendered Markdown the docs generator
  produces — one source of truth.
- Editor-agnostic: stdio-based LSP server. VSCode + Neovim + JetBrains
  fork-of-the-day all work via standard config.

## Non-goals (initial release)

- Refactor catalog beyond rename. (No "extract method", no "move to
  module".)
- Inlay hints. (Phase 4.)
- Debugger / DAP. Separate effort.
- Performance work below "feels fluid on a 50-file project". Big-repo
  optimization (incremental re-parse, salsa-style memoization) is
  Phase 4.

## Architecture

```
editor ──LSP/stdio──► wlift_lsp::server
                            │
                            ├── workspace store
                            │     ├── open documents (Rope per file)
                            │     ├── hatchfile manifest cache
                            │     └── installed dep bundles (in-memory)
                            │
                            ├── frontend
                            │     ├── parse::parse_module        (existing)
                            │     ├── sema::resolve              (existing)
                            │     └── span → symbol table (new)
                            │
                            └── handlers
                                  ├── textDocument/diagnostic
                                  ├── textDocument/hover
                                  ├── textDocument/definition
                                  ├── textDocument/references
                                  ├── textDocument/completion
                                  └── textDocument/rename
```

The server is a thin orchestrator over the existing `wren_lift::parse`
and `wren_lift::sema` crates. Each open file maps to a `Document`
state; when the editor sends `didChange`, we re-parse + re-resolve
that file (and any file whose imports it touches via a coarse
"who-imports-me" map).

## Hatch awareness

Two layers:

1. **Local resolution** — workspace files inside the user's package.
   Resolved by walking the hatchfile's `[modules]` and the on-disk
   tree. Same as `wlift --check`.
2. **Dep resolution** — `@hatch:*` and other scoped imports. We reuse
   `hatch::resolve_deps` (or its wasm shim) to fetch each declared
   dep's `.hatch` bundle and load its `Wlbc` + `Source` sections into
   the workspace store. The bundles' source sections give us
   spans into the dep's original Wren files for goto-definition.
   Without bundle source sections we degrade to "hover shows
   declared signature only".

Cache lifetime: deps are fetched once per workspace boot; a manual
`wlift-lsp reload-deps` command (exposed via `workspace/executeCommand`)
re-fetches when the user bumps a version.

Symbol resolution across hatch boundaries:

- A symbol's "home module" is the package name, e.g. `@hatch:gpu`.
- Goto-def jumps to the bundled source section's span; if missing,
  falls back to the package's published docs URL (opened in browser
  via `window/showDocument`).
- Find-references searches the workspace plus any open dep file.

## Diagnostic catalog

Reuses ariadne `Diagnostic`s the runtime already produces. The LSP
adapter:

1. Parse pass → `parse::ParseError` → LSP `Diagnostic` with severity
   `Error`, code from the error variant, related notes attached.
2. Sema pass → `sema::Diagnostic` (typed-AST resolution errors:
   unresolved imports, name conflicts, type-mismatch in method
   signatures). Same translation.
3. Optional warning passes (pluggable):
   - Unused locals.
   - Unused imports.
   - Shadowed fields.
   These start as opt-in via `[lsp.lints]` in the hatchfile.

The runtime-only errors (e.g. method-not-found at dispatch) don't
surface in the LSP; they need execution. `hatch test` is the right
home for those.

## Hover content

Two sources, joined:

1. Signature line (rendered from sema's resolved type info).
2. Doc Markdown — the JSON manifest the docs generator emits. Server
   loads `manifest.json` from each installed bundle's docs section
   (Phase v2 of the docs plan).

Fallback when no doc manifest is present: just the signature line.

## Completion

Trigger contexts:

- After `.` on an expression — list members of the resolved class.
- After `import "@` — list known package names from the local
  `hatch::index` cache.
- After `import "@hatch:foo" for ` — list public symbols of foo.
- Inside method body — locals + class fields + module-level decls.

Completion items carry doc-summary previews (the first paragraph of
the symbol's docstring) when available.

## Editor integrations

| Editor    | Path                                              |
|-----------|---------------------------------------------------|
| VSCode    | New `vscode-wrenlift` extension wrapping the LSP. |
| Neovim    | `nvim-lspconfig` recipe, no extension code.       |
| Helix     | Built-in config block.                            |
| JetBrains | LSP4IJ plugin recipe.                             |

VSCode is the only one we ship a full extension for; the others get a
README snippet. The extension also bundles a `.tmGrammar` for syntax
highlighting because LSP doesn't carry tokens.

## Phasing

### v0 — diagnostic-only server

1. `wlift_lsp` crate: tower-lsp wrapper around parse+sema.
2. `textDocument/{didOpen,didChange,didSave,didClose}` plumbing.
3. `textDocument/diagnostic` (or push diagnostics) reflecting parse
   errors.
4. VSCode extension shell — installs the binary, starts it, surfaces
   diagnostics.

Deliverable: editing a `.wren` file in VSCode shows red squigglies on
syntax errors with the same messages the CLI emits.

### v1 — semantic + hover

5. Sema pass wired into the diagnostic stream.
6. Span → symbol table: every identifier carries its resolved binding
   (decl site, type).
7. `textDocument/hover` showing signature + (when present) doc
   Markdown extracted from the user's local `///` comments.

### v2 — goto-def + references

8. Workspace symbol index. Cross-file references inside the user's
   package.
9. `textDocument/definition`, `textDocument/references`.
10. `textDocument/documentSymbol` for the editor's outline view.

### v3 — hatch deps + prelude

11. Fetch + cache `@hatch:*` bundles using the existing dep walker.
12. Surface dep symbols in hover/goto-def — every cached bundle's
    `Source` sections feed `wren_lift::docs::collect_module`; the
    resulting `ModuleDoc`s sit in a workspace-level `HashMap<pkg,
    ModuleDoc>` keyed by package name. The hover handler walks
    *that* table when the receiver of a `.method` access binds to
    an imported name (e.g. hovering on `.toBe` inside an
    `Expect.that(x).toBe(y)` call falls through to
    `@hatch:assert`'s docs).
13. Prelude docs. The runtime's core classes (Num / String / List
    / Map / Fiber / System / Browser / Dom / …) are Rust-defined
    so the generator can't see them today. Ship Rust-side
    `///`-doc-only Wren stubs at `src/runtime/prelude/*.wren`,
    collect them at build time into a static `prelude_doc_model`,
    and prepend it to the workspace lookup table. Same blob
    embedded into `wlift_wasm` so the playground hover /
    completion gets prelude docs without a network round-trip.
14. Completion across hatch deps after `import "@hatch:foo" for `.
15. `workspace/executeCommand`: `wlift-lsp.reloadDeps`.

### v4 — completion + rename + inlay

15. General-purpose completion (members, locals, scoped imports).
16. `textDocument/rename` — workspace-wide identifier rename, scoped
    to the user's package (won't rewrite hatch deps).
17. Optional inlay hints for inferred types (`var foo = ...` →
    `: Class`).

### v5 — performance

18. Incremental re-parse using ropes + dirty-region tracking.
19. Salsa-style memoization keyed on (file_id, content_hash) for sema
    pass.
20. Multi-file projects of ~500 files should respond to keystrokes in
    < 50 ms p95.

## Crate layout

```
crates/
  wlift_lsp/                # The server binary + library
  wlift_lsp/src/server.rs   # tower-lsp glue
  wlift_lsp/src/workspace.rs# document store + dep cache
  wlift_lsp/src/symbols.rs  # span → symbol resolver
  wlift_lsp/src/handlers/   # one file per LSP method
```

Re-exports `wren_lift::parse` and `wren_lift::sema`. Depends on
`wlift_docs` (separate plan) for hover content rendering.

## Distribution

- The server binary ships in the standard wlift release archive
  alongside `wlift` and `hatch`.
- VSCode extension published to the marketplace (separate workflow,
  versioned independently).
- The install script gains a `--with-lsp` flag (or always installs
  it; the binary is small).

## Open questions

- Do we expose a "force-reparse all" command to debug stuck states,
  or rely on workspace reload? Lean: ship the command behind a
  hidden config flag.
- Diagnostic source attribution: should sema warnings be tagged as
  coming from `wlift-sema` vs `wlift-parse` so editors group them?
  Lean yes.
- Workspace boundary detection: hatchfile root, or LSP `rootUri`?
  Should be hatchfile if present, fall back to rootUri. Multi-root
  workspaces with separate hatchfiles need one server instance per
  hatchfile (handled by spawning a child server, or by namespacing
  workspaces in one server). Decide in v2.
- Can sema run safely on partial / mid-edit source? It currently
  asserts well-formed AST. Need a "best-effort" mode that returns
  Ok with diagnostics instead of panicking on malformed input. v0
  blocker for the diagnostic pass.

## Dependencies on the docs plan

- v1 hover content needs the docs generator's symbol model. Without
  it, hover degrades to signature-only.
- v3 hatch-dep hover requires the docs JSON manifest to be packaged
  into each `.hatch` release artifact (covered in docs-plan v2).
- The crate split (`wlift_docs` separate from CLI glue) is what
  makes these dependencies clean.
