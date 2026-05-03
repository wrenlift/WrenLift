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

### v0 — diagnostic-only server (DONE)

1. `wlift_lsp` crate: tower-lsp wrapper around parse+sema. ✓
2. `textDocument/{didOpen,didChange,didSave,didClose}` plumbing. ✓
3. `textDocument/diagnostic` (push diagnostics) reflecting parse
   errors. ✓

### v0.5 — VSCode extension shell

Split out from v0 because the server side landed first and the
extension is independently versioned and shipped through the
marketplace. Bundles the LSP client (spawns `wlift_lsp` from
PATH), a TextMate grammar (`syntaxes/wren.tmLanguage.json`) for
syntax highlighting since LSP doesn't carry tokens, and a
language configuration (brackets, comments, indent rules).

Deliverable: editing a `.wren` file in VSCode shows red
squigglies on syntax errors with the same messages the CLI
emits, plus syntax highlighting on every `.wren` file.

### v1 — semantic + hover

4. Sema pass wired into the diagnostic stream. ✓ (parse path
   live; sema-warning attribution still pending)
5. Span → symbol table: every identifier carries its resolved
   binding (decl site, type). ✓ (via `Analysis` in
   `src/docs/hover.rs`)
6. `textDocument/hover` showing signature + (when present) doc
   Markdown extracted from the user's local `///` comments. ✓

### v2 — goto-def + references

8. Workspace symbol index. Cross-file references inside the user's
   package. ✓ (`workspace_modules` in `wlift_lsp::main`; rebuilt
   on file create / delete)
9. `textDocument/definition`. ✓ — covers class names, member
   calls (typed receiver), scoped imports into bundled
   `@hatch:*` source, workspace cross-file class/member matches,
   and bare local references (`var`, `for` binding, method
   parameter, closure parameter).
   `textDocument/references` still pending.
10. `textDocument/documentSymbol` for the editor's outline view.

### v3 — hatch deps + prelude

11. Fetch + cache `@hatch:*` bundles using the existing dep walker. ✓
12. Surface dep symbols in hover/goto-def. ✓ — every cached
    bundle's `Source` sections feed `wren_lift::docs::collect_module`;
    the resulting `ModuleDoc`s sit in a workspace-level vector
    (`dep_docs` in `wlift_lsp::main`). Hover, goto-def, and
    completion all walk that table for imported scoped names.
13. Prelude docs. ✓ — `///`-doc-only Wren stubs at
    `src/runtime/prelude/*.wren` (Object, Class, Bool, Null,
    System, Num, String, Range, Sequence, List, Map, Fiber, Fn,
    TypedArrays, Browser, Dom, Future, WebSocket, Storage)
    collected lazily by `wren_lift::docs::prelude::prelude_docs`.
    The same blob is embedded into `wlift_wasm`.
14. Completion across hatch deps after `import "@hatch:foo" for `. ✓
15. `workspace/executeCommand`: `wlift-lsp.reloadDeps` — pending.
    Today deps refresh on workspace boot (or whenever the
    extension restarts the server).

### v4 — completion + rename + inlay

15. General-purpose completion. ✓ partial — members on a typed
    receiver (sema-resolved local var, field, method param,
    `this`, class name, or bundled-dep ident), literal receivers
    (`(10).`, `"foo".`, `[1,2].`, `{}.`, `true.`, `null.`), and
    bare class names from local + workspace + dep + prelude
    sources. General locals + module decls outside a `.` trigger
    are still pending.
16. `textDocument/rename` — workspace-wide identifier rename,
    scoped to the user's package. Pending.
17. Optional inlay hints for inferred types — pending.

### v5 — codelens + project surface (shipped, unplanned in v0)

18. `textDocument/codeLens` ✓ — file-level `▶ Run <filename>`
    above `main.wren` and `*.spec.wren`, plus per-block `▶ Run
    "<name>"` lenses on every `Test.describe(...)` / `Test.it(...)`
    line in spec files. Lenses dispatch the
    `wrenlift.runFile` command in the VS Code extension.
19. Activity Bar surface ✓ (extension-side). One `wrenlift.runner`
    view container with two contextual welcome states: a
    project scaffolder ("Create new project" → folder picker +
    `hatch init`) when the workspace has no `.wren` /
    `hatchfile`, and a tree of describe/it blocks parsed from
    the active spec file when one is open. Tree nodes carry
    inline ▶ Run / 📄 View output buttons. Per-test filtering
    needs `@hatch:test` runtime support that doesn't exist yet
    (Wren has no env access without a plugin), so today's run
    button executes the whole spec file. The tree node wiring
    is already in place; the upgrade is local once the
    `Test.filter` setter lands.

### v6 — performance

20. Incremental re-parse using ropes + dirty-region tracking.
21. Salsa-style memoization keyed on (file_id, content_hash) for sema
    pass.
22. Multi-file projects of ~500 files should respond to keystrokes in
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
- Can sema run safely on partial / mid-edit source? **Resolved.**
  Sema runs unconditionally now (`pr.errors.is_empty()` gate
  removed in `publish_diagnostics`); the parser's recovery
  preserves enough of the AST that undefined-name checks still
  light up on unaffected regions of the file when an unrelated
  parse error sits elsewhere. The `module_docs` build (hover
  cache) still gates on a clean parse.

## Dependencies on the docs plan

- v1 hover content needs the docs generator's symbol model. Without
  it, hover degrades to signature-only.
- v3 hatch-dep hover requires the docs JSON manifest to be packaged
  into each `.hatch` release artifact (covered in docs-plan v2).
- The crate split (`wlift_docs` separate from CLI glue) is what
  makes these dependencies clean.
