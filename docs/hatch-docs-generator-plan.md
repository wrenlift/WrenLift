# Hatch docs generator — plan

A `hatch docs` subcommand that turns a hatch package's Wren source into
browsable API documentation. Companion to the LSP (separate plan): both
share the same parser + semantic frontend, so the doc generator gets
type-resolved hover-style content for free.

## Goals

- Zero-config: `hatch docs` in a package directory writes a static HTML
  site under `./docs/`. No additional config file required.
- Markdown-first comment format. Authors write CommonMark in `///` doc
  comments; the generator handles rendering, syntax-highlighting code
  fences, and resolving cross-references.
- Aggregate index: a single registry build can collect docs from every
  published `@hatch:*` package and link them together (`@hatch:gpu`'s
  `Renderer2D.draw` linkable from `@hatch:game`'s tutorials).
- Same source the LSP uses for hover. No second parser, no doc-only
  AST. The LSP's hover handler asks the doc renderer for the same
  Markdown bytes that the static site embeds.

## Non-goals (initial release)

- Running examples to verify they compile / pass. (Phase 4.)
- Multi-version docs on the same site. (Phase 5.)
- Bidirectional source links (clicking a method jumps to the line).
  Source links are easy; we just don't ship them in v0.
- Theming. The generator ships one cohesive theme matching
  `wrenlift.com`'s palette; users can override CSS but template
  customization isn't a v0 surface.

## Comment format spec

We standardize on a small Rust-influenced convention. Authors only need
to learn three rules.

### Doc comment kinds

| Prefix  | Position                                  | Renders as          |
|---------|-------------------------------------------|---------------------|
| `///`   | Immediately above a decl                  | Doc for that decl   |
| `//!`   | Top of a file (no leading non-`//!` text) | Module-level doc    |
| `//`    | Anywhere                                  | Internal — ignored  |

Existing `//` comments remain as code annotations; tooling looks at
`///` and `//!` only. Migration is opt-in per file.

### Body is CommonMark

```wren
/// Spawns a new fruit at the bottom of the stage.
///
/// `vy` is sampled uniformly in [-1100, -900] so the fruit clears
/// the canopy. The launch position is randomized within the
/// playable bounds.
///
/// ## Example
///
/// ```wren
/// game.spawn(40, 520)
/// ```
///
/// See also: [Game.update], [@hatch:gpu Renderer2D].
spawn(x, y) { ... }
```

### Special blocks (optional)

Treated as Markdown headings, not bespoke tags:

- `## Example` / `## Examples` — promoted into the example slot. Code
  fences inside this block get the "run in playground" affordance in
  Phase 4.
- `## Parameters` / `## Returns` / `## Throws` — picked up for the
  signature panel sidebar.

### Cross-references

Bracketed identifiers are auto-resolved against the package's symbol
table:

- `[ClassName]` → link to that class page.
- `[Class.method]` → link to the method anchor.
- `[@hatch:foo]` → link to the foo package's index page (or unresolved
  if the registry index isn't built).
- `[#tag]` → link to a heading in the same doc.

If a reference can't resolve, the rendered output keeps the brackets
and emits a build warning. Unresolved cross-refs do not block the
build.

### One-sentence summary

The first paragraph of any doc is treated as the symbol's summary —
shown in indices, in LSP completion previews, and in the parent
class's method list. Keep it short.

## Generator architecture

```
hatch source files          ┐
hatchfile                   ├─►  collector  ─►  symbol table
sema-resolved types         │                      │
                            │                      ▼
                            │                  resolver  ─►  doc model
                            │                      │
                            ▼                      ▼
                        markdown                renderer  ─►  HTML / JSON
                       (commentmark)
```

### `collector`

Walks the package's modules using the existing `parse::parse_module` +
`sema::resolve` pipeline (so we get class membership, method
arities, inferred types). For each declaration it captures:

- Span (file, line, column range).
- `///` comment block immediately above the declaration token.
- `//!` block at the top of the file (module level).
- Visibility — currently Wren classes are all public; we expose a
  hidden-from-docs marker via a `@private` line tag.

### `resolver`

Cross-reference pass. Parses bracketed identifiers in each doc
Markdown blob and tries to bind them to symbols collected above. Falls
back to looking up the package registry index for `@hatch:*` references.

### `renderer`

Two output modes:

1. **Static HTML site** (`hatch docs --out docs/`) — one HTML page per
   class plus a package index, sidebar with nav, search index built
   from the symbol table.
2. **JSON manifest** (`hatch docs --json`) — machine-readable model
   the LSP consumes for hover content. No HTML rendering on the hot
   path.

Same code path; the JSON output is a serialization step before the
HTML pass.

### Markdown engine

`pulldown_cmark` (already in the dependency tree of `ariadne`-adjacent
crates). Configure for:

- GFM-flavoured tables.
- Smart-quote pass off (we need verbatim quotes inside code samples).
- Custom code-fence handler that runs Wren highlighting via the same
  highlighter the playground uses.

## Phasing

### v0 — parse + render

1. Add `///` and `//!` recognition to the lexer/parser. Keep them as
   trivia attached to the next token; don't mutate AST shape.
2. `hatch docs` subcommand. Collects, resolves, renders one HTML page
   per public class, plus a package index. No cross-package linking
   yet.
3. Sidebar nav, theme matching the wrenlift.com landing page.

Deliverable: `hatch docs` in `@hatch:gpu` writes a browsable site
showing every class and method with its summary + body Markdown.

### v1 — cross-references

4. Symbol table indexes class names + method names per package.
   Bracketed `[Class]` / `[Class.method]` rewrites as links.
5. CLI flag `--registry-index <url>` loads a JSON aggregate of
   published packages so `[@hatch:foo Bar]` links work.

### v2 — registry aggregate

6. CI step in `publish-pkg.yml` runs `hatch docs --json`, attaches
   the manifest to the GitHub release alongside the `.hatch` bundle.
7. Aggregate builder fetches all current package manifests, emits the
   registry index above + a top-level "browse all packages" site at
   `wrenlift.com/docs/`.

### v3 — example validation

8. Each `## Example` Wren code fence is run through `wlift --check`
   (parse + sema only). Errors fail the doc build, with a labelled
   ariadne span pointing at the bad example.
9. Optional `--run-examples` flag actually executes them and asserts
   their output matches the next prose paragraph (a la Rust
   doctests).

### v4 — LSP integration

10. The LSP server (separate plan) imports the generator's collector +
    renderer crates and serves rendered Markdown blobs in
    `textDocument/hover` responses. No second doc parser.

## Standardization workstream

Concurrent with v0:

- Audit existing `@hatch:*` packages. Today most have a top-of-file
  `//` comment block; convert those to `//!` (mechanical rename).
  Method docs are sparse and need authoring.
- Publish a `STYLE.md` in the hatch repo describing the convention,
  with three or four worked examples.
- Update the `wlift package new` template (if/when it exists, else
  the closest scaffold) to include the standardized doc skeleton.

## Built-in / prelude docs

The generator only sees `///` comments in user-authored Wren
source. The runtime's prelude — `Num`, `String`, `List`, `Map`,
`Fiber`, `System`, `Browser`, `Dom`, etc. — lives in Rust, so its
methods are invisible to the collector and to every consumer
(LSP hover, playground autocomplete, generated package sites).

Resolution path:

1. **Rust-side stub modules.** Each Rust core class registers a
   parallel "prelude stub" — a `///`-doc-only Wren file that
   declares the same surface (`class Num { /* ... */ }`) without
   bodies. The generator collects from those stubs the same way it
   collects from real source. Lives at `src/runtime/prelude/*.wren`,
   built into the runtime as a static blob, exposed via a new
   `wren_lift::docs::prelude_doc_model()` accessor.
2. **Bundled into every consumer.** The CLI's `--docs` flag emits
   prelude docs alongside user-package docs (or as a sibling
   `prelude.json`). The wasm `wlift_wasm` binary embeds the same
   blob so the playground hover / completion can answer "what's
   `String.split` do?" without a network round-trip.
3. **Imported-dep docs.** Same model the LSP plan v3 hatch-deps
   pass already uses — every installed `.hatch` bundle's source
   sections feed `collect_module`, the resulting `ModuleDoc`s
   land in a workspace-level lookup keyed by package name. Hover
   on `@hatch:assert` `Expect.that(x).toBe(y)` then reaches into
   the assert package's docs.

Phase placement: prelude stubs are a v3 deliverable (after the
v2 registry aggregate so the cross-package linker has a target
to point at). The wasm-side embed is its own follow-up.

## Open questions

- Should `///` doc comments mutate semantics anywhere (e.g. the
  registry's `description` field of a package's hatchfile)? Lean
  no — keep them purely additive.
- Multi-arity methods (`foo()`, `foo(x)`, `foo(x, y)`) each get their
  own anchor or are merged on a single page? Lean: merge, with a
  signatures table at the top.
- Field docs: Wren fields are private and don't have decl sites.
  Doc properties on fields via the getter? Yes — getter is the public
  surface.

## Crate layout

```
crates/
  wlift_docs/         # Library: collector, resolver, renderer
  wlift_docs/cli.rs   # `hatch docs` entry point glue
  wlift_docs/web/     # HTML templates + CSS
```

Splitting from `hatch` proper so the LSP can depend on it without
pulling in the CLI's host concerns.
