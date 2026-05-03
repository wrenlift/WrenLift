# `hatch test` — first-party `@hatch:test` integration

Today `hatch test <dir>` is a thin wrapper: walk for
`*.spec.wren`, spawn `wlift <spec>` per file, parse the
`ok: N/M passed` line out of stdout. The runner inside
`@hatch:test` writes plain text and exits non-zero on failure;
that's enough for "did anything fail in this directory" but
nothing finer.

The VS Code extension's spec-runner sidebar + per-block
codelens already render every `Test.describe` / `Test.it` in
the file as its own clickable row, but clicking ▶ runs the
whole spec because the runner has no per-case selection.
Fixing the per-test surface end-to-end means treating
`@hatch:test` as a first-party package the `hatch` CLI knows
about, not just a registered dependency.

## Goals

- `hatch test` accepts a `--filter "<group> > <name>"` flag
  that runs only the matching cases.
- `hatch test --json` emits one structured event per case
  start / pass / fail, plus a final summary, so editors can
  render live results.
- The VS Code spec-runner sidebar's per-row ▶ button calls
  `hatch test --json --filter "<label>" <spec>` and the
  extension overlays pass / fail status inline on each tree
  node.
- A failing case prints the same diagnostic the runtime
  emits, anchored to the `Test.it(...)` line in the editor.

## Non-goals (initial release)

- Watch mode (`hatch test --watch` triggers reruns on file
  save). Comes later — depends on the editor / file-system
  watcher work that the LSP already does.
- Parallel case execution. Serial keeps output deterministic;
  parallelism is an opt-in flag once stability is locked in.
- Coverage reporting. Separate effort.

## Architecture

```
hatch test [--filter F] [--json] <path>
        │
        ├── enumerate *.spec.wren under <path>
        ├── for each spec:
        │     ├── build a tiny Wren bootstrap source:
        │     │     import "@hatch:test" for Test
        │     │     Test.filter = "F"            // when --filter
        │     │     Test.reporter = "json"       // when --json
        │     │     import "<absolute spec path>"
        │     ├── write the bootstrap to a tmp file
        │     ├── spawn `wlift --mode interpreter <bootstrap>`
        │     │     in the spec's directory so relative imports
        │     │     resolve and the hatchfile resolver picks
        │     │     up `@hatch:*` deps the spec uses
        │     └── stream stdout — JSON-Lines events when
        │           --json, plain text otherwise
        └── aggregate: summary line + non-zero exit on any fail
```

Two independent runtime surfaces in `@hatch:test`:

1. **`Test.filter = "..."` setter.** When non-empty,
   `Test.run()` skips cases whose `<group> > <name>` label
   doesn't contain the substring.
2. **`Test.reporter = "json"` setter.** Routes case-start /
   case-pass / case-fail to `System.print` as JSON-Lines
   instead of the human-readable `ok: N/M` shape.

Both are static fields on the existing `Test` class; the
runner code reads them inside `Test.run()` and the bootstrap
file the CLI writes is the only thing that ever flips them on.
Plain `wlift foo.spec.wren` still emits the human-readable
output untouched.

## CLI surface

```sh
# Run every case under the current dir (existing behaviour):
$ hatch test .

# Filter by substring against `<group> > <name>` label:
$ hatch test . --filter "toBe on primitives"
$ hatch test . --filter "equality"          # matches the whole group

# Structured JSON output for editor consumption:
$ hatch test ./assert.spec.wren --json
{"event":"start","group":"equality","name":"toBe on primitives"}
{"event":"pass","group":"equality","name":"toBe on primitives","duration_ms":0.4}
{"event":"start","group":"equality","name":"toEqual compares lists structurally"}
{"event":"fail","group":"equality","name":"toEqual compares lists structurally",
 "message":"expected toEqual([1, 2, 3]) but got [1, 2]","span":{"file":"assert.spec.wren","line":13,"col":18}}
{"event":"summary","passed":5,"failed":1,"skipped":18,"duration_ms":2.1}
```

## VS Code integration

The extension's `wrenlift.runSpecCase(node)` command takes the
current `SpecNode` argument unchanged. After this work lands:

- For a `case` or `group` node, build the `<group> > <name>`
  filter (or `<group>` for a describe-block run) and shell out
  to `hatch test --json --filter "<label>" <spec>`.
- Pipe the JSON-Lines stream into a `vscode.Pseudoterminal`
  bound to the existing "WrenLift Run" terminal, decorated
  per-event so the user sees live tick / cross marks instead
  of a wall of text.
- Mirror the per-case status (pass / fail / running) onto the
  spec-runner tree node's `description` field so the sidebar
  reflects the live state.

## Phasing

### Phase 1 — `Test.filter` + `Test.reporter` runtime hooks

Change scope: `hatch/packages/hatch-test/test.wren`. Bump
`@hatch:test` to 0.2.0. Two static setters, two checks inside
`Test.run()`. Existing `Test.run()` callers (every spec file
that ends with `Test.run()`) keep working unchanged — the
defaults match today's behaviour.

### Phase 2 — `hatch test --filter`

Change scope: `src/bin/hatch.rs::cmd_test`. When `--filter` is
present, write a per-spec bootstrap file under `tempfile::tempdir()`,
spawn `wlift` against the bootstrap instead of the raw spec,
parse the same human-readable `ok: N/M` shape on the way out.
Cleanup the tempdir on exit.

### Phase 3 — `hatch test --json`

Add the JSON reporter to `@hatch:test`. Update `cmd_test`'s
output parser to deserialise JSON-Lines when `--json` is set,
and pass through to stdout so the caller (an editor) can
consume directly. The text reporter stays the default.

### Phase 4 — VS Code spec-runner wires it up

Extension changes in `wrenlift.runSpecCase` /
`wrenlift.viewSpecOutput`. Drop the "runs the whole file"
note from the README + sidebar tooltip when this lands.

## Open questions

- Should the filter accept a regex, a glob, or stay
  substring-only? Lean: substring for the filter flag; let
  the editor compose precise labels. Regex is a separate
  flag if it earns its keep.
- Bootstrap file lifetime — pre-write per spec or generate
  per-run? Per-run keeps cleanup simple; per-spec lets us
  skip writes when re-running the same case repeatedly.
- Streaming watch-mode triggers — share the LSP's file
  watcher, or stand up a separate `notify` watcher inside
  `hatch test --watch`? Decide when watch lands.
