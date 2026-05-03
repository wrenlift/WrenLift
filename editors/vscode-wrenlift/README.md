# WrenLift for Visual Studio Code

[Wren](https://wren.io) language support powered by the
[WrenLift](https://github.com/wrenlift/WrenLift) runtime and its
companion language server.

## Features

- **Syntax highlighting** for `.wren` files: keywords, classes,
  fields (`_name` instance, `__name` static), strings with
  `%(…)` interpolation, doc comments (`///`, `//!`), and Hatch
  cross-target attributes (`#!native`, `#!wasm`).
- **Diagnostics** as you type: parse errors with the same messages
  the CLI emits.
- **Hover** for identifiers: signature + rendered Markdown from
  `///` doc comments.
- **Goto-definition, completion, signature help** — landing in
  upcoming versions.

## Requirements

The extension expects `wlift-lsp` on `$PATH`. The standard install
script ships it alongside `wlift` and `hatch`:

```sh
curl -fsSL https://raw.githubusercontent.com/wrenlift/WrenLift/main/install.sh | bash
```

If you've built from source, point
`wrenlift.serverPath` at your `target/release/wlift-lsp`
binary in VS Code settings.

## Settings

| Setting | Default | Description |
|---|---|---|
| `wrenlift.serverPath` | `wlift-lsp` | Path to the language-server binary. |
| `wrenlift.trace.server` | `off` | Trace LSP messages in the Output panel. |

## Commands

| Command | Description |
|---|---|
| `WrenLift: Restart Server` | Stop and re-spawn the language server. |

## Repository

Source lives at
[wrenlift/WrenLift](https://github.com/wrenlift/WrenLift) under
`editors/vscode-wrenlift/`. Bug reports and pull requests welcome.

## Local development

```sh
cd editors/vscode-wrenlift
npm install
npm run compile
```

Open the folder in VS Code and press **F5**. A second VS Code
window opens with the sample workspace under `sample/`
(`hatchfile`, `main.wren`, a copy of `@hatch:fmt`'s source).
Diagnostics + hover should fire on the real `.wren` files.

The sample window picks up `wlift-lsp` from `$PATH`. To test
against a local debug build, set `wrenlift.serverPath` in the
sample window's settings.json to your build output, e.g.
`/Users/you/.../wren_lift/target/release/wlift-lsp`.

`npm run watch` keeps the TS compiler running in the
background; the Extension Development Host picks up the
recompiled bundle on the next `Developer: Reload Window`.
