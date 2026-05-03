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
