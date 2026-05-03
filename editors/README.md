# Editor support

Wren / Hatch tooling integrations live here. The Visual Studio
Code extension is the only one we publish to a marketplace; the
other editors get configuration recipes that point at the same
`wlift-lsp` binary and the shared TextMate grammar.

## Layout

```
editors/
  vscode-wrenlift/         # marketplace-published extension
    syntaxes/
      wren.tmLanguage.json # TextMate grammar — reused below
```

## Editor matrix

| Editor    | Setup |
|-----------|-------|
| VS Code   | Install **WrenLift** from the marketplace. |
| Neovim    | `nvim-lspconfig` → `wlift-lsp` cmd, or use [`mason-lspconfig`] when packaged. Source `wren.tmLanguage.json` via `nvim-treesitter` once a tree-sitter grammar lands; for now, point Neovim at the JSON via [`vim-tmgrammar`]. |
| Helix     | `languages.toml`: `language-server = { command = "wlift-lsp" }` and `language = "wren"`. Helix consumes textmate grammars natively via the JSON. |
| Sublime   | Drop `wren.tmLanguage.json` into a Sublime package; pair with [`LSP`] pointing at `wlift-lsp`. |
| JetBrains | [`LSP4IJ`] plugin: register `wlift-lsp` as the server, scope `*.wren`. |

`wlift-lsp` ships in the standard install:

```sh
curl -fsSL https://raw.githubusercontent.com/wrenlift/WrenLift/main/install.sh | bash
```

[`mason-lspconfig`]: https://github.com/williamboman/mason-lspconfig.nvim
[`vim-tmgrammar`]: https://github.com/dunstontc/vim-tmgrammar
[`LSP`]: https://github.com/sublimelsp/LSP
[`LSP4IJ`]: https://plugins.jetbrains.com/plugin/23257-lsp4ij
