import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;

/// Resolve `${workspaceFolder}` and `${workspaceFolder:NAME}` in a
/// user-supplied path. VS Code only substitutes these tokens for
/// `launch.json` / `tasks.json` itself; arbitrary extension
/// settings come back as literal strings, so we do it here.
function resolveVariables(input: string): string {
  return input
    .replace(/\$\{workspaceFolder\}/g, () => {
      const f = vscode.workspace.workspaceFolders?.[0];
      return f ? f.uri.fsPath : "";
    })
    .replace(/\$\{workspaceFolder:([^}]+)\}/g, (_m, name) => {
      const f = vscode.workspace.workspaceFolders?.find(
        (w) => w.name === name,
      );
      return f ? f.uri.fsPath : "";
    });
}

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("wrenlift");
  const rawCommand = config.get<string>("serverPath") || "wlift-lsp";
  const command = resolveVariables(rawCommand);

  // The wlift-lsp binary is built from the same workspace as the
  // wlift runtime. The standard install script (install.sh) drops
  // it on $PATH alongside `wlift` and `hatch`; users who built
  // from source can point `wrenlift.serverPath` at their target
  // directory.
  const serverOptions: ServerOptions = {
    run: { command, transport: TransportKind.stdio },
    debug: { command, transport: TransportKind.stdio },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "wren" }],
    synchronize: {
      configurationSection: "wrenlift",
      // Watch the workspace's hatchfile so the server can pick
      // up version bumps + new dependencies without a manual
      // reload.
      fileEvents: vscode.workspace.createFileSystemWatcher("**/hatchfile"),
    },
    outputChannelName: "WrenLift",
  };

  client = new LanguageClient(
    "wrenlift",
    "WrenLift Language Server",
    serverOptions,
    clientOptions,
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("wrenlift.restartServer", async () => {
      if (!client) return;
      await client.stop();
      await client.start();
    }),
  );

  client.start().catch((err) => {
    vscode.window.showErrorMessage(
      `Failed to start wlift-lsp at "${command}": ${err}. ` +
        `Install via curl -fsSL https://raw.githubusercontent.com/wrenlift/WrenLift/main/install.sh | bash, ` +
        `or set "wrenlift.serverPath" to an absolute path.`,
    );
  });
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
