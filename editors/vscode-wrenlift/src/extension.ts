import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  State,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let statusItem: vscode.StatusBarItem | undefined;

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

function buildClient(): LanguageClient {
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

  return new LanguageClient(
    "wrenlift",
    "WrenLift Language Server",
    serverOptions,
    clientOptions,
  );
}

function refreshStatus(): void {
  if (!statusItem) return;
  if (!client) {
    statusItem.text = "$(circle-slash) WrenLift";
    statusItem.tooltip = "WrenLift LSP not started";
    statusItem.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.warningBackground",
    );
    return;
  }
  switch (client.state) {
    case State.Starting:
      statusItem.text = "$(loading~spin) WrenLift";
      statusItem.tooltip = "WrenLift LSP starting…";
      statusItem.backgroundColor = undefined;
      break;
    case State.Running:
      statusItem.text = "$(check) WrenLift";
      statusItem.tooltip =
        "WrenLift LSP running. Click for actions (start / stop / restart).";
      statusItem.backgroundColor = undefined;
      break;
    case State.Stopped:
    default:
      statusItem.text = "$(debug-stop) WrenLift";
      statusItem.tooltip = "WrenLift LSP stopped. Click to start.";
      statusItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground",
      );
      break;
  }
}

async function startServer(): Promise<void> {
  if (client && client.state !== State.Stopped) {
    return;
  }
  if (!client) {
    client = buildClient();
    client.onDidChangeState(refreshStatus);
  }
  refreshStatus();
  try {
    await client.start();
  } catch (err) {
    const config = vscode.workspace.getConfiguration("wrenlift");
    const cmd = resolveVariables(config.get<string>("serverPath") || "wlift-lsp");
    vscode.window.showErrorMessage(
      `Failed to start wlift-lsp at "${cmd}": ${err}. ` +
        `Install via curl -fsSL https://raw.githubusercontent.com/wrenlift/WrenLift/main/install.sh | bash, ` +
        `or set "wrenlift.serverPath" to an absolute path.`,
    );
  }
  refreshStatus();
}

async function stopServer(): Promise<void> {
  if (!client) return;
  await client.stop();
  refreshStatus();
}

async function restartServer(): Promise<void> {
  await stopServer();
  await startServer();
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  // Status bar item: shows the server's lifecycle and acts as the
  // entry point for start / stop / restart via a quick-pick menu.
  // Click target is `wrenlift.showServerActions` so a single
  // command surface drives both keybinding and click access.
  statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100,
  );
  statusItem.name = "WrenLift";
  statusItem.command = "wrenlift.showServerActions";
  context.subscriptions.push(statusItem);
  statusItem.show();

  context.subscriptions.push(
    vscode.commands.registerCommand("wrenlift.startServer", startServer),
    vscode.commands.registerCommand("wrenlift.stopServer", stopServer),
    vscode.commands.registerCommand("wrenlift.restartServer", restartServer),
    vscode.commands.registerCommand("wrenlift.showServerActions", async () => {
      const running = client?.state === State.Running;
      const items: (vscode.QuickPickItem & { id: string })[] = [
        running
          ? {
              id: "restart",
              label: "$(refresh) Restart server",
              description: "Stop then start the language server",
            }
          : {
              id: "start",
              label: "$(play) Start server",
              description: "Spawn wlift-lsp and connect",
            },
        running
          ? {
              id: "stop",
              label: "$(debug-stop) Stop server",
              description: "Disconnect and terminate the language server",
            }
          : {
              id: "noop-stop",
              label: "$(debug-stop) Stop server",
              description: "(server isn't running)",
            },
        {
          id: "output",
          label: "$(output) Open Output panel",
          description: "Show the WrenLift LSP transcript",
        },
      ];
      const picked = await vscode.window.showQuickPick(items, {
        placeHolder: "WrenLift server actions",
      });
      switch (picked?.id) {
        case "start":
          await startServer();
          break;
        case "stop":
          await stopServer();
          break;
        case "restart":
          await restartServer();
          break;
        case "output":
          client?.outputChannel?.show();
          break;
      }
    }),
  );

  await startServer();
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
