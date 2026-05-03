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
  const command = resolveBinary(
    "wlift-lsp",
    config.get<string>("serverPath"),
    "wlift-lsp",
  );

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

/// Probe a binary for its `--version` output. Returns the
/// trimmed first line on success; rejects with the spawn
/// error or the binary's stderr when it fails. Used by the
/// "WrenLift: Show Toolchain Versions" command so users can
/// confirm the runtime + CLI + LSP they have configured.
function probeVersion(cmd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const cp = require("child_process") as typeof import("child_process");
    cp.execFile(cmd, ["--version"], { timeout: 4000 }, (err, stdout, stderr) => {
      if (err) {
        reject((stderr || err.message).toString().trim() || "spawn failed");
        return;
      }
      const out = (stdout || stderr || "").toString().trim().split("\n")[0];
      resolve(out || "(no version output)");
    });
  });
}

/// Walk upward from `filePath`'s directory looking for a
/// sibling `hatchfile`. Returns the directory containing the
/// hatchfile, or undefined if none exists between the file
/// and the filesystem root. Used by `wrenlift.runFile` to
/// decide whether to dispatch through `hatch run` (project-
/// aware) or `wlift <file>` (single-file).
function findHatchfileRoot(filePath: string): string | undefined {
  const path = require("path") as typeof import("path");
  const fs = require("fs") as typeof import("fs");
  let dir = path.dirname(filePath);
  while (true) {
    if (fs.existsSync(path.join(dir, "hatchfile"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) return undefined;
    dir = parent;
  }
}

/// Resolve a toolchain binary name to a usable command. The
/// `install.sh` script drops `wlift`, `hatch`, and `wlift-lsp`
/// into `$HOME/.local/bin` by default, which isn't always on
/// PATH (zsh on macOS, sandboxed shells, fresh installs that
/// haven't sourced their rc yet). Probing that directory first
/// means a curl-installed user gets hits without any extra
/// configuration. Explicit overrides win — when the user has
/// pinned `wrenlift.<name>Path`, we honour it verbatim.
function resolveBinary(
  binaryName: string,
  configured: string | undefined,
  defaultName: string,
): string {
  if (configured && configured !== defaultName) {
    return resolveVariables(configured);
  }
  const os = require("os") as typeof import("os");
  const path = require("path") as typeof import("path");
  const fs = require("fs") as typeof import("fs");
  const localBin = path.join(os.homedir(), ".local", "bin", binaryName);
  if (fs.existsSync(localBin)) {
    return localBin;
  }
  return binaryName;
}

function quoteShell(s: string): string {
  // Single-quote and escape any embedded single quotes so paths
  // with spaces / weird chars survive the shell. Keeps the
  // generated `wlift ...` line copy-pasteable from the terminal.
  return `'${s.replace(/'/g, "'\\''")}'`;
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
    const cmd = resolveBinary(
      "wlift-lsp",
      config.get<string>("serverPath"),
      "wlift-lsp",
    );
    await offerLspInstall(cmd, String(err));
  }
  refreshStatus();
}

/// Actionable failure dialog when the LSP binary can't be
/// found or fails to spawn. The default install path drops
/// `wlift`, `hatch`, and `wlift-lsp` into `$HOME/.local/bin`,
/// but a fresh user pulled in via the marketplace won't have
/// run anything yet — surface a one-click "Install" that runs
/// the same curl-pipe-bash the README documents, so the
/// happy path is "click button → server starts on next
/// retry". `Configure path` opens the relevant setting for
/// people who already have a custom build at hand. Dismiss
/// is the third button so the user can opt out without
/// closing every modal.
async function offerLspInstall(cmd: string, err: string): Promise<void> {
  const INSTALL_URL =
    "https://raw.githubusercontent.com/wrenlift/WrenLift/main/install.sh";
  const choice = await vscode.window.showErrorMessage(
    `WrenLift can't reach the language server at "${cmd}".\n\n${err}`,
    { modal: false },
    "Install via install.sh",
    "Browse to binary…",
    "Open settings",
  );
  if (!choice) return;
  if (choice === "Install via install.sh") {
    if (process.platform === "win32") {
      // install.sh is POSIX bash; PowerShell / cmd users
      // need to grab a binary from the GitHub Releases page
      // by hand. Direct them there instead of typing a
      // command that won't run.
      await vscode.env.openExternal(
        vscode.Uri.parse("https://github.com/wrenlift/WrenLift/releases/latest"),
      );
      vscode.window.showInformationMessage(
        "Windows users: download the matching tarball from the Releases page, drop wlift-lsp on PATH, then run 'WrenLift: Restart Language Server'.",
      );
      return;
    }
    const TERM_NAME = "WrenLift Install";
    let terminal = vscode.window.terminals.find((t) => t.name === TERM_NAME);
    if (!terminal) {
      // No `shellPath` override — VS Code uses the user's
      // configured default integrated-terminal profile (zsh
      // on most macOS, bash/fish on Linux). The curl line
      // explicitly pipes to `bash`, so the host shell only
      // needs to be POSIX-ish to handle the pipe; the
      // installer body runs under bash regardless of what's
      // hosting the terminal.
      terminal = vscode.window.createTerminal({ name: TERM_NAME });
    }
    terminal.show(true);
    terminal.sendText(`curl -fsSL ${INSTALL_URL} | bash`);
    vscode.window.showInformationMessage(
      "Once the install finishes in the terminal, run 'WrenLift: Restart Language Server' to retry.",
    );
  } else if (choice === "Browse to binary…") {
    const picked = await vscode.window.showOpenDialog({
      canSelectFiles: true,
      canSelectFolders: false,
      canSelectMany: false,
      openLabel: "Use this binary",
      title: "Select wlift-lsp binary",
    });
    if (picked && picked[0]) {
      const target =
        vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
          ? vscode.ConfigurationTarget.Workspace
          : vscode.ConfigurationTarget.Global;
      await vscode.workspace
        .getConfiguration("wrenlift")
        .update("serverPath", picked[0].fsPath, target);
      vscode.window.showInformationMessage(
        `wrenlift.serverPath set to ${picked[0].fsPath}. Run 'WrenLift: Restart Language Server' to retry.`,
      );
    }
  } else if (choice === "Open settings") {
    await vscode.commands.executeCommand(
      "workbench.action.openSettings",
      "wrenlift.serverPath",
    );
  }
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
    vscode.commands.registerCommand("wrenlift.showVersions", async () => {
      const cfg = vscode.workspace.getConfiguration("wrenlift");
      const wliftCmd = resolveBinary("wlift", cfg.get<string>("wliftPath"), "wlift");
      const hatchCmd = resolveBinary("hatch", cfg.get<string>("hatchPath"), "hatch");
      const lspCmd = resolveBinary(
        "wlift-lsp",
        cfg.get<string>("serverPath"),
        "wlift-lsp",
      );
      const lines = await Promise.all(
        [
          { name: "wlift", cmd: wliftCmd },
          { name: "hatch", cmd: hatchCmd },
          { name: "wlift-lsp", cmd: lspCmd },
        ].map(async ({ name, cmd }) => {
          try {
            const v = await probeVersion(cmd);
            return `${name}: ${v} (${cmd})`;
          } catch (e) {
            return `${name}: not found at "${cmd}" (${e})`;
          }
        }),
      );
      vscode.window.showInformationMessage(lines.join("\n"), { modal: true });
    }),

    vscode.commands.registerCommand(
      "wrenlift.runFile",
      async (uri: string | vscode.Uri | undefined) => {
        const target =
          typeof uri === "string"
            ? vscode.Uri.parse(uri)
            : uri ?? vscode.window.activeTextEditor?.document.uri;
        if (!target) {
          vscode.window.showWarningMessage(
            "WrenLift: no file to run. Open a .wren file and try again.",
          );
          return;
        }
        const filePath = target.fsPath;
        const cfg = vscode.workspace.getConfiguration("wrenlift");
        const wliftCmd = resolveBinary("wlift", cfg.get<string>("wliftPath"), "wlift");
        const hatchCmd = resolveBinary("hatch", cfg.get<string>("hatchPath"), "hatch");

        // Pick `hatch run` when a hatchfile sits in or above
        // the file's directory and the file is `main.wren`
        // (the conventional entry point) — that gets `hatch`'s
        // dep-resolution + bundle path. Otherwise spawn
        // `wlift <file>` directly. Both cases are overridable
        // via `wrenlift.wliftPath` / `wrenlift.hatchPath`.
        const fileName = filePath.split(/[\\/]/).pop() || "";
        const hatchRoot = findHatchfileRoot(filePath);
        const useHatch = !!hatchRoot && fileName === "main.wren";

        const TERM_NAME = "WrenLift Run";
        let terminal = vscode.window.terminals.find(
          (t) => t.name === TERM_NAME,
        );
        if (!terminal) {
          terminal = vscode.window.createTerminal(TERM_NAME);
        }
        terminal.show(true);
        if (useHatch && hatchRoot) {
          terminal.sendText(
            `cd ${quoteShell(hatchRoot)} && ${quoteShell(hatchCmd)} run`,
          );
        } else {
          terminal.sendText(`${quoteShell(wliftCmd)} ${quoteShell(filePath)}`);
        }
      },
    ),
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
