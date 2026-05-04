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
/// Re-entrancy guard for `startServer`. The post-install poll
/// fires `startServer` from a setTimeout chain while the dialog
/// path may still be awaiting `client.start()` (or holding the
/// user inside `offerLspInstall`). Two starts in flight collide
/// on the language client's internal state machine and produce
/// silent failures — the second `start()` resolves immediately
/// against a half-initialised client and never reaches running.
let starting = false;

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

/// Read `client.state` through a function so TypeScript's
/// control-flow narrowing doesn't carry an early-exit check
/// past an `await` boundary inside the post-install poller.
/// The poller calls `startServer` mid-loop, which can rebuild
/// `client` entirely — TS can't see across that mutation.
function clientIsRunning(): boolean {
  return client?.state === State.Running;
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
  if (starting) return;
  if (client && client.state === State.Running) {
    return;
  }
  starting = true;
  try {
    // Always build a fresh client so `resolveBinary` re-runs
    // — if the user just installed the LSP via the dialog,
    // the path needs to be re-resolved against the current
    // filesystem state. Reusing a previously-failed client
    // also lands the runtime in a non-restartable state in
    // some vscode-languageclient versions.
    if (client) {
      // Only call stop() when there's actually a live process
      // to talk to. After a crash loop, vscode-languageclient
      // disposes the connection but `client.stop()` still tries
      // to handshake a shutdown with the dead process and hangs
      // until manually killed. The 2s ceiling is a belt-and-
      // braces guard for the live case where shutdown ack
      // genuinely takes a moment.
      if (
        client.state === State.Running ||
        client.state === State.Starting
      ) {
        try {
          await client.stop(2000);
        } catch {
          // Stop timeouts surface as rejection — fine, we're
          // about to drop the client reference anyway.
        }
      }
      client = undefined;
    }
    client = buildClient();
    client.onDidChangeState(refreshStatus);
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
  } finally {
    starting = false;
  }
}

/// Poll for the install.sh-default binary path (`~/.local/bin/
/// wlift-lsp`) — that's where the curl-piped installer drops
/// it regardless of the user's configured `serverPath`. Once
/// the binary lands AND is executable, clear any stale
/// absolute-path `serverPath` override so `resolveBinary`'s
/// fall-through picks the fresh install up; then start the
/// server.
///
/// Race window: install.sh runs `mv` then `chmod +x` as two
/// separate steps. Between them, `existsSync` returns true but
/// the file may not yet be executable, so spawning produces an
/// EACCES that surfaces as a silent client.start() rejection.
/// We gate on `accessSync(X_OK)` instead, plus a brief settle
/// delay, and we keep polling until either the client reaches
/// `Running` or the timeout expires — a single `start()`
/// failure no longer terminates the loop.
function pollForBinaryAndRestart(): void {
  const fs = require("fs") as typeof import("fs");
  const os = require("os") as typeof import("os");
  const path = require("path") as typeof import("path");
  const installTarget = path.join(os.homedir(), ".local", "bin", "wlift-lsp");
  const start = Date.now();
  const TIMEOUT_MS = 60000;
  const POLL_MS = 1000;
  const SETTLE_MS = 250;
  let pathReset = false;

  const isReady = (): boolean => {
    try {
      fs.accessSync(installTarget, fs.constants.X_OK);
      return true;
    } catch {
      return false;
    }
  };

  const tick = async () => {
    if (Date.now() - start > TIMEOUT_MS) return;
    if (clientIsRunning()) return;

    if (!isReady()) {
      setTimeout(tick, POLL_MS);
      return;
    }

    // Reset a stale absolute `serverPath` once — only when the
    // configured override points somewhere the binary never
    // landed (a deleted debug build, a tarball path that no
    // longer exists). A healthy override that resolves to the
    // freshly-installed binary stays put.
    if (!pathReset) {
      pathReset = true;
      const cfg = vscode.workspace.getConfiguration("wrenlift");
      const configured = cfg.get<string>("serverPath");
      if (
        configured &&
        configured !== "wlift-lsp" &&
        resolveVariables(configured) !== installTarget &&
        !fs.existsSync(resolveVariables(configured))
      ) {
        const target =
          vscode.workspace.workspaceFolders &&
          vscode.workspace.workspaceFolders.length > 0
            ? vscode.ConfigurationTarget.Workspace
            : vscode.ConfigurationTarget.Global;
        await cfg.update("serverPath", "wlift-lsp", target);
      }
      // Give chmod / fs a beat to settle before the first spawn.
      await new Promise((r) => setTimeout(r, SETTLE_MS));
    }

    await startServer();
    if (!clientIsRunning()) {
      setTimeout(tick, POLL_MS);
    }
  };
  setTimeout(tick, POLL_MS);
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
    // Auto-detect when the install finishes by polling for
    // the binary at its install path. install.sh writes
    // atomically, so the moment `existsSync` returns true the
    // file is fully laid down. 60s window covers a slow
    // network on a modern connection (binary is ~10-20 MB);
    // beyond that the user can hit "Restart Server" by hand
    // — surface that fallback explicitly via the action
    // button below.
    pollForBinaryAndRestart();
    void vscode.window
      .showInformationMessage(
        "Wait for the install to finish in the terminal, then restart the language server. WrenLift will also auto-restart for the next 60 seconds once the binary lands on disk.",
        "Restart server now",
      )
      .then((choice) => {
        if (choice === "Restart server now") {
          void restartServer();
        }
      });
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
  // Cap the stop call: a crashed server's `stop()` can hang
  // forever waiting for an ack from a process that's gone.
  try {
    await client.stop(2000);
  } catch {
    // Timeout / rejection is fine — caller wanted the server
    // gone and we're about to reflect that in the UI anyway.
  }
  refreshStatus();
}

async function restartServer(): Promise<void> {
  // Don't go through stopServer here — startServer's own prelude
  // tears the old client down with the same timeout-bounded
  // shutdown, AND skips the stop entirely when the client is
  // already in a Stopped/Crashed state. That path is what
  // unblocks restart-after-crash, which the simpler stop+start
  // sequence used to deadlock on.
  await startServer();
}

// ---------------------------------------------------------------
// Spec runner sidebar
// ---------------------------------------------------------------

type SpecNode = SpecFileNode | SpecGroupNode | SpecCaseNode;

interface SpecFileNode {
  kind: "file";
  uri: vscode.Uri;
  label: string;
}

interface SpecGroupNode {
  kind: "group";
  uri: vscode.Uri;
  line: number;
  label: string;
}

interface SpecCaseNode {
  kind: "case";
  uri: vscode.Uri;
  line: number;
  group: string | null;
  label: string;
}

interface ParsedSpecBlock {
  kind: "describe" | "it";
  name: string;
  line: number;
}

/// Best-effort line scan for `Test.describe("name") {` and
/// `Test.it("name") {` blocks. Mirrors the LSP-side scan in
/// `wlift_lsp::code_lens` — both are cosmetic; clicking a
/// codelens or a tree node still runs the whole spec file.
/// Per-test filtering needs `@hatch:test` runtime support that
/// hasn't landed yet (Wren has no env access without a plugin),
/// so the granularity here is "we know the case exists, we know
/// where it lives in the file" — not "we can run only this case
/// and capture only its result."
export function scanSpecBlocks(text: string): ParsedSpecBlock[] {
  const out: ParsedSpecBlock[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trimStart();
    let kind: "describe" | "it" | null = null;
    if (trimmed.startsWith("Test.describe(")) kind = "describe";
    else if (trimmed.startsWith("Test.it(")) kind = "it";
    if (!kind) continue;
    const name = extractFirstStringArg(trimmed) ?? "(unnamed)";
    out.push({ kind, name, line: i });
  }
  return out;
}

/// Extract the first string-literal argument from a call line.
/// Handles `\"` escapes inside the literal; falls back to `null`
/// for non-string first args (variable names, expressions).
export function extractFirstStringArg(s: string): string | null {
  const open = s.indexOf("(");
  if (open < 0) return null;
  let i = open + 1;
  while (i < s.length && /\s/.test(s[i])) i++;
  if (s[i] !== '"') return null;
  i++;
  let out = "";
  while (i < s.length) {
    const c = s[i];
    if (c === "\\" && i + 1 < s.length) {
      out += s[i + 1];
      i += 2;
      continue;
    }
    if (c === '"') return out;
    out += c;
    i++;
  }
  return null;
}

class SpecRunnerProvider implements vscode.TreeDataProvider<SpecNode> {
  private readonly _onDidChange = new vscode.EventEmitter<SpecNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChange.event;
  private currentUri: vscode.Uri | undefined;
  /// Keyed by the Test.describe line number so getChildren on
  /// a group can hand back its registered cases without
  /// re-parsing. Rebuilt every refresh.
  private groups = new Map<number, SpecCaseNode[]>();

  refresh(uri?: vscode.Uri): void {
    this.currentUri = uri ?? vscode.window.activeTextEditor?.document.uri;
    if (this.currentUri && !this.currentUri.fsPath.endsWith(".spec.wren")) {
      this.currentUri = undefined;
    }
    this._onDidChange.fire(undefined);
  }

  getTreeItem(node: SpecNode): vscode.TreeItem {
    if (node.kind === "file") {
      const item = new vscode.TreeItem(
        node.label,
        vscode.TreeItemCollapsibleState.Expanded,
      );
      item.iconPath = new vscode.ThemeIcon("file-code");
      item.contextValue = "specFile";
      item.tooltip = node.uri.fsPath;
      return item;
    }
    if (node.kind === "group") {
      const item = new vscode.TreeItem(
        node.label,
        vscode.TreeItemCollapsibleState.Expanded,
      );
      item.iconPath = new vscode.ThemeIcon("symbol-namespace");
      item.contextValue = "specGroup";
      item.command = {
        command: "vscode.open",
        title: "Open",
        arguments: [
          node.uri,
          {
            selection: new vscode.Range(node.line, 0, node.line, 0),
            preserveFocus: true,
          },
        ],
      };
      return item;
    }
    const item = new vscode.TreeItem(
      node.label,
      vscode.TreeItemCollapsibleState.None,
    );
    item.iconPath = new vscode.ThemeIcon("symbol-method");
    item.contextValue = "specCase";
    item.tooltip = node.group ? `${node.group} > ${node.label}` : node.label;
    item.command = {
      command: "vscode.open",
      title: "Open",
      arguments: [
        node.uri,
        {
          selection: new vscode.Range(node.line, 0, node.line, 0),
          preserveFocus: true,
        },
      ],
    };
    return item;
  }

  async getChildren(parent?: SpecNode): Promise<SpecNode[]> {
    if (!this.currentUri) return [];
    const uri = this.currentUri;
    if (!parent) {
      const label = uri.fsPath.split(/[\\/]/).pop() || "";
      return [{ kind: "file", uri, label }];
    }
    if (parent.kind === "file") {
      const bytes = await vscode.workspace.fs.readFile(uri);
      const text = Buffer.from(bytes).toString("utf8");
      const blocks = scanSpecBlocks(text);
      this.groups.clear();
      const groupNodes: SpecGroupNode[] = [];
      const orphanCases: SpecCaseNode[] = [];
      let currentGroup: SpecGroupNode | null = null;
      for (const b of blocks) {
        if (b.kind === "describe") {
          const g: SpecGroupNode = {
            kind: "group",
            uri,
            line: b.line,
            label: b.name,
          };
          groupNodes.push(g);
          this.groups.set(b.line, []);
          currentGroup = g;
        } else {
          const c: SpecCaseNode = {
            kind: "case",
            uri,
            line: b.line,
            group: currentGroup?.label ?? null,
            label: b.name,
          };
          if (currentGroup) {
            this.groups.get(currentGroup.line)!.push(c);
          } else {
            orphanCases.push(c);
          }
        }
      }
      return [...groupNodes, ...orphanCases];
    }
    if (parent.kind === "group") {
      return this.groups.get(parent.line) ?? [];
    }
    return [];
  }
}

/// Refresh the `wrenlift.workspaceEmpty` context flag. Drives the
/// welcome-view visibility — the Activity Bar's WrenLift panel
/// shows the scaffolding prompt when there's nothing Wren-shaped
/// in the workspace, and the spec runner instead when the user
/// already has a project. Excludes `.git`, `node_modules`,
/// `target` so a Rust-host workspace with `wren_lift` as a dep
/// doesn't trip "empty" -> "non-empty" on its own build artifacts.
async function refreshWorkspaceEmptyContext(): Promise<void> {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    await vscode.commands.executeCommand(
      "setContext",
      "wrenlift.workspaceEmpty",
      true,
    );
    return;
  }
  const skip = "**/{node_modules,target,.git,out,.vscode-test}/**";
  const wren = await vscode.workspace.findFiles("**/*.wren", skip, 1);
  if (wren.length > 0) {
    await vscode.commands.executeCommand(
      "setContext",
      "wrenlift.workspaceEmpty",
      false,
    );
    return;
  }
  const hatch = await vscode.workspace.findFiles("**/hatchfile", skip, 1);
  await vscode.commands.executeCommand(
    "setContext",
    "wrenlift.workspaceEmpty",
    hatch.length === 0,
  );
}

function refreshSpecOpenContext(): void {
  const ed = vscode.window.activeTextEditor;
  const isSpec = !!ed && ed.document.uri.fsPath.endsWith(".spec.wren");
  void vscode.commands.executeCommand(
    "setContext",
    "wrenlift.specOpen",
    isSpec,
  );
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

  // Activity Bar: spec runner tree + welcome view. Visibility is
  // gated by `wrenlift.specOpen` / `wrenlift.workspaceEmpty`
  // contexts (see view contributions in package.json). The tree
  // refreshes on active-editor change and on file save so editing
  // a spec adds/removes blocks live.
  const specProvider = new SpecRunnerProvider();
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("wrenlift.runner", specProvider),
  );
  specProvider.refresh();
  refreshSpecOpenContext();
  void refreshWorkspaceEmptyContext();

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => {
      specProvider.refresh();
      refreshSpecOpenContext();
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      const active = vscode.window.activeTextEditor?.document.uri;
      if (active && doc.uri.toString() === active.toString()) {
        specProvider.refresh();
      }
    }),
  );

  // Workspace-empty context: refresh on workspace-folder change
  // and when .wren / hatchfile content lands or disappears. The
  // FileSystemWatcher fires on create/delete; rename surfaces as
  // delete-then-create so it's covered.
  const wrenWatcher = vscode.workspace.createFileSystemWatcher("**/*.wren");
  const hatchWatcher = vscode.workspace.createFileSystemWatcher("**/hatchfile");
  const onFsChange = () => {
    void refreshWorkspaceEmptyContext();
  };
  context.subscriptions.push(
    wrenWatcher,
    hatchWatcher,
    wrenWatcher.onDidCreate(onFsChange),
    wrenWatcher.onDidDelete(onFsChange),
    hatchWatcher.onDidCreate(onFsChange),
    hatchWatcher.onDidDelete(onFsChange),
    vscode.workspace.onDidChangeWorkspaceFolders(onFsChange),
  );

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
        const fileName = filePath.split(/[\\/]/).pop() || "";

        // Refuse non-Wren targets up-front. Without the guard, the
        // welcome view's "run the active file" link or a stale
        // editor focus could fire `wlift <hatchfile>` and the
        // parser would mis-attribute the syntax error to the
        // user's source. The two file shapes WrenLift knows how
        // to run are `*.wren` (any) and the workspace manifest's
        // canonical `main.wren` entry point.
        if (fileName === "hatchfile") {
          vscode.window.showWarningMessage(
            "WrenLift: the hatchfile is the workspace manifest, not Wren source. Open main.wren or a *.spec.wren and run that.",
          );
          return;
        }
        if (!fileName.endsWith(".wren")) {
          vscode.window.showWarningMessage(
            `WrenLift: '${fileName}' isn't a .wren file.`,
          );
          return;
        }

        const cfg = vscode.workspace.getConfiguration("wrenlift");
        const wliftCmd = resolveBinary("wlift", cfg.get<string>("wliftPath"), "wlift");
        const hatchCmd = resolveBinary("hatch", cfg.get<string>("hatchPath"), "hatch");

        // Dispatch policy:
        //
        //   * `main.wren` inside a hatch project → `hatch run <root>`.
        //     Hatch builds + dep-resolves through its bundler, which
        //     is the canonical "run the app" path.
        //   * any other `*.wren` (specs, helper modules, scripts) →
        //     `wlift <file>` directly. wlift's loader walks up from
        //     the file's directory looking for a hatchfile, so
        //     `@hatch:foo` imports still resolve when one is in
        //     scope. This matches what `hatch test` does for spec
        //     files internally.
        //   * `*.wren` with no hatchfile in scope → `wlift <file>`
        //     from the file's dir, no project context.
        //
        // Both binary paths are overridable via
        // `wrenlift.wliftPath` / `wrenlift.hatchPath`.
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
    vscode.commands.registerCommand("wrenlift.refreshSpecs", () => {
      specProvider.refresh();
    }),
    vscode.commands.registerCommand(
      "wrenlift.runSpecCase",
      // Two callers feed different argument shapes:
      //   * Sidebar tree click → first arg is a `SpecNode`,
      //     second arg unset.
      //   * LSP per-block codelens → first arg is the spec
      //     URI as a string, second arg is the
      //     `<group> > <name>` filter label.
      // Resolve both into a `target` URI plus a `filter`
      // substring before the dispatch.
      async (
        arg1: SpecNode | string | undefined,
        arg2: string | undefined,
      ) => {
        let target: vscode.Uri | undefined;
        let filter = "";
        if (typeof arg1 === "string") {
          try {
            target = vscode.Uri.parse(arg1);
          } catch {
            target = undefined;
          }
          filter = arg2 ?? "";
        } else if (arg1 && typeof arg1 === "object" && "uri" in arg1) {
          target = arg1.uri;
          filter =
            arg1.kind === "case"
              ? arg1.group
                ? `${arg1.group} > ${arg1.label}`
                : arg1.label
              : arg1.kind === "group"
                ? arg1.label
                : "";
        } else {
          target = vscode.window.activeTextEditor?.document.uri;
        }

        if (!target) {
          vscode.window.showWarningMessage(
            "WrenLift: open a *.spec.wren file to run.",
          );
          return;
        }
        const filePath = target.fsPath;
        const fileName = filePath.split(/[\\/]/).pop() || "";
        if (!fileName.endsWith(".spec.wren")) {
          // Fall back to the whole-file runner when the user
          // triggered this against something that isn't a spec
          // (active editor focus drift, command-palette
          // invocation outside the sidebar).
          await vscode.commands.executeCommand(
            "wrenlift.runFile",
            target.toString(),
          );
          return;
        }

        const cfg = vscode.workspace.getConfiguration("wrenlift");
        const hatchCmd = resolveBinary("hatch", cfg.get<string>("hatchPath"), "hatch");

        const TERM_NAME = "WrenLift Run";
        let terminal = vscode.window.terminals.find(
          (t) => t.name === TERM_NAME,
        );
        if (!terminal) {
          terminal = vscode.window.createTerminal(TERM_NAME);
        }
        terminal.show(true);
        // `hatch test <spec>` accepts a single *.spec.wren
        // file and writes a per-spec runner that prepends the
        // `Test.filter = ...` setter ahead of the spec source —
        // shared module, ordered before the spec's trailing
        // `Test.run()` so only the matching case actually runs.
        if (filter) {
          terminal.sendText(
            `${quoteShell(hatchCmd)} test ${quoteShell(filePath)} --filter ${quoteShell(filter)}`,
          );
        } else {
          terminal.sendText(
            `${quoteShell(hatchCmd)} test ${quoteShell(filePath)}`,
          );
        }
      },
    ),
    vscode.commands.registerCommand(
      "wrenlift.viewSpecOutput",
      async (_node: SpecNode | undefined) => {
        // The whole-file run dumps into the shared "WrenLift Run"
        // terminal. Until per-case filtering lands, "View output"
        // just focuses that terminal.
        const term = vscode.window.terminals.find(
          (t) => t.name === "WrenLift Run",
        );
        if (term) {
          term.show(false);
        } else {
          vscode.window.showInformationMessage(
            "No spec output yet — click ▶ to run a case first.",
          );
        }
      },
    ),
    vscode.commands.registerCommand("wrenlift.newProject", async () => {
      const name = await vscode.window.showInputBox({
        prompt: "Project name",
        placeHolder: "my-app",
        validateInput: (v) => {
          const t = v.trim();
          if (!t) return "Required.";
          if (!/^[a-zA-Z0-9_-]+$/.test(t)) {
            return "Use letters, numbers, dashes, or underscores only.";
          }
          return null;
        },
      });
      if (!name) return;
      const trimmed = name.trim();

      // Folder picker for the parent location. Defaults to the
      // currently-open workspace if there is one, otherwise the
      // home directory — works whether the user invoked this
      // from a fresh `code` instance with no folder, from inside
      // an existing project, or from the welcome view.
      const cwdFolder = vscode.workspace.workspaceFolders?.[0]?.uri;
      const picked = await vscode.window.showOpenDialog({
        canSelectFiles: false,
        canSelectFolders: true,
        canSelectMany: false,
        defaultUri: cwdFolder,
        openLabel: "Create here",
        title: `Choose where '${trimmed}' should live`,
      });
      if (!picked || picked.length === 0) return;
      const parent = picked[0];
      const projectUri = vscode.Uri.joinPath(parent, trimmed);

      // Refuse upfront if the target subdirectory already exists
      // — `hatch init` would also bail, but surfacing it here
      // gives a cleaner message than a terminal stderr line.
      try {
        const stat = await vscode.workspace.fs.stat(projectUri);
        if (stat) {
          vscode.window.showErrorMessage(
            `'${projectUri.fsPath}' already exists. Pick a different name or location.`,
          );
          return;
        }
      } catch {
        // ENOENT — good, we can scaffold here.
      }

      const cfg = vscode.workspace.getConfiguration("wrenlift");
      const hatchCmd = resolveBinary("hatch", cfg.get<string>("hatchPath"), "hatch");

      // Exec `hatch init <name>` directly so we can await
      // completion and open the resulting folder. A terminal
      // makes the flow feel async + manual; this is fast (no
      // network, just file ops) so a synchronous spawn is fine.
      const cp = require("child_process") as typeof import("child_process");
      try {
        await new Promise<void>((resolve, reject) => {
          cp.execFile(
            hatchCmd,
            ["init", trimmed],
            { cwd: parent.fsPath, timeout: 15000 },
            (err, _stdout, stderr) => {
              if (err) {
                reject(new Error(stderr?.toString().trim() || err.message));
                return;
              }
              resolve();
            },
          );
        });
      } catch (e) {
        vscode.window.showErrorMessage(`hatch init failed: ${(e as Error).message}`);
        return;
      }

      // Open the new project. If a workspace is already open we
      // route to a new window so the user doesn't lose their
      // current context; on a fresh `code` instance with no
      // folder, reuse the current window. The `vscode.openFolder`
      // command takes a URI and an `forceNewWindow` boolean.
      const forceNewWindow = !!cwdFolder;
      await vscode.commands.executeCommand(
        "vscode.openFolder",
        projectUri,
        forceNewWindow,
      );
    }),
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
