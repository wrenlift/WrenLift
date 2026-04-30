//! `wlift-lsp` — Language Server for Wren / WrenLift / Hatch.
//!
//! v0: diagnostic-only. Server reads stdin, accepts open / change /
//! save / close notifications, re-parses on every edit, and pushes
//! parse + lex errors back to the editor as LSP diagnostics.
//!
//! Hover / goto-def / completion / rename land in later phases per
//! the [LSP plan](../../../docs/lsp-plan.md). The JSON doc model
//! from `wren_lift::docs` is the same model the hover handler will
//! consume — no second source of truth.

use std::path::PathBuf;
use std::sync::RwLock;

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use wren_lift::diagnostics::{Diagnostic as WlDiagnostic, Severity as WlSeverity};
use wren_lift::hatch::{Dependency, Manifest};
use wren_lift::parse::parser::parse;
use wren_lift::sema::resolve as sema_resolve;

#[derive(Debug)]
struct Backend {
    client: Client,
    /// Per-document state. `Url` keyed because that's what the
    /// editor sends; we store the raw text + a precomputed line-start
    /// index so we can map byte offsets to LSP `Position`s without
    /// re-scanning the file on every diagnostic.
    docs: DashMap<Url, Document>,
    /// Workspace state — the project's hatchfile root + parsed
    /// manifest. Single workspace per server for now; multi-root
    /// support lives in v2 of the LSP plan.
    workspace: RwLock<Workspace>,
}

#[derive(Debug, Default)]
struct Workspace {
    /// Root directory of the project (where the hatchfile lives).
    /// `None` when the editor opened a stand-alone file.
    root: Option<PathBuf>,
    /// Parsed `hatchfile` manifest, or `None` if absent / failed
    /// to parse. We don't surface parse errors here — the user
    /// edits the manifest in a different editor flow.
    manifest: Option<Manifest>,
}

#[derive(Debug)]
struct Document {
    text: String,
    /// Byte offsets of the start of each line (line 0 = 0,
    /// line 1 = byte after the first '\n', …). Always has at
    /// least one entry.
    line_starts: Vec<usize>,
    /// Cached doc model — populated when parse + sema produce a
    /// usable AST. Used by `textDocument/hover` to look up class /
    /// method docs without re-parsing.
    docs: Option<wren_lift::docs::ModuleDoc>,
}

impl Document {
    fn new(text: String) -> Self {
        let line_starts = compute_line_starts(&text);
        Self {
            text,
            line_starts,
            docs: None,
        }
    }

    /// Convert a byte offset into a UTF-16-encoded LSP `Position`.
    /// The protocol uses UTF-16 code units in column counts; we
    /// translate by re-encoding the prefix of the line.
    fn byte_to_position(&self, byte: usize) -> Position {
        let byte = byte.min(self.text.len());
        // Binary search for the line that contains `byte`.
        let line = match self.line_starts.binary_search(&byte) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };
        let line_start = self.line_starts[line];
        let prefix = &self.text[line_start..byte];
        let utf16_col: u32 = prefix.encode_utf16().count() as u32;
        Position {
            line: line as u32,
            character: utf16_col,
        }
    }

    fn byte_range_to_lsp(&self, span: std::ops::Range<usize>) -> Range {
        Range {
            start: self.byte_to_position(span.start),
            end: self.byte_to_position(span.end),
        }
    }

    /// Convert an LSP `Position` (UTF-16 line + column) into a UTF-8
    /// byte offset. Inverse of `byte_to_position`.
    fn position_to_byte(&self, pos: Position) -> usize {
        let line_idx = (pos.line as usize).min(self.line_starts.len().saturating_sub(1));
        let line_start = self.line_starts[line_idx];
        let line_end = self
            .line_starts
            .get(line_idx + 1)
            .copied()
            .unwrap_or(self.text.len());
        let line_text = &self.text[line_start..line_end];
        let mut utf16_left = pos.character as usize;
        for (byte_off, ch) in line_text.char_indices() {
            let units = ch.len_utf16();
            if utf16_left < units {
                return line_start + byte_off;
            }
            utf16_left -= units;
        }
        line_end
    }
}

fn compute_line_starts(text: &str) -> Vec<usize> {
    let mut out = Vec::with_capacity(64);
    out.push(0);
    for (i, ch) in text.char_indices() {
        if ch == '\n' {
            out.push(i + 1);
        }
    }
    out
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, p: InitializeParams) -> Result<InitializeResult> {
        // Workspace root: prefer the modern `workspace_folders`,
        // fall back to the deprecated single `root_uri`. We only
        // care about file:// URIs — remote workspaces don't have
        // a hatchfile we can read.
        let root: Option<PathBuf> = p
            .workspace_folders
            .as_ref()
            .and_then(|folders| folders.first())
            .map(|f| f.uri.clone())
            .or(p.root_uri.clone())
            .and_then(|uri| uri.to_file_path().ok());

        let mut manifest = None;
        if let Some(ref root_path) = root {
            let hatchfile = root_path.join("hatchfile");
            if let Ok(text) = std::fs::read_to_string(&hatchfile) {
                match toml::from_str::<Manifest>(&text) {
                    Ok(m) => manifest = Some(m),
                    Err(e) => {
                        // Surface this as a one-off log message so
                        // the user knows their hatchfile failed to
                        // parse — without it, hover-on-import
                        // silently shows nothing.
                        self.client
                            .log_message(
                                MessageType::WARNING,
                                format!("hatchfile parse failed: {e}"),
                            )
                            .await;
                    }
                }
            }
        }
        let manifest_loaded = manifest.is_some();
        if let Ok(mut ws) = self.workspace.write() {
            ws.root = root.clone();
            ws.manifest = manifest;
        }
        // One-shot log so an editor session can confirm the
        // workspace + manifest were actually picked up.
        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "wlift-lsp workspace: root={:?} manifest_loaded={}",
                    root, manifest_loaded
                ),
            )
            .await;

        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "wlift-lsp".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        // Full sync — v0 keeps a per-document
                        // string and re-parses from scratch. Move
                        // to incremental sync + rope storage in v5
                        // when perf actually matters.
                        change: Some(TextDocumentSyncKind::FULL),
                        save: Some(TextDocumentSyncSaveOptions::Supported(true)),
                        ..Default::default()
                    },
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                ..Default::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "wlift-lsp ready")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, p: DidOpenTextDocumentParams) {
        let uri = p.text_document.uri.clone();
        self.docs
            .insert(uri.clone(), Document::new(p.text_document.text));
        self.publish_diagnostics(&uri).await;
    }

    async fn did_change(&self, p: DidChangeTextDocumentParams) {
        let uri = p.text_document.uri.clone();
        // FULL sync mode → exactly one change carrying the whole
        // document text. (TextDocumentContentChangeEvent.range is
        // `None` in that mode.)
        if let Some(change) = p.content_changes.into_iter().next() {
            self.docs.insert(uri.clone(), Document::new(change.text));
            self.publish_diagnostics(&uri).await;
        }
    }

    async fn did_save(&self, p: DidSaveTextDocumentParams) {
        // The didChange handler already keeps the doc fresh; the
        // save hook is here so future passes (e.g. running
        // expensive sema-only checks) can hook on save without
        // re-routing.
        let uri = p.text_document.uri.clone();
        self.publish_diagnostics(&uri).await;
    }

    async fn did_close(&self, p: DidCloseTextDocumentParams) {
        let uri = p.text_document.uri;
        self.docs.remove(&uri);
        // Per protocol, clear diagnostics for files we no longer
        // own — otherwise stale red squigglies stay in the
        // editor's problem list.
        self.client
            .publish_diagnostics(uri, Vec::new(), None)
            .await;
    }

    async fn hover(&self, p: HoverParams) -> Result<Option<Hover>> {
        let uri = p.text_document_position_params.text_document.uri;
        let pos = p.text_document_position_params.position;
        let Some(doc) = self.docs.get(&uri) else {
            return Ok(None);
        };
        let byte = doc.position_to_byte(pos);

        // Check for hover-on-import-string first — that's what
        // turns workspace hatchfile state into a useful tooltip,
        // and it doesn't need the doc model to be ready.
        if let Some(hover) = self.hover_import_at(&doc, byte) {
            return Ok(Some(hover));
        }

        let Some(module) = doc.docs.as_ref() else {
            return Ok(None);
        };
        Ok(hover_at(module, byte, &doc))
    }
}

impl Backend {
    /// Detect `import "..."` strings in the source and produce a
    /// hover when the cursor lands inside one. Resolution against
    /// the workspace hatchfile gives the editor a one-line summary
    /// of the imported package — version, description (if any).
    fn hover_import_at(&self, doc: &Document, byte: usize) -> Option<Hover> {
        let span = find_import_string(&doc.text, byte)?;
        let imported = doc.text.get(span.clone())?;
        let ws = self.workspace.read().ok()?;
        let manifest = ws.manifest.as_ref()?;
        let dep = manifest.dependencies.get(imported)?;
        let mut markdown = format!("```toml\n\"{imported}\" = ");
        match dep {
            Dependency::Version(v) => {
                markdown.push_str(&format!("\"{v}\"\n```"));
            }
            Dependency::Path { path, .. } => {
                markdown.push_str(&format!("{{ path = \"{path}\" }}\n```"));
            }
            Dependency::Git {
                git, tag, rev, branch, ..
            } => {
                let mut parts = vec![format!("git = \"{git}\"")];
                if let Some(t) = tag {
                    parts.push(format!("tag = \"{t}\""));
                }
                if let Some(r) = rev {
                    parts.push(format!("rev = \"{r}\""));
                }
                if let Some(b) = branch {
                    parts.push(format!("branch = \"{b}\""));
                }
                markdown.push_str(&format!("{{ {} }}\n```", parts.join(", ")));
            }
            // Direct-URL deps are wasm-runtime-only (the host CLI
            // doesn't accept them) — show the URL anyway so the
            // hover stays useful when the LSP runs against a web
            // workspace.
            other => {
                markdown.push_str(&format!("{:?}\n```", other));
            }
        }
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: Some(doc.byte_range_to_lsp(span)),
        })
    }
}

/// Walk the source for an `import "..."` whose quoted string
/// contains `byte`. Returns the byte range *inside* the quotes
/// (excluding them) so the hover lights up the package name only.
fn find_import_string(src: &str, byte: usize) -> Option<std::ops::Range<usize>> {
    // Scan line-by-line — there's at most one `import` per line in
    // idiomatic Wren. Edge cases (the same import appearing in a
    // string literal somewhere) don't matter for v0; we err on
    // the side of "no hover" in those cases.
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            i += 1;
        }
        if !src[i..].starts_with("import") {
            // Skip to next line.
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            i += 1;
            continue;
        }
        // Find the opening quote.
        let mut j = i + "import".len();
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'"' {
            i = j;
            continue;
        }
        let quote_start = j + 1;
        let mut k = quote_start;
        while k < bytes.len() && bytes[k] != b'"' && bytes[k] != b'\n' {
            k += 1;
        }
        if k >= bytes.len() || bytes[k] != b'"' {
            i = k;
            continue;
        }
        // The cursor counts as on-the-string when it's anywhere
        // from the opening quote through the closing one.
        if (quote_start - 1) <= byte && byte <= k {
            return Some(quote_start..k);
        }
        i = k + 1;
    }
    None
}

/// Find the smallest declaration whose span contains `byte` and
/// build an LSP `Hover` for it. Returns `None` when the cursor
/// isn't on any documented decl.
fn hover_at(
    module: &wren_lift::docs::ModuleDoc,
    byte: usize,
    doc: &Document,
) -> Option<Hover> {
    for class in &module.classes {
        if !contains(&class.span, byte) {
            continue;
        }
        // Cursor inside class body — try the methods first.
        for member in &class.members {
            if contains(&member.span, byte) {
                return Some(member_hover(class, member, doc));
            }
        }
        // Class itself.
        return Some(class_hover(class, doc));
    }
    None
}

fn contains(span: &std::ops::Range<usize>, byte: usize) -> bool {
    span.start <= byte && byte < span.end
}

fn class_hover(class: &wren_lift::docs::ClassDoc, doc: &Document) -> Hover {
    let mut markdown = format!("```wren\nclass {}\n```\n", class.name);
    if !class.doc.is_empty() {
        markdown.push('\n');
        markdown.push_str(&class.doc);
    }
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: markdown,
        }),
        range: Some(doc.byte_range_to_lsp(class.span.clone())),
    }
}

fn member_hover(
    class: &wren_lift::docs::ClassDoc,
    member: &wren_lift::docs::MemberDoc,
    doc: &Document,
) -> Hover {
    let mut markdown = format!(
        "```wren\n{}.{}\n```\n",
        class.name, member.signature
    );
    if !member.doc.is_empty() {
        markdown.push('\n');
        markdown.push_str(&member.doc);
    }
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: markdown,
        }),
        range: Some(doc.byte_range_to_lsp(member.span.clone())),
    }
}

impl Backend {
    async fn publish_diagnostics(&self, uri: &Url) {
        // Snapshot the source text + clear any prior cached doc
        // before re-running the pipeline. The collected ModuleDoc
        // gets written back at the end so the hover handler sees
        // the freshest version.
        let text = match self.docs.get(uri) {
            Some(d) => d.text.clone(),
            None => return,
        };
        let pr = parse(&text);

        // Translate parse errors first.
        let mut diags: Vec<Diagnostic> = {
            let doc = self.docs.get(uri).unwrap();
            pr.errors
                .iter()
                .map(|d| translate_diagnostic(&doc, d))
                .collect()
        };

        // Only run sema + collect docs when the parser produced a
        // usable AST. If parse errored, sema would emit confused
        // follow-on diagnostics that overwhelm the editor's
        // problem list.
        let mut module_docs: Option<wren_lift::docs::ModuleDoc> = None;
        if pr.errors.is_empty() {
            let sema = sema_resolve::resolve(&pr.module, &pr.interner);
            {
                let doc = self.docs.get(uri).unwrap();
                for d in &sema.errors {
                    diags.push(translate_diagnostic(&doc, d));
                }
            }
            // Build the doc model from the same parse — feeds the
            // hover handler.
            let module_name = uri
                .path_segments()
                .and_then(|mut s| s.next_back())
                .unwrap_or("module")
                .trim_end_matches(".wren")
                .to_string();
            module_docs = Some(wren_lift::docs::collect_module(
                module_name,
                &text,
                &pr.module,
                &pr.docs,
                &pr.interner,
            ));
        }

        // Write back the doc cache.
        if let Some(mut entry) = self.docs.get_mut(uri) {
            entry.docs = module_docs;
        }

        self.client
            .publish_diagnostics(uri.clone(), diags, None)
            .await;
    }
}

/// Translate a `wren_lift::diagnostics::Diagnostic` into the LSP
/// `Diagnostic` shape, mapping byte spans to UTF-16 positions and
/// stamping `source: "wlift"` so editor problem lists group ours.
fn translate_diagnostic(doc: &Document, diag: &WlDiagnostic) -> Diagnostic {
    let range = match diag.labels.first() {
        Some(l) => doc.byte_range_to_lsp(l.span.clone()),
        None => {
            // Span-less diagnostic. Fall back to position 0:0 —
            // better than dropping the message.
            Range {
                start: Position::default(),
                end: Position::default(),
            }
        }
    };
    Diagnostic {
        range,
        severity: Some(map_severity(diag.severity)),
        source: Some("wlift".into()),
        message: diag.message.clone(),
        ..Default::default()
    }
}

fn map_severity(sev: WlSeverity) -> DiagnosticSeverity {
    match sev {
        WlSeverity::Error => DiagnosticSeverity::ERROR,
        WlSeverity::Warning => DiagnosticSeverity::WARNING,
        WlSeverity::Info => DiagnosticSeverity::INFORMATION,
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        docs: DashMap::new(),
        workspace: RwLock::new(Workspace::default()),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
