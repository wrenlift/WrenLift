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

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use wren_lift::diagnostics::{Diagnostic as WlDiagnostic, Severity as WlSeverity};
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
}

#[derive(Debug)]
struct Document {
    text: String,
    /// Byte offsets of the start of each line (line 0 = 0,
    /// line 1 = byte after the first '\n', …). Always has at
    /// least one entry.
    line_starts: Vec<usize>,
}

impl Document {
    fn new(text: String) -> Self {
        let line_starts = compute_line_starts(&text);
        Self { text, line_starts }
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
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
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
}

impl Backend {
    async fn publish_diagnostics(&self, uri: &Url) {
        let Some(doc) = self.docs.get(uri) else {
            return;
        };
        let pr = parse(&doc.text);
        let mut diags: Vec<Diagnostic> = pr
            .errors
            .iter()
            .map(|d| translate_diagnostic(&doc, d))
            .collect();

        // Only run sema when the parser produced a usable AST. If
        // parse errored, sema would emit confused follow-on
        // diagnostics that overwhelm the editor's problem list.
        if pr.errors.is_empty() {
            let sema = sema_resolve::resolve(&pr.module, &pr.interner);
            for d in &sema.errors {
                diags.push(translate_diagnostic(&doc, d));
            }
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
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
