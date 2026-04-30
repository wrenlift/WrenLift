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
        let Some(module) = doc.docs.as_ref() else {
            return Ok(None);
        };
        Ok(hover_at(module, byte, &doc))
    }
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
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
