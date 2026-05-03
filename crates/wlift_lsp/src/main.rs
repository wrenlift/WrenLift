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
use wren_lift::hatch::{self, Dependency, Manifest, SectionKind};
use wren_lift::hatch_registry;
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
    /// Doc models for every `[dependencies]` package the workspace
    /// pulls in. Populated by `resolve_dep_docs` after `initialize`
    /// reads the manifest. Hover walks this alongside the prelude
    /// when an `import "@hatch:foo" for X` resolves locally to a
    /// dep symbol.
    dep_docs: RwLock<Vec<wren_lift::docs::ModuleDoc>>,
    /// Doc models for every `*.wren` file under the workspace
    /// root, indexed by file URI. Populated at init by a
    /// shallow walk of the workspace tree, refreshed on
    /// `did_change` for each open document. Goto-definition
    /// uses it to jump from a `Fmt` use in `main.wren` to
    /// `fmt.wren`'s `class Fmt` even when `fmt.wren` isn't
    /// already open in the editor.
    workspace_modules: RwLock<std::collections::HashMap<Url, wren_lift::docs::ModuleDoc>>,
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

        // Kick off dep resolution synchronously during init so the
        // first hover after the editor finishes opening can see the
        // dep docs. The walk hits the local hatch cache when warm
        // and only falls back to the registry CDN on cache miss,
        // so a returning user pays no network latency.
        if manifest_loaded {
            self.resolve_dep_docs().await;
        }

        // Index every `*.wren` file under the workspace so
        // goto-definition can jump into files the user hasn't
        // opened yet. Bounded depth keeps the scan cheap on
        // large checkouts; rare nested layouts open the file
        // first and fall through to the live `did_open` path.
        if let Some(ref root_path) = root {
            self.scan_workspace_modules(root_path).await;
        }

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
                definition_provider: Some(OneOf::Left(true)),
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
        self.client.publish_diagnostics(uri, Vec::new(), None).await;
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
            // No doc model yet — try the prelude path on a raw
            // identifier-under-cursor. (Pre-cache state.)
            return Ok(prelude_hover(&doc, byte));
        };

        // Identifier-under-cursor lookup wins over span-based
        // enclosing-decl: when the cursor sits on a *use* of a
        // name (Renderer2D.new, System.print) we should show
        // *that* class's docs, not the docs of the method we
        // happen to be inside.
        let dep_docs_snapshot: Vec<wren_lift::docs::ModuleDoc> = self
            .dep_docs
            .read()
            .map(|v| v.clone())
            .unwrap_or_default();
        if let Some(h) = identifier_hover(module, &dep_docs_snapshot, &doc, byte) {
            return Ok(Some(h));
        }
        if let Some(h) = prelude_hover(&doc, byte) {
            return Ok(Some(h));
        }
        // Span-based fallback for hovers on the keyword itself
        // (e.g. on the `class` token) or whitespace.
        Ok(hover_at(module, byte, &doc))
    }

    async fn goto_definition(
        &self,
        p: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = p.text_document_position_params.text_document.uri;
        let pos = p.text_document_position_params.position;
        let Some(doc) = self.docs.get(&uri) else {
            return Ok(None);
        };
        let byte = doc.position_to_byte(pos);

        // Cursor inside an `import "..."` string → resolve the
        // path and return the imported file's location. Handles
        // both relative paths (`./fmt`, `../bar/baz`) and absolute
        // module names (which we can't resolve from the file
        // system today; left for v3 of the LSP plan).
        if let Some(loc) = Self::goto_import_at(&uri, &doc, byte) {
            return Ok(Some(GotoDefinitionResponse::Scalar(loc)));
        }

        let Some(module) = doc.docs.as_ref() else {
            return Ok(None);
        };

        // Identifier under cursor.
        let Some(span) = wren_lift::docs::hover::identifier_at(&doc.text, byte) else {
            return Ok(None);
        };
        let Some(ident) = doc.text.get(span.clone()) else {
            return Ok(None);
        };

        // Receiver-scoped member lookup: when the cursor sits on
        // `<recv>.<member>` we look up only that receiver class's
        // member table, not every class in the module.
        let analysis = wren_lift::docs::hover::Analysis::run(&doc.text);
        let receiver_class: Option<String> = analysis.as_ref().and_then(|a| {
            a.receiver_type_for_call_at(byte)
                .and_then(|t| wren_lift::docs::hover::inferred_to_class_name(&t, &a.interner))
        });

        if let Some(recv) = &receiver_class {
            for class in &module.classes {
                if class.name != *recv {
                    continue;
                }
                for member in &class.members {
                    if member.name == ident {
                        return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                            uri: uri.clone(),
                            range: doc.byte_range_to_lsp(member.span.clone()),
                        })));
                    }
                }
            }
        }

        // Class-name jump: cursor on a class identifier itself.
        for class in &module.classes {
            if class.name == ident {
                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range: doc.byte_range_to_lsp(class.span.clone()),
                })));
            }
        }

        // Last resort: any same-named member anywhere in the
        // module. Less precise than the receiver-scoped path but
        // covers bare references in doc bodies and prelude-style
        // free functions sitting inside a single class.
        for class in &module.classes {
            for member in &class.members {
                if member.name == ident {
                    return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                        uri: uri.clone(),
                        range: doc.byte_range_to_lsp(member.span.clone()),
                    })));
                }
            }
        }

        // Cross-file lookup against every other indexed `.wren`
        // in the workspace. Class match wins over member match
        // (jumping to a class definition is more useful than
        // landing inside one of its methods that happens to
        // share a name with the cursor).
        let workspace_index = self
            .workspace_modules
            .read()
            .ok()
            .map(|m| m.clone())
            .unwrap_or_default();
        if let Some(recv) = &receiver_class {
            for (other_uri, other_module) in &workspace_index {
                if other_uri == &uri {
                    continue;
                }
                for class in &other_module.classes {
                    if class.name != *recv {
                        continue;
                    }
                    for member in &class.members {
                        if member.name == ident {
                            return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                                uri: other_uri.clone(),
                                range: byte_range_to_lsp(other_uri, member.span.clone()),
                            })));
                        }
                    }
                }
            }
        }
        for (other_uri, other_module) in &workspace_index {
            if other_uri == &uri {
                continue;
            }
            for class in &other_module.classes {
                if class.name == ident {
                    return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                        uri: other_uri.clone(),
                        range: byte_range_to_lsp(other_uri, class.span.clone()),
                    })));
                }
                for member in &class.members {
                    if member.name == ident {
                        return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                            uri: other_uri.clone(),
                            range: byte_range_to_lsp(other_uri, member.span.clone()),
                        })));
                    }
                }
            }
        }

        Ok(None)
    }
}

/// Translate a byte span into an LSP `Range` against a file we
/// don't have an open `Document` for. Reads the file fresh and
/// computes line starts; cheap enough at goto-def time and
/// avoids carrying a parallel cache for every workspace .wren.
fn byte_range_to_lsp(uri: &Url, span: std::ops::Range<usize>) -> Range {
    let path = match uri.to_file_path() {
        Ok(p) => p,
        Err(_) => return Range::default(),
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Range::default(),
    };
    let line_starts = compute_line_starts(&text);
    let to_pos = |byte: usize| {
        let byte = byte.min(text.len());
        let line = match line_starts.binary_search(&byte) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };
        let col = text[line_starts[line]..byte].encode_utf16().count() as u32;
        Position {
            line: line as u32,
            character: col,
        }
    };
    Range {
        start: to_pos(span.start),
        end: to_pos(span.end),
    }
}

/// Identifier-match version of `hover_at`. Walks the local
/// module's classes / members and the prelude for a name equal
/// to the identifier under the cursor, regardless of whether
/// that identifier appears at a decl site or a use site.
///
/// When the cursor sits on `<Receiver>.<member>`, restricts the
/// member lookup to the class named by the receiver. Without
/// this, `Renderer2D.new` would match FruitSlicer's `new`
/// constructor (or any other `new`) before reaching the right
/// class.
fn identifier_hover(
    module: &wren_lift::docs::ModuleDoc,
    dep_docs: &[wren_lift::docs::ModuleDoc],
    doc: &Document,
    byte: usize,
) -> Option<Hover> {
    let span = wren_lift::docs::hover::identifier_at(&doc.text, byte)?;
    let ident = doc.text.get(span.clone())?;
    if wren_lift::docs::hover::is_keyword(ident) {
        return None;
    }
    let text_receiver = wren_lift::docs::hover::receiver_before(&doc.text, span.start);
    // Sema's typed AST. Drives field/local type splices and the
    // typed-receiver-class lookup below — sema knows `_trail: List`
    // when the field was assigned `[]`, so `_trail.count` resolves
    // against List's `count` instead of the first class with a
    // `count` member.
    let analysis = wren_lift::docs::hover::Analysis::run(&doc.text);
    let typed_receiver_class: Option<String> = analysis.as_ref().and_then(|a| {
        a.receiver_type_for_call_at(byte)
            .and_then(|t| wren_lift::docs::hover::inferred_to_class_name(&t, &a.interner))
    });

    let mut all: Vec<&wren_lift::docs::ModuleDoc> =
        wren_lift::docs::prelude_docs().iter().collect();
    // Order: local module first (shadows everything), then deps,
    // then the prelude. A user-defined `Fmt` in the open file
    // wins over `@hatch:fmt`'s `Fmt`, which in turn wins over
    // any prelude class with the same name.
    for d in dep_docs {
        all.insert(0, d);
    }
    all.insert(0, module);

    // Class-name hover (cursor on the class identifier itself).
    if text_receiver.is_none() {
        for m in &all {
            for class in &m.classes {
                if class.name == ident {
                    let body = format!(
                        "```wren\nclass {}\n```{}{}",
                        class.name,
                        if class.doc.is_empty() { "" } else { "\n\n" },
                        class.doc,
                    );
                    return Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: body,
                        }),
                        range: Some(doc.byte_range_to_lsp(span)),
                    });
                }
            }
        }
        // Bare member reference in the local module — covers
        // doc bodies mentioning a sibling method by name.
        // Prelude is skipped here to avoid `name`/`count` /
        // `print` matching params or locals.
        for class in &module.classes {
            for member in &class.members {
                if member.name == ident {
                    let body = format!(
                        "```wren\n{}\n```{}{}",
                        wren_lift::docs::hover::format_member_sig(&class.name, &member.signature),
                        if member.doc.is_empty() { "" } else { "\n\n" },
                        member.doc,
                    );
                    return Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: body,
                        }),
                        range: Some(doc.byte_range_to_lsp(span)),
                    });
                }
            }
        }
    }

    // Effective receiver class for member lookup. Order:
    //   1. sema's inferred receiver class (typed-AST truth);
    //   2. text receiver, when class-shaped uppercase.
    let effective_receiver: Option<String> = typed_receiver_class.clone().or_else(|| {
        text_receiver.and_then(|r| {
            let first = r.chars().next()?;
            if first.is_ascii_uppercase() {
                Some(r.to_string())
            } else {
                None
            }
        })
    });

    if let Some(recv_class) = effective_receiver.as_deref() {
        for m in &all {
            for class in &m.classes {
                if class.name != recv_class {
                    continue;
                }
                for member in &class.members {
                    if member.name == ident {
                        let body = format!(
                            "```wren\n{}\n```{}{}",
                            wren_lift::docs::hover::format_member_sig(
                                &class.name,
                                &member.signature
                            ),
                            if member.doc.is_empty() { "" } else { "\n\n" },
                            member.doc,
                        );
                        return Some(Hover {
                            contents: HoverContents::Markup(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: body,
                            }),
                            range: Some(doc.byte_range_to_lsp(span)),
                        });
                    }
                }
            }
        }
    }

    // Last-resort scan-all only when sema couldn't pin the
    // receiver type AND the text receiver isn't class-shaped.
    if effective_receiver.is_none() {
        if let Some(recv) = text_receiver {
            let first = recv.chars().next().unwrap_or('_');
            let receiver_looks_like_value = first.is_ascii_lowercase() || first == '_';
            if receiver_looks_like_value {
                for m in &all {
                    for class in &m.classes {
                        for member in &class.members {
                            if member.name == ident {
                                let body = format!(
                                    "```wren\n{}\n```{}{}",
                                    wren_lift::docs::hover::format_member_sig(
                                        &class.name,
                                        &member.signature
                                    ),
                                    if member.doc.is_empty() { "" } else { "\n\n" },
                                    member.doc,
                                );
                                return Some(Hover {
                                    contents: HoverContents::Markup(MarkupContent {
                                        kind: MarkupKind::Markdown,
                                        value: body,
                                    }),
                                    range: Some(doc.byte_range_to_lsp(span)),
                                });
                            }
                        }
                    }
                }
            } else {
                let body_md = format!(
                    "```wren\n{}.{}\n```\n\n`{}` belongs to a class we don't have local docs for — likely an imported `@hatch:*` dep the workspace hasn't loaded yet.",
                    recv, ident, ident
                );
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: body_md,
                    }),
                    range: Some(doc.byte_range_to_lsp(span)),
                });
            }
        }
    }

    // Pass 3 — identifier-kind classifier. Sema's TypeEnv
    // splices the inferred type for fields and locals.
    let (signature, body_text) = wren_lift::docs::hover::identifier_kind_hint(
        &doc.text,
        ident,
        span.start,
        wren_lift::docs::prelude_docs(),
        analysis.as_ref(),
    )?;
    let body_md = format!(
        "```wren\n{}\n```{}{}",
        signature,
        if body_text.is_empty() { "" } else { "\n\n" },
        body_text,
    );
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: body_md,
        }),
        range: Some(doc.byte_range_to_lsp(span)),
    })
}

/// Identifier-under-cursor lookup against the runtime prelude
/// stubs. Mirrors `hover_wren`'s prelude path so the desktop LSP
/// answers the same questions as the playground.
fn prelude_hover(doc: &Document, byte: usize) -> Option<Hover> {
    let span = wren_lift::docs::hover::identifier_at(&doc.text, byte)?;
    let ident = doc.text.get(span.clone())?;
    if wren_lift::docs::hover::is_keyword(ident) {
        return None;
    }
    for prelude_module in wren_lift::docs::prelude_docs() {
        for class in &prelude_module.classes {
            if class.name == ident {
                let body = format!(
                    "```wren\nclass {}\n```{}{}",
                    class.name,
                    if class.doc.is_empty() { "" } else { "\n\n" },
                    class.doc,
                );
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: body,
                    }),
                    range: Some(doc.byte_range_to_lsp(span)),
                });
            }
            for member in &class.members {
                if member.name == ident {
                    let body = format!(
                        "```wren\n{}\n```{}{}",
                        wren_lift::docs::hover::format_member_sig(&class.name, &member.signature),
                        if member.doc.is_empty() { "" } else { "\n\n" },
                        member.doc,
                    );
                    return Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: body,
                        }),
                        range: Some(doc.byte_range_to_lsp(span)),
                    });
                }
            }
        }
    }
    None
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
                git,
                tag,
                rev,
                branch,
                ..
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

    /// Resolve `import "..."` cursors to the imported file's
    /// location. Relative paths (`./fmt`, `../bar/baz`) resolve
    /// against the importing file's directory; scoped names
    /// (`@hatch:foo`) need dep-source URIs we don't have for
    /// `.hatch`-bundled sources, so they return `None` for now.
    fn goto_import_at(uri: &Url, doc: &Document, byte: usize) -> Option<Location> {
        let span = find_import_string(&doc.text, byte)?;
        let imported = doc.text.get(span)?;
        if !imported.starts_with("./") && !imported.starts_with("../") {
            return None;
        }
        let importer_path = uri.to_file_path().ok()?;
        let parent = importer_path.parent()?;
        let mut target = parent.join(imported);
        if target.extension().is_none() {
            target.set_extension("wren");
        }
        if !target.exists() {
            return None;
        }
        let target_uri = Url::from_file_path(&target).ok()?;
        Some(Location {
            uri: target_uri,
            range: Range {
                start: Position { line: 0, character: 0 },
                end: Position { line: 0, character: 0 },
            },
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
fn hover_at(module: &wren_lift::docs::ModuleDoc, byte: usize, doc: &Document) -> Option<Hover> {
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
        "```wren\n{}\n```\n",
        wren_lift::docs::hover::format_member_sig(&class.name, &member.signature)
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
    /// Recursively walk the workspace tree, parse every `*.wren`
    /// file, and stash its `ModuleDoc` in `workspace_modules`
    /// keyed by file URI. Skips `target/`, `node_modules/`,
    /// `.git/`, and `out/` to keep large repos cheap. Files
    /// the user has open as live documents take precedence at
    /// lookup time — those re-parse on every keystroke.
    async fn scan_workspace_modules(&self, root: &std::path::Path) {
        let mut indexed: std::collections::HashMap<Url, wren_lift::docs::ModuleDoc> =
            std::collections::HashMap::new();
        let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
        const SKIP_DIRS: &[&str] = &["target", "node_modules", ".git", "out", ".vscode-test"];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    if path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| SKIP_DIRS.contains(&n) || n.starts_with('.'))
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    stack.push(path);
                } else if file_type.is_file()
                    && path.extension().and_then(|e| e.to_str()) == Some("wren")
                {
                    let Ok(text) = std::fs::read_to_string(&path) else {
                        continue;
                    };
                    let pr = parse(&text);
                    if !pr.errors.is_empty() {
                        continue;
                    }
                    let module_name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    let module = wren_lift::docs::collect::collect_module(
                        &module_name,
                        &text,
                        &pr.module,
                        &pr.docs,
                        &pr.interner,
                    );
                    if let Ok(uri) = Url::from_file_path(&path) {
                        indexed.insert(uri, module);
                    }
                }
            }
        }
        let count = indexed.len();
        if let Ok(mut slot) = self.workspace_modules.write() {
            *slot = indexed;
        }
        self.client
            .log_message(
                MessageType::INFO,
                format!("wlift-lsp: indexed {count} workspace .wren file(s)"),
            )
            .await;
    }

    /// Walk the workspace's `[dependencies]`, fetch each one's
    /// `.hatch` bundle through the registry cache, decode the
    /// `Source` sections, and run `docs::collect::collect_module`
    /// on each so hover can resolve `@hatch:foo`-imported
    /// symbols against the dep's actual `///` blocks. Best-effort:
    /// individual failures (network, parse, missing source) are
    /// logged and skipped so a broken dep doesn't take the whole
    /// workspace's hover off.
    async fn resolve_dep_docs(&self) {
        let manifest = match self.workspace.read() {
            Ok(ws) => ws.manifest.clone(),
            Err(_) => return,
        };
        let Some(manifest) = manifest else { return };

        let registry = hatch_registry::registry_url();
        let cache = match hatch_registry::cache_root() {
            Ok(c) => c,
            Err(e) => {
                self.client
                    .log_message(
                        MessageType::WARNING,
                        format!("wlift-lsp: cache_root unavailable: {e}"),
                    )
                    .await;
                return;
            }
        };
        if let Err(e) = std::fs::create_dir_all(&cache) {
            self.client
                .log_message(
                    MessageType::WARNING,
                    format!(
                        "wlift-lsp: couldn't create cache dir {}: {e}",
                        cache.display()
                    ),
                )
                .await;
            return;
        }

        let mut collected: Vec<wren_lift::docs::ModuleDoc> = Vec::new();
        for (name, dep) in &manifest.dependencies {
            // Only registry-style version pins resolve here. `path`
            // / `git` / `url` deps are out of scope for v3 — those
            // need workspace-relative source loading.
            let version = match dep {
                Dependency::Version(v) => v.clone(),
                _ => continue,
            };
            let path = match hatch_registry::ensure_in_cache_dir(
                &cache, &registry, name, &version,
            ) {
                Ok(p) => p,
                Err(e) => {
                    self.client
                        .log_message(
                            MessageType::INFO,
                            format!("wlift-lsp: skipping {name}@{version}: {e}"),
                        )
                        .await;
                    continue;
                }
            };
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    self.client
                        .log_message(
                            MessageType::INFO,
                            format!(
                                "wlift-lsp: read {} failed: {e}",
                                path.display()
                            ),
                        )
                        .await;
                    continue;
                }
            };
            let bundle = match hatch::load(&bytes) {
                Ok(h) => h,
                Err(e) => {
                    self.client
                        .log_message(
                            MessageType::INFO,
                            format!(
                                "wlift-lsp: decode {} failed: {e}",
                                path.display()
                            ),
                        )
                        .await;
                    continue;
                }
            };
            for section in &bundle.sections {
                if !matches!(section.kind, SectionKind::Source) {
                    continue;
                }
                let Ok(source) = std::str::from_utf8(&section.data) else {
                    continue;
                };
                let pr = parse(source);
                if !pr.errors.is_empty() {
                    continue;
                }
                let module = wren_lift::docs::collect::collect_module(
                    &section.name,
                    source,
                    &pr.module,
                    &pr.docs,
                    &pr.interner,
                );
                collected.push(module);
            }
        }

        let count = collected.len();
        if let Ok(mut slot) = self.dep_docs.write() {
            *slot = collected;
        }
        self.client
            .log_message(
                MessageType::INFO,
                format!("wlift-lsp: resolved {count} dep module(s)"),
            )
            .await;
    }

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
            // Match the runtime's prelude — same names sema's
            // resolve sees in the wasm `run()` path. Without
            // this, every `System.print` / `Fiber.new` shows
            // up as "undefined".
            let mut interner = pr.interner;
            let prelude_names: [&str; 16] = [
                "Object",
                "Class",
                "Bool",
                "Num",
                "String",
                "List",
                "Map",
                "Range",
                "Null",
                "Fn",
                "Fiber",
                "System",
                "Sequence",
                "ByteArray",
                "Float32Array",
                "Float64Array",
            ];
            let prelude: Vec<wren_lift::intern::SymbolId> =
                prelude_names.iter().map(|n| interner.intern(n)).collect();
            let sema = sema_resolve::resolve_with_prelude(&pr.module, &interner, &prelude);
            // Restore the (possibly mutated) interner so the doc
            // collector below sees the same identifier ids.
            let pr_interner = interner;
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
                &pr_interner,
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
        dep_docs: RwLock::new(Vec::new()),
        workspace_modules: RwLock::new(std::collections::HashMap::new()),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
