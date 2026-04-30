//! Doc model — the data shape both the renderer and (later) the
//! LSP hover handler consume. Kept deliberately small: a class, its
//! members, and the Markdown body for each.
//!
//! The collector populates this from `(ParseResult, source)`; the
//! renderer reads it and produces HTML.

use crate::ast::Span;

/// A whole module's docs — the things published as one HTML page
/// per file in the static-site output.
#[derive(Debug, Clone, Default)]
pub struct ModuleDoc {
    /// Module name (file stem with directory dotted in, e.g.
    /// `foo.bar` for `foo/bar.wren`). The CLI fills this in;
    /// the collector defaults to empty so library callers can set
    /// it themselves.
    pub name: String,
    /// `//!` body joined into a single Markdown blob. Empty when
    /// the file has no module-level docs.
    pub doc: String,
    pub classes: Vec<ClassDoc>,
}

#[derive(Debug, Clone)]
pub struct ClassDoc {
    pub name: String,
    pub span: Span,
    /// `///` body for the class itself, Markdown.
    pub doc: String,
    pub members: Vec<MemberDoc>,
}

#[derive(Debug, Clone)]
pub struct MemberDoc {
    pub name: String,
    pub kind: MemberKind,
    /// `///` body, Markdown. Empty when the member has no doc
    /// comment.
    pub doc: String,
    pub span: Span,
    /// Pretty-printed signature line — `static foo(x, y)`,
    /// `bar=(value)`, `baz { _baz }`, etc. Renderer prints this
    /// above the body.
    pub signature: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberKind {
    Method,
    StaticMethod,
    Getter,
    Setter,
    Constructor,
    Field,
}

impl ModuleDoc {
    /// Pull the first paragraph of `doc` out for use as a one-line
    /// summary in indices / completion previews. Empty string if
    /// the doc body is empty.
    pub fn summary(&self) -> &str {
        first_paragraph(&self.doc)
    }
}

impl ClassDoc {
    pub fn summary(&self) -> &str {
        first_paragraph(&self.doc)
    }
}

impl MemberDoc {
    pub fn summary(&self) -> &str {
        first_paragraph(&self.doc)
    }
}

fn first_paragraph(body: &str) -> &str {
    match body.find("\n\n") {
        Some(end) => body[..end].trim_end(),
        None => body.trim_end(),
    }
}
