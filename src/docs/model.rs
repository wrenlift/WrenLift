//! Doc model — the data shape both the renderer and (later) the
//! LSP hover handler consume. Kept deliberately small: a class, its
//! members, and the Markdown body for each.
//!
//! Output IR is JSON. The wrenlift.com docs viewer owns the visual
//! design and renders the Markdown bodies in its own theme — the
//! generator's job stops at producing well-typed structured data.

use crate::ast::Span;
use serde::Serialize;

/// A whole module's docs — emitted as one JSON document per file.
#[derive(Debug, Clone, Default, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
pub struct ClassDoc {
    pub name: String,
    #[serde(serialize_with = "ser_span")]
    pub span: Span,
    /// `///` body for the class itself, Markdown.
    pub doc: String,
    pub members: Vec<MemberDoc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemberDoc {
    pub name: String,
    pub kind: MemberKind,
    /// `///` body, Markdown. Empty when the member has no doc
    /// comment.
    pub doc: String,
    #[serde(serialize_with = "ser_span")]
    pub span: Span,
    /// Pretty-printed signature line — `static foo(x, y)`,
    /// `bar=(value)`, `baz { _baz }`, etc. Consumers print it
    /// above the body.
    pub signature: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemberKind {
    Method,
    StaticMethod,
    Getter,
    Setter,
    Constructor,
    Field,
}

/// Wrap the foreign `Span` (a `Range<usize>`) as a 2-element JSON
/// array: `[start, end]`. Lets consumers do `span[0]`, `span[1]`
/// without naming the field.
fn ser_span<S: serde::Serializer>(span: &Span, s: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeTuple;
    let mut t = s.serialize_tuple(2)?;
    t.serialize_element(&span.start)?;
    t.serialize_element(&span.end)?;
    t.end()
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
