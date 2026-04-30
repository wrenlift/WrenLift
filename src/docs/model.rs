//! Doc model — the data shape both the renderer and (later) the
//! LSP hover handler consume. Kept deliberately small: a class, its
//! members, and the Markdown body for each.
//!
//! Output IR is JSON. The wrenlift.com docs viewer owns the visual
//! design and renders the Markdown bodies in its own theme — the
//! generator's job stops at producing well-typed structured data.

use crate::ast::Span;
use serde::{Deserialize, Serialize};

/// A whole module's docs — emitted as one JSON document per file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Cross-references found across every Markdown body in this
    /// module. The consumer site uses these to turn bracketed
    /// identifiers into clickable links — the generator does the
    /// resolution; the site does the rendering.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub cross_refs: Vec<CrossRef>,
}

/// A single bracket reference (`[Class]`, `[Class.method]`,
/// `[#anchor]`, `[@hatch:pkg Name]`) discovered in some Markdown
/// body inside this module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRef {
    /// The raw bracketed text *without* the surrounding `[]`,
    /// exactly as the author wrote it. Consumers match on this
    /// when rewriting the Markdown.
    pub text: String,
    /// Where this reference lives — the body it was scanned from
    /// (so the site doesn't accidentally rewrite a string of the
    /// same shape that happens to live in a different doc).
    pub origin: RefOrigin,
    /// What the reference resolved to. `Unresolved` keeps the
    /// reference in the output so the site can render it as
    /// plain bracketed text.
    pub target: RefTarget,
}

/// Where a cross-reference was found inside the module.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RefOrigin {
    /// The module's `//!` doc body.
    Module,
    /// A class doc body.
    Class { class: String },
    /// A member doc body. `class` is the enclosing class; `member`
    /// is the member's display name.
    Member { class: String, member: String },
}

/// Resolved target of a cross-reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RefTarget {
    /// Same-module class. `class` is the class name.
    Class { class: String },
    /// Same-module member. `class` + `member` identify it.
    Member { class: String, member: String },
    /// Same-doc heading anchor (`[#some-section]`).
    Anchor { anchor: String },
    /// Cross-package reference (`[@hatch:foo]` or
    /// `[@hatch:foo Bar]`). Resolution against a registry index
    /// happens at site-build time; we just preserve the package
    /// name and optional symbol so the site can hyperlink.
    External {
        package: String,
        symbol: Option<String>,
    },
    /// Bracket pattern that didn't match any known shape. The
    /// site renders these as literal text — the bracket is part
    /// of the body's prose, not a reference.
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassDoc {
    pub name: String,
    #[serde(serialize_with = "ser_span", deserialize_with = "de_span")]
    pub span: Span,
    /// `///` body for the class itself, Markdown.
    pub doc: String,
    pub members: Vec<MemberDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberDoc {
    pub name: String,
    pub kind: MemberKind,
    /// `///` body, Markdown. Empty when the member has no doc
    /// comment.
    pub doc: String,
    #[serde(serialize_with = "ser_span", deserialize_with = "de_span")]
    pub span: Span,
    /// Pretty-printed signature line — `static foo(x: Num, y: Num)`,
    /// `bar=(value)`, `baz { _baz }`, etc. When the body has
    /// `@param {Type} name` annotations, the parser splices them
    /// into the param list here so the hover signature alone
    /// communicates types.
    pub signature: String,
    /// Per-parameter type annotations parsed from the doc body's
    /// `@param {Type} name [— description]` lines. Empty when no
    /// annotations were authored.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub param_types: Vec<ParamTypeInfo>,
    /// `@returns {Type}` annotation, if present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub return_type: Option<String>,
}

/// One row from a `@param {Type} name — description` line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamTypeInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Counterpart to [`ser_span`]: parse a 2-element JSON array
/// back into a `Range<usize>`. Used when the runtime
/// deserialises the `Docs` section out of a published bundle.
fn de_span<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Span, D::Error> {
    let pair: [usize; 2] = serde::Deserialize::deserialize(d)?;
    Ok(pair[0]..pair[1])
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
