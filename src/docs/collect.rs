//! Walk a parsed module and pair each declaration with its
//! preceding doc-comment block.
//!
//! Proximity rule (matches [comment-format-spec.md](../../../../docs/comment-format-spec.md)):
//!
//! * `///` blocks immediately above a declaration belong to that
//!   declaration. "Immediately above" = the comment's end byte is
//!   on the line just above the declaration's start byte, or the
//!   comment is part of a contiguous run of `///` lines whose final
//!   line is.
//! * `//!` blocks anywhere in the file aggregate into the module's
//!   own doc body, in source order.
//! * Anything between a `///` block and the declaration that isn't
//!   another `///` line breaks the association — the doc is
//!   discarded with a (TODO) warning. We accept blank lines.

use crate::ast::{ClassDecl, Method, MethodSig, Op, Spanned, Stmt};
use crate::intern::Interner;
use crate::parse::lexer::{DocComment, DocKind};

use super::model::{ClassDoc, CrossRef, MemberDoc, MemberKind, ModuleDoc, RefOrigin, RefTarget};

/// Build a [`ModuleDoc`] from a parsed result + the original source.
///
/// `name` is the module's qualified name (e.g. `"foo.bar"` for
/// `foo/bar.wren`); leave empty when the caller is the LSP and the
/// concept doesn't apply yet.
pub fn collect_module(
    name: impl Into<String>,
    source: &str,
    module: &[Spanned<Stmt>],
    docs: &[DocComment],
    interner: &Interner,
) -> ModuleDoc {
    let mut module_doc = ModuleDoc {
        name: name.into(),
        ..Default::default()
    };

    // Module-level body: every `//!` block, joined by blank lines so
    // multi-paragraph notes survive.
    module_doc.doc = docs
        .iter()
        .filter(|d| d.kind == DocKind::Module)
        .map(|d| d.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    // Decl docs: walk the module's top-level items, pulling each
    // class out and recursing into its methods.
    for stmt in module {
        if let Stmt::Class(class) = &stmt.0 {
            module_doc
                .classes
                .push(collect_class(class, &stmt.1, source, docs, interner));
        }
        // `var` declarations are module-level data, not part of the
        // public API surface for v0. Skip them.
    }

    // Second pass: resolve `[X]` bracket cross-references in every
    // body. Builds a tiny symbol table from the classes we just
    // collected and walks each Markdown blob for bracketed text.
    module_doc.cross_refs = resolve_cross_refs(&module_doc);

    module_doc
}

fn collect_class(
    class: &ClassDecl,
    decl_span: &crate::ast::Span,
    source: &str,
    docs: &[DocComment],
    interner: &Interner,
) -> ClassDoc {
    let name = interner.resolve(class.name.0).to_string();
    let doc = doc_for_decl(decl_span, source, docs);

    let mut members = Vec::with_capacity(class.methods.len());
    for spanned in &class.methods {
        members.push(collect_member(&spanned.0, &spanned.1, source, docs, interner));
    }

    ClassDoc {
        name,
        span: decl_span.clone(),
        doc,
        members,
    }
}

fn collect_member(
    method: &Method,
    decl_span: &crate::ast::Span,
    source: &str,
    docs: &[DocComment],
    interner: &Interner,
) -> MemberDoc {
    let kind = if method.is_static {
        MemberKind::StaticMethod
    } else {
        match &method.signature {
            MethodSig::Construct { .. } => MemberKind::Constructor,
            MethodSig::Getter(_) => MemberKind::Getter,
            MethodSig::Setter { .. } => MemberKind::Setter,
            _ => MemberKind::Method,
        }
    };

    let (name, signature) = signature_text(&method.signature, method.is_static, interner);

    MemberDoc {
        name,
        kind,
        doc: doc_for_decl(decl_span, source, docs),
        span: decl_span.clone(),
        signature,
    }
}

fn signature_text(
    sig: &MethodSig,
    is_static: bool,
    interner: &Interner,
) -> (String, String) {
    let prefix = if is_static { "static " } else { "" };
    match sig {
        MethodSig::Named { name, params } => {
            let n = interner.resolve(*name).to_string();
            let plist = render_params(params, interner);
            (n.clone(), format!("{}{}({})", prefix, n, plist))
        }
        MethodSig::Getter(name) => {
            let n = interner.resolve(*name).to_string();
            (n.clone(), format!("{}{}", prefix, n))
        }
        MethodSig::Setter { name, param } => {
            let n = interner.resolve(*name).to_string();
            let p = interner.resolve(param.0).to_string();
            (n.clone(), format!("{}{}=({})", prefix, n, p))
        }
        MethodSig::Subscript { params } => {
            let plist = render_params(params, interner);
            ("[]".into(), format!("{}[{}]", prefix, plist))
        }
        MethodSig::SubscriptSetter { params, value } => {
            let plist = render_params(params, interner);
            let v = interner.resolve(value.0).to_string();
            ("[]=".into(), format!("{}[{}]=({})", prefix, plist, v))
        }
        MethodSig::Operator { op, params } => {
            let sym = op_symbol(*op);
            let plist = render_params(params, interner);
            if params.is_empty() {
                (sym.into(), format!("{}{}", prefix, sym))
            } else {
                (sym.into(), format!("{}{}({})", prefix, sym, plist))
            }
        }
        MethodSig::Construct { name, params } => {
            let n = interner.resolve(*name).to_string();
            let plist = render_params(params, interner);
            (n.clone(), format!("construct {}({})", n, plist))
        }
    }
}

fn render_params(params: &[Spanned<crate::intern::SymbolId>], interner: &Interner) -> String {
    params
        .iter()
        .map(|p| interner.resolve(p.0).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn op_symbol(op: Op) -> &'static str {
    match op {
        Op::Plus => "+",
        Op::Minus => "-",
        Op::Star => "*",
        Op::Slash => "/",
        Op::Percent => "%",
        Op::Lt => "<",
        Op::Gt => ">",
        Op::LtEq => "<=",
        Op::GtEq => ">=",
        Op::EqEq => "==",
        Op::BangEq => "!=",
        Op::BitAnd => "&",
        Op::BitOr => "|",
        Op::BitXor => "^",
        Op::Shl => "<<",
        Op::Shr => ">>",
        Op::DotDot => "..",
        Op::DotDotDot => "...",
        Op::Neg => "-",
        Op::Bang => "!",
        Op::Tilde => "~",
    }
}

/// Find the `///` block that ends just above `decl_span`, joined into
/// a single Markdown body. Returns an empty string when nothing is
/// adjacent.
///
/// "Adjacent" = the comment's start sits on a previous line and the
/// only bytes between its end and the declaration are whitespace +
/// newlines + further `///` lines (which the lexer already coalesces
/// into separate `DocComment` entries).
fn doc_for_decl(decl_span: &crate::ast::Span, source: &str, docs: &[DocComment]) -> String {
    // Collect every Decl-kind comment whose span ends before the
    // declaration; pick the longest contiguous tail.
    let mut block: Vec<&DocComment> = Vec::new();
    for c in docs {
        if c.kind != DocKind::Decl {
            continue;
        }
        if c.span.end > decl_span.start {
            // Comment appears at or after the declaration; can't
            // belong to it.
            break;
        }
        match block.last() {
            None => block.push(c),
            Some(prev) => {
                if is_contiguous(prev.span.end, c.span.start, source) {
                    block.push(c);
                } else {
                    block.clear();
                    block.push(c);
                }
            }
        }
    }
    if let Some(last) = block.last() {
        if !is_adjacent_to(last.span.end, decl_span.start, source) {
            return String::new();
        }
    } else {
        return String::new();
    }

    block
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Two `///` lines are part of the same block iff the bytes between
/// them are exactly one newline + whatever leading whitespace the
/// follow-up line has before its `///`.
fn is_contiguous(prev_end: usize, next_start: usize, source: &str) -> bool {
    let between = &source.as_bytes()[prev_end..next_start];
    // Expect: `\n` + (spaces|tabs) — no blank line.
    let mut saw_newline = false;
    for &b in between {
        match b {
            b'\n' => {
                if saw_newline {
                    return false; // blank line breaks the block
                }
                saw_newline = true;
            }
            b' ' | b'\t' | b'\r' => {}
            _ => return false,
        }
    }
    saw_newline
}

/// The doc block ends "adjacent" to the declaration when only
/// whitespace + at most one newline sit between them. We tolerate
/// tabs / leading spaces but reject blank-line gaps.
fn is_adjacent_to(comment_end: usize, decl_start: usize, source: &str) -> bool {
    let between = &source.as_bytes()[comment_end..decl_start];
    let mut newlines = 0;
    for &b in between {
        match b {
            b'\n' => {
                newlines += 1;
                if newlines > 1 {
                    return false;
                }
            }
            b' ' | b'\t' | b'\r' => {}
            _ => return false,
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Cross-reference resolution
// ---------------------------------------------------------------------------

/// Walk every Markdown body in `module` and resolve bracketed
/// references against a tiny symbol table built from the module's
/// own classes. Returns one `CrossRef` per discovered bracket.
///
/// The site is the rendering layer — it'll consult this list when
/// emitting links, so unresolved brackets are kept (with kind =
/// `Unresolved`) rather than dropped.
fn resolve_cross_refs(module: &ModuleDoc) -> Vec<CrossRef> {
    let mut symbols = SymbolTable::default();
    for class in &module.classes {
        symbols.classes.push(class.name.clone());
        for member in &class.members {
            symbols
                .members
                .push((class.name.clone(), member.name.clone()));
        }
    }

    let mut refs = Vec::new();
    let mut emit = |body: &str, origin: RefOrigin| {
        for raw in scan_brackets(body) {
            let target = resolve_one(&raw, &symbols);
            refs.push(CrossRef {
                text: raw,
                origin: origin.clone(),
                target,
            });
        }
    };

    emit(&module.doc, RefOrigin::Module);
    for class in &module.classes {
        emit(
            &class.doc,
            RefOrigin::Class {
                class: class.name.clone(),
            },
        );
        for member in &class.members {
            emit(
                &member.doc,
                RefOrigin::Member {
                    class: class.name.clone(),
                    member: member.name.clone(),
                },
            );
        }
    }
    refs
}

#[derive(Default)]
struct SymbolTable {
    classes: Vec<String>,
    members: Vec<(String, String)>, // (class, member)
}

/// Match a single bracket reference against the symbol table.
///
/// Recognized shapes:
/// * `Class`              → same-module class
/// * `Class.member`       → same-module member
/// * `#anchor`            → same-doc heading anchor
/// * `@scope:pkg`         → external package
/// * `@scope:pkg Symbol`  → external symbol inside a package
fn resolve_one(text: &str, symbols: &SymbolTable) -> RefTarget {
    let trimmed = text.trim();

    if let Some(rest) = trimmed.strip_prefix('#') {
        if !rest.is_empty() {
            return RefTarget::Anchor {
                anchor: rest.to_string(),
            };
        }
    }

    if let Some(rest) = trimmed.strip_prefix('@') {
        // External package. Two forms:
        //   `@hatch:foo`           — package only
        //   `@hatch:foo Symbol`    — symbol inside package
        let mut parts = rest.splitn(2, char::is_whitespace);
        let package = parts.next().unwrap_or("").to_string();
        let symbol = parts.next().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        if !package.is_empty() {
            return RefTarget::External { package, symbol };
        }
    }

    if let Some((class, member)) = trimmed.split_once('.') {
        if symbols
            .members
            .iter()
            .any(|(c, m)| c == class && m == member)
        {
            return RefTarget::Member {
                class: class.to_string(),
                member: member.to_string(),
            };
        }
    }

    if symbols.classes.iter().any(|c| c == trimmed) {
        return RefTarget::Class {
            class: trimmed.to_string(),
        };
    }

    RefTarget::Unresolved
}

/// Pull every `[X]` bracket out of a Markdown body, ignoring those
/// inside fenced code blocks (so a code sample doesn't accidentally
/// register `[Class]` as a cross-reference) and `\[escaped\]` ones.
///
/// Returns the inner text of each bracket, in source order. We
/// don't try to be cute here — heuristic match is fine, the site
/// re-checks against the cross_refs list when rewriting.
fn scan_brackets(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    let mut in_code = false;
    while i < bytes.len() {
        // Toggle on triple-backtick fences. Single backticks for
        // inline code are handled by the same skip path below.
        if i + 2 < bytes.len() && &bytes[i..i + 3] == b"```" {
            in_code = !in_code;
            i += 3;
            continue;
        }
        if in_code {
            i += 1;
            continue;
        }
        // Inline code (`...`) — skip up to the next backtick.
        if bytes[i] == b'`' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'`' {
                i += 1;
            }
            i += 1; // step past the closing backtick
            continue;
        }
        // Escaped bracket (`\[`) — skip both bytes.
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            continue;
        }
        if bytes[i] == b'[' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b']' && bytes[end] != b'\n' {
                end += 1;
            }
            if end < bytes.len() && bytes[end] == b']' {
                // Skip the `[text](...)` standard-link form — those
                // are real markdown links, not cross-refs.
                if end + 1 < bytes.len() && bytes[end + 1] == b'(' {
                    i = end + 1;
                    continue;
                }
                let inner = body[start..end].to_string();
                if !inner.is_empty() {
                    out.push(inner);
                }
                i = end + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    fn collect(src: &str) -> ModuleDoc {
        let pr = parse(src);
        collect_module("test", src, &pr.module, &pr.docs, &pr.interner)
    }

    #[test]
    fn captures_module_doc_from_bang_comments() {
        let doc = collect("//! Top of file.\n//! Second line.\n\nclass Foo {}\n");
        assert!(doc.doc.contains("Top of file."));
        assert!(doc.doc.contains("Second line."));
    }

    #[test]
    fn captures_class_doc_from_triple_slash() {
        let src = "/// Class summary.\nclass Foo {}\n";
        let doc = collect(src);
        assert_eq!(doc.classes.len(), 1);
        assert_eq!(doc.classes[0].name, "Foo");
        assert_eq!(doc.classes[0].doc, "Class summary.");
    }

    #[test]
    fn captures_member_doc() {
        let src = "class Foo {\n  /// Says hi.\n  hi() {}\n}\n";
        let doc = collect(src);
        let foo = &doc.classes[0];
        assert_eq!(foo.members.len(), 1);
        assert_eq!(foo.members[0].name, "hi");
        assert_eq!(foo.members[0].doc, "Says hi.");
    }

    #[test]
    fn blank_line_breaks_association() {
        let src = "/// orphaned doc.\n\nclass Foo {}\n";
        let doc = collect(src);
        assert!(doc.classes[0].doc.is_empty(), "blank line should detach the doc");
    }

    #[test]
    fn signature_includes_static_prefix() {
        let src = "class Foo {\n  static greet(name) {}\n}\n";
        let doc = collect(src);
        assert_eq!(doc.classes[0].members[0].signature, "static greet(name)");
    }

    #[test]
    fn resolves_local_class_and_member_refs() {
        let src = "\
/// See [Foo] and [Foo.greet] for details. [Bar] is unknown.
class Foo {
  greet(name) {}
}
";
        let doc = collect(src);
        let class_refs: Vec<&CrossRef> = doc
            .cross_refs
            .iter()
            .filter(|r| matches!(r.origin, RefOrigin::Class { .. }))
            .collect();
        assert_eq!(class_refs.len(), 3);
        assert!(matches!(
            class_refs[0].target,
            RefTarget::Class { ref class } if class == "Foo"
        ));
        assert!(matches!(
            class_refs[1].target,
            RefTarget::Member { ref class, ref member } if class == "Foo" && member == "greet"
        ));
        assert!(matches!(class_refs[2].target, RefTarget::Unresolved));
    }

    #[test]
    fn resolves_anchor_and_external_refs() {
        let src = "\
//! See [#intro] in this doc, and [@hatch:gpu Renderer2D] elsewhere.
class Foo {}
";
        let doc = collect(src);
        let anchor = doc
            .cross_refs
            .iter()
            .find(|r| r.text == "#intro")
            .expect("anchor ref present");
        match &anchor.target {
            RefTarget::Anchor { anchor } => assert_eq!(anchor, "intro"),
            other => panic!("expected anchor, got {:?}", other),
        }
        let ext = doc
            .cross_refs
            .iter()
            .find(|r| r.text.starts_with("@hatch:gpu"))
            .expect("external ref present");
        match &ext.target {
            RefTarget::External { package, symbol } => {
                assert_eq!(package, "hatch:gpu");
                assert_eq!(symbol.as_deref(), Some("Renderer2D"));
            }
            other => panic!("expected external, got {:?}", other),
        }
    }

    #[test]
    fn ignores_brackets_inside_code_fences() {
        let src = "\
/// Outside [Foo] resolves.
///
/// ```
/// var x = list[0]   // [NotAClass] inside code is left alone
/// ```
class Foo {}
";
        let doc = collect(src);
        let names: Vec<_> = doc.cross_refs.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(names, vec!["Foo"]);
    }
}
