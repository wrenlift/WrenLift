//! Shared hover helpers — identifier detection, receiver
//! tracking, scope-aware var-decl lookup, RHS type inference,
//! and signature formatting. Used by both the wasm bridge
//! (`wlift_wasm::hover_wren`) and the desktop LSP server
//! (`wlift_lsp`).
//!
//! The single-source rule keeps the playground and the editor
//! LSP in lock-step: a fix in scope detection or a new
//! receiver heuristic shows up in both surfaces without a
//! parallel edit.

use std::ops::Range;

use super::{ModuleDoc, ParamTypeInfo};

/// Glue a class name to a member signature for hover display.
/// The collector encodes static-ness as a `static ` prefix on
/// the signature; hoist that to the front of the rendered string
/// so we get `static Class.method(args)` rather than the
/// nonsensical `Class.static method(args)`.
pub fn format_member_sig(class_name: &str, signature: &str) -> String {
    if let Some(rest) = signature.strip_prefix("static ") {
        format!("static {}.{}", class_name, rest)
    } else if let Some(rest) = signature.strip_prefix("construct ") {
        format!("{}.{}", class_name, rest)
    } else {
        format!("{}.{}", class_name, signature)
    }
}

/// Walk back / forward from `byte` to find an identifier-shaped
/// run (`[A-Za-z_][A-Za-z0-9_]*`). Returns the byte range or
/// `None` when the cursor isn't on an identifier.
pub fn identifier_at(source: &str, byte: usize) -> Option<Range<usize>> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    if byte > len {
        return None;
    }
    let is_id = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    // If `byte` sits between two identifiers (e.g. on a `.`),
    // prefer the one to the left so hover-on-`.` still works.
    let pivot = if byte < len && is_id(bytes[byte]) {
        byte
    } else if byte > 0 && is_id(bytes[byte - 1]) {
        byte - 1
    } else {
        return None;
    };
    let mut start = pivot;
    while start > 0 && is_id(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = pivot + 1;
    while end < len && is_id(bytes[end]) {
        end += 1;
    }
    if bytes[start].is_ascii_digit() {
        return None;
    }
    Some(start..end)
}

/// Identifier sitting immediately before a `.` at `before_byte`.
/// Used to detect receiver expressions (`Renderer2D` in
/// `Renderer2D.new`) so the hover lookup can restrict member
/// matches to the receiver's class instead of taking the first
/// hit from any class with the same method name.
pub fn receiver_before(source: &str, before_byte: usize) -> Option<&str> {
    let bytes = source.as_bytes();
    if before_byte == 0 {
        return None;
    }
    let mut i = before_byte;
    while i > 0 && bytes[i - 1] == b' ' {
        i -= 1;
    }
    if i == 0 || bytes[i - 1] != b'.' {
        return None;
    }
    i -= 1;
    while i > 0 && bytes[i - 1] == b' ' {
        i -= 1;
    }
    let recv_end = i;
    while i > 0 {
        let b = bytes[i - 1];
        if b.is_ascii_alphanumeric() || b == b'_' {
            i -= 1;
        } else {
            break;
        }
    }
    if i == recv_end {
        return None;
    }
    if bytes[i].is_ascii_digit() {
        return None;
    }
    Some(&source[i..recv_end])
}

/// True when `ident` is a Wren reserved word — never produce a
/// hover hint for these. Sourced directly from the lexer: we
/// re-tokenise `ident` with `logos` and treat anything that
/// doesn't come back as an `Ident` / `Field` / `StaticField` as
/// a keyword. That way the lexer is the single source of truth
/// — adding a new `#[token]` to `parse::lexer::Token` flows
/// here automatically.
pub fn is_keyword(ident: &str) -> bool {
    use crate::parse::lexer::Token;
    use logos::Logos;
    let mut lex = Token::lexer(ident);
    let Some(Ok(tok)) = lex.next() else {
        return false;
    };
    if lex.next().is_some() {
        // The string had more than one token — it isn't a bare
        // identifier shape, so the keyword check doesn't apply.
        return false;
    }
    !matches!(tok, Token::Ident | Token::Field | Token::StaticField)
}

/// Identifier-kind hint for cursors that don't match a class /
/// member name. Returns `(signature, body)`. Body is empty for
/// signature-only buckets (fields, generic fallback) so we
/// don't ship template boilerplate on every hover.
pub fn identifier_kind_hint(
    source: &str,
    ident: &str,
    byte: usize,
    prelude: &[ModuleDoc],
) -> Option<(String, String)> {
    if is_keyword(ident) {
        return None;
    }
    if ident.starts_with("__") {
        return Some((format!("static field {}", ident), String::new()));
    }
    if ident.starts_with('_') {
        return Some((format!("instance field {}", ident), String::new()));
    }
    if ident.chars().next().map_or(false, |c| c.is_ascii_uppercase()) {
        return Some((
            format!("class {}", ident),
            "Likely imported from an `@hatch:*` package whose docs the workspace hasn't loaded.".into(),
        ));
    }
    if let Some((line_no, line)) = find_var_decl_line(source, ident, byte) {
        let rhs = decl_rhs(line.trim());
        let ty = rhs.and_then(|r| infer_rhs_type(r, prelude));
        let signature = match &ty {
            Some(t) => format!("local {}: {}", ident, t),
            None => format!("local {}", ident),
        };
        return Some((
            signature,
            format!(
                "Declared on line {}:\n\n```wren\n{}\n```",
                line_no,
                line.trim()
            ),
        ));
    }
    Some((format!("identifier {}", ident), String::new()))
}

/// Pull the right-hand side of a `var name = <rhs>` line.
fn decl_rhs(line: &str) -> Option<&str> {
    let after_var = line.trim_start().strip_prefix("var ")?;
    let eq = after_var.find('=')?;
    Some(after_var[eq + 1..].trim())
}

/// Best-effort type inference from a var-decl's RHS expression.
/// Recognised shapes:
///
/// * `"..."`         → `String`
/// * `42` / `-3.1`   → `Num`
/// * `true`/`false`  → `Bool`
/// * `null`          → `Null`
/// * `[...]`         → `List`
/// * `{...}`         → `Map`
/// * `Class.new(...)`              → `Class`
/// * `Class.staticMethod(...)`     → method's `@returns {Type}` from `prelude`
/// * trailing `.await` unwraps known wrapper types
///   (`ByteFuture` → `ByteArray`, `Future` → `Object`).
pub fn infer_rhs_type(rhs: &str, prelude: &[ModuleDoc]) -> Option<String> {
    let r = rhs.trim_start();
    let first = r.chars().next()?;
    match first {
        '"' | '\'' => return Some("String".into()),
        '[' => return Some("List".into()),
        '{' => return Some("Map".into()),
        '0'..='9' => return Some("Num".into()),
        '-' if r.chars().nth(1).map_or(false, |c| c.is_ascii_digit()) => {
            return Some("Num".into());
        }
        _ => {}
    }
    if r.starts_with("true") || r.starts_with("false") {
        return Some("Bool".into());
    }
    if r.starts_with("null") {
        return Some("Null".into());
    }
    let dot = r.find('.')?;
    let class_name = r[..dot].trim();
    if class_name.is_empty() || !class_name.chars().next()?.is_ascii_uppercase() {
        return None;
    }
    let after_dot = &r[dot + 1..];
    let mut name_end = 0;
    for (i, ch) in after_dot.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            name_end = i + ch.len_utf8();
        } else {
            break;
        }
    }
    if name_end == 0 {
        return None;
    }
    let method_name = &after_dot[..name_end];
    let mut base_type = if method_name == "new" {
        Some(class_name.to_string())
    } else {
        let mut found: Option<String> = None;
        for module in prelude {
            for class in &module.classes {
                if class.name != class_name {
                    continue;
                }
                for member in &class.members {
                    if member.name == method_name {
                        found = member.return_type.clone();
                    }
                }
            }
        }
        found
    };
    let mut tail = after_dot[name_end..].trim_start();
    if tail.starts_with('(') {
        let mut depth = 0;
        let close = tail.bytes().enumerate().find_map(|(i, b)| match b {
            b'(' => {
                depth += 1;
                None
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    Some(i)
                } else {
                    None
                }
            }
            _ => None,
        });
        if let Some(end) = close {
            tail = tail[end + 1..].trim_start();
        }
    }
    while tail.starts_with('.') {
        let after = tail[1..].trim_start();
        let mut len = 0;
        for (i, ch) in after.char_indices() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                len = i + ch.len_utf8();
            } else {
                break;
            }
        }
        if len == 0 {
            break;
        }
        let chained = &after[..len];
        base_type = match (base_type.as_deref(), chained) {
            (Some("ByteFuture"), "await") => Some("ByteArray".into()),
            (Some("Future"), "await") => Some("Object".into()),
            _ => base_type,
        };
        let mut rest = after[len..].trim_start();
        if rest.starts_with('(') {
            let mut depth = 0;
            let close = rest.bytes().enumerate().find_map(|(i, b)| match b {
                b'(' => {
                    depth += 1;
                    None
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        Some(i)
                    } else {
                        None
                    }
                }
                _ => None,
            });
            if let Some(end) = close {
                rest = rest[end + 1..].trim_start();
            } else {
                break;
            }
        }
        tail = rest;
    }
    base_type
}

/// Search for the closest declaration of `ident` in scope at
/// `byte`. Three forms are recognised:
///
///   1. `for (<ident> in <expr>) { ... }` — the loop variable.
///   2. `<ident>` in the enclosing method's parameter list.
///   3. `var <ident> = ...` inside the enclosing block.
///
/// The "enclosing block" is found by walking back from the
/// cursor counting unmatched `{` / `}`, with strings and `//`
/// comments masked so braces inside them don't unbalance the
/// count. Stops scope leakage where a `var foo = ...` in
/// another method body would otherwise show up as the hover
/// for a same-named variable in the current scope.
pub fn find_var_decl_line(source: &str, ident: &str, byte: usize) -> Option<(usize, String)> {
    let block = enclosing_block_range(source, byte);
    let scope_start = block.as_ref().map(|r| r.start).unwrap_or(0);
    let scope_end = block.as_ref().map(|r| r.end).unwrap_or(source.len());

    if let Some(hit) = find_for_binding(source, ident, scope_start) {
        return Some(hit);
    }
    if let Some(hit) = find_param_in_method_header(source, ident, scope_start) {
        return Some(hit);
    }

    let mut backward: Option<(usize, String)> = None;
    let mut forward: Option<(usize, String)> = None;
    for (idx, line) in source.lines().enumerate() {
        let line_no = idx + 1;
        if !line_matches_var_decl(line, ident) {
            continue;
        }
        let line_start = source
            .lines()
            .take(idx)
            .map(|l| l.len() + 1)
            .sum::<usize>();
        if line_start < scope_start || line_start >= scope_end {
            continue;
        }
        if line_start <= byte {
            backward = Some((line_no, line.to_string()));
        } else if forward.is_none() {
            forward = Some((line_no, line.to_string()));
        }
    }
    backward.or(forward)
}

fn line_matches_var_decl(line: &str, ident: &str) -> bool {
    let trimmed = line.trim_start();
    let after = match trimmed.strip_prefix("var ") {
        Some(s) => s.trim_start(),
        None => return false,
    };
    if !after.starts_with(ident) {
        return false;
    }
    let tail = &after[ident.len()..];
    tail.is_empty()
        || tail.starts_with('=')
        || tail.starts_with(',')
        || tail.starts_with(' ')
}

fn enclosing_block_range(source: &str, byte: usize) -> Option<Range<usize>> {
    let masked = mask_strings_and_comments(source);
    let mb = masked.as_bytes();
    let cursor = byte.min(mb.len());
    let mut depth: i32 = 0;
    let mut open_at: Option<usize> = None;
    let mut i = cursor;
    while i > 0 {
        i -= 1;
        match mb[i] {
            b'}' => depth += 1,
            b'{' => {
                if depth == 0 {
                    open_at = Some(i);
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    let open = open_at?;
    let mut depth: i32 = 1;
    let mut j = open + 1;
    while j < mb.len() {
        match mb[j] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + 1..j);
                }
            }
            _ => {}
        }
        j += 1;
    }
    Some(open + 1..mb.len())
}

fn mask_strings_and_comments(source: &str) -> String {
    let mut out: Vec<u8> = source.bytes().collect();
    let mut i = 0;
    while i < out.len() {
        match out[i] {
            b'"' => {
                out[i] = b'_';
                i += 1;
                while i < out.len() {
                    let b = out[i];
                    out[i] = if b == b'\n' { b'\n' } else { b'_' };
                    if b == b'"' {
                        i += 1;
                        break;
                    }
                    if b == b'\\' && i + 1 < out.len() {
                        out[i + 1] = b'_';
                        i += 2;
                        continue;
                    }
                    i += 1;
                }
            }
            b'/' if i + 1 < out.len() && out[i + 1] == b'/' => {
                while i < out.len() && out[i] != b'\n' {
                    out[i] = b'_';
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| source.to_string())
}

fn find_for_binding(
    source: &str,
    ident: &str,
    scope_start: usize,
) -> Option<(usize, String)> {
    if scope_start == 0 {
        return None;
    }
    let line_start = source[..scope_start]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    let line_end = source[scope_start..]
        .find('\n')
        .map(|i| scope_start + i)
        .unwrap_or(source.len());
    let line = &source[line_start..line_end];
    let trimmed = line.trim_start();
    if !trimmed.starts_with("for") {
        return None;
    }
    let after_for = trimmed["for".len()..].trim_start();
    let after_paren = match after_for.strip_prefix('(') {
        Some(s) => s.trim_start(),
        None => return None,
    };
    if after_paren.starts_with(ident) {
        let tail = &after_paren[ident.len()..];
        if tail.starts_with(' ') || tail.starts_with('\t') || tail.starts_with("in") {
            let line_no = source[..line_start].matches('\n').count() + 1;
            return Some((line_no, line.to_string()));
        }
    }
    None
}

fn find_param_in_method_header(
    source: &str,
    ident: &str,
    scope_start: usize,
) -> Option<(usize, String)> {
    if scope_start == 0 {
        return None;
    }
    let line_start = source[..scope_start - 1]
        .rfind('\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    let line_end = source[scope_start..]
        .find('\n')
        .map(|i| scope_start + i)
        .unwrap_or(source.len());
    let line = &source[line_start..line_end];
    let open = line.find('(')?;
    let close = line[open..].find(')')?;
    let params = &line[open + 1..open + close];
    for p in params.split(',') {
        let p = p.trim();
        if p == ident {
            let line_no = source[..line_start].matches('\n').count() + 1;
            return Some((line_no, line.to_string()));
        }
    }
    None
}

// Silence an unused-warning: `ParamTypeInfo` is here so future
// callers (LSP completion previews, etc.) can build typed
// signatures via `infer_rhs_type` + JSDoc data without
// re-importing.
#[doc(hidden)]
pub fn _touch_param_type(_: &ParamTypeInfo) {}

#[cfg(test)]
mod tests {
    use super::is_keyword;

    #[test]
    fn keywords_match_lexer() {
        // Every reserved word the lexer recognises must be
        // rejected by `is_keyword`. The list is duplicated here
        // intentionally — drift between this test and the lexer
        // is the signal the keyword-guard is broken.
        let keywords = [
            "as", "break", "class", "construct", "continue", "else", "false", "for", "foreign",
            "if", "import", "in", "is", "null", "return", "static", "super", "this", "true",
            "var", "while",
        ];
        for kw in keywords {
            assert!(is_keyword(kw), "expected `{kw}` to be a keyword");
        }
    }

    #[test]
    fn identifiers_pass_through() {
        for id in ["foo", "_field", "__static", "Foo", "myVar123"] {
            assert!(!is_keyword(id), "expected `{id}` to be an identifier");
        }
    }

    #[test]
    fn non_atomic_strings_are_not_keywords() {
        // Multi-token strings (`var x`) shouldn't be classified
        // as keywords — `is_keyword` is only meaningful for the
        // bare-identifier shape returned by `identifier_at`.
        assert!(!is_keyword("var x"));
        assert!(!is_keyword(""));
    }
}
