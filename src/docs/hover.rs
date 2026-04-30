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

use crate::ast::{Expr, Module, Spanned, Stmt};
use crate::intern::{Interner, SymbolId};
use crate::sema::types::{infer_types, InferredType, TypeEnv};

use super::{ModuleDoc, ParamTypeInfo};

/// Parsed module + interner + inferred type environment, kept
/// together so hover / completion can consult sema's typed AST
/// instead of falling back to text heuristics. Built once per
/// hover request via [`Analysis::run`]; cheap to throw away.
pub struct Analysis {
    pub module: Module,
    pub interner: Interner,
    pub type_env: TypeEnv,
}

impl Analysis {
    /// Parse + run the type inferrer. Returns `None` when the
    /// parser couldn't produce a usable module — callers fall
    /// back to text heuristics in that case.
    pub fn run(source: &str) -> Option<Self> {
        let pr = crate::parse::parser::parse(source);
        if !pr.errors.is_empty() {
            return None;
        }
        // 3-pass inference: pass 1 collects field assignment
        // types, pass 2 records method return types from the
        // field-aware inference, pass 3 emits the final
        // per-expression type map. Single-pass left field
        // references as `Any` because the field hadn't been
        // recorded yet when the use site was visited.
        let type_env = infer_types(&pr.module);
        Some(Analysis {
            module: pr.module,
            interner: pr.interner,
            type_env,
        })
    }

    /// Symbol id of the class whose body contains `byte`, or
    /// `None` when the cursor sits outside any class.
    pub fn enclosing_class(&self, byte: usize) -> Option<SymbolId> {
        for stmt in &self.module {
            if !span_contains(&stmt.1, byte) {
                continue;
            }
            if let Stmt::Class(class) = &stmt.0 {
                return Some(class.name.0);
            }
        }
        None
    }

    /// Inferred type for the field `name` (sans leading `_`)
    /// inside the class containing `byte`. `None` when sema
    /// hasn't recorded a type — i.e. the field has no
    /// initialiser or is only assigned values whose types didn't
    /// converge.
    pub fn field_type_at(&self, byte: usize, field_name: &str) -> Option<InferredType> {
        let class = self.enclosing_class(byte)?;
        let field = self.interner.lookup(field_name)?;
        let ty = self.type_env.get_field_type(class, field).clone();
        ty.is_known().then_some(ty)
    }

    /// Inferred type recorded at a `var <name>` declaration's
    /// span_start. Look up via [`find_var_decl_span`].
    pub fn var_type_at_span(&self, decl_span_start: usize) -> Option<InferredType> {
        let ty = self.type_env.get_var_type(decl_span_start).clone();
        ty.is_known().then_some(ty)
    }

    /// Walk the AST to find a method-call whose method-name span
    /// covers `byte`, and return the inferred type of its
    /// receiver. Used to resolve `<recv>.<member>` hovers
    /// against the actual receiver class instead of taking the
    /// first same-named member from any class.
    pub fn receiver_type_for_call_at(&self, byte: usize) -> Option<InferredType> {
        let recv = self.find_call_receiver_at(byte)?;
        // Choose the look-up that's robust to multi-pass
        // inference. `expr_types` is "ephemeral" and gets cleared
        // between passes, so a Field reference often comes back
        // as `Any` even when the underlying field type is known.
        // The field-table and the enclosing class together give
        // a stable answer.
        let ty = match &recv.0 {
            Expr::Field(field_sym) => {
                let class = self.enclosing_class(byte)?;
                self.type_env.get_field_type(class, *field_sym).clone()
            }
            Expr::This => InferredType::Class(self.enclosing_class(byte)?),
            _ => self.type_env.get_expr_type(recv.1.start).clone(),
        };
        ty.is_known().then_some(ty)
    }

    /// Walk the AST to find the receiver expression of the call
    /// whose method-name span covers `byte`. Returned as a
    /// reference into the analysed module so the caller can
    /// inspect the AST node directly (the receiver might be a
    /// field, `this`, a chained call, etc.).
    fn find_call_receiver_at(&self, byte: usize) -> Option<&Spanned<Expr>> {
        let mut found: Option<&Spanned<Expr>> = None;
        for stmt in &self.module {
            walk_stmt_for_call(stmt, byte, &mut found);
            if found.is_some() {
                break;
            }
        }
        found
    }
}

/// Map a sema [`InferredType`] to a printable class name suitable
/// for hover signatures (`local x: List`, etc.). Built-in types
/// resolve to their prelude class name; user classes resolve via
/// the interner. `None` for `Any`.
pub fn inferred_to_class_name(ty: &InferredType, interner: &Interner) -> Option<String> {
    match ty {
        InferredType::Num => Some("Num".into()),
        InferredType::Bool => Some("Bool".into()),
        InferredType::Null => Some("Null".into()),
        InferredType::String => Some("String".into()),
        InferredType::List => Some("List".into()),
        InferredType::Map => Some("Map".into()),
        InferredType::Range => Some("Range".into()),
        InferredType::Fn => Some("Fn".into()),
        InferredType::Class(sym) => Some(interner.resolve(*sym).to_string()),
        InferredType::Any => None,
    }
}

fn span_contains(span: &Range<usize>, byte: usize) -> bool {
    span.start <= byte && byte < span.end
}

fn walk_stmt_for_call<'a>(
    stmt: &'a Spanned<Stmt>,
    byte: usize,
    out: &mut Option<&'a Spanned<Expr>>,
) {
    // No span gating — see `find_call_receiver_at`. Descend
    // through every node and let the leaf check (the method
    // name span) gate the actual match.
    match &stmt.0 {
        Stmt::Expr(expr) => walk_expr_for_call(expr, byte, out),
        Stmt::Var { initializer, .. } => {
            if let Some(init) = initializer {
                walk_expr_for_call(init, byte, out);
            }
        }
        Stmt::Class(class) => {
            for method in &class.methods {
                if let Some(body) = &method.0.body {
                    walk_stmt_for_call(body, byte, out);
                    if out.is_some() {
                        return;
                    }
                }
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                walk_stmt_for_call(s, byte, out);
                if out.is_some() {
                    return;
                }
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            walk_expr_for_call(condition, byte, out);
            if out.is_none() {
                walk_stmt_for_call(then_branch, byte, out);
            }
            if out.is_none() {
                if let Some(eb) = else_branch {
                    walk_stmt_for_call(eb, byte, out);
                }
            }
        }
        Stmt::While { condition, body } => {
            walk_expr_for_call(condition, byte, out);
            if out.is_none() {
                walk_stmt_for_call(body, byte, out);
            }
        }
        Stmt::For {
            iterator, body, ..
        } => {
            walk_expr_for_call(iterator, byte, out);
            if out.is_none() {
                walk_stmt_for_call(body, byte, out);
            }
        }
        Stmt::Return(Some(e)) => walk_expr_for_call(e, byte, out),
        Stmt::Return(None) | Stmt::Break | Stmt::Continue | Stmt::Import { .. } => {}
    }
}

fn walk_expr_for_call<'a>(
    expr: &'a Spanned<Expr>,
    byte: usize,
    out: &mut Option<&'a Spanned<Expr>>,
) {
    // Descend into children first so the *innermost* matching
    // call wins — `a.b().c().d()` with the cursor on `d` should
    // resolve against the inner `.c()`'s return type, not the
    // outer chain head.
    match &expr.0 {
        Expr::Call {
            receiver,
            method,
            args,
            block_arg,
            ..
        } => {
            for arg in args {
                walk_expr_for_call(arg, byte, out);
                if out.is_some() {
                    return;
                }
            }
            if let Some(b) = block_arg {
                walk_expr_for_call(b, byte, out);
                if out.is_some() {
                    return;
                }
            }
            if let Some(recv) = receiver {
                walk_expr_for_call(recv, byte, out);
                if out.is_some() {
                    return;
                }
            }
            if span_contains(&method.1, byte) {
                if let Some(recv) = receiver {
                    *out = Some(recv);
                }
            }
        }
        Expr::UnaryOp { operand, .. } => walk_expr_for_call(operand, byte, out),
        Expr::BinaryOp { left, right, .. } | Expr::LogicalOp { left, right, .. } => {
            walk_expr_for_call(left, byte, out);
            if out.is_none() {
                walk_expr_for_call(right, byte, out);
            }
        }
        Expr::Is { value, type_name } => {
            walk_expr_for_call(value, byte, out);
            if out.is_none() {
                walk_expr_for_call(type_name, byte, out);
            }
        }
        Expr::Assign { target, value } | Expr::CompoundAssign { target, value, .. } => {
            walk_expr_for_call(target, byte, out);
            if out.is_none() {
                walk_expr_for_call(value, byte, out);
            }
        }
        Expr::Subscript { receiver, args } => {
            walk_expr_for_call(receiver, byte, out);
            for a in args {
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(a, byte, out);
            }
        }
        Expr::SubscriptSet {
            receiver,
            index_args,
            value,
        } => {
            walk_expr_for_call(receiver, byte, out);
            for a in index_args {
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(a, byte, out);
            }
            if out.is_none() {
                walk_expr_for_call(value, byte, out);
            }
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            walk_expr_for_call(condition, byte, out);
            if out.is_none() {
                walk_expr_for_call(then_expr, byte, out);
            }
            if out.is_none() {
                walk_expr_for_call(else_expr, byte, out);
            }
        }
        Expr::ListLiteral(items) => {
            for it in items {
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(it, byte, out);
            }
        }
        Expr::MapLiteral(pairs) => {
            for (k, v) in pairs {
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(k, byte, out);
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(v, byte, out);
            }
        }
        Expr::Range { from, to, .. } => {
            walk_expr_for_call(from, byte, out);
            if out.is_none() {
                walk_expr_for_call(to, byte, out);
            }
        }
        Expr::Closure { body, .. } => walk_stmt_for_call(body, byte, out),
        Expr::Interpolation(parts) => {
            for p in parts {
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(p, byte, out);
            }
        }
        Expr::SuperCall { args, .. } => {
            for a in args {
                if out.is_some() {
                    return;
                }
                walk_expr_for_call(a, byte, out);
            }
        }
        // Leaves — no children to descend into.
        Expr::Num(_)
        | Expr::Str(_)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::This
        | Expr::Ident(_)
        | Expr::Field(_)
        | Expr::StaticField(_) => {}
    }
}

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
///
/// `analysis` is the parsed AST + sema's [`TypeEnv`]. When
/// present, fields and `var` locals get typed signatures
/// (`instance field _trail: List`, `local count: Num`) sourced
/// from the typed AST instead of regex over the source text.
/// Pass `None` only when the parser couldn't produce a usable
/// module — a literal text-fallback path for half-typed code.
pub fn identifier_kind_hint(
    source: &str,
    ident: &str,
    byte: usize,
    prelude: &[ModuleDoc],
    analysis: Option<&Analysis>,
) -> Option<(String, String)> {
    if is_keyword(ident) {
        return None;
    }
    if ident.starts_with("__") {
        let sig = field_signature("static field", ident, ident.trim_start_matches('_'), byte, analysis);
        return Some((sig, String::new()));
    }
    if ident.starts_with('_') {
        let sig = field_signature("instance field", ident, ident.trim_start_matches('_'), byte, analysis);
        return Some((sig, String::new()));
    }
    if ident.chars().next().map_or(false, |c| c.is_ascii_uppercase()) {
        return Some((
            format!("class {}", ident),
            "Likely imported from an `@hatch:*` package whose docs the workspace hasn't loaded.".into(),
        ));
    }
    if let Some((line_no, line)) = find_var_decl_line(source, ident, byte) {
        // Prefer the typed AST: sema records every `var x = ...`
        // initializer at the decl's span_start. Fall back to the
        // RHS regex only when sema couldn't run (parser broken)
        // or the type came back as Any.
        let line_byte = source.lines().take(line_no - 1).map(|l| l.len() + 1).sum::<usize>();
        let typed = analysis.and_then(|a| {
            // The decl span starts at the `var` keyword's first
            // byte; the var-name token sits a few chars in. Find
            // the name's offset on the line and look it up.
            let trimmed = line.trim_start();
            let var_at = line_byte + (line.len() - line.trim_start().len());
            let _ = trimmed;
            a.var_type_at_span(var_at).and_then(|t| inferred_to_class_name(&t, &a.interner))
        });
        let signature = match typed {
            Some(t) => format!("local {}: {}", ident, t),
            None => {
                let rhs = decl_rhs(line.trim());
                let fallback = rhs.and_then(|r| infer_rhs_type(r, prelude));
                match fallback {
                    Some(t) => format!("local {}: {}", ident, t),
                    None => format!("local {}", ident),
                }
            }
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

/// Build a field-style signature with the inferred type spliced
/// in when sema knows it. `kind` is the human-readable bucket
/// (`"instance field"` or `"static field"`); `display` is the
/// full underscore-prefixed identifier; `bare` is the field name
/// with leading underscores stripped (the form sema interns).
fn field_signature(
    kind: &str,
    display: &str,
    bare: &str,
    byte: usize,
    analysis: Option<&Analysis>,
) -> String {
    let typed = analysis.and_then(|a| {
        a.field_type_at(byte, bare)
            .as_ref()
            .and_then(|t| inferred_to_class_name(t, &a.interner))
    });
    match typed {
        Some(t) => format!("{} {}: {}", kind, display, t),
        None => format!("{} {}", kind, display),
    }
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

    use super::{inferred_to_class_name, Analysis};

    #[test]
    fn field_type_resolves_via_sema_list_literal() {
        // A field assigned `[]` inside the constructor must
        // surface as `List` on hover. This is the
        // `_fruitNames = []` case the user flagged.
        let src = r#"
class Foo {
  construct new() {
    _items = []
  }
  use() {
    var c = _items.count
  }
}
"#;
        let a = Analysis::run(src).expect("parse + sema");
        // Cursor on the `_items` token inside `use()`.
        let byte = src.find("var c = _items").unwrap() + "var c = ".len() + 2; // mid-_items
        let ty = a.field_type_at(byte, "items").expect("field type known");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("List")
        );
    }

    #[test]
    fn receiver_type_resolves_for_list_field() {
        // `_items.count` with cursor on `count` should report
        // the receiver as `List` — fixes the
        // `_trail.count → String.count` mis-hover.
        let src = r#"
class Foo {
  construct new() { _items = [] }
  use() {
    var c = _items.count
  }
}
"#;
        let a = Analysis::run(src).expect("parse + sema");
        let dot = src.find(".count").unwrap();
        let cursor = dot + 1 + 2; // inside `count`
        let ty = a.receiver_type_for_call_at(cursor).expect("recv typed");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("List")
        );
    }

    #[test]
    fn local_var_type_records_at_decl_span() {
        let src = "var nums = []\n";
        let a = Analysis::run(src).expect("parse + sema");
        // The `var` keyword's first byte is index 0; sema
        // records the var's type at the *name* span, not the
        // statement span. Pick out the `nums` start.
        let name_at = src.find("nums").unwrap();
        let ty = a.var_type_at_span(name_at).expect("var type known");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("List")
        );
    }
}
