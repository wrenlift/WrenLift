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
use crate::sema::types::{infer_types_with_classes, InferredType, TypeEnv};

use super::{ModuleDoc, ParamTypeInfo};

/// Parsed module + interner + inferred type environment, kept
/// together so hover / completion can consult sema's typed AST
/// instead of falling back to text heuristics. Built once per
/// hover request via [`Analysis::run`]; cheap to throw away.
pub struct Analysis {
    pub module: Module,
    pub interner: Interner,
    pub type_env: TypeEnv,
    /// Names that resolve to a class in this analysis: every
    /// local `class Foo {}` decl, every `import for X` binding,
    /// plus any extra class names the caller supplied. Cached
    /// alongside `type_env` because a method-call's `Expr::Call`
    /// shares its `span.start` with its receiver, and sema's
    /// `infer_expr` clobbers the receiver's recorded type with
    /// the Call's (usually `Any`) return type. Receiver hover
    /// and goto can't trust the per-expr table for
    /// class-receiver idents and consult this set instead.
    pub known_classes: std::collections::HashSet<SymbolId>,
}

impl Analysis {
    /// Parse + run the type inferrer. Returns `None` when the
    /// parser couldn't produce a usable module — callers fall
    /// back to text heuristics in that case.
    /// Convenience: run with only the prelude class names plus
    /// names sema can derive from the source itself (local class
    /// decls, `import for X`). Most callers should prefer
    /// [`Analysis::run_with_extra_classes`] so dep-package
    /// classes (`Renderer2D`, `Game`, …) also flow through.
    pub fn run(source: &str) -> Option<Self> {
        Self::run_with_extra_classes(source, std::iter::empty::<&str>())
    }

    /// Same as [`Analysis::run`], plus extra class names the
    /// caller knows about (typically every class name the
    /// playground / LSP has loaded from `@hatch:*` packages).
    /// Sema then treats matching `Expr::Ident`s as `Class(sym)`
    /// and `<Sym>.new(args)` as a constructor call returning
    /// that class — so `_renderer = Renderer2D.new(...)` infers
    /// `_renderer: Renderer2D` even though sema has no source
    /// for the `Renderer2D` class.
    pub fn run_with_extra_classes<'a>(
        source: &str,
        extra_class_names: impl IntoIterator<Item = &'a str>,
    ) -> Option<Self> {
        let pr = crate::parse::parser::parse(source);
        if !pr.errors.is_empty() {
            return None;
        }
        let known_classes = build_known_classes(&pr.module, &pr.interner, extra_class_names);
        let new_symbol = pr.interner.lookup("new");
        // 3-pass inference: pass 1 collects field assignment
        // types, pass 2 records method return types from the
        // field-aware inference, pass 3 emits the final
        // per-expression type map. Single-pass left field
        // references as `Any` because the field hadn't been
        // recorded yet when the use site was visited.
        let type_env = infer_types_with_classes(&pr.module, known_classes.clone(), new_symbol);
        Some(Analysis {
            module: pr.module,
            interner: pr.interner,
            type_env,
            known_classes,
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
            Expr::Ident(sym) => {
                // Class-shaped idents (local class decls, scoped
                // imports, prelude) come straight from the
                // cached `known_classes` set — sema's per-expr
                // table can't be trusted for these because a
                // method-call's `Expr::Call` shares its
                // `span.start` with the receiver and overwrites
                // the receiver's `Class(sym)` with the Call's
                // `Any` return.
                if self.known_classes.contains(sym) {
                    InferredType::Class(*sym)
                } else {
                    let direct = self.type_env.get_expr_type(recv.1.start).clone();
                    if direct.is_known() {
                        direct
                    } else {
                        self.local_var_type(byte, *sym).unwrap_or(InferredType::Any)
                    }
                }
            }
            // Literal-receiver shapes get their type directly from
            // the AST node — no per-expr table lookup needed.
            // Sema only re-records these spans during a full pass
            // through the receiver expr, which a hover cursor
            // sitting on the method name doesn't always trigger
            // (the receiver's span_start lands inside parens or
            // bracket whitespace that sema doesn't index).
            Expr::Num(_) => InferredType::Num,
            Expr::Str(_) | Expr::Interpolation(_) => InferredType::String,
            Expr::Bool(_) => InferredType::Bool,
            Expr::ListLiteral(_) => InferredType::List,
            Expr::MapLiteral(_) => InferredType::Map,
            Expr::Range { .. } => InferredType::Range,
            Expr::Closure { .. } => InferredType::Fn,
            _ => self.type_env.get_expr_type(recv.1.start).clone(),
        };
        ty.is_known().then_some(ty)
    }

    /// Find the nearest `var <sym>` declaration that's lexically
    /// in scope at `byte`, and return the type sema recorded for
    /// it. Walks the AST so the lookup is structure-aware
    /// (different `var x` in different methods don't collide).
    fn local_var_type(&self, byte: usize, sym: SymbolId) -> Option<InferredType> {
        let span = self.local_var_decl_span(byte, sym)?;
        let ty = self.type_env.get_var_type(span.start).clone();
        ty.is_known().then_some(ty)
    }

    /// Walk-shared helper that returns the span of the
    /// nearest in-scope `var <sym>`, `for (<sym> in …)`, method
    /// parameter, or closure parameter declaration. The span
    /// covers just the binding name so goto-def lands the
    /// cursor on the binder, not the surrounding statement.
    fn local_var_decl_span(&self, byte: usize, sym: SymbolId) -> Option<std::ops::Range<usize>> {
        let mut found: Option<std::ops::Range<usize>> = None;
        for stmt in &self.module {
            walk_stmt_for_var_decl(stmt, byte, sym, &mut found);
        }
        found
    }

    /// Public goto-def hook. Returns the binder span for the
    /// nearest in-scope local with name `name`, or `None` when
    /// no such binding exists at `byte`.
    pub fn local_var_decl_span_by_name(
        &self,
        byte: usize,
        name: &str,
    ) -> Option<std::ops::Range<usize>> {
        let sym = self.interner.lookup(name)?;
        self.local_var_decl_span(byte, sym)
    }

    /// Same as [`local_var_type`](Self::local_var_type) but takes
    /// the variable name as a string. Looks the symbol up in
    /// this analysis's interner and routes through the AST walk
    /// — used by `identifier_kind_hint` so a `var quad = _quad`
    /// hover can splice the field's type into the local's
    /// signature without any regex over the source.
    pub fn local_var_type_by_name(&self, byte: usize, name: &str) -> Option<InferredType> {
        let sym = self.interner.lookup(name)?;
        self.local_var_type(byte, sym)
    }

    /// Walk the AST to find the receiver expression of the call
    /// whose method-name span covers `byte`. Returned as a
    /// reference into the analysed module so the caller can
    /// inspect the AST node directly (the receiver might be a
    /// field, `this`, a chained call, etc.).
    pub fn find_call_receiver_at(&self, byte: usize) -> Option<&Spanned<Expr>> {
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

/// Walk the module's top-level imports looking for `name` (or
/// any alias). Returns the source module string from the matching
/// `import "..." for ...` so hover can say *"imported from
/// @hatch:gpu"* instead of the generic "workspace hasn't loaded"
/// when the class is sitting in a dep that just hasn't finished
/// fetching.
pub fn imported_from(module: &Module, interner: &Interner, name: &str) -> Option<String> {
    let sym = interner.lookup(name)?;
    for stmt in module {
        if let Stmt::Import { module: src, names } = &stmt.0 {
            for n in names {
                let bind = n.alias.as_ref().unwrap_or(&n.name);
                if bind.0 == sym {
                    return Some(src.0.clone());
                }
            }
        }
    }
    None
}

/// Symbols sema should treat as classes during type inference:
///
/// 1. Every `class Foo {...}` in the source.
/// 2. Every `import "..." for X, Y as Z` — imported names are
///    almost always classes, and treating them as such gives
///    `X.new(args) → X` for free.
/// 3. Every entry in `extra_class_names` whose name is interned
///    in this source's interner. Hover passes the union of
///    prelude class names + dep-package class names here so
///    `Renderer2D`, `Game`, `Num`, `String`, … all resolve.
fn build_known_classes<'a>(
    module: &Module,
    interner: &Interner,
    extra_class_names: impl IntoIterator<Item = &'a str>,
) -> std::collections::HashSet<SymbolId> {
    let mut out = std::collections::HashSet::new();
    for stmt in module {
        match &stmt.0 {
            Stmt::Class(decl) => {
                out.insert(decl.name.0);
            }
            Stmt::Import { names, .. } => {
                for n in names {
                    let sym = n.alias.as_ref().unwrap_or(&n.name);
                    out.insert(sym.0);
                }
            }
            _ => {}
        }
    }
    for name in extra_class_names {
        if let Some(sym) = interner.lookup(name) {
            out.insert(sym);
        }
    }
    out
}

/// Find the nearest `var <sym> = ...` decl that's lexically
/// in scope at `byte`. Out-param is the decl's *name span
/// start* (the same key sema's TypeEnv uses for var types).
fn walk_stmt_for_var_decl(
    stmt: &Spanned<Stmt>,
    byte: usize,
    sym: SymbolId,
    out: &mut Option<std::ops::Range<usize>>,
) {
    match &stmt.0 {
        Stmt::Var {
            name, initializer, ..
        } => {
            // Var must be declared *before* the cursor and at
            // module level (or in an enclosing block — which
            // walk_stmt_for_var_decl handles by recursing into
            // Block / If / While / For below).
            if name.0 == sym && name.1.start <= byte {
                *out = Some(name.1.clone());
            }
            // Still descend into the initializer in case it
            // contains a closure with its own decls.
            if let Some(init) = initializer {
                walk_expr_for_var_decl(init, byte, sym, out);
            }
        }
        Stmt::Class(class) => {
            for method in &class.methods {
                // Method parameters declare locals at the
                // parameter span. Walk the signature first.
                walk_method_sig_for_var_decl(&method.0.signature, sym, byte, out);
                if let Some(body) = &method.0.body {
                    walk_stmt_for_var_decl(body, byte, sym, out);
                }
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                // Block variables only shadow up to the cursor.
                walk_stmt_for_var_decl(s, byte, sym, out);
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            walk_expr_for_var_decl(condition, byte, sym, out);
            walk_stmt_for_var_decl(then_branch, byte, sym, out);
            if let Some(eb) = else_branch {
                walk_stmt_for_var_decl(eb, byte, sym, out);
            }
        }
        Stmt::While { condition, body } => {
            walk_expr_for_var_decl(condition, byte, sym, out);
            walk_stmt_for_var_decl(body, byte, sym, out);
        }
        Stmt::For {
            variable,
            iterator,
            body,
        } => {
            if variable.0 == sym && variable.1.start <= byte {
                *out = Some(variable.1.clone());
            }
            walk_expr_for_var_decl(iterator, byte, sym, out);
            walk_stmt_for_var_decl(body, byte, sym, out);
        }
        Stmt::Expr(e) => walk_expr_for_var_decl(e, byte, sym, out),
        Stmt::Return(Some(e)) => walk_expr_for_var_decl(e, byte, sym, out),
        Stmt::Return(None) | Stmt::Break | Stmt::Continue | Stmt::Import { .. } => {}
    }
}

fn walk_expr_for_var_decl(
    expr: &Spanned<Expr>,
    byte: usize,
    sym: SymbolId,
    out: &mut Option<std::ops::Range<usize>>,
) {
    match &expr.0 {
        Expr::Closure { params, body } => {
            for p in params {
                if p.0 == sym && p.1.start <= byte {
                    *out = Some(p.1.clone());
                }
            }
            walk_stmt_for_var_decl(body, byte, sym, out);
        }
        Expr::Call {
            receiver,
            args,
            block_arg,
            ..
        } => {
            if let Some(r) = receiver {
                walk_expr_for_var_decl(r, byte, sym, out);
            }
            for a in args {
                walk_expr_for_var_decl(a, byte, sym, out);
            }
            if let Some(b) = block_arg {
                walk_expr_for_var_decl(b, byte, sym, out);
            }
        }
        Expr::UnaryOp { operand, .. } => walk_expr_for_var_decl(operand, byte, sym, out),
        Expr::BinaryOp { left, right, .. } | Expr::LogicalOp { left, right, .. } => {
            walk_expr_for_var_decl(left, byte, sym, out);
            walk_expr_for_var_decl(right, byte, sym, out);
        }
        Expr::Is { value, type_name } => {
            walk_expr_for_var_decl(value, byte, sym, out);
            walk_expr_for_var_decl(type_name, byte, sym, out);
        }
        Expr::Assign { target, value } | Expr::CompoundAssign { target, value, .. } => {
            walk_expr_for_var_decl(target, byte, sym, out);
            walk_expr_for_var_decl(value, byte, sym, out);
        }
        Expr::Subscript { receiver, args } => {
            walk_expr_for_var_decl(receiver, byte, sym, out);
            for a in args {
                walk_expr_for_var_decl(a, byte, sym, out);
            }
        }
        Expr::SubscriptSet {
            receiver,
            index_args,
            value,
        } => {
            walk_expr_for_var_decl(receiver, byte, sym, out);
            for a in index_args {
                walk_expr_for_var_decl(a, byte, sym, out);
            }
            walk_expr_for_var_decl(value, byte, sym, out);
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            walk_expr_for_var_decl(condition, byte, sym, out);
            walk_expr_for_var_decl(then_expr, byte, sym, out);
            walk_expr_for_var_decl(else_expr, byte, sym, out);
        }
        Expr::ListLiteral(items) => {
            for it in items {
                walk_expr_for_var_decl(it, byte, sym, out);
            }
        }
        Expr::MapLiteral(pairs) => {
            for (k, v) in pairs {
                walk_expr_for_var_decl(k, byte, sym, out);
                walk_expr_for_var_decl(v, byte, sym, out);
            }
        }
        Expr::Range { from, to, .. } => {
            walk_expr_for_var_decl(from, byte, sym, out);
            walk_expr_for_var_decl(to, byte, sym, out);
        }
        Expr::Interpolation(parts) => {
            for p in parts {
                walk_expr_for_var_decl(p, byte, sym, out);
            }
        }
        Expr::SuperCall { args, .. } => {
            for a in args {
                walk_expr_for_var_decl(a, byte, sym, out);
            }
        }
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

fn walk_method_sig_for_var_decl(
    sig: &crate::ast::MethodSig,
    sym: SymbolId,
    byte: usize,
    out: &mut Option<std::ops::Range<usize>>,
) {
    use crate::ast::MethodSig::*;
    let params: &[Spanned<SymbolId>] = match sig {
        Named { params, .. }
        | Subscript { params }
        | Operator { params, .. }
        | Construct { params, .. } => params,
        SubscriptSetter { params, value } => {
            walk_param(value, sym, byte, out);
            params
        }
        Setter { param, .. } => {
            walk_param(param, sym, byte, out);
            return;
        }
        Getter(_) => return,
    };
    for p in params {
        walk_param(p, sym, byte, out);
    }
}

fn walk_param(
    p: &Spanned<SymbolId>,
    sym: SymbolId,
    byte: usize,
    out: &mut Option<std::ops::Range<usize>>,
) {
    if p.0 == sym && p.1.start <= byte {
        *out = Some(p.1.clone());
    }
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
        Stmt::For { iterator, body, .. } => {
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
        let sig = field_signature(
            "static field",
            ident,
            ident.trim_start_matches('_'),
            byte,
            analysis,
        );
        return Some((sig, String::new()));
    }
    if ident.starts_with('_') {
        let sig = field_signature(
            "instance field",
            ident,
            ident.trim_start_matches('_'),
            byte,
            analysis,
        );
        return Some((sig, String::new()));
    }
    if ident.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        // Class-shaped name we couldn't match against any class
        // in the local module, dep_docs, or the prelude. Tailor
        // the body to whether the source actually imports it:
        //   * imported here → "imported from <pkg>; docs may
        //     still be loading" (true mid-prefetch).
        //   * otherwise → no docs for this name; could be a
        //     class the workspace doesn't know about yet.
        let body = match analysis.and_then(|a| imported_from(&a.module, &a.interner, ident)) {
            Some(pkg) => format!(
                "Imported from `{}`. Docs may still be loading — they appear once the workspace finishes resolving the dep graph.",
                pkg
            ),
            None => "No matching class in the local module, loaded `@hatch:*` packages, or the prelude.".into(),
        };
        return Some((format!("class {}", ident), body));
    }
    if let Some((line_no, line)) = find_var_decl_line(source, ident, byte) {
        // Prefer the typed AST: walk to the var's *name* span
        // (sema records var types there, not at `var`'s span)
        // and look up the recorded type. Pointing at `v` of
        // `var` would miss every lookup and degrade to
        // `local quad` instead of `local quad: Sprite`.
        let typed = analysis
            .and_then(|a| a.local_var_type_by_name(byte, ident))
            .and_then(|t| analysis.and_then(|a| inferred_to_class_name(&t, &a.interner)));
        let signature = match typed {
            Some(t) => format!("local {}: {}", ident, t),
            None => {
                // Last-resort RHS regex — only fires when sema
                // couldn't pin a type (e.g. the initializer
                // chains through an untyped builder).
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
        '-' if r.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) => {
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
        let line_start = source.lines().take(idx).map(|l| l.len() + 1).sum::<usize>();
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
    tail.is_empty() || tail.starts_with('=') || tail.starts_with(',') || tail.starts_with(' ')
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

fn find_for_binding(source: &str, ident: &str, scope_start: usize) -> Option<(usize, String)> {
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
    let after_paren = after_for.strip_prefix('(')?.trim_start();
    if let Some(tail) = after_paren.strip_prefix(ident) {
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
            "as",
            "break",
            "class",
            "construct",
            "continue",
            "else",
            "false",
            "for",
            "foreign",
            "if",
            "import",
            "in",
            "is",
            "null",
            "return",
            "static",
            "super",
            "this",
            "true",
            "var",
            "while",
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

    #[test]
    fn imported_class_constructor_propagates_to_field() {
        // The exact shape that broke before: an imported class
        // (`Sprite`) has its constructor called and the result
        // assigned to a field. With the imported name in the
        // class pool, sema infers the field as `Sprite`.
        let src = r#"
import "@hatch:gpu" for Sprite

class Slicer {
  setup(g) {
    _quad = Sprite.new(g)
  }
  draw() {
    _quad.draw(g)
  }
}
"#;
        let a = Analysis::run(src).expect("parse + sema");
        // Cursor on the `_quad` reference inside `draw()`.
        let cursor = src.find("_quad.draw").unwrap();
        let ty = a.field_type_at(cursor, "quad").expect("field typed");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("Sprite")
        );
    }

    #[test]
    fn local_var_type_via_local_var_type_by_name() {
        // `var quad = _quad` where `_quad: Sprite`. The
        // identifier_kind_hint path now drives type lookup
        // through `local_var_type_by_name`, which finds the
        // var's *name* span (sema's recording key) instead of
        // the previous regex-derived line-byte that pointed at
        // `var`.
        let src = r#"
import "@hatch:gpu" for Sprite

class Slicer {
  setup(g) { _quad = Sprite.new(g) }
  drawAll() {
    var quad = _quad
  }
}
"#;
        let a = Analysis::run(src).expect("parse + sema");
        let cursor = src.find("quad = _quad").unwrap() + 1;
        let ty = a
            .local_var_type_by_name(cursor, "quad")
            .expect("local typed");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("Sprite")
        );
    }

    #[test]
    fn arithmetic_with_unknown_operand_is_num() {
        // `var n = some.getter * 22`. Sema can't pin the
        // method-return type for `some.getter` (that needs the
        // primitive method-return table that hasn't landed
        // yet), but the multiplication should still infer
        // `Num` — Wren's arithmetic ops only accept Num
        // operands, so the well-typed result is Num regardless.
        let src = "var n = 1.cos * 22\n";
        let a = Analysis::run(src).expect("parse + sema");
        let name_at = src.find("n =").unwrap();
        let ty = a.var_type_at_span(name_at).expect("var typed");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("Num")
        );
    }

    #[test]
    fn local_assigned_from_field_keeps_class_type() {
        // `var quad = _quad` where `_quad: Sprite` — the local
        // should also be `Sprite`. This is the chain the user
        // hit when `quad.draw(_renderer)` resolved to
        // `FruitSlicer.draw(g)` instead of `Sprite.draw`.
        let src = r#"
import "@hatch:gpu" for Sprite

class Slicer {
  setup(g) { _quad = Sprite.new(g) }
  drawAll() {
    var quad = _quad
    quad.draw(g)
  }
}
"#;
        let a = Analysis::run(src).expect("parse + sema");
        // The receiver of `quad.draw(...)` is the Ident `quad`.
        let cursor = src.find("quad.draw").unwrap() + "quad.dr".len();
        let ty = a.receiver_type_for_call_at(cursor).expect("recv typed");
        assert_eq!(
            inferred_to_class_name(&ty, &a.interner).as_deref(),
            Some("Sprite")
        );
    }
}

#[cfg(test)]
mod imported_class_receiver_tests {
    use super::*;

    fn class_name_under(src: &str, pat: &str) -> Option<String> {
        let byte = src.find(pat).unwrap() + 1;
        let a = Analysis::run(src)?;
        let ty = a.receiver_type_for_call_at(byte)?;
        inferred_to_class_name(&ty, &a.interner)
    }

    #[test]
    fn imported_class_static_method_receiver_resolves() {
        let src = r#"import "@hatch:fmt" for Fmt
class App {
  static run() {
    System.print(Fmt.green("hello"))
  }
}
"#;
        assert_eq!(class_name_under(src, "green"), Some("Fmt".into()));
    }
}

#[cfg(test)]
mod literal_receiver_tests {
    use super::*;

    fn class_name_under(src: &str, pat: &str) -> Option<String> {
        let byte = src.find(pat).unwrap() + 1;
        let a = Analysis::run(src)?;
        let ty = a.receiver_type_for_call_at(byte)?;
        inferred_to_class_name(&ty, &a.interner)
    }

    #[test]
    fn paren_num_literal_receiver_resolves_to_num() {
        assert_eq!(
            class_name_under("var x = (10).ceil\n", "ceil"),
            Some("Num".into())
        );
    }

    #[test]
    fn list_literal_receiver_resolves_to_list() {
        assert_eq!(
            class_name_under("var x = [1, 2, 3].count\n", "count"),
            Some("List".into())
        );
    }

    #[test]
    fn string_literal_receiver_resolves_to_string() {
        assert_eq!(
            class_name_under("var x = \"hi\".count\n", "count"),
            Some("String".into())
        );
    }
}

#[cfg(test)]
mod gpu_smoke {
    #[test]
    #[ignore]
    fn parses_published_gpu_source() {
        let src = std::fs::read_to_string("hatch/packages/hatch-gpu/gpu_web.wren").unwrap();
        let pr = crate::parse::parser::parse(&src);
        if !pr.errors.is_empty() {
            for e in pr.errors.iter().take(5) {
                eprintln!("parse err: {:?}", e);
            }
            panic!("{} parse errors", pr.errors.len());
        }
        let mut classes: Vec<&str> = Vec::new();
        for stmt in &pr.module {
            if let crate::ast::Stmt::Class(c) = &stmt.0 {
                classes.push(pr.interner.resolve(c.name.0));
            }
        }
        eprintln!("classes: {:?}", classes);
        assert!(classes.contains(&"Camera2D"));
        assert!(classes.contains(&"Renderer2D"));
        assert!(classes.contains(&"Sprite"));
    }
}

#[cfg(test)]
mod gpu_bundle_smoke {
    #[test]
    #[ignore]
    fn camera2d_in_collected_module_doc() {
        let src = std::fs::read_to_string("hatch/packages/hatch-gpu/gpu_web.wren").unwrap();
        let pr = crate::parse::parser::parse(&src);
        assert!(
            pr.errors.is_empty(),
            "gpu_web.wren has parse errors: {}",
            pr.errors.len()
        );
        let m = crate::docs::collect::collect_module(
            "gpu_web",
            &src,
            &pr.module,
            &pr.docs,
            &pr.interner,
        );
        let class_names: Vec<&str> = m.classes.iter().map(|c| c.name.as_str()).collect();
        eprintln!("collected classes: {:?}", class_names);
        assert!(class_names.contains(&"Camera2D"), "Camera2D missing");
        let r2d = m.classes.iter().find(|c| c.name == "Renderer2D").unwrap();
        let new_members: Vec<&str> = r2d
            .members
            .iter()
            .filter(|mb| mb.name == "new")
            .map(|mb| mb.signature.as_str())
            .collect();
        eprintln!("Renderer2D.new members: {:?}", new_members);
        assert!(!new_members.is_empty(), "Renderer2D has no `new` member");
    }

    #[test]
    #[ignore]
    fn input_mousejustpressed_in_collected_game_docs() {
        let raw = std::fs::read_to_string("hatch/packages/hatch-game/game.wren").unwrap();
        // Same cfg pre-pass that `register_hatch_docs` runs on
        // wasm — the published source has `#!wasm` attributes
        // that the parser rejects in place but the cfg pass
        // either keeps or strips depending on target.
        let src = crate::parse::cfg::apply(&raw, Some("wasm32"));
        let pr = crate::parse::parser::parse(&src);
        for e in pr.errors.iter().take(5) {
            eprintln!("parse err: {:?}", e);
        }
        assert!(pr.errors.is_empty(), "{} parse errors", pr.errors.len());
        let m =
            crate::docs::collect::collect_module("game", &src, &pr.module, &pr.docs, &pr.interner);
        let input = m
            .classes
            .iter()
            .find(|c| c.name == "Input")
            .expect("no Input class in collected docs");
        let mjp: Vec<&str> = input
            .members
            .iter()
            .filter(|mb| mb.name == "mouseJustPressed")
            .map(|mb| mb.signature.as_str())
            .collect();
        eprintln!("Input.mouseJustPressed: {:?}", mjp);
        assert!(!mjp.is_empty(), "Input has no mouseJustPressed");
    }
}
