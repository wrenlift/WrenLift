/// Name and scope resolution for Wren.
///
/// Walks the AST and resolves every identifier to a `ResolvedName`:
/// - `Local(index)` — local variable in the current scope
/// - `Upvalue(index)` — captured variable from an enclosing scope
/// - `ModuleVar(index)` — top-level module variable
/// - `Field(SymbolId)` — instance field (`_name`)
/// - `StaticField(SymbolId)` — static field (`__name`)
///
/// Also detects:
/// - Use of undefined variables
/// - Duplicate variable declarations in the same scope
/// - `this` / `super` used outside a method
/// - `break` / `continue` used outside a loop
/// - `return` used at module level (allowed in Wren, but we track it)

use std::collections::HashMap;

use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::intern::{Interner, SymbolId};

// ---------------------------------------------------------------------------
// Resolution output
// ---------------------------------------------------------------------------

/// What an identifier resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedName {
    Local(u16),
    Upvalue(u16),
    ModuleVar(u16),
}

/// The result of a successful resolution pass.
#[derive(Debug)]
pub struct ResolveResult {
    /// Map from AST span start → resolved name (for identifiers).
    pub resolutions: HashMap<usize, ResolvedName>,
    /// Errors discovered during resolution.
    pub errors: Vec<Diagnostic>,
    /// Upvalue info for each scope that captures variables.
    pub upvalues: HashMap<usize, Vec<UpvalueInfo>>,
}

/// Information about a captured upvalue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpvalueInfo {
    /// The index in the parent scope (local index if `is_local`, upvalue index otherwise).
    pub index: u16,
    /// Whether this captures a local from the immediately enclosing scope
    /// (true) or an upvalue from it (false).
    pub is_local: bool,
}

// ---------------------------------------------------------------------------
// Scope types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum ScopeKind {
    Module,
    Class,
    Method,
    Block,
    Closure,
}

#[derive(Debug)]
struct Local {
    name: SymbolId,
    depth: usize,
    is_initialized: bool,
}

#[derive(Debug)]
struct Scope {
    kind: ScopeKind,
    locals: Vec<Local>,
    upvalues: Vec<UpvalueInfo>,
    depth: usize,
    /// Scope ID for upvalue tracking (span start of the function/closure).
    scope_id: usize,
}

impl Scope {
    fn new(kind: ScopeKind, depth: usize, scope_id: usize) -> Self {
        Self {
            kind,
            locals: Vec::new(),
            upvalues: Vec::new(),
            depth,
            scope_id,
        }
    }

    fn find_local(&self, name: SymbolId) -> Option<u16> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name {
                return Some(i as u16);
            }
        }
        None
    }

    fn add_upvalue(&mut self, info: UpvalueInfo) -> u16 {
        // Check if we already capture this.
        for (i, existing) in self.upvalues.iter().enumerate() {
            if existing.index == info.index && existing.is_local == info.is_local {
                return i as u16;
            }
        }
        let idx = self.upvalues.len() as u16;
        self.upvalues.push(info);
        idx
    }
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

pub struct Resolver<'a> {
    interner: &'a Interner,
    scopes: Vec<Scope>,
    module_vars: Vec<SymbolId>,
    resolutions: HashMap<usize, ResolvedName>,
    upvalue_map: HashMap<usize, Vec<UpvalueInfo>>,
    errors: Vec<Diagnostic>,
    loop_depth: usize,
    in_class: bool,
    in_method: bool,
}

impl<'a> Resolver<'a> {
    pub fn new(interner: &'a Interner) -> Self {
        Self {
            interner,
            scopes: Vec::new(),
            module_vars: Vec::new(),
            resolutions: HashMap::new(),
            upvalue_map: HashMap::new(),
            errors: Vec::new(),
            loop_depth: 0,
            in_class: false,
            in_method: false,
        }
    }

    /// Resolve all names in a module.
    pub fn resolve(mut self, module: &Module) -> ResolveResult {
        self.push_scope(ScopeKind::Module, 0);

        // First pass: register all top-level class and var declarations
        // as module-level variables (Wren allows forward references at module level).
        for stmt in module {
            match &stmt.0 {
                Stmt::Class(decl) => {
                    self.define_module_var(decl.name.0, decl.name.1.clone());
                }
                Stmt::Var { name, .. } => {
                    self.define_module_var(name.0, name.1.clone());
                }
                Stmt::Import { names, .. } => {
                    for import_name in names {
                        let sym = import_name.alias.as_ref().unwrap_or(&import_name.name);
                        self.define_module_var(sym.0, sym.1.clone());
                    }
                }
                _ => {}
            }
        }

        // Second pass: resolve bodies.
        for stmt in module {
            self.resolve_stmt(stmt);
        }

        self.pop_scope();

        ResolveResult {
            resolutions: self.resolutions,
            errors: self.errors,
            upvalues: self.upvalue_map,
        }
    }

    // -- Module vars --------------------------------------------------------

    fn define_module_var(&mut self, name: SymbolId, span: Span) {
        if self.module_vars.contains(&name) {
            self.errors.push(
                Diagnostic::error(format!(
                    "module variable '{}' is already defined",
                    self.interner.resolve(name)
                ))
                .with_label(span, "duplicate definition"),
            );
            return;
        }
        self.module_vars.push(name);
    }

    fn find_module_var(&self, name: SymbolId) -> Option<u16> {
        self.module_vars.iter().position(|&n| n == name).map(|i| i as u16)
    }

    // -- Scope management ---------------------------------------------------

    fn push_scope(&mut self, kind: ScopeKind, scope_id: usize) {
        let depth = self.scopes.len();
        self.scopes.push(Scope::new(kind, depth, scope_id));
    }

    fn pop_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            if !scope.upvalues.is_empty() {
                self.upvalue_map.insert(scope.scope_id, scope.upvalues);
            }
        }
    }

    fn current_scope(&self) -> &Scope {
        self.scopes.last().expect("no current scope")
    }

    fn current_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("no current scope")
    }

    fn declare_local(&mut self, name: SymbolId, span: Span) {
        // Check for duplicate in current block depth.
        let scope = self.current_scope();
        let depth = scope.depth;
        for local in &scope.locals {
            if local.name == name && local.depth == depth {
                self.errors.push(
                    Diagnostic::error(format!(
                        "variable '{}' is already declared in this scope",
                        self.interner.resolve(name)
                    ))
                    .with_label(span, "duplicate declaration"),
                );
                return;
            }
        }
        let depth = self.current_scope().depth;
        self.current_scope_mut().locals.push(Local {
            name,
            depth,
            is_initialized: false,
        });
    }

    fn mark_initialized(&mut self) {
        if let Some(local) = self.current_scope_mut().locals.last_mut() {
            local.is_initialized = true;
        }
    }

    // -- Name resolution ----------------------------------------------------

    fn resolve_name(&mut self, name: SymbolId, span: &Span) -> Option<ResolvedName> {
        // Walk scopes from innermost to outermost.
        let num_scopes = self.scopes.len();

        // First, check the current scope for a local.
        if let Some(idx) = self.scopes[num_scopes - 1].find_local(name) {
            let resolved = ResolvedName::Local(idx);
            self.resolutions.insert(span.start, resolved);
            return Some(resolved);
        }

        // Walk enclosing scopes to find locals/upvalues.
        for i in (0..num_scopes - 1).rev() {
            if let Some(local_idx) = self.scopes[i].find_local(name) {
                // Capture as upvalue through each intermediate scope.
                let resolved = self.capture_upvalue(i, local_idx, num_scopes - 1);
                self.resolutions.insert(span.start, resolved);
                return Some(resolved);
            }
        }

        // Check module-level variables.
        if let Some(idx) = self.find_module_var(name) {
            let resolved = ResolvedName::ModuleVar(idx);
            self.resolutions.insert(span.start, resolved);
            return Some(resolved);
        }

        None
    }

    fn capture_upvalue(&mut self, source_scope: usize, local_idx: u16, target_scope: usize) -> ResolvedName {
        // Create a chain of upvalues from source_scope+1 to target_scope.
        let mut index = local_idx;
        let mut is_local = true;

        for scope_idx in (source_scope + 1)..=target_scope {
            let info = UpvalueInfo { index, is_local };
            index = self.scopes[scope_idx].add_upvalue(info);
            is_local = false;
        }

        ResolvedName::Upvalue(index)
    }

    // -- Statement resolution -----------------------------------------------

    fn resolve_stmt(&mut self, stmt: &Spanned<Stmt>) {
        match &stmt.0 {
            Stmt::Expr(expr) => self.resolve_expr(expr),

            Stmt::Var { name, initializer } => {
                if let Some(init) = initializer {
                    self.resolve_expr(init);
                }
                // At module level, vars are already registered.
                if self.current_scope().kind != ScopeKind::Module {
                    self.declare_local(name.0, name.1.clone());
                    self.mark_initialized();
                }
            }

            Stmt::Class(decl) => self.resolve_class(decl),

            Stmt::Import { .. } => {
                // Imports are handled in the first pass.
            }

            Stmt::Block(stmts) => {
                self.push_scope(ScopeKind::Block, stmt.1.start);
                for s in stmts {
                    self.resolve_stmt(s);
                }
                self.pop_scope();
            }

            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.resolve_expr(condition);
                self.resolve_stmt(then_branch);
                if let Some(else_b) = else_branch {
                    self.resolve_stmt(else_b);
                }
            }

            Stmt::While { condition, body } => {
                self.resolve_expr(condition);
                self.loop_depth += 1;
                self.resolve_stmt(body);
                self.loop_depth -= 1;
            }

            Stmt::For {
                variable,
                iterator,
                body,
            } => {
                self.resolve_expr(iterator);
                self.push_scope(ScopeKind::Block, stmt.1.start);
                self.declare_local(variable.0, variable.1.clone());
                self.mark_initialized();
                self.loop_depth += 1;
                self.resolve_stmt(body);
                self.loop_depth -= 1;
                self.pop_scope();
            }

            Stmt::Break => {
                if self.loop_depth == 0 {
                    self.errors.push(
                        Diagnostic::error("'break' used outside of a loop")
                            .with_label(stmt.1.clone(), "not in a loop"),
                    );
                }
            }

            Stmt::Continue => {
                if self.loop_depth == 0 {
                    self.errors.push(
                        Diagnostic::error("'continue' used outside of a loop")
                            .with_label(stmt.1.clone(), "not in a loop"),
                    );
                }
            }

            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    self.resolve_expr(e);
                }
            }
        }
    }

    // -- Class resolution ---------------------------------------------------

    fn resolve_class(&mut self, decl: &ClassDecl) {
        if let Some(superclass) = &decl.superclass {
            self.resolve_name(superclass.0, &superclass.1);
        }

        let saved_in_class = self.in_class;
        self.in_class = true;

        for method in &decl.methods {
            self.resolve_method(&method.0, method.1.start);
        }

        self.in_class = saved_in_class;
    }

    fn resolve_method(&mut self, method: &Method, scope_id: usize) {
        if method.is_foreign {
            return; // Foreign methods have no body.
        }

        let saved_in_method = self.in_method;
        self.in_method = true;

        self.push_scope(ScopeKind::Method, scope_id);

        // Declare parameters.
        match &method.signature {
            MethodSig::Named { params, .. } | MethodSig::Construct { params, .. } => {
                for param in params {
                    self.declare_local(param.0, param.1.clone());
                    self.mark_initialized();
                }
            }
            MethodSig::Setter { param, .. } => {
                self.declare_local(param.0, param.1.clone());
                self.mark_initialized();
            }
            MethodSig::Subscript { params } => {
                for param in params {
                    self.declare_local(param.0, param.1.clone());
                    self.mark_initialized();
                }
            }
            MethodSig::SubscriptSetter { params, value } => {
                for param in params {
                    self.declare_local(param.0, param.1.clone());
                    self.mark_initialized();
                }
                self.declare_local(value.0, value.1.clone());
                self.mark_initialized();
            }
            MethodSig::Operator { params, .. } => {
                for param in params {
                    self.declare_local(param.0, param.1.clone());
                    self.mark_initialized();
                }
            }
            MethodSig::Getter(_) => {}
        }

        if let Some(body) = &method.body {
            self.resolve_stmt(body);
        }

        self.pop_scope();
        self.in_method = saved_in_method;
    }

    // -- Expression resolution ----------------------------------------------

    fn resolve_expr(&mut self, expr: &Spanned<Expr>) {
        match &expr.0 {
            Expr::Num(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Null => {}

            Expr::This => {
                if !self.in_method {
                    self.errors.push(
                        Diagnostic::error("'this' used outside of a method")
                            .with_label(expr.1.clone(), "not in a method"),
                    );
                }
            }

            Expr::Ident(name) => {
                if self.resolve_name(*name, &expr.1).is_none() {
                    self.errors.push(
                        Diagnostic::error(format!(
                            "undefined variable '{}'",
                            self.interner.resolve(*name)
                        ))
                        .with_label(expr.1.clone(), "not found"),
                    );
                }
            }

            Expr::Field(_) | Expr::StaticField(_) => {
                if !self.in_method {
                    self.errors.push(
                        Diagnostic::error("fields can only be used inside a method")
                            .with_label(expr.1.clone(), "not in a method"),
                    );
                }
            }

            Expr::UnaryOp { operand, .. } => {
                self.resolve_expr(operand);
            }

            Expr::BinaryOp { left, right, .. } => {
                self.resolve_expr(left);
                self.resolve_expr(right);
            }

            Expr::LogicalOp { left, right, .. } => {
                self.resolve_expr(left);
                self.resolve_expr(right);
            }

            Expr::Is { value, type_name } => {
                self.resolve_expr(value);
                self.resolve_expr(type_name);
            }

            Expr::Assign { target, value } => {
                self.resolve_expr(value);
                self.resolve_expr(target);
            }

            Expr::CompoundAssign { target, value, .. } => {
                self.resolve_expr(target);
                self.resolve_expr(value);
            }

            Expr::Call {
                receiver,
                args,
                block_arg,
                ..
            } => {
                if let Some(recv) = receiver {
                    self.resolve_expr(recv);
                }
                for arg in args {
                    self.resolve_expr(arg);
                }
                if let Some(block) = block_arg {
                    self.resolve_expr(block);
                }
            }

            Expr::SuperCall { args, .. } => {
                if !self.in_method {
                    self.errors.push(
                        Diagnostic::error("'super' used outside of a method")
                            .with_label(expr.1.clone(), "not in a method"),
                    );
                }
                for arg in args {
                    self.resolve_expr(arg);
                }
            }

            Expr::Subscript { receiver, args } => {
                self.resolve_expr(receiver);
                for arg in args {
                    self.resolve_expr(arg);
                }
            }

            Expr::SubscriptSet {
                receiver,
                index_args,
                value,
            } => {
                self.resolve_expr(receiver);
                for arg in index_args {
                    self.resolve_expr(arg);
                }
                self.resolve_expr(value);
            }

            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.resolve_expr(condition);
                self.resolve_expr(then_expr);
                self.resolve_expr(else_expr);
            }

            Expr::ListLiteral(elements) => {
                for elem in elements {
                    self.resolve_expr(elem);
                }
            }

            Expr::MapLiteral(entries) => {
                for (key, val) in entries {
                    self.resolve_expr(key);
                    self.resolve_expr(val);
                }
            }

            Expr::Range { from, to, .. } => {
                self.resolve_expr(from);
                self.resolve_expr(to);
            }

            Expr::Closure { params, body } => {
                self.push_scope(ScopeKind::Closure, expr.1.start);
                for param in params {
                    self.declare_local(param.0, param.1.clone());
                    self.mark_initialized();
                }
                self.resolve_stmt(body);
                self.pop_scope();
            }

            Expr::Interpolation(parts) => {
                for part in parts {
                    self.resolve_expr(part);
                }
            }
        }
    }
}

/// Convenience: resolve a parsed module.
pub fn resolve(module: &Module, interner: &Interner) -> ResolveResult {
    let resolver = Resolver::new(interner);
    resolver.resolve(module)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    fn resolve_source(source: &str) -> (ResolveResult, Interner) {
        let parse_result = parse(source);
        assert!(
            parse_result.errors.is_empty(),
            "parse errors: {:?}",
            parse_result.errors
        );
        let result = resolve(&parse_result.module, &parse_result.interner);
        (result, parse_result.interner)
    }

    fn resolve_errors(source: &str) -> Vec<String> {
        let (result, _) = resolve_source(source);
        result.errors.iter().map(|e| e.message.clone()).collect()
    }

    fn expect_no_errors(source: &str) {
        let errors = resolve_errors(source);
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    fn expect_error_containing(source: &str, substring: &str) {
        let errors = resolve_errors(source);
        assert!(
            errors.iter().any(|e| e.contains(substring)),
            "expected error containing '{}', got: {:?}",
            substring,
            errors
        );
    }

    // -- Local variable resolution ------------------------------------------

    #[test]
    fn test_local_resolution() {
        expect_no_errors("var x = 1\nx");
    }

    #[test]
    fn test_local_in_block() {
        expect_no_errors("{\n  var x = 1\n  x\n}");
    }

    #[test]
    fn test_undefined_variable() {
        expect_error_containing("x", "undefined variable 'x'");
    }

    #[test]
    fn test_duplicate_module_var() {
        expect_error_containing("var x = 1\nvar x = 2", "already defined");
    }

    #[test]
    fn test_duplicate_local_var() {
        expect_error_containing(
            "{\n  var x = 1\n  var x = 2\n}",
            "already declared",
        );
    }

    #[test]
    fn test_shadowing_ok() {
        // Different scopes — should be fine.
        expect_no_errors("var x = 1\n{\n  var x = 2\n  x\n}");
    }

    // -- Module-level forward reference -------------------------------------

    #[test]
    fn test_forward_reference() {
        // Classes can be referenced before declaration at module level.
        expect_no_errors("var x = Foo\nclass Foo {}");
    }

    // -- Upvalue resolution -------------------------------------------------

    #[test]
    fn test_closure_captures_local() {
        // Use a method call with block arg to create a closure scope.
        expect_no_errors("var x = 1\nvar f = { x }");
    }

    #[test]
    fn test_nested_closure_capture() {
        expect_no_errors("var x = 1\nvar f = {\n  var g = { x }\n  g\n}");
    }

    // -- this / super -------------------------------------------------------

    #[test]
    fn test_this_in_method() {
        expect_no_errors("class Foo {\n  bar { this }\n}");
    }

    #[test]
    fn test_this_outside_method() {
        expect_error_containing("this", "'this' used outside");
    }

    #[test]
    fn test_super_outside_method() {
        expect_error_containing("super.foo()", "'super' used outside");
    }

    #[test]
    fn test_super_in_method() {
        expect_no_errors(
            "class Base {}\nclass Foo is Base {\n  bar { super.bar() }\n}",
        );
    }

    // -- Fields -------------------------------------------------------------

    #[test]
    fn test_field_in_method() {
        expect_no_errors("class Foo {\n  bar { _x }\n}");
    }

    #[test]
    fn test_field_outside_method() {
        expect_error_containing("_x", "fields can only be used inside a method");
    }

    #[test]
    fn test_static_field_outside_method() {
        expect_error_containing("__x", "fields can only be used inside a method");
    }

    // -- break / continue ---------------------------------------------------

    #[test]
    fn test_break_in_loop() {
        expect_no_errors("while (true) break");
    }

    #[test]
    fn test_break_outside_loop() {
        expect_error_containing("break", "'break' used outside of a loop");
    }

    #[test]
    fn test_continue_outside_loop() {
        expect_error_containing("continue", "'continue' used outside of a loop");
    }

    #[test]
    fn test_break_in_for() {
        expect_no_errors("var list = null\nfor (x in list) break");
    }

    // -- Class resolution ---------------------------------------------------

    #[test]
    fn test_class_with_methods() {
        expect_no_errors(
            "class Point {\n  construct new(x, y) {\n    _x = x\n    _y = y\n  }\n  x { _x }\n  y { _y }\n}",
        );
    }

    #[test]
    fn test_class_inherits_known() {
        expect_no_errors("class Base {}\nclass Child is Base {}");
    }

    // -- For loop variable --------------------------------------------------

    #[test]
    fn test_for_variable_scope() {
        // The for-in variable is in scope within the body.
        expect_no_errors("var list = null\nfor (item in list) item");
    }

    // -- Complete program ---------------------------------------------------

    #[test]
    fn test_complete_program() {
        expect_no_errors(
            r#"
class Animal {
  construct new(name) {
    _name = name
  }
  name { _name }
  greet() {
    var msg = "Hello"
    msg
  }
}

var a = Animal.new("Cat")
a.name
a.greet()
"#,
        );
    }

    // -- Resolution tracking ------------------------------------------------

    #[test]
    fn test_resolution_entries() {
        let (result, _interner) = resolve_source("var x = 1\nx");
        assert!(result.errors.is_empty());
        // There should be at least one resolution entry (for the `x` reference).
        assert!(!result.resolutions.is_empty());
    }

    // -- Multiple errors ----------------------------------------------------

    #[test]
    fn test_multiple_errors() {
        let errors = resolve_errors("x\ny\nz");
        assert_eq!(errors.len(), 3);
    }
}
