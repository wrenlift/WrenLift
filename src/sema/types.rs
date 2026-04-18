/// Speculative type inference for Wren.
///
/// This pass walks the AST and infers types for expressions where possible,
/// producing an `InferredType` for each expression. The type system is a
/// lattice with widening:
///
/// - Concrete: `Num`, `Bool`, `Null`, `String`, `List`, `Map`, `Range`
/// - Named class: `Class(SymbolId)`
/// - `Any` — unknown or too complex to infer
///
/// The inference is *speculative* — it's used to guide optimization (e.g.,
/// emitting unboxed f64 arithmetic), not to enforce correctness. Type errors
/// at this stage are warnings, not hard errors.
///
/// Uses a 3-pass approach for class-heavy code:
///   Pass 1: Collect field types from assignments (_field = value).
///   Pass 2: Re-infer with field types → record method return types.
///   Pass 3: Re-infer with field + method types → final TypeEnv.
///
/// The key win: tight numeric loops AND class methods with numeric fields
/// get unboxed f64 ops in codegen.
use std::collections::HashMap;

use crate::ast::*;
use crate::intern::SymbolId;

// ---------------------------------------------------------------------------
// Type lattice
// ---------------------------------------------------------------------------

/// An inferred type for an expression or variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InferredType {
    Num,
    Bool,
    Null,
    String,
    List,
    Map,
    Range,
    Fn,
    /// A known class type.
    Class(SymbolId),
    /// Unknown or too complex to infer.
    Any,
}

impl InferredType {
    /// Join two types (least upper bound in the lattice).
    /// Used when a variable can hold either type (e.g., if/else branches).
    pub fn join(&self, other: &InferredType) -> InferredType {
        if self == other {
            self.clone()
        } else {
            InferredType::Any
        }
    }

    /// Is this a numeric type?
    pub fn is_num(&self) -> bool {
        matches!(self, InferredType::Num)
    }

    /// Is this type fully known (not Any)?
    pub fn is_known(&self) -> bool {
        !matches!(self, InferredType::Any)
    }
}

// ---------------------------------------------------------------------------
// Type environment
// ---------------------------------------------------------------------------

/// Maps variable locations (span start) to their inferred types.
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Variable types (keyed by declaration span start).
    vars: HashMap<usize, InferredType>,
    /// Expression types (keyed by expression span start).
    exprs: HashMap<usize, InferredType>,
    /// Field types per class: class_name → field_name → InferredType.
    field_types: HashMap<SymbolId, HashMap<SymbolId, InferredType>>,
    /// Method return types: (class_name, method_name) → InferredType.
    method_return_types: HashMap<(SymbolId, SymbolId), InferredType>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            exprs: HashMap::new(),
            field_types: HashMap::new(),
            method_return_types: HashMap::new(),
        }
    }

    pub fn set_var_type(&mut self, span_start: usize, ty: InferredType) {
        self.vars.insert(span_start, ty);
    }

    pub fn get_var_type(&self, span_start: usize) -> &InferredType {
        self.vars.get(&span_start).unwrap_or(&InferredType::Any)
    }

    pub fn set_expr_type(&mut self, span_start: usize, ty: InferredType) {
        self.exprs.insert(span_start, ty);
    }

    pub fn get_expr_type(&self, span_start: usize) -> &InferredType {
        self.exprs.get(&span_start).unwrap_or(&InferredType::Any)
    }

    /// Widen a variable type (join with new type on reassignment).
    pub fn widen_var(&mut self, span_start: usize, new_ty: &InferredType) {
        let current = self.get_var_type(span_start).clone();
        let widened = current.join(new_ty);
        self.vars.insert(span_start, widened);
    }

    /// Record a field type for a class (joins with existing type).
    pub fn record_field_type(&mut self, class: SymbolId, field: SymbolId, ty: &InferredType) {
        let entry = self.field_types.entry(class).or_default().entry(field);
        let current = entry.or_insert_with(|| ty.clone());
        if current != ty {
            *current = current.join(ty);
        }
    }

    /// Get the inferred type for a class field.
    pub fn get_field_type(&self, class: SymbolId, field: SymbolId) -> &InferredType {
        self.field_types
            .get(&class)
            .and_then(|f| f.get(&field))
            .unwrap_or(&InferredType::Any)
    }

    /// Record a method return type (joins with existing).
    pub fn record_method_return_type(
        &mut self,
        class: SymbolId,
        method: SymbolId,
        ty: &InferredType,
    ) {
        let entry = self.method_return_types.entry((class, method));
        let current = entry.or_insert_with(|| ty.clone());
        if current != ty {
            *current = current.join(ty);
        }
    }

    /// Get the inferred return type for a class method.
    pub fn get_method_return_type(&self, class: SymbolId, method: SymbolId) -> &InferredType {
        self.method_return_types
            .get(&(class, method))
            .unwrap_or(&InferredType::Any)
    }

    /// Clear per-expression and per-variable types (keep field + method types).
    fn clear_ephemeral(&mut self) {
        self.vars.clear();
        self.exprs.clear();
    }

    pub fn var_types(&self) -> &HashMap<usize, InferredType> {
        &self.vars
    }

    pub fn expr_types(&self) -> &HashMap<usize, InferredType> {
        &self.exprs
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Type inferrer
// ---------------------------------------------------------------------------

pub struct TypeInferrer {
    env: TypeEnv,
    /// Current class being inferred (for field + method return tracking).
    current_class: Option<SymbolId>,
    /// Current method being inferred (for return type recording).
    current_method: Option<SymbolId>,
}

impl Default for TypeInferrer {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeInferrer {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            current_class: None,
            current_method: None,
        }
    }

    /// Create an inferrer that carries forward field/method type knowledge.
    fn with_env(env: TypeEnv) -> Self {
        Self {
            env,
            current_class: None,
            current_method: None,
        }
    }

    /// Run type inference on a module, returning the type environment.
    pub fn infer(mut self, module: &Module) -> TypeEnv {
        self.infer_module(module);
        self.env
    }

    fn infer_module(&mut self, module: &Module) {
        for stmt in module {
            self.infer_stmt(stmt);
        }
    }

    // -- Statements ---------------------------------------------------------

    fn infer_stmt(&mut self, stmt: &Spanned<Stmt>) {
        match &stmt.0 {
            Stmt::Expr(expr) => {
                self.infer_expr(expr);
            }

            Stmt::Var {
                name,
                initializer,
                attributes: _,
            } => {
                let ty = if let Some(init) = initializer {
                    self.infer_expr(init)
                } else {
                    InferredType::Null
                };
                self.env.set_var_type(name.1.start, ty);
            }

            Stmt::Class(decl) => {
                let prev_class = self.current_class;
                self.current_class = Some(decl.name.0);
                for method in &decl.methods {
                    // Extract method name for return type tracking.
                    let method_name = match &method.0.signature {
                        MethodSig::Named { name, .. } => Some(*name),
                        MethodSig::Getter(name) => Some(*name),
                        MethodSig::Construct { name, .. } => Some(*name),
                        MethodSig::Setter { name, .. } => Some(*name),
                        _ => None,
                    };
                    let prev_method = self.current_method;
                    self.current_method = method_name;
                    if let Some(ref body) = method.0.body {
                        self.infer_stmt(body);
                    }
                    self.current_method = prev_method;
                }
                self.current_class = prev_class;
            }

            Stmt::Import { .. } => {}

            Stmt::Block(stmts) => {
                for s in stmts {
                    self.infer_stmt(s);
                }
            }

            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.infer_expr(condition);
                self.infer_stmt(then_branch);
                if let Some(else_b) = else_branch {
                    self.infer_stmt(else_b);
                }
            }

            Stmt::While { condition, body } => {
                self.infer_expr(condition);
                self.infer_stmt(body);
            }

            Stmt::For {
                variable,
                iterator,
                body,
            } => {
                let iter_ty = self.infer_expr(iterator);
                // For Range iterators, the loop variable is Num.
                let var_ty = match iter_ty {
                    InferredType::Range => InferredType::Num,
                    _ => InferredType::Any,
                };
                self.env.set_var_type(variable.1.start, var_ty);
                self.infer_stmt(body);
            }

            Stmt::Break | Stmt::Continue => {}

            Stmt::Return(expr) => {
                let ret_ty = if let Some(e) = expr {
                    self.infer_expr(e)
                } else {
                    InferredType::Null
                };
                // Record return type for the current method.
                if let (Some(class), Some(method)) = (self.current_class, self.current_method) {
                    self.env.record_method_return_type(class, method, &ret_ty);
                }
            }
        }
    }

    // -- Expressions --------------------------------------------------------

    fn infer_expr(&mut self, expr: &Spanned<Expr>) -> InferredType {
        let ty = self.infer_expr_inner(expr);
        self.env.set_expr_type(expr.1.start, ty.clone());
        ty
    }

    fn infer_expr_inner(&mut self, expr: &Spanned<Expr>) -> InferredType {
        match &expr.0 {
            Expr::Num(_) => InferredType::Num,
            Expr::Str(_) => InferredType::String,
            Expr::Interpolation(parts) => {
                for part in parts {
                    self.infer_expr(part);
                }
                InferredType::String
            }
            Expr::Bool(_) => InferredType::Bool,
            Expr::Null => InferredType::Null,
            Expr::This => {
                if let Some(class) = self.current_class {
                    InferredType::Class(class)
                } else {
                    InferredType::Any
                }
            }

            Expr::Ident(_) => {
                // Look up from var types if we tracked it.
                self.env.get_expr_type(expr.1.start).clone()
            }

            Expr::Field(field_name) => {
                // Return tracked field type if available (from previous pass).
                if let Some(class) = self.current_class {
                    self.env.get_field_type(class, *field_name).clone()
                } else {
                    InferredType::Any
                }
            }
            Expr::StaticField(_) => InferredType::Any,

            Expr::UnaryOp { op, operand } => {
                let operand_ty = self.infer_expr(operand);
                match op {
                    UnaryOp::Neg | UnaryOp::BNot => {
                        if operand_ty.is_num() {
                            InferredType::Num
                        } else {
                            InferredType::Any
                        }
                    }
                    UnaryOp::Not => InferredType::Bool,
                }
            }

            Expr::BinaryOp { op, left, right } => {
                let left_ty = self.infer_expr(left);
                let right_ty = self.infer_expr(right);
                self.infer_binary_op(*op, &left_ty, &right_ty)
            }

            Expr::LogicalOp { left, right, .. } => {
                let left_ty = self.infer_expr(left);
                let right_ty = self.infer_expr(right);
                // Logical ops return one of their operands.
                left_ty.join(&right_ty)
            }

            Expr::Is { value, type_name } => {
                self.infer_expr(value);
                self.infer_expr(type_name);
                InferredType::Bool
            }

            Expr::Assign { target, value } => {
                let val_ty = self.infer_expr(value);
                // Track field types when assigning to _field inside a class.
                if let Expr::Field(field_name) = &target.0 {
                    if let Some(class) = self.current_class {
                        self.env.record_field_type(class, *field_name, &val_ty);
                    }
                }
                self.infer_expr(target);
                val_ty
            }

            Expr::CompoundAssign { op, target, value } => {
                let target_ty = self.infer_expr(target);
                let val_ty = self.infer_expr(value);
                let result_ty = self.infer_binary_op(*op, &target_ty, &val_ty);
                // Track compound assignment to fields.
                if let Expr::Field(field_name) = &target.0 {
                    if let Some(class) = self.current_class {
                        self.env.record_field_type(class, *field_name, &result_ty);
                    }
                }
                result_ty
            }

            Expr::Call {
                receiver,
                method,
                args,
                block_arg,
                ..
            } => {
                let recv_ty = if let Some(recv) = receiver {
                    self.infer_expr(recv)
                } else {
                    // Bare call: implicit this.
                    if let Some(class) = self.current_class {
                        InferredType::Class(class)
                    } else {
                        InferredType::Any
                    }
                };
                for arg in args {
                    self.infer_expr(arg);
                }
                if let Some(block) = block_arg {
                    self.infer_expr(block);
                }
                // Look up method return type if receiver class is known.
                if let InferredType::Class(class_sym) = &recv_ty {
                    let ret_ty = self
                        .env
                        .get_method_return_type(*class_sym, method.0)
                        .clone();
                    if ret_ty.is_known() {
                        return ret_ty;
                    }
                }
                InferredType::Any
            }

            Expr::SuperCall { args, .. } => {
                for arg in args {
                    self.infer_expr(arg);
                }
                InferredType::Any
            }

            Expr::Subscript { receiver, args } => {
                self.infer_expr(receiver);
                for arg in args {
                    self.infer_expr(arg);
                }
                InferredType::Any
            }

            Expr::SubscriptSet {
                receiver,
                index_args,
                value,
            } => {
                self.infer_expr(receiver);
                for arg in index_args {
                    self.infer_expr(arg);
                }
                self.infer_expr(value)
            }

            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.infer_expr(condition);
                let then_ty = self.infer_expr(then_expr);
                let else_ty = self.infer_expr(else_expr);
                then_ty.join(&else_ty)
            }

            Expr::ListLiteral(elements) => {
                for elem in elements {
                    self.infer_expr(elem);
                }
                InferredType::List
            }

            Expr::MapLiteral(entries) => {
                for (key, val) in entries {
                    self.infer_expr(key);
                    self.infer_expr(val);
                }
                InferredType::Map
            }

            Expr::Range { from, to, .. } => {
                self.infer_expr(from);
                self.infer_expr(to);
                InferredType::Range
            }

            Expr::Closure { params: _, body } => {
                self.infer_stmt(body);
                InferredType::Fn
            }
        }
    }

    // -- Binary op type rules -----------------------------------------------

    fn infer_binary_op(
        &self,
        op: BinaryOp,
        left: &InferredType,
        right: &InferredType,
    ) -> InferredType {
        match op {
            // Arithmetic ops: Num × Num → Num
            BinaryOp::Add => {
                if left.is_num() && right.is_num() {
                    InferredType::Num
                } else if *left == InferredType::String || *right == InferredType::String {
                    InferredType::String
                } else {
                    InferredType::Any
                }
            }
            BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                if left.is_num() && right.is_num() {
                    InferredType::Num
                } else {
                    InferredType::Any
                }
            }

            // Bitwise ops: Num × Num → Num
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => {
                if left.is_num() && right.is_num() {
                    InferredType::Num
                } else {
                    InferredType::Any
                }
            }

            // Comparison ops → Bool
            BinaryOp::Lt
            | BinaryOp::Gt
            | BinaryOp::LtEq
            | BinaryOp::GtEq
            | BinaryOp::Eq
            | BinaryOp::NotEq => InferredType::Bool,
        }
    }
}

/// 3-pass type inference for a parsed module.
///
/// Pass 1: Walk class bodies to collect field assignment types.
/// Pass 2: Re-infer with field type knowledge → record method return types.
/// Pass 3: Re-infer with field + method return types → final TypeEnv.
pub fn infer_types(module: &Module) -> TypeEnv {
    // Pass 1: Collect field types + initial method return types.
    let inferrer1 = TypeInferrer::new();
    let mut env = inferrer1.infer(module);

    // Pass 2: Re-infer with field types → better method return types.
    env.clear_ephemeral();
    let mut inferrer2 = TypeInferrer::with_env(env);
    inferrer2.infer_module(module);
    env = inferrer2.env;

    // Pass 3: Re-infer with field + method types → final expression types.
    env.clear_ephemeral();
    let mut inferrer3 = TypeInferrer::with_env(env);
    inferrer3.infer_module(module);
    inferrer3.env
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    fn infer_source(source: &str) -> TypeEnv {
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "parse errors: {:?}",
            result.errors
        );
        infer_types(&result.module)
    }

    fn first_expr_type(source: &str) -> InferredType {
        let env = infer_source(source);
        // The first expression starts at offset 0
        env.get_expr_type(0).clone()
    }

    // -- Literal types ------------------------------------------------------

    #[test]
    fn test_num_literal() {
        assert_eq!(first_expr_type("42"), InferredType::Num);
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(first_expr_type("\"hello\""), InferredType::String);
    }

    #[test]
    fn test_bool_literal() {
        assert_eq!(first_expr_type("true"), InferredType::Bool);
        assert_eq!(first_expr_type("false"), InferredType::Bool);
    }

    #[test]
    fn test_null_literal() {
        assert_eq!(first_expr_type("null"), InferredType::Null);
    }

    // -- Arithmetic ---------------------------------------------------------

    #[test]
    fn test_num_add() {
        assert_eq!(first_expr_type("1 + 2"), InferredType::Num);
    }

    #[test]
    fn test_num_sub() {
        assert_eq!(first_expr_type("5 - 3"), InferredType::Num);
    }

    #[test]
    fn test_num_mul() {
        assert_eq!(first_expr_type("2 * 3"), InferredType::Num);
    }

    #[test]
    fn test_num_div() {
        assert_eq!(first_expr_type("10 / 2"), InferredType::Num);
    }

    #[test]
    fn test_nested_arithmetic() {
        assert_eq!(first_expr_type("1 + 2 * 3"), InferredType::Num);
    }

    // -- Comparison ---------------------------------------------------------

    #[test]
    fn test_comparison() {
        assert_eq!(first_expr_type("1 < 2"), InferredType::Bool);
        assert_eq!(first_expr_type("1 == 2"), InferredType::Bool);
        assert_eq!(first_expr_type("1 != 2"), InferredType::Bool);
    }

    // -- Unary ops ----------------------------------------------------------

    #[test]
    fn test_negation() {
        assert_eq!(first_expr_type("-42"), InferredType::Num);
    }

    #[test]
    fn test_not() {
        assert_eq!(first_expr_type("!true"), InferredType::Bool);
    }

    // -- Is expression ------------------------------------------------------

    #[test]
    fn test_is_type() {
        let env = infer_source("var x = 1\nx is Num");
        // The `is` expression should be Bool.
        // It starts after "var x = 1\n" (10 bytes)
        let is_type = env.get_expr_type(10);
        assert_eq!(*is_type, InferredType::Bool);
    }

    // -- Collections --------------------------------------------------------

    #[test]
    fn test_list_literal() {
        assert_eq!(first_expr_type("[1, 2, 3]"), InferredType::List);
    }

    #[test]
    fn test_map_literal() {
        let env = infer_source("{\"a\": 1}");
        let ty = env.get_expr_type(0);
        assert_eq!(*ty, InferredType::Map);
    }

    #[test]
    fn test_range() {
        assert_eq!(first_expr_type("1..10"), InferredType::Range);
    }

    // -- Closure ------------------------------------------------------------

    #[test]
    fn test_closure_type() {
        let env = infer_source("Fn.new { 1 + 2 }");
        // The closure block arg is a Fn type — but the Call wrapping it is Any.
        let ty = env.get_expr_type(0);
        assert_eq!(*ty, InferredType::Any);
    }

    // -- Conditional --------------------------------------------------------

    #[test]
    fn test_conditional_same_type() {
        assert_eq!(first_expr_type("true ? 1 : 2"), InferredType::Num);
    }

    #[test]
    fn test_conditional_different_types() {
        assert_eq!(first_expr_type("true ? 1 : \"a\""), InferredType::Any);
    }

    // -- Var type tracking --------------------------------------------------

    #[test]
    fn test_var_type_inference() {
        let env = infer_source("var x = 42");
        // The var declaration records the type at the name's span start.
        // "var x" — 'x' starts at offset 4
        let ty = env.get_var_type(4);
        assert_eq!(*ty, InferredType::Num);
    }

    #[test]
    fn test_var_no_init() {
        let env = infer_source("var x");
        let ty = env.get_var_type(4);
        assert_eq!(*ty, InferredType::Null);
    }

    // -- For loop variable --------------------------------------------------

    #[test]
    fn test_for_range_variable() {
        let env = infer_source("for (i in 1..10) i");
        // 'i' in "for (i" starts at offset 5
        let ty = env.get_var_type(5);
        assert_eq!(*ty, InferredType::Num);
    }

    // -- Type join ----------------------------------------------------------

    #[test]
    fn test_type_join_same() {
        assert_eq!(
            InferredType::Num.join(&InferredType::Num),
            InferredType::Num
        );
    }

    #[test]
    fn test_type_join_different() {
        assert_eq!(
            InferredType::Num.join(&InferredType::String),
            InferredType::Any
        );
    }

    #[test]
    fn test_type_join_any() {
        assert_eq!(
            InferredType::Num.join(&InferredType::Any),
            InferredType::Any
        );
    }

    // -- Interpolation type -------------------------------------------------

    #[test]
    fn test_interpolation_type() {
        let env = infer_source("var x = 1\n\"value: %(x)\"");
        // The interpolation starts at offset 10
        let ty = env.get_expr_type(10);
        assert_eq!(*ty, InferredType::String);
    }
}
