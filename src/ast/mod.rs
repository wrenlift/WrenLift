use std::ops::Range;

use crate::intern::SymbolId;

/// Byte offset range into source text.
pub type Span = Range<usize>;

/// A value paired with its source location.
pub type Spanned<T> = (T, Span);

// ---------------------------------------------------------------------------
// Top-level program
// ---------------------------------------------------------------------------

/// A module is a sequence of statements.
pub type Module = Vec<Spanned<Stmt>>;

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Expression used as a statement.
    Expr(Spanned<Expr>),

    /// `var name = initializer`
    Var {
        name: Spanned<SymbolId>,
        initializer: Option<Spanned<Expr>>,
    },

    /// Class declaration.
    Class(ClassDecl),

    /// `import "module"` or `import "module" for Name, Name as Alias`
    Import {
        /// Module path is kept as a String (not interned — it's a path, not
        /// an identifier, and is only used once during module resolution).
        module: Spanned<String>,
        names: Vec<ImportName>,
    },

    /// `{ stmts }`
    Block(Vec<Spanned<Stmt>>),

    /// `if (condition) then_branch else else_branch`
    If {
        condition: Spanned<Expr>,
        then_branch: Box<Spanned<Stmt>>,
        else_branch: Option<Box<Spanned<Stmt>>>,
    },

    /// `while (condition) body`
    While {
        condition: Spanned<Expr>,
        body: Box<Spanned<Stmt>>,
    },

    /// `for (variable in iterator) body`
    For {
        variable: Spanned<SymbolId>,
        iterator: Spanned<Expr>,
        body: Box<Spanned<Stmt>>,
    },

    /// `break`
    Break,

    /// `continue`
    Continue,

    /// `return` or `return expr`
    Return(Option<Spanned<Expr>>),
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ImportName {
    pub name: Spanned<SymbolId>,
    pub alias: Option<Spanned<SymbolId>>,
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ClassDecl {
    pub name: Spanned<SymbolId>,
    pub superclass: Option<Spanned<SymbolId>>,
    pub is_foreign: bool,
    pub methods: Vec<Spanned<Method>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    pub is_static: bool,
    pub is_foreign: bool,
    pub signature: MethodSig,
    /// `None` for foreign methods.
    pub body: Option<Spanned<Stmt>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MethodSig {
    /// `name(params)` — regular method.
    Named {
        name: SymbolId,
        params: Vec<Spanned<SymbolId>>,
    },

    /// `name` — getter (no parens).
    Getter(SymbolId),

    /// `name=(param)` — setter.
    Setter {
        name: SymbolId,
        param: Spanned<SymbolId>,
    },

    /// `[params]` — subscript getter.
    Subscript {
        params: Vec<Spanned<SymbolId>>,
    },

    /// `[params]=(value)` — subscript setter.
    SubscriptSetter {
        params: Vec<Spanned<SymbolId>>,
        value: Spanned<SymbolId>,
    },

    /// Operator overload: `+(other)`, `-`, `!`, etc.
    Operator {
        op: Op,
        params: Vec<Spanned<SymbolId>>,
    },

    /// `construct name(params)`
    Construct {
        name: SymbolId,
        params: Vec<Spanned<SymbolId>>,
    },
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,   // -
    Not,   // !
    BNot,  // ~
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Eq,
    NotEq,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
}

/// Operators that can appear in method signatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    // Infix
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Lt,
    Gt,
    LtEq,
    GtEq,
    EqEq,
    BangEq,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    DotDot,
    DotDotDot,
    // Prefix (unary)
    Neg,
    Bang,
    Tilde,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Number literal.
    Num(f64),

    /// String literal (kept as String since string *values* can be arbitrary
    /// content — they are interned at the runtime level, not the AST level).
    Str(String),

    /// String interpolation: alternating string parts and embedded expressions.
    /// e.g. `"hello %(name)!"` → `[Str("hello "), Ident("name"), Str("!")]`
    Interpolation(Vec<Spanned<Expr>>),

    /// `true` or `false`.
    Bool(bool),

    /// `null`
    Null,

    /// `this`
    This,

    /// Variable or class name reference.
    Ident(SymbolId),

    /// Instance field: `_name`
    Field(SymbolId),

    /// Static field: `__name`
    StaticField(SymbolId),

    /// Unary operation: `-x`, `!x`, `~x`
    UnaryOp {
        op: UnaryOp,
        operand: Box<Spanned<Expr>>,
    },

    /// Binary operation: `a + b`, `a < b`, etc.
    BinaryOp {
        op: BinaryOp,
        left: Box<Spanned<Expr>>,
        right: Box<Spanned<Expr>>,
    },

    /// Short-circuit logical: `a && b`, `a || b`
    LogicalOp {
        op: LogicalOp,
        left: Box<Spanned<Expr>>,
        right: Box<Spanned<Expr>>,
    },

    /// Type test: `value is ClassName`
    Is {
        value: Box<Spanned<Expr>>,
        type_name: Box<Spanned<Expr>>,
    },

    /// Assignment: `target = value`
    Assign {
        target: Box<Spanned<Expr>>,
        value: Box<Spanned<Expr>>,
    },

    /// Compound assignment: `target += value`, `target -= value`, etc.
    CompoundAssign {
        op: BinaryOp,
        target: Box<Spanned<Expr>>,
        value: Box<Spanned<Expr>>,
    },

    /// Method call: `receiver.method(args)`
    /// For bare function-style calls, receiver is an implicit `this`.
    Call {
        receiver: Option<Box<Spanned<Expr>>>,
        method: Spanned<SymbolId>,
        args: Vec<Spanned<Expr>>,
        block_arg: Option<Box<Spanned<Expr>>>,
        /// Whether call was made with explicit parens: `foo.bar()` vs `foo.bar`
        has_parens: bool,
    },

    /// `super.method(args)` or `super(args)`
    SuperCall {
        method: Option<Spanned<SymbolId>>,
        args: Vec<Spanned<Expr>>,
        /// Whether call was made with explicit parens: `super.bar()` vs `super.bar`
        has_parens: bool,
    },

    /// Subscript getter: `receiver[args]`
    Subscript {
        receiver: Box<Spanned<Expr>>,
        args: Vec<Spanned<Expr>>,
    },

    /// Subscript setter: `receiver[index_args] = value`
    SubscriptSet {
        receiver: Box<Spanned<Expr>>,
        index_args: Vec<Spanned<Expr>>,
        value: Box<Spanned<Expr>>,
    },

    /// Ternary conditional: `condition ? then_expr : else_expr`
    Conditional {
        condition: Box<Spanned<Expr>>,
        then_expr: Box<Spanned<Expr>>,
        else_expr: Box<Spanned<Expr>>,
    },

    /// List literal: `[a, b, c]`
    ListLiteral(Vec<Spanned<Expr>>),

    /// Map literal: `{key: value, key: value}`
    MapLiteral(Vec<(Spanned<Expr>, Spanned<Expr>)>),

    /// Range: `from..to` (inclusive) or `from...to` (exclusive)
    Range {
        from: Box<Spanned<Expr>>,
        to: Box<Spanned<Expr>>,
        inclusive: bool,
    },

    /// Closure / block argument: `{ |params| body }` or `Fn.new { body }`
    Closure {
        params: Vec<Spanned<SymbolId>>,
        body: Box<Spanned<Stmt>>,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;

    fn span(start: usize, end: usize) -> Span {
        start..end
    }

    #[test]
    fn test_ast_construction_exprs() {
        let mut i = Interner::new();
        let foo = i.intern("foo");
        let name = i.intern("name");
        let x = i.intern("x");
        let count = i.intern("count");

        let _ = Expr::Num(42.0);
        let _ = Expr::Str("hello".into());
        let _ = Expr::Bool(true);
        let _ = Expr::Null;
        let _ = Expr::This;
        let _ = Expr::Ident(foo);
        let _ = Expr::Field(x);
        let _ = Expr::StaticField(count);
        let _ = Expr::ListLiteral(vec![]);
        let _ = Expr::MapLiteral(vec![]);
        let _ = Expr::Interpolation(vec![
            (Expr::Str("hello ".into()), span(0, 6)),
            (Expr::Ident(name), span(8, 12)),
        ]);
    }

    #[test]
    fn test_ast_construction_stmts() {
        let mut i = Interner::new();
        let x = i.intern("x");
        let sqrt = i.intern("sqrt");

        let _ = Stmt::Break;
        let _ = Stmt::Continue;
        let _ = Stmt::Return(None);
        let _ = Stmt::Return(Some((Expr::Num(1.0), span(7, 8))));
        let _ = Stmt::Var {
            name: (x, span(4, 5)),
            initializer: Some((Expr::Num(42.0), span(8, 10))),
        };
        let _ = Stmt::Block(vec![]);
        let _ = Stmt::Import {
            module: ("math".into(), span(8, 14)),
            names: vec![ImportName {
                name: (sqrt, span(19, 23)),
                alias: None,
            }],
        };
    }

    #[test]
    fn test_ast_spans_preserved() {
        let mut i = Interner::new();
        let x = i.intern("x");

        let expr: Spanned<Expr> = (Expr::Num(3.14), span(0, 4));
        assert_eq!(expr.1, 0..4);

        let stmt: Spanned<Stmt> = (
            Stmt::Expr((Expr::Ident(x), span(0, 1))),
            span(0, 1),
        );
        assert_eq!(stmt.1, 0..1);
    }

    #[test]
    fn test_ast_clone_eq() {
        let a = Expr::BinaryOp {
            op: BinaryOp::Add,
            left: Box::new((Expr::Num(1.0), span(0, 1))),
            right: Box::new((Expr::Num(2.0), span(4, 5))),
        };
        let b = a.clone();
        assert_eq!(a, b);

        let c = Expr::Num(99.0);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ast_nested_deep() {
        let inner = Expr::BinaryOp {
            op: BinaryOp::Add,
            left: Box::new((Expr::Num(1.0), span(1, 2))),
            right: Box::new((Expr::Num(2.0), span(5, 6))),
        };
        let outer = Expr::BinaryOp {
            op: BinaryOp::Mul,
            left: Box::new((inner, span(0, 7))),
            right: Box::new((Expr::Num(3.0), span(10, 11))),
        };
        match &outer {
            Expr::BinaryOp { op, .. } => assert_eq!(*op, BinaryOp::Mul),
            _ => panic!("expected BinaryOp"),
        }
    }

    #[test]
    fn test_ast_class_full() {
        let mut i = Interner::new();
        let animal = i.intern("Animal");
        let object = i.intern("Object");
        let new = i.intern("new");
        let name_sym = i.intern("name");
        let species = i.intern("species");
        let other = i.intern("other");
        let index = i.intern("index");

        let class = ClassDecl {
            name: (animal, span(6, 12)),
            superclass: Some((object, span(16, 22))),
            is_foreign: false,
            methods: vec![
                (
                    Method {
                        is_static: false,
                        is_foreign: false,
                        signature: MethodSig::Construct {
                            name: new,
                            params: vec![(name_sym, span(40, 44))],
                        },
                        body: Some((Stmt::Block(vec![]), span(46, 48))),
                    },
                    span(30, 48),
                ),
                (
                    Method {
                        is_static: false,
                        is_foreign: false,
                        signature: MethodSig::Getter(name_sym),
                        body: Some((
                            Stmt::Expr((Expr::Field(name_sym), span(55, 60))),
                            span(55, 60),
                        )),
                    },
                    span(50, 62),
                ),
                (
                    Method {
                        is_static: true,
                        is_foreign: false,
                        signature: MethodSig::Named {
                            name: species,
                            params: vec![],
                        },
                        body: Some((
                            Stmt::Return(Some((Expr::Str("Animal".into()), span(90, 98)))),
                            span(83, 98),
                        )),
                    },
                    span(70, 100),
                ),
                (
                    Method {
                        is_static: false,
                        is_foreign: false,
                        signature: MethodSig::Operator {
                            op: Op::EqEq,
                            params: vec![(other, span(110, 115))],
                        },
                        body: Some((Stmt::Block(vec![]), span(117, 119))),
                    },
                    span(105, 119),
                ),
                (
                    Method {
                        is_static: false,
                        is_foreign: false,
                        signature: MethodSig::Subscript {
                            params: vec![(index, span(125, 130))],
                        },
                        body: Some((Stmt::Block(vec![]), span(132, 134))),
                    },
                    span(120, 134),
                ),
            ],
        };

        assert_eq!(i.resolve(class.name.0), "Animal");
        assert_eq!(i.resolve(class.superclass.as_ref().unwrap().0), "Object");
        assert!(!class.is_foreign);
        assert_eq!(class.methods.len(), 5);
    }

    #[test]
    fn test_ast_closure() {
        let mut i = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");

        let closure = Expr::Closure {
            params: vec![
                (a, span(3, 4)),
                (b, span(6, 7)),
            ],
            body: Box::new((
                Stmt::Expr((
                    Expr::BinaryOp {
                        op: BinaryOp::Add,
                        left: Box::new((Expr::Ident(a), span(10, 11))),
                        right: Box::new((Expr::Ident(b), span(14, 15))),
                    },
                    span(10, 15),
                )),
                span(10, 15),
            )),
        };
        match &closure {
            Expr::Closure { params, .. } => assert_eq!(params.len(), 2),
            _ => panic!("expected Closure"),
        }
    }

    #[test]
    fn test_ast_call_with_block_arg() {
        let mut i = Interner::new();
        let list = i.intern("list");
        let where_sym = i.intern("where");
        let x = i.intern("x");

        let call = Expr::Call {
            receiver: Some(Box::new((Expr::Ident(list), span(0, 4)))),
            method: (where_sym, span(5, 10)),
            args: vec![],
            block_arg: Some(Box::new((
                Expr::Closure {
                    params: vec![(x, span(14, 15))],
                    body: Box::new((
                        Stmt::Expr((
                            Expr::BinaryOp {
                                op: BinaryOp::Gt,
                                left: Box::new((Expr::Ident(x), span(18, 19))),
                                right: Box::new((Expr::Num(3.0), span(22, 23))),
                            },
                            span(18, 23),
                        )),
                        span(18, 23),
                    )),
                },
                span(12, 25),
            ))),
            has_parens: false,
        };
        match &call {
            Expr::Call { block_arg, .. } => assert!(block_arg.is_some()),
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn test_ast_symbol_dedup() {
        let mut i = Interner::new();
        let a1 = i.intern("foo");
        let a2 = i.intern("foo");
        // Same identifier → same SymbolId → O(1) equality
        assert_eq!(a1, a2);
        // Can use directly in AST nodes
        let e1 = Expr::Ident(a1);
        let e2 = Expr::Ident(a2);
        assert_eq!(e1, e2);
    }
}
