# Abstract Syntax Tree

Pure data types representing all AST nodes for the Wren language. This module contains only type definitions -- no logic, no traversal, no lowering. Downstream passes (parser, compiler, etc.) consume and produce these types.

## Core Abstractions

- `Span` = `Range<usize>` -- byte offset range into source text.
- `Spanned<T>` = `(T, Span)` -- pairs any node with its source location.
- `Module` = `Vec<Spanned<Stmt>>` -- a parsed source file.

Identifiers are stored as `SymbolId` (from `crate::intern`), not raw strings. This gives O(1) equality checks and compact storage. String literals and module paths are the exception; they remain as `String` since they represent runtime values / paths rather than reusable identifiers.

## Stmt variants

| Variant | Description |
|---------|-------------|
| `Expr` | Expression used as a statement |
| `Var` | `var name = initializer` |
| `Class` | Class declaration (delegates to `ClassDecl`) |
| `Import` | `import "module"` with optional `for Name, Name as Alias` |
| `Block` | `{ stmts }` |
| `If` | `if (cond) then else` |
| `While` | `while (cond) body` |
| `For` | `for (variable in iterator) body` |
| `Break` | `break` |
| `Continue` | `continue` |
| `Return` | `return` with optional expression |

## Expr variants

| Variant | Description |
|---------|-------------|
| `Num`, `Str`, `Bool`, `Null`, `This` | Literals and `this` |
| `Interpolation` | String interpolation, alternating string parts and expressions |
| `Ident`, `Field`, `StaticField` | Name references (`x`, `_x`, `__x`) |
| `UnaryOp`, `BinaryOp`, `LogicalOp` | Operators; logical ops are separate for short-circuit semantics |
| `Is` | Type test (`value is ClassName`) |
| `Assign`, `CompoundAssign` | `target = value`, `target += value`, etc. |
| `Call` | Method call with optional receiver and optional trailing block arg |
| `SuperCall` | `super.method(args)` or bare `super(args)` |
| `Subscript`, `SubscriptSet` | `receiver[args]` getter and setter |
| `Conditional` | Ternary `cond ? a : b` |
| `ListLiteral`, `MapLiteral` | Collection literals |
| `Range` | `from..to` (inclusive) / `from...to` (exclusive) |
| `Closure` | `{ |params| body }` |

## ClassDecl structure

```
ClassDecl
  name:        Spanned<SymbolId>
  superclass:  Option<Spanned<SymbolId>>
  is_foreign:  bool
  methods:     Vec<Spanned<Method>>

Method
  is_static:   bool
  is_foreign:  bool
  signature:   MethodSig
  body:        Option<Spanned<Stmt>>   -- None for foreign methods

MethodSig = Named | Getter | Setter | Subscript | SubscriptSetter | Operator | Construct
```

`MethodSig` covers all Wren method shapes: regular named methods, getters, setters, subscript operators (`[]`, `[]=`), operator overloads (`+`, `==`, `!`, etc.), and constructors.

## Operator types

Three separate enums keep expression-level operators (`UnaryOp`, `BinaryOp`, `LogicalOp`) distinct from the method-signature operator set (`Op`). `Op` includes operators that can appear in Wren method signatures (e.g., `..`, `...`) and is used exclusively in `MethodSig::Operator`.

## Design decisions

- **Pure data, no behaviour.** The types derive `Debug`, `Clone`, `PartialEq` and nothing else. Pattern matching is the intended consumption mechanism.
- **Pervasive span tracking.** Every node that originates from source text is wrapped in `Spanned<T>`, enabling precise error reporting and source maps without a separate side-table.
- **Interned identifiers.** `SymbolId` keeps the AST compact and comparison-fast. Only string literal values and module paths use `String`.
- **Boxed recursion.** Recursive positions (`left`, `right`, `body`, `then_branch`, etc.) use `Box<Spanned<T>>` to keep enum sizes bounded.
