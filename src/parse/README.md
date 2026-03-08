# Wren Frontend (Lexer + Parser)

This module implements the lexing and parsing frontend for the Wren language.
Source text goes in; a `Vec<Spanned<Stmt>>` AST (aliased as `Module`) comes out.

## Lexer

Logos-based tokenizer for Wren source code.

The `Token` enum (~60 variants) covers all Wren syntax:

- **Punctuation**: `( ) [ ] { } , . .. ... : # ?`
- **Operators**: single-char (`+ - * / % < > = ! ~ & | ^`), multi-char (`== != <= >= << >> && ||`), compound assignment (`+= -= *= /= %= &= |= ^= <<= >>=`)
- **Keywords**: `as break class construct continue else false for foreign if import in is null return static super this true var while`
- **Literals**: numbers (integer, float, hex `0x`, scientific `1e5`), strings (simple, raw `"""..."""`, interpolated `%(expr)`)
- **Identifiers**: plain (`foo`), instance fields (`_name`), static fields (`__name`)
- **Newlines**: emitted as tokens (significant in Wren grammar)

Logos handles simple token recognition via `#[token]`/`#[regex]` attributes. Custom post-processing on top of logos handles:

- **String literals** with escape sequences (`\n \t \r \\ \" \0 \a \b \e \f \v \% \xHH \uHHHH \UHHHHHHHH`)
- **String interpolation** (`%(expr)`), including nested strings inside interpolation, emitting `InterpolationStart`/`InterpolationMid`/`InterpolationEnd` tokens
- **Raw strings** (`"""..."""`) with no escape processing
- **Nested block comments** (`/* ... /* ... */ ... */`) with depth tracking
- **Line comments** (`//`) and shebang lines (`#!`)

Malformed input (unterminated strings, unterminated block comments, unknown escapes, unexpected characters) produces `Token::Error` and a `Diagnostic`.

### Key types

- `Token` -- enum of all token kinds
- `Lexeme` -- a token paired with its `Span` (byte range) and extracted `text: String`
- `fn lex(source: &str) -> (Vec<Lexeme>, Vec<Diagnostic>)` -- public entry point

## Parser

Recursive-descent parser that consumes a `Vec<Lexeme>` and produces a typed AST.

### Expression precedence (lowest to highest)

1. Assignment (`=`, compound `+= -= ...`)
2. Conditional (ternary `? :`)
3. Logical OR (`||`)
4. Logical AND (`&&`)
5. Equality (`== !=`)
6. Type check (`is`)
7. Comparison (`< > <= >=`)
8. Bitwise OR (`|`)
9. Bitwise XOR (`^`)
10. Bitwise AND (`&`)
11. Shift (`<< >>`)
12. Range (`.. ...`)
13. Term (`+ -`)
14. Factor (`* / %`)
15. Unary (`- ! ~`)
16. Postfix (`.method`, `[subscript]`, call `(args)`)
17. Primary (literals, identifiers, fields, `this`, `super`, grouping, list/map/closure)

### Declarations and statements

- `var` declarations with optional initializer
- `import` with optional `for` bindings and `as` aliases
- `class` declarations: `foreign` modifier, `is` superclass, method bodies
- Method signatures: `construct`, named, getter, setter (`name=(val)`), subscript (`[i]`/`[i]=(val)`), operator overloads
- Control flow: `if/else`, `while`, `for...in`, `return`, `break`, `continue`
- Block statements `{ ... }`

### Expression forms

- Literals: numbers, strings, raw strings, booleans, `null`, list `[...]`, map `{k: v}`
- String interpolation `"text %(expr) text"`
- Closures `{ |params| body }` (single-expression or multi-statement)
- Method calls with optional block arguments: `obj.method(args) { block }`
- Subscript get/set: `obj[i]`, `obj[i] = val`
- `super` calls: `super`, `super.method(args)`, `super(args)`
- Field access: `_field`, `__staticField`

Map vs. closure disambiguation uses speculative parsing: peek past `{`, try to parse an expression, and check for `:`.

Error recovery: on parse failure, the parser skips to the next statement boundary and continues.

### Key types

- `Parser` -- holds token stream, cursor position, `Interner`, and error accumulator
- `ParseResult` -- contains `module: Module`, `errors: Vec<Diagnostic>`, `interner: Interner`
- `fn parse(source: &str) -> ParseResult` -- convenience function that lexes then parses in one call

