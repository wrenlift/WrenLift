# Diagnostics

Error reporting module with source-context rendering for the Wren compiler.

## Overview

This module provides structured diagnostics that pair error messages with labeled source spans, rendered as annotated terminal output via [ariadne](https://docs.rs/ariadne). It is used by both the lexer and parser to report errors, and by semantic analysis to report resolution failures.

## Core Types

### `Severity`

```
Error | Warning | Info
```

Controls report kind and default label color (Red, Yellow, Blue respectively). Only `Error`-severity diagnostics increment the emitter's error counter.

### `Diagnostic`

A single diagnostic consisting of:

- **severity** -- one of `Error`, `Warning`, or `Info`
- **message** -- top-level description of the problem
- **labels** (`Vec<DiagLabel>`) -- labeled byte-offset spans (`Range<usize>`) into source text, each with a message and optional `ariadne::Color` override
- **notes** (`Vec<String>`) -- additional context appended after the rendered source snippet

Constructed via builder methods:

```rust
Diagnostic::error("type mismatch")
    .with_label(8..12, "this is Bool")
    .with_colored_label(15..16, "expected Bool", Color::Cyan)
    .with_note("Bool and Num are not compatible")
```

`render(&self, source, writer)` produces ariadne output to any `io::Write`. `eprint` is a convenience that writes to stderr.

### `DiagnosticEmitter`

Accumulator for diagnostics during a compilation pass.

- `emit(diag)` -- records a diagnostic; increments the atomic error counter for `Error` severity
- `has_errors()` / `error_count()` -- query error state (uses `AtomicUsize` with relaxed ordering)
- `render_all(source, writer)` / `eprint_all(source)` -- render every accumulated diagnostic

## Parser and Lexer Integration

The parser and lexer do not use `DiagnosticEmitter` directly. Instead, they collect `Vec<Diagnostic>` internally:

- **Lexer** (`parse::lexer::lex`) -- pushes `Diagnostic::error(...)` for unterminated strings, block comments, invalid escapes, unexpected characters, and unterminated interpolations.
- **Parser** (`parse::parser::Parser`) -- stores errors in a `Vec<Diagnostic>` field and pushes labeled error diagnostics on syntax failures (missing identifiers, unexpected tokens, etc.). The `parse()` method returns a `ParseResult` containing the AST module and the accumulated error vector.

The top-level `parse::parser::parse(source)` function combines lexer and parser errors into a single `ParseResult.errors` list.

## Semantic Analysis

The `sema::resolve` module imports `Diagnostic` to report name-resolution errors with source spans.
