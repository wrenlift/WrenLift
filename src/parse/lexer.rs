use logos::Logos;

use crate::ast::Span;

/// Token type for the Wren language.
///
/// Logos handles simple token recognition. String interpolation and nested
/// block comments are handled by post-processing in [`lex`].
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r]+")]
pub enum Token {
    // -- Punctuation --------------------------------------------------------
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("#")]
    Hash,
    #[token("?")]
    Question,

    // -- Dots (order matters: longest first) ---------------------------------
    #[token("...")]
    DotDotDot,
    #[token("..")]
    DotDot,
    #[token(".")]
    Dot,

    // -- Compound assignment (before simple operators) -----------------------
    #[token("<<=")]
    LtLtEq,
    #[token(">>=")]
    GtGtEq,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("&=")]
    AmpEq,
    #[token("|=")]
    PipeEq,
    #[token("^=")]
    CaretEq,

    // -- Multi-char operators -----------------------------------------------
    #[token("<<")]
    LtLt,
    #[token(">>")]
    GtGt,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,

    // -- Single-char operators ----------------------------------------------
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("=")]
    Eq,
    #[token("!")]
    Bang,
    #[token("~")]
    Tilde,
    #[token("&")]
    Amp,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,

    // -- Keywords -----------------------------------------------------------
    #[token("as")]
    As,
    #[token("break")]
    Break,
    #[token("class")]
    Class,
    #[token("construct")]
    Construct,
    #[token("continue")]
    Continue,
    #[token("else")]
    Else,
    #[token("false")]
    False,
    #[token("for")]
    For,
    #[token("foreign")]
    Foreign,
    #[token("if")]
    If,
    #[token("import")]
    Import,
    #[token("in")]
    In,
    #[token("is")]
    Is,
    #[token("null")]
    Null,
    #[token("return")]
    Return,
    #[token("static")]
    Static,
    #[token("super")]
    Super,
    #[token("this")]
    This,
    #[token("true")]
    True,
    #[token("var")]
    Var,
    #[token("while")]
    While,

    // -- Literals -----------------------------------------------------------
    /// Number: integer, float, hex, scientific notation.
    #[regex(r"0[xX][0-9a-fA-F]+", priority = 3)]
    #[regex(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?", priority = 2)]
    Number,

    // -- Identifiers --------------------------------------------------------
    /// Static field: `__name`
    #[regex(r"__[a-zA-Z_][a-zA-Z0-9_]*", priority = 4)]
    StaticField,

    /// Instance field: `_name` (single underscore + at least one more char)
    #[regex(r"_[a-zA-Z][a-zA-Z0-9_]*", priority = 3)]
    Field,

    /// Identifier (also matches keywords, but logos prioritizes exact tokens).
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,

    // -- Newlines -----------------------------------------------------------
    #[token("\n")]
    Newline,

    // -- Tokens produced by custom lexing (not by logos directly) ------------
    /// String literal (simple, no interpolation).
    StringLit,

    /// Start of an interpolation segment: the string part before `%(`.
    InterpolationStart,

    /// End of an interpolation segment: the string part after `)`.
    InterpolationEnd,

    /// Middle of interpolation: string part between `)` and next `%(`.
    InterpolationMid,

    /// Raw string `"""..."""`.
    RawString,

    /// Line comment (skipped, not emitted).
    LineComment,

    /// Block comment (skipped, not emitted).
    BlockComment,

    /// An error token for unrecognized input.
    Error,
}

/// A token with its source span and extracted string slice.
#[derive(Debug, Clone)]
pub struct Lexeme {
    pub token: Token,
    pub span: Span,
    pub text: String,
}

/// Lex Wren source code into a sequence of [`Lexeme`]s.
///
/// Handles string interpolation, nested block comments, and raw strings
/// via a custom pass on top of logos.
pub fn lex(source: &str) -> (Vec<Lexeme>, Vec<crate::diagnostics::Diagnostic>) {
    let mut lexemes = Vec::new();
    let mut errors = Vec::new();
    let bytes = source.as_bytes();
    let mut pos = 0;

    while pos < source.len() {
        // Skip whitespace (not newlines)
        if bytes[pos] == b' ' || bytes[pos] == b'\t' || bytes[pos] == b'\r' {
            pos += 1;
            continue;
        }

        // Newline
        if bytes[pos] == b'\n' {
            lexemes.push(Lexeme {
                token: Token::Newline,
                span: pos..pos + 1,
                text: "\n".into(),
            });
            pos += 1;
            continue;
        }

        // Line comment
        if pos + 1 < source.len() && bytes[pos] == b'/' && bytes[pos + 1] == b'/' {
            let start = pos;
            pos += 2;
            while pos < source.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            let _ = start; // comment skipped
            continue;
        }

        // Shebang on first line
        if pos == 0 && pos + 1 < source.len() && bytes[0] == b'#' && bytes[1] == b'!' {
            while pos < source.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }

        // Block comment (with nesting)
        if pos + 1 < source.len() && bytes[pos] == b'/' && bytes[pos + 1] == b'*' {
            let start = pos;
            pos += 2;
            let mut depth = 1;
            while pos + 1 < source.len() && depth > 0 {
                if bytes[pos] == b'/' && bytes[pos + 1] == b'*' {
                    depth += 1;
                    pos += 2;
                } else if bytes[pos] == b'*' && bytes[pos + 1] == b'/' {
                    depth -= 1;
                    pos += 2;
                } else {
                    pos += 1;
                }
            }
            if depth > 0 {
                errors.push(
                    crate::diagnostics::Diagnostic::error("unterminated block comment")
                        .with_label(start..pos, "comment starts here"),
                );
            }
            continue;
        }

        // String literal (with interpolation support)
        if bytes[pos] == b'"' {
            // Check for raw string `"""`
            if pos + 2 < source.len() && bytes[pos + 1] == b'"' && bytes[pos + 2] == b'"' {
                let start = pos;
                pos += 3;
                loop {
                    if pos + 2 < source.len()
                        && bytes[pos] == b'"'
                        && bytes[pos + 1] == b'"'
                        && bytes[pos + 2] == b'"'
                    {
                        pos += 3;
                        break;
                    }
                    if pos >= source.len() {
                        errors.push(
                            crate::diagnostics::Diagnostic::error("unterminated raw string")
                                .with_label(start..pos, "raw string starts here"),
                        );
                        break;
                    }
                    pos += 1;
                }
                let text = &source[start..pos];
                // Strip the triple quotes
                let inner = &text[3..text.len().saturating_sub(3)];
                lexemes.push(Lexeme {
                    token: Token::RawString,
                    span: start..pos,
                    text: inner.to_string(),
                });
                continue;
            }

            // Regular string (may contain interpolation)
            lex_string(source, &mut pos, &mut lexemes, &mut errors);
            continue;
        }

        // Use logos for everything else — lex one token
        let remaining = &source[pos..];
        let mut logos_lexer = Token::lexer(remaining);

        if let Some(result) = logos_lexer.next() {
            let logo_span = logos_lexer.span();
            let text = &remaining[logo_span.clone()];
            let abs_start = pos + logo_span.start;
            let abs_end = pos + logo_span.end;

            match result {
                Ok(tok) => {
                    lexemes.push(Lexeme {
                        token: tok,
                        span: abs_start..abs_end,
                        text: text.to_string(),
                    });
                }
                Err(()) => {
                    errors.push(
                        crate::diagnostics::Diagnostic::error(format!(
                            "unexpected character '{}'",
                            text.chars().next().unwrap_or('?')
                        ))
                        .with_label(abs_start..abs_end, "here"),
                    );
                    lexemes.push(Lexeme {
                        token: Token::Error,
                        span: abs_start..abs_end,
                        text: text.to_string(),
                    });
                }
            }
            pos = abs_end;
        } else {
            // Logos couldn't match anything — skip one byte
            errors.push(
                crate::diagnostics::Diagnostic::error(format!(
                    "unexpected character '{}'",
                    source[pos..].chars().next().unwrap_or('?')
                ))
                .with_label(pos..pos + 1, "here"),
            );
            lexemes.push(Lexeme {
                token: Token::Error,
                span: pos..pos + 1,
                text: source[pos..pos + 1].to_string(),
            });
            pos += 1;
        }
    }

    (lexemes, errors)
}

/// Lex a regular string starting at `"`, handling escape sequences and
/// `%(expr)` interpolation.
fn lex_string(
    source: &str,
    pos: &mut usize,
    lexemes: &mut Vec<Lexeme>,
    errors: &mut Vec<crate::diagnostics::Diagnostic>,
) {
    let bytes = source.as_bytes();
    let start = *pos;
    *pos += 1; // skip opening `"`
    let mut buf = String::new();
    let mut has_interpolation = false;

    loop {
        if *pos >= source.len() {
            errors.push(
                crate::diagnostics::Diagnostic::error("unterminated string")
                    .with_label(start..*pos, "string starts here"),
            );
            break;
        }

        let ch = bytes[*pos];

        if ch == b'"' {
            *pos += 1; // skip closing `"`
            break;
        }

        if ch == b'\\' {
            *pos += 1;
            if *pos >= source.len() {
                errors.push(
                    crate::diagnostics::Diagnostic::error("unterminated escape sequence")
                        .with_label(*pos - 1..*pos, "here"),
                );
                break;
            }
            let esc = bytes[*pos];
            match esc {
                b'"' => buf.push('"'),
                b'\\' => buf.push('\\'),
                b'%' => buf.push('%'),
                b'0' => buf.push('\0'),
                b'a' => buf.push('\x07'),
                b'b' => buf.push('\x08'),
                b'e' => buf.push('\x1b'),
                b'f' => buf.push('\x0c'),
                b'n' => buf.push('\n'),
                b'r' => buf.push('\r'),
                b't' => buf.push('\t'),
                b'v' => buf.push('\x0b'),
                b'x' => {
                    if let Some(c) = lex_hex_escape(source, pos, 2) {
                        buf.push(c);
                    } else {
                        errors.push(
                            crate::diagnostics::Diagnostic::error("invalid hex escape")
                                .with_label(*pos - 1..*pos + 2, "here"),
                        );
                    }
                }
                b'u' => {
                    if let Some(c) = lex_hex_escape(source, pos, 4) {
                        buf.push(c);
                    } else {
                        errors.push(
                            crate::diagnostics::Diagnostic::error("invalid unicode escape")
                                .with_label(*pos - 1..*pos + 4, "here"),
                        );
                    }
                }
                b'U' => {
                    if let Some(c) = lex_hex_escape(source, pos, 8) {
                        buf.push(c);
                    } else {
                        errors.push(
                            crate::diagnostics::Diagnostic::error("invalid unicode escape")
                                .with_label(*pos - 1..*pos + 8, "here"),
                        );
                    }
                }
                _ => {
                    errors.push(
                        crate::diagnostics::Diagnostic::error(format!(
                            "unknown escape sequence '\\{}'",
                            esc as char
                        ))
                        .with_label(*pos - 1..*pos + 1, "here"),
                    );
                    buf.push(esc as char);
                }
            }
            *pos += 1;
            continue;
        }

        // Interpolation: `%(`
        if ch == b'%' && *pos + 1 < source.len() && bytes[*pos + 1] == b'(' {
            has_interpolation = true;
            // Emit the string part before interpolation
            let seg_token = if lexemes.last().is_none_or(|l| {
                !matches!(l.token, Token::InterpolationStart | Token::InterpolationMid)
            }) {
                Token::InterpolationStart
            } else {
                Token::InterpolationMid
            };
            lexemes.push(Lexeme {
                token: seg_token,
                span: start..*pos + 2,
                text: std::mem::take(&mut buf),
            });
            *pos += 2; // skip `%(`

            // Lex the interpolated expression tokens until matching `)`
            let mut paren_depth = 1;
            while *pos < source.len() && paren_depth > 0 {
                // Skip whitespace
                if bytes[*pos] == b' ' || bytes[*pos] == b'\t' || bytes[*pos] == b'\r' {
                    *pos += 1;
                    continue;
                }
                if bytes[*pos] == b'\n' {
                    lexemes.push(Lexeme {
                        token: Token::Newline,
                        span: *pos..*pos + 1,
                        text: "\n".into(),
                    });
                    *pos += 1;
                    continue;
                }
                if bytes[*pos] == b'"' {
                    // Nested string inside interpolation
                    lex_string(source, pos, lexemes, errors);
                    continue;
                }

                let remaining = &source[*pos..];
                let mut inner_lexer = Token::lexer(remaining);
                if let Some(result) = inner_lexer.next() {
                    let span = inner_lexer.span();
                    let text = &remaining[span.clone()];
                    let abs_start = *pos + span.start;
                    let abs_end = *pos + span.end;

                    match result {
                        Ok(tok) => {
                            match tok {
                                Token::LeftParen => paren_depth += 1,
                                Token::RightParen => {
                                    paren_depth -= 1;
                                    if paren_depth == 0 {
                                        *pos = abs_end;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            lexemes.push(Lexeme {
                                token: tok,
                                span: abs_start..abs_end,
                                text: text.to_string(),
                            });
                        }
                        Err(()) => {
                            lexemes.push(Lexeme {
                                token: Token::Error,
                                span: abs_start..abs_end,
                                text: text.to_string(),
                            });
                        }
                    }
                    *pos = abs_end;
                } else {
                    *pos += 1;
                }
            }

            if paren_depth > 0 {
                errors.push(
                    crate::diagnostics::Diagnostic::error("unterminated string interpolation")
                        .with_label(start..*pos, "interpolation starts here"),
                );
                return;
            }

            // Continue parsing the rest of the string
            continue;
        }

        // Regular character
        buf.push(ch as char);
        *pos += 1;
    }

    if has_interpolation {
        lexemes.push(Lexeme {
            token: Token::InterpolationEnd,
            span: start..*pos,
            text: buf,
        });
    } else {
        lexemes.push(Lexeme {
            token: Token::StringLit,
            span: start..*pos,
            text: buf,
        });
    }
}

/// Read `count` hex digits at `*pos + 1` and return the character.
fn lex_hex_escape(source: &str, pos: &mut usize, count: usize) -> Option<char> {
    let start = *pos + 1;
    let end = start + count;
    if end > source.len() {
        return None;
    }
    let hex = &source[start..end];
    let code = u32::from_str_radix(hex, 16).ok()?;
    *pos += count; // advance past the hex digits (caller will +1 more)
    char::from_u32(code)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_tokens(source: &str) -> Vec<Token> {
        let (lexemes, errors) = lex(source);
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
        lexemes.iter().map(|l| l.token.clone()).collect()
    }

    fn lex_texts(source: &str) -> Vec<String> {
        let (lexemes, _) = lex(source);
        lexemes.iter().map(|l| l.text.clone()).collect()
    }

    #[test]
    fn test_single_char_tokens() {
        let tokens = lex_tokens("( ) [ ] { } , . : # ? ~ + - * / % < > = ! & | ^");
        assert_eq!(
            tokens,
            vec![
                Token::LeftParen,
                Token::RightParen,
                Token::LeftBracket,
                Token::RightBracket,
                Token::LeftBrace,
                Token::RightBrace,
                Token::Comma,
                Token::Dot,
                Token::Colon,
                Token::Hash,
                Token::Question,
                Token::Tilde,
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Percent,
                Token::Lt,
                Token::Gt,
                Token::Eq,
                Token::Bang,
                Token::Amp,
                Token::Pipe,
                Token::Caret,
            ]
        );
    }

    #[test]
    fn test_two_char_tokens() {
        let tokens = lex_tokens("== != <= >= << >> || && .. ...");
        assert_eq!(
            tokens,
            vec![
                Token::EqEq,
                Token::BangEq,
                Token::LtEq,
                Token::GtEq,
                Token::LtLt,
                Token::GtGt,
                Token::PipePipe,
                Token::AmpAmp,
                Token::DotDot,
                Token::DotDotDot,
            ]
        );
    }

    #[test]
    fn test_compound_assignment() {
        let tokens = lex_tokens("+= -= *= /= %= &= |= ^= <<= >>=");
        assert_eq!(
            tokens,
            vec![
                Token::PlusEq,
                Token::MinusEq,
                Token::StarEq,
                Token::SlashEq,
                Token::PercentEq,
                Token::AmpEq,
                Token::PipeEq,
                Token::CaretEq,
                Token::LtLtEq,
                Token::GtGtEq,
            ]
        );
    }

    #[test]
    fn test_keywords() {
        let tokens = lex_tokens(
            "as break class construct continue else false for foreign if import in is null return static super this true var while",
        );
        assert_eq!(
            tokens,
            vec![
                Token::As,
                Token::Break,
                Token::Class,
                Token::Construct,
                Token::Continue,
                Token::Else,
                Token::False,
                Token::For,
                Token::Foreign,
                Token::If,
                Token::Import,
                Token::In,
                Token::Is,
                Token::Null,
                Token::Return,
                Token::Static,
                Token::Super,
                Token::This,
                Token::True,
                Token::Var,
                Token::While,
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        let tokens = lex_tokens("foo bar123 camelCase PascalCase _x");
        assert_eq!(
            tokens,
            vec![
                Token::Ident,
                Token::Ident,
                Token::Ident,
                Token::Ident,
                Token::Field
            ]
        );
    }

    #[test]
    fn test_fields() {
        let tokens = lex_tokens("_field __staticField");
        assert_eq!(tokens, vec![Token::Field, Token::StaticField]);
    }

    #[test]
    fn test_numbers_integer() {
        let tokens = lex_tokens("0 1 42 1000000");
        assert_eq!(tokens, vec![Token::Number; 4]);
        let texts = lex_texts("0 1 42 1000000");
        assert_eq!(texts, vec!["0", "1", "42", "1000000"]);
    }

    #[test]
    fn test_numbers_float() {
        let texts = lex_texts("3.14 0.5");
        assert_eq!(texts, vec!["3.14", "0.5"]);
    }

    #[test]
    fn test_numbers_hex() {
        let texts = lex_texts("0xff 0xDEADBEEF 0X1A");
        assert_eq!(texts, vec!["0xff", "0xDEADBEEF", "0X1A"]);
    }

    #[test]
    fn test_numbers_scientific() {
        let texts = lex_texts("1e5 1.5e-3 2E+10");
        assert_eq!(texts, vec!["1e5", "1.5e-3", "2E+10"]);
    }

    #[test]
    fn test_strings_simple() {
        let (lexemes, errors) = lex(r#""hello" "" "with spaces""#);
        assert!(errors.is_empty());
        let texts: Vec<_> = lexemes.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "", "with spaces"]);
        assert!(lexemes.iter().all(|l| l.token == Token::StringLit));
    }

    #[test]
    fn test_strings_escapes() {
        let (lexemes, errors) = lex(r#""\n\t\r\\\"\0\a\b\e\f\v\%""#);
        assert!(errors.is_empty());
        assert_eq!(lexemes[0].text, "\n\t\r\\\"\0\x07\x08\x1b\x0c\x0b%");
    }

    #[test]
    fn test_strings_hex_escape() {
        let (lexemes, errors) = lex(r#""\x41\x42""#);
        assert!(errors.is_empty());
        assert_eq!(lexemes[0].text, "AB");
    }

    #[test]
    fn test_strings_unicode_escape() {
        let (lexemes, errors) = lex(r#""\u0041\u00e9""#);
        assert!(errors.is_empty());
        assert_eq!(lexemes[0].text, "A\u{e9}");
    }

    #[test]
    fn test_strings_interpolation() {
        let (lexemes, errors) = lex(r#""a %(b) c""#);
        assert!(errors.is_empty(), "errors: {:?}", errors);

        let tokens: Vec<_> = lexemes.iter().map(|l| &l.token).collect();
        assert_eq!(
            tokens,
            vec![
                &Token::InterpolationStart,
                &Token::Ident,
                &Token::InterpolationEnd,
            ]
        );
        assert_eq!(lexemes[0].text, "a ");
        assert_eq!(lexemes[1].text, "b");
        assert_eq!(lexemes[2].text, " c");
    }

    #[test]
    fn test_strings_nested_interpolation() {
        let (lexemes, errors) = lex(r#""a %(b + "c %(d) e") f""#);
        assert!(errors.is_empty(), "errors: {:?}", errors);

        // Should produce: InterpStart("a "), Ident(b), Plus, InterpStart("c "), Ident(d), InterpEnd(" e"), InterpEnd(" f")
        let tokens: Vec<_> = lexemes.iter().map(|l| &l.token).collect();
        assert!(tokens.contains(&&Token::InterpolationStart));
        assert!(tokens.contains(&&Token::Ident));
        assert!(tokens.contains(&&Token::Plus));
    }

    #[test]
    fn test_raw_strings() {
        let (lexemes, errors) = lex(r#""""raw\nstring""""#);
        assert!(errors.is_empty());
        assert_eq!(lexemes[0].token, Token::RawString);
        assert_eq!(lexemes[0].text, r"raw\nstring");
    }

    #[test]
    fn test_line_comments() {
        let tokens = lex_tokens("a // comment\nb");
        assert_eq!(tokens, vec![Token::Ident, Token::Newline, Token::Ident]);
    }

    #[test]
    fn test_block_comments() {
        let tokens = lex_tokens("a /* comment */ b");
        assert_eq!(tokens, vec![Token::Ident, Token::Ident]);
    }

    #[test]
    fn test_block_comments_nested() {
        let tokens = lex_tokens("a /* /* inner */ outer */ b");
        assert_eq!(tokens, vec![Token::Ident, Token::Ident]);
    }

    #[test]
    fn test_newlines_significant() {
        let tokens = lex_tokens("a\nb\nc");
        assert_eq!(
            tokens,
            vec![
                Token::Ident,
                Token::Newline,
                Token::Ident,
                Token::Newline,
                Token::Ident,
            ]
        );
    }

    #[test]
    fn test_whitespace_skipped() {
        let tokens = lex_tokens("  a  \t  b  ");
        assert_eq!(tokens, vec![Token::Ident, Token::Ident]);
    }

    #[test]
    fn test_spans_correct() {
        let (lexemes, _) = lex("var x = 42");
        assert_eq!(lexemes[0].span, 0..3); // var
        assert_eq!(lexemes[1].span, 4..5); // x
        assert_eq!(lexemes[2].span, 6..7); // =
        assert_eq!(lexemes[3].span, 8..10); // 42
    }

    #[test]
    fn test_complete_program() {
        let source = r#"class Greeter {
  construct new(name) {
    _name = name
  }
  greet() {
    System.print("Hello, %(_name)!")
  }
}"#;
        let (lexemes, errors) = lex(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        // Just verify it produces tokens without errors
        assert!(lexemes.len() > 10);
        // First token should be `class`
        assert_eq!(lexemes[0].token, Token::Class);
    }

    #[test]
    fn test_shebang() {
        let tokens = lex_tokens("#!/usr/bin/env wren\nvar x = 1");
        assert_eq!(tokens[0], Token::Newline);
        assert_eq!(tokens[1], Token::Var);
    }

    #[test]
    fn test_unterminated_string_error() {
        let (_, errors) = lex(r#""hello"#);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unterminated"));
    }

    #[test]
    fn test_unterminated_block_comment_error() {
        let (_, errors) = lex("/* no end");
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unterminated"));
    }
}
