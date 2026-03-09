use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::intern::{Interner, SymbolId};
use crate::parse::lexer::{Lexeme, Token};

/// Parse result: AST module + any diagnostics.
pub struct ParseResult {
    pub module: Module,
    pub errors: Vec<Diagnostic>,
    pub interner: Interner,
}

/// Recursive-descent parser for Wren.
pub struct Parser {
    tokens: Vec<Lexeme>,
    pos: usize,
    interner: Interner,
    errors: Vec<Diagnostic>,
}

impl Parser {
    pub fn new(tokens: Vec<Lexeme>, interner: Interner) -> Self {
        Self {
            tokens,
            pos: 0,
            interner,
            errors: Vec::new(),
        }
    }

    /// Parse the entire module.
    pub fn parse(mut self) -> ParseResult {
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !self.is_at_end() {
            match self.declaration() {
                Some(stmt) => stmts.push(stmt),
                None => {
                    // Error recovery: skip to next statement
                    self.advance();
                    self.skip_newlines();
                }
            }
            self.skip_newlines();
        }
        ParseResult {
            module: stmts,
            errors: self.errors,
            interner: self.interner,
        }
    }

    /// Consume the interner for later use.
    pub fn into_interner(self) -> Interner {
        self.interner
    }

    // -----------------------------------------------------------------------
    // Token helpers
    // -----------------------------------------------------------------------

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|l| &l.token)
    }

    fn peek_lexeme(&self) -> Option<&Lexeme> {
        self.tokens.get(self.pos)
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn advance(&mut self) -> Option<&Lexeme> {
        if self.pos < self.tokens.len() {
            let lexeme = &self.tokens[self.pos];
            self.pos += 1;
            Some(lexeme)
        } else {
            None
        }
    }

    fn check(&self, token: &Token) -> bool {
        self.peek() == Some(token)
    }

    fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: &Token, msg: &str) -> bool {
        if self.match_token(token) {
            true
        } else {
            let span = self.current_span();
            self.errors
                .push(Diagnostic::error(msg).with_label(span, "here"));
            false
        }
    }

    fn previous(&self) -> &Lexeme {
        &self.tokens[self.pos - 1]
    }

    fn current_span(&self) -> Span {
        if let Some(l) = self.tokens.get(self.pos) {
            l.span.clone()
        } else if let Some(l) = self.tokens.last() {
            l.span.end..l.span.end
        } else {
            0..0
        }
    }

    fn previous_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span.clone()
        } else {
            0..0
        }
    }

    fn skip_newlines(&mut self) {
        while self.check(&Token::Newline) {
            self.advance();
        }
    }

    fn expect_statement_end(&mut self) {
        if self.is_at_end() {
            return;
        }
        if self.check(&Token::Newline) || self.check(&Token::RightBrace) {
            if self.check(&Token::Newline) {
                self.advance();
            }
            return;
        }
        // `else` terminates the then-branch body of an if statement
        if self.check(&Token::Else) {
            return;
        }
        // Allow semicolons if we ever add them, but Wren uses newlines
        let span = self.current_span();
        self.errors.push(
            Diagnostic::error("expected newline after statement").with_label(span, "here"),
        );
    }

    /// Probe whether a `{` at current position starts a map literal.
    /// Saves and restores position and errors. Returns true if `{ expr : ...`.
    fn is_map_literal(&mut self) -> bool {
        let saved_pos = self.pos;
        let saved_errors = self.errors.len();
        self.advance(); // consume `{`
        self.skip_newlines();

        // Empty braces `{}` — not a map
        if self.check(&Token::RightBrace) {
            self.pos = saved_pos;
            self.errors.truncate(saved_errors);
            return false;
        }
        // `{ | ...` — closure with params
        if self.check(&Token::Pipe) {
            self.pos = saved_pos;
            self.errors.truncate(saved_errors);
            return false;
        }
        // Try parsing an expression, check if `:` follows
        let result = self.expression().is_some() && self.check(&Token::Colon);
        self.pos = saved_pos;
        self.errors.truncate(saved_errors);
        result
    }

    fn intern(&mut self, s: &str) -> SymbolId {
        self.interner.intern(s)
    }

    fn intern_prev(&mut self) -> SymbolId {
        let text = self.previous().text.clone();
        self.interner.intern(&text)
    }

    // -----------------------------------------------------------------------
    // Declarations / Statements
    // -----------------------------------------------------------------------

    fn declaration(&mut self) -> Option<Spanned<Stmt>> {
        self.skip_newlines();
        let start = self.current_span().start;

        match self.peek()? {
            Token::Class | Token::Foreign => self.class_declaration(start),
            Token::Import => self.import_declaration(start),
            Token::Var => self.var_declaration(start),
            _ => self.statement(start),
        }
    }

    fn var_declaration(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        self.advance(); // consume `var`
        self.skip_newlines();

        if !self.check(&Token::Ident) {
            let span = self.current_span();
            self.errors.push(
                Diagnostic::error("expected variable name").with_label(span, "here"),
            );
            return None;
        }
        self.advance();
        let name_text = self.previous().text.clone();
        let name_span = self.previous_span();
        let name_sym = self.intern(&name_text);

        let initializer = if self.match_token(&Token::Eq) {
            self.skip_newlines();
            Some(self.expression()?)
        } else {
            None
        };

        let end = self.previous_span().end;
        self.expect_statement_end();
        Some((
            Stmt::Var {
                name: (name_sym, name_span),
                initializer,
            },
            start..end,
        ))
    }

    fn import_declaration(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        self.advance(); // consume `import`
        self.skip_newlines();

        // Module path string
        if !self.check(&Token::StringLit) {
            let span = self.current_span();
            self.errors.push(
                Diagnostic::error("expected module path string").with_label(span, "here"),
            );
            return None;
        }
        self.advance();
        let module_text = self.previous().text.clone();
        let module_span = self.previous_span();

        let mut names = Vec::new();
        if self.match_token(&Token::For) {
            self.skip_newlines();
            loop {
                if !self.check(&Token::Ident) {
                    let span = self.current_span();
                    self.errors.push(
                        Diagnostic::error("expected import name").with_label(span, "here"),
                    );
                    break;
                }
                self.advance();
                let name_text = self.previous().text.clone();
                let name_span = self.previous_span();
                let name_sym = self.intern(&name_text);

                let alias = if self.match_token(&Token::As) {
                    self.skip_newlines();
                    if !self.check(&Token::Ident) {
                        let span = self.current_span();
                        self.errors.push(
                            Diagnostic::error("expected alias name").with_label(span, "here"),
                        );
                        break;
                    }
                    self.advance();
                    let alias_text = self.previous().text.clone();
                    let alias_span = self.previous_span();
                    let alias_sym = self.intern(&alias_text);
                    Some((alias_sym, alias_span))
                } else {
                    None
                };

                names.push(ImportName {
                    name: (name_sym, name_span),
                    alias,
                });

                if !self.match_token(&Token::Comma) {
                    break;
                }
                self.skip_newlines();
            }
        }

        let end = self.previous_span().end;
        self.expect_statement_end();
        Some((
            Stmt::Import {
                module: (module_text, module_span),
                names,
            },
            start..end,
        ))
    }

    fn class_declaration(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        let is_foreign = self.match_token(&Token::Foreign);
        self.expect(&Token::Class, "expected 'class'");
        self.skip_newlines();

        if !self.check(&Token::Ident) {
            let span = self.current_span();
            self.errors
                .push(Diagnostic::error("expected class name").with_label(span, "here"));
            return None;
        }
        self.advance();
        let name_text = self.previous().text.clone();
        let name_span = self.previous_span();
        let name_sym = self.intern(&name_text);

        let superclass = if self.match_token(&Token::Is) {
            self.skip_newlines();
            if !self.check(&Token::Ident) {
                let span = self.current_span();
                self.errors.push(
                    Diagnostic::error("expected superclass name").with_label(span, "here"),
                );
                return None;
            }
            self.advance();
            let sc_text = self.previous().text.clone();
            let sc_span = self.previous_span();
            let sc_sym = self.intern(&sc_text);
            Some((sc_sym, sc_span))
        } else {
            None
        };

        self.skip_newlines();
        self.expect(&Token::LeftBrace, "expected '{' after class name");
        self.skip_newlines();

        let mut methods = Vec::new();
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if let Some(method) = self.method_definition(is_foreign) {
                methods.push(method);
            } else {
                self.advance();
            }
            self.skip_newlines();
        }
        self.expect(&Token::RightBrace, "expected '}' after class body");
        let end = self.previous_span().end;
        self.expect_statement_end();

        Some((
            Stmt::Class(ClassDecl {
                name: (name_sym, name_span),
                superclass,
                is_foreign,
                methods,
            }),
            start..end,
        ))
    }

    fn method_definition(&mut self, class_is_foreign: bool) -> Option<Spanned<Method>> {
        let start = self.current_span().start;
        let is_foreign = self.match_token(&Token::Foreign);
        let is_static = self.match_token(&Token::Static);
        self.skip_newlines();

        let signature = self.method_signature()?;

        let body = if is_foreign || (class_is_foreign && !matches!(signature, MethodSig::Construct { .. })) {
            self.expect_statement_end();
            None
        } else {
            self.skip_newlines();
            Some(self.block()?)
        };

        let end = self.previous_span().end;
        self.skip_newlines();

        Some((
            Method {
                is_static,
                is_foreign: is_foreign || class_is_foreign,
                signature,
                body,
            },
            start..end,
        ))
    }

    fn method_signature(&mut self) -> Option<MethodSig> {
        // construct name(params)
        if self.match_token(&Token::Construct) {
            self.skip_newlines();
            if !self.check(&Token::Ident) {
                let span = self.current_span();
                self.errors.push(
                    Diagnostic::error("expected constructor name").with_label(span, "here"),
                );
                return None;
            }
            self.advance();
            let name = self.intern_prev();
            let params = self.parameter_list()?;
            return Some(MethodSig::Construct { name, params });
        }

        // [params] or [params]=(value) — subscript
        if self.match_token(&Token::LeftBracket) {
            let params = self.subscript_params()?;
            self.expect(&Token::RightBracket, "expected ']'");
            if self.match_token(&Token::Eq) {
                self.expect(&Token::LeftParen, "expected '(' after ']='");
                if !self.check(&Token::Ident) {
                    let span = self.current_span();
                    self.errors.push(
                        Diagnostic::error("expected parameter name").with_label(span, "here"),
                    );
                    return None;
                }
                self.advance();
                let value_text = self.previous().text.clone();
                let value_span = self.previous_span();
                let value_sym = self.intern(&value_text);
                self.expect(&Token::RightParen, "expected ')'");
                return Some(MethodSig::SubscriptSetter {
                    params,
                    value: (value_sym, value_span),
                });
            }
            return Some(MethodSig::Subscript { params });
        }

        // Operator methods
        if let Some(op) = self.try_operator() {
            let params = if self.check(&Token::LeftParen) {
                self.parameter_list()?
            } else {
                Vec::new()
            };
            return Some(MethodSig::Operator { op, params });
        }

        // Named method, getter, or setter
        if !self.check(&Token::Ident) {
            let span = self.current_span();
            self.errors.push(
                Diagnostic::error("expected method name").with_label(span, "here"),
            );
            return None;
        }
        self.advance();
        let name = self.intern_prev();

        // setter: name=(param)
        if self.match_token(&Token::Eq) {
            self.expect(&Token::LeftParen, "expected '(' after '='");
            if !self.check(&Token::Ident) {
                let span = self.current_span();
                self.errors.push(
                    Diagnostic::error("expected parameter name").with_label(span, "here"),
                );
                return None;
            }
            self.advance();
            let param_text = self.previous().text.clone();
            let param_span = self.previous_span();
            let param_sym = self.intern(&param_text);
            self.expect(&Token::RightParen, "expected ')'");
            return Some(MethodSig::Setter {
                name,
                param: (param_sym, param_span),
            });
        }

        // method with params: name(params)
        if self.check(&Token::LeftParen) {
            let params = self.parameter_list()?;
            return Some(MethodSig::Named { name, params });
        }

        // getter: name
        Some(MethodSig::Getter(name))
    }

    fn try_operator(&mut self) -> Option<Op> {
        let op = match self.peek()? {
            Token::Plus => Op::Plus,
            Token::Minus => Op::Minus,
            Token::Star => Op::Star,
            Token::Slash => Op::Slash,
            Token::Percent => Op::Percent,
            Token::Lt => Op::Lt,
            Token::Gt => Op::Gt,
            Token::LtEq => Op::LtEq,
            Token::GtEq => Op::GtEq,
            Token::EqEq => Op::EqEq,
            Token::BangEq => Op::BangEq,
            Token::Amp => Op::BitAnd,
            Token::Pipe => Op::BitOr,
            Token::Caret => Op::BitXor,
            Token::LtLt => Op::Shl,
            Token::GtGt => Op::Shr,
            Token::DotDot => Op::DotDot,
            Token::DotDotDot => Op::DotDotDot,
            Token::Bang => Op::Bang,
            Token::Tilde => Op::Tilde,
            _ => return None,
        };
        self.advance();
        Some(op)
    }

    fn parameter_list(&mut self) -> Option<Vec<Spanned<SymbolId>>> {
        self.expect(&Token::LeftParen, "expected '('");
        let mut params = Vec::new();
        if !self.check(&Token::RightParen) {
            loop {
                self.skip_newlines();
                if !self.check(&Token::Ident) {
                    let span = self.current_span();
                    self.errors.push(
                        Diagnostic::error("expected parameter name").with_label(span, "here"),
                    );
                    break;
                }
                self.advance();
                let text = self.previous().text.clone();
                let span = self.previous_span();
                let sym = self.intern(&text);
                params.push((sym, span));

                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(&Token::RightParen, "expected ')'");
        Some(params)
    }

    fn subscript_params(&mut self) -> Option<Vec<Spanned<SymbolId>>> {
        let mut params = Vec::new();
        if !self.check(&Token::RightBracket) {
            loop {
                self.skip_newlines();
                if !self.check(&Token::Ident) {
                    let span = self.current_span();
                    self.errors.push(
                        Diagnostic::error("expected parameter name").with_label(span, "here"),
                    );
                    break;
                }
                self.advance();
                let text = self.previous().text.clone();
                let span = self.previous_span();
                let sym = self.intern(&text);
                params.push((sym, span));

                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        Some(params)
    }

    // -----------------------------------------------------------------------
    // Statements
    // -----------------------------------------------------------------------

    fn statement(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        match self.peek()? {
            Token::If => self.if_statement(start),
            Token::While => self.while_statement(start),
            Token::For => self.for_statement(start),
            Token::Break => {
                self.advance();
                let end = self.previous_span().end;
                self.expect_statement_end();
                Some((Stmt::Break, start..end))
            }
            Token::Continue => {
                self.advance();
                let end = self.previous_span().end;
                self.expect_statement_end();
                Some((Stmt::Continue, start..end))
            }
            Token::Return => self.return_statement(start),
            Token::LeftBrace => {
                // Could be a block, map literal, or closure.
                // Probe: if first meaningful content after `{` is expr followed by `:`,
                // it's a map literal. Otherwise parse as a block statement.
                if self.is_map_literal() {
                    let expr = self.expression()?;
                    let end = expr.1.end;
                    self.expect_statement_end();
                    Some((Stmt::Expr(expr), start..end))
                } else {
                    let blk = self.block()?;
                    Some(blk)
                }
            }
            _ => {
                let expr = self.expression()?;
                let end = expr.1.end;
                self.expect_statement_end();
                Some((Stmt::Expr(expr), start..end))
            }
        }
    }

    fn if_statement(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        self.advance(); // consume `if`
        self.expect(&Token::LeftParen, "expected '(' after 'if'");
        self.skip_newlines();
        let condition = self.expression()?;
        self.skip_newlines();
        self.expect(&Token::RightParen, "expected ')' after condition");
        self.skip_newlines();

        let then_start = self.current_span().start;
        let then_branch = self.statement(then_start)?;

        self.skip_newlines();
        let else_branch = if self.match_token(&Token::Else) {
            self.skip_newlines();
            let else_start = self.current_span().start;
            Some(Box::new(self.statement(else_start)?))
        } else {
            None
        };

        let end = else_branch
            .as_ref()
            .map(|e| e.1.end)
            .unwrap_or(then_branch.1.end);

        Some((
            Stmt::If {
                condition,
                then_branch: Box::new(then_branch),
                else_branch,
            },
            start..end,
        ))
    }

    fn while_statement(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        self.advance(); // consume `while`
        self.expect(&Token::LeftParen, "expected '(' after 'while'");
        self.skip_newlines();
        let condition = self.expression()?;
        self.skip_newlines();
        self.expect(&Token::RightParen, "expected ')' after condition");
        self.skip_newlines();

        let body_start = self.current_span().start;
        let body = self.statement(body_start)?;
        let end = body.1.end;

        Some((
            Stmt::While {
                condition,
                body: Box::new(body),
            },
            start..end,
        ))
    }

    fn for_statement(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        self.advance(); // consume `for`
        self.expect(&Token::LeftParen, "expected '(' after 'for'");
        self.skip_newlines();

        if !self.check(&Token::Ident) {
            let span = self.current_span();
            self.errors.push(
                Diagnostic::error("expected variable name in for loop").with_label(span, "here"),
            );
            return None;
        }
        self.advance();
        let var_text = self.previous().text.clone();
        let var_span = self.previous_span();
        let var_sym = self.intern(&var_text);

        self.expect(&Token::In, "expected 'in' after variable");
        self.skip_newlines();
        let iterator = self.expression()?;
        self.skip_newlines();
        self.expect(&Token::RightParen, "expected ')' after iterator");
        self.skip_newlines();

        let body_start = self.current_span().start;
        let body = self.statement(body_start)?;
        let end = body.1.end;

        Some((
            Stmt::For {
                variable: (var_sym, var_span),
                iterator,
                body: Box::new(body),
            },
            start..end,
        ))
    }

    fn return_statement(&mut self, start: usize) -> Option<Spanned<Stmt>> {
        self.advance(); // consume `return`

        // Check if there's an expression on the same line
        if self.is_at_end()
            || self.check(&Token::Newline)
            || self.check(&Token::RightBrace)
        {
            let end = self.previous_span().end;
            self.expect_statement_end();
            return Some((Stmt::Return(None), start..end));
        }

        let expr = self.expression()?;
        let end = expr.1.end;
        self.expect_statement_end();
        Some((Stmt::Return(Some(expr)), start..end))
    }

    fn block(&mut self) -> Option<Spanned<Stmt>> {
        let start = self.current_span().start;
        self.expect(&Token::LeftBrace, "expected '{'");
        self.skip_newlines();

        let mut stmts = Vec::new();
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if let Some(stmt) = self.declaration() {
                stmts.push(stmt);
            } else {
                self.advance();
            }
            self.skip_newlines();
        }

        self.expect(&Token::RightBrace, "expected '}'");
        let end = self.previous_span().end;
        Some((Stmt::Block(stmts), start..end))
    }

    // -----------------------------------------------------------------------
    // Expressions — Pratt parser
    // -----------------------------------------------------------------------

    fn expression(&mut self) -> Option<Spanned<Expr>> {
        self.assignment()
    }

    fn assignment(&mut self) -> Option<Spanned<Expr>> {
        let expr = self.conditional()?;

        if let Some(op) = self.match_compound_assign() {
            self.skip_newlines();
            let value = self.assignment()?;
            let span = expr.1.start..value.1.end;
            return Some((
                Expr::CompoundAssign {
                    op,
                    target: Box::new(expr),
                    value: Box::new(value),
                },
                span,
            ));
        }

        if self.match_token(&Token::Eq) {
            self.skip_newlines();
            let value = self.assignment()?;
            let span = expr.1.start..value.1.end;
            return Some((
                Expr::Assign {
                    target: Box::new(expr),
                    value: Box::new(value),
                },
                span,
            ));
        }

        Some(expr)
    }

    fn match_compound_assign(&mut self) -> Option<BinaryOp> {
        let op = match self.peek()? {
            Token::PlusEq => BinaryOp::Add,
            Token::MinusEq => BinaryOp::Sub,
            Token::StarEq => BinaryOp::Mul,
            Token::SlashEq => BinaryOp::Div,
            Token::PercentEq => BinaryOp::Mod,
            Token::AmpEq => BinaryOp::BitAnd,
            Token::PipeEq => BinaryOp::BitOr,
            Token::CaretEq => BinaryOp::BitXor,
            Token::LtLtEq => BinaryOp::Shl,
            Token::GtGtEq => BinaryOp::Shr,
            _ => return None,
        };
        self.advance();
        Some(op)
    }

    fn conditional(&mut self) -> Option<Spanned<Expr>> {
        let expr = self.logical_or()?;

        if self.match_token(&Token::Question) {
            self.skip_newlines();
            let then_expr = self.expression()?;
            self.skip_newlines();
            self.expect(&Token::Colon, "expected ':' in ternary");
            self.skip_newlines();
            let else_expr = self.expression()?;
            let span = expr.1.start..else_expr.1.end;
            return Some((
                Expr::Conditional {
                    condition: Box::new(expr),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                },
                span,
            ));
        }

        Some(expr)
    }

    fn logical_or(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.logical_and()?;
        while self.match_token(&Token::PipePipe) {
            self.skip_newlines();
            let right = self.logical_and()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::LogicalOp {
                    op: LogicalOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn logical_and(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.equality()?;
        while self.match_token(&Token::AmpAmp) {
            self.skip_newlines();
            let right = self.equality()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::LogicalOp {
                    op: LogicalOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn equality(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.is_check()?;
        loop {
            let op = match self.peek() {
                Some(Token::EqEq) => BinaryOp::Eq,
                Some(Token::BangEq) => BinaryOp::NotEq,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let right = self.is_check()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn is_check(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.comparison()?;
        while self.match_token(&Token::Is) {
            self.skip_newlines();
            let right = self.comparison()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::Is {
                    value: Box::new(left),
                    type_name: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn comparison(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.bitwise_or()?;
        loop {
            let op = match self.peek() {
                Some(Token::Lt) => BinaryOp::Lt,
                Some(Token::Gt) => BinaryOp::Gt,
                Some(Token::LtEq) => BinaryOp::LtEq,
                Some(Token::GtEq) => BinaryOp::GtEq,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let right = self.bitwise_or()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn bitwise_or(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.bitwise_xor()?;
        while self.check(&Token::Pipe) {
            self.advance();
            self.skip_newlines();
            let right = self.bitwise_xor()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op: BinaryOp::BitOr,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn bitwise_xor(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.bitwise_and()?;
        while self.match_token(&Token::Caret) {
            self.skip_newlines();
            let right = self.bitwise_and()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op: BinaryOp::BitXor,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn bitwise_and(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.shift()?;
        while self.check(&Token::Amp) {
            self.advance();
            self.skip_newlines();
            let right = self.shift()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op: BinaryOp::BitAnd,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn shift(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.range()?;
        loop {
            let op = match self.peek() {
                Some(Token::LtLt) => BinaryOp::Shl,
                Some(Token::GtGt) => BinaryOp::Shr,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let right = self.range()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn range(&mut self) -> Option<Spanned<Expr>> {
        let left = self.term()?;
        if self.check(&Token::DotDot) || self.check(&Token::DotDotDot) {
            let inclusive = self.check(&Token::DotDot);
            self.advance();
            self.skip_newlines();
            let right = self.term()?;
            let span = left.1.start..right.1.end;
            return Some((
                Expr::Range {
                    from: Box::new(left),
                    to: Box::new(right),
                    inclusive,
                },
                span,
            ));
        }
        Some(left)
    }

    fn term(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.factor()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinaryOp::Add,
                Some(Token::Minus) => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let right = self.factor()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn factor(&mut self) -> Option<Spanned<Expr>> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Star) => BinaryOp::Mul,
                Some(Token::Slash) => BinaryOp::Div,
                Some(Token::Percent) => BinaryOp::Mod,
                _ => break,
            };
            self.advance();
            self.skip_newlines();
            let right = self.unary()?;
            let span = left.1.start..right.1.end;
            left = (
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Some(left)
    }

    fn unary(&mut self) -> Option<Spanned<Expr>> {
        let start = self.current_span().start;
        let op = match self.peek() {
            Some(Token::Minus) => UnaryOp::Neg,
            Some(Token::Bang) => UnaryOp::Not,
            Some(Token::Tilde) => UnaryOp::BNot,
            _ => return self.postfix(),
        };
        self.advance();
        self.skip_newlines();
        let operand = self.unary()?;
        let end = operand.1.end;
        Some((
            Expr::UnaryOp {
                op,
                operand: Box::new(operand),
            },
            start..end,
        ))
    }

    fn postfix(&mut self) -> Option<Spanned<Expr>> {
        let mut expr = self.primary()?;

        loop {
            if self.match_token(&Token::Dot) {
                self.skip_newlines();
                // Method call or field access
                if !self.check(&Token::Ident) {
                    let span = self.current_span();
                    self.errors.push(
                        Diagnostic::error("expected method name after '.'")
                            .with_label(span, "here"),
                    );
                    return Some(expr);
                }
                self.advance();
                let method_text = self.previous().text.clone();
                let method_span = self.previous_span();
                let method_sym = self.intern(&method_text);

                if self.check(&Token::LeftParen) {
                    // method call with parens
                    let args = self.argument_list()?;
                    let block_arg = self.try_block_arg();
                    let end = self.previous_span().end;
                    let span = expr.1.start..end;
                    expr = (
                        Expr::Call {
                            receiver: Some(Box::new(expr)),
                            method: (method_sym, method_span),
                            args,
                            block_arg,
                            has_parens: true,
                        },
                        span,
                    );
                } else if self.check(&Token::LeftBrace) {
                    // method call with block arg only (no parens)
                    let block_arg = self.try_block_arg();
                    let end = self.previous_span().end;
                    let span = expr.1.start..end;
                    expr = (
                        Expr::Call {
                            receiver: Some(Box::new(expr)),
                            method: (method_sym, method_span),
                            args: Vec::new(),
                            block_arg,
                            has_parens: false,
                        },
                        span,
                    );
                } else if self.match_token(&Token::Eq) {
                    // setter: obj.name = value
                    self.skip_newlines();
                    let value = self.expression()?;
                    let span = expr.1.start..value.1.end;
                    // Desugar setter to a call
                    let setter_name = format!("{}=", method_text);
                    let setter_sym = self.intern(&setter_name);
                    expr = (
                        Expr::Call {
                            receiver: Some(Box::new(expr)),
                            method: (setter_sym, method_span),
                            args: vec![value],
                            block_arg: None,
                            has_parens: true,
                        },
                        span,
                    );
                } else {
                    // Getter
                    let end = method_span.end;
                    let span = expr.1.start..end;
                    expr = (
                        Expr::Call {
                            receiver: Some(Box::new(expr)),
                            method: (method_sym, method_span),
                            args: Vec::new(),
                            block_arg: None,
                            has_parens: false,
                        },
                        span,
                    );
                }
            } else if self.match_token(&Token::LeftBracket) {
                // Subscript
                self.skip_newlines();
                let mut args = Vec::new();
                if !self.check(&Token::RightBracket) {
                    loop {
                        self.skip_newlines();
                        args.push(self.expression()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.skip_newlines();
                self.expect(&Token::RightBracket, "expected ']'");

                if self.match_token(&Token::Eq) {
                    self.skip_newlines();
                    let value = self.expression()?;
                    let span = expr.1.start..value.1.end;
                    expr = (
                        Expr::SubscriptSet {
                            receiver: Box::new(expr),
                            index_args: args,
                            value: Box::new(value),
                        },
                        span,
                    );
                } else {
                    let end = self.previous_span().end;
                    let span = expr.1.start..end;
                    expr = (
                        Expr::Subscript {
                            receiver: Box::new(expr),
                            args,
                        },
                        span,
                    );
                }
            } else {
                break;
            }
        }

        Some(expr)
    }

    fn argument_list(&mut self) -> Option<Vec<Spanned<Expr>>> {
        self.expect(&Token::LeftParen, "expected '('");
        let mut args = Vec::new();
        if !self.check(&Token::RightParen) {
            loop {
                self.skip_newlines();
                args.push(self.expression()?);
                self.skip_newlines();
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.expect(&Token::RightParen, "expected ')'");
        Some(args)
    }

    fn try_block_arg(&mut self) -> Option<Box<Spanned<Expr>>> {
        if !self.check(&Token::LeftBrace) {
            return None;
        }
        let closure = self.closure_expr()?;
        Some(Box::new(closure))
    }

    // -----------------------------------------------------------------------
    // Primary expressions
    // -----------------------------------------------------------------------

    fn primary(&mut self) -> Option<Spanned<Expr>> {
        let start = self.current_span().start;

        match self.peek()? {
            Token::Number => {
                self.advance();
                let text = &self.previous().text;
                let n: f64 = if text.starts_with("0x") || text.starts_with("0X") {
                    u64::from_str_radix(&text[2..], 16).unwrap_or(0) as f64
                } else {
                    text.parse().unwrap_or(0.0)
                };
                let span = self.previous_span();
                Some((Expr::Num(n), span))
            }

            Token::StringLit => {
                self.advance();
                let text = self.previous().text.clone();
                let span = self.previous_span();
                Some((Expr::Str(text), span))
            }

            Token::RawString => {
                self.advance();
                let text = self.previous().text.clone();
                let span = self.previous_span();
                Some((Expr::Str(text), span))
            }

            Token::InterpolationStart => {
                self.parse_interpolation(start)
            }

            Token::True => {
                self.advance();
                Some((Expr::Bool(true), self.previous_span()))
            }

            Token::False => {
                self.advance();
                Some((Expr::Bool(false), self.previous_span()))
            }

            Token::Null => {
                self.advance();
                Some((Expr::Null, self.previous_span()))
            }

            Token::This => {
                self.advance();
                Some((Expr::This, self.previous_span()))
            }

            Token::Super => {
                self.advance();
                let super_span = self.previous_span();
                if self.match_token(&Token::Dot) {
                    if !self.check(&Token::Ident) {
                        let span = self.current_span();
                        self.errors.push(
                            Diagnostic::error("expected method name after 'super.'")
                                .with_label(span, "here"),
                        );
                        return None;
                    }
                    self.advance();
                    let method_text = self.previous().text.clone();
                    let method_span = self.previous_span();
                    let method_sym = self.intern(&method_text);
                    let args = if self.check(&Token::LeftParen) {
                        self.argument_list()?
                    } else {
                        Vec::new()
                    };
                    let end = self.previous_span().end;
                    Some((
                        Expr::SuperCall {
                            method: Some((method_sym, method_span)),
                            args,
                        },
                        start..end,
                    ))
                } else if self.check(&Token::LeftParen) {
                    let args = self.argument_list()?;
                    let end = self.previous_span().end;
                    Some((
                        Expr::SuperCall {
                            method: None,
                            args,
                        },
                        start..end,
                    ))
                } else {
                    Some((
                        Expr::SuperCall {
                            method: None,
                            args: Vec::new(),
                        },
                        super_span,
                    ))
                }
            }

            Token::Ident => {
                self.advance();
                let text = self.previous().text.clone();
                let ident_span = self.previous_span();
                let sym = self.intern(&text);

                // Bare function call: name(args)
                if self.check(&Token::LeftParen) {
                    let args = self.argument_list()?;
                    let block_arg = self.try_block_arg();
                    let end = self.previous_span().end;
                    return Some((
                        Expr::Call {
                            receiver: None,
                            method: (sym, ident_span.clone()),
                            args,
                            block_arg,
                            has_parens: true,
                        },
                        start..end,
                    ));
                }

                // Bare call with block arg: name { }
                if self.check(&Token::LeftBrace) {
                    let block_arg = self.try_block_arg();
                    let end = self.previous_span().end;
                    return Some((
                        Expr::Call {
                            receiver: None,
                            method: (sym, ident_span.clone()),
                            args: Vec::new(),
                            block_arg,
                            has_parens: false,
                        },
                        start..end,
                    ));
                }

                Some((Expr::Ident(sym), ident_span))
            }

            Token::Field => {
                self.advance();
                let text = self.previous().text.clone();
                let span = self.previous_span();
                // Strip leading underscore for the field name
                let name = &text[1..];
                let sym = self.intern(name);
                Some((Expr::Field(sym), span))
            }

            Token::StaticField => {
                self.advance();
                let text = self.previous().text.clone();
                let span = self.previous_span();
                let name = &text[2..];
                let sym = self.intern(name);
                Some((Expr::StaticField(sym), span))
            }

            Token::LeftParen => {
                self.advance(); // consume `(`
                self.skip_newlines();
                let expr = self.expression()?;
                self.skip_newlines();
                self.expect(&Token::RightParen, "expected ')'");
                Some(expr)
            }

            Token::LeftBracket => {
                self.advance(); // consume `[`
                self.skip_newlines();
                let mut elements = Vec::new();
                if !self.check(&Token::RightBracket) {
                    loop {
                        self.skip_newlines();
                        elements.push(self.expression()?);
                        self.skip_newlines();
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.skip_newlines();
                self.expect(&Token::RightBracket, "expected ']'");
                let end = self.previous_span().end;
                Some((Expr::ListLiteral(elements), start..end))
            }

            Token::LeftBrace => {
                // Could be a map literal or a closure.
                // Map literal: { key: value, ... }
                // Closure: { |params| body } or { body }
                // Heuristic: if first non-newline token after { is |, it's a closure.
                // If it's expr followed by :, it's a map. Otherwise, closure body.
                self.parse_brace_expr(start)
            }

            _ => {
                let span = self.current_span();
                let text = self
                    .peek_lexeme()
                    .map(|l| l.text.clone())
                    .unwrap_or_default();
                self.errors.push(
                    Diagnostic::error(format!("unexpected token '{}'", text))
                        .with_label(span, "here"),
                );
                None
            }
        }
    }

    fn parse_interpolation(&mut self, start: usize) -> Option<Spanned<Expr>> {
        let mut parts = Vec::new();

        // First segment
        self.advance(); // consume InterpolationStart
        let text = self.previous().text.clone();
        if !text.is_empty() {
            let span = self.previous_span();
            parts.push((Expr::Str(text), span));
        }

        // Interpolated expressions and middle/end segments
        loop {
            // Parse the interpolated expression
            if !self.is_at_end()
                && !self.check(&Token::InterpolationEnd)
                && !self.check(&Token::InterpolationMid)
            {
                if let Some(expr) = self.expression() {
                    parts.push(expr);
                }
            }

            if self.check(&Token::InterpolationMid) {
                self.advance();
                let text = self.previous().text.clone();
                if !text.is_empty() {
                    let span = self.previous_span();
                    parts.push((Expr::Str(text), span));
                }
                continue;
            }

            if self.check(&Token::InterpolationEnd) {
                self.advance();
                let text = self.previous().text.clone();
                if !text.is_empty() {
                    let span = self.previous_span();
                    parts.push((Expr::Str(text), span));
                }
                break;
            }

            // Shouldn't reach here in valid input
            break;
        }

        let end = self.previous_span().end;
        Some((Expr::Interpolation(parts), start..end))
    }

    fn parse_brace_expr(&mut self, start: usize) -> Option<Spanned<Expr>> {
        // Look ahead to determine if this is a map or closure
        let saved = self.pos;

        // Peek past the `{` and newlines
        self.advance(); // consume `{`
        self.skip_newlines();

        // Empty braces: closure with empty body
        if self.check(&Token::RightBrace) {
            self.advance();
            let end = self.previous_span().end;
            return Some((
                Expr::Closure {
                    params: Vec::new(),
                    body: Box::new((Stmt::Block(Vec::new()), start..end)),
                },
                start..end,
            ));
        }

        // Pipe means closure params
        if self.check(&Token::Pipe) {
            self.pos = saved;
            return self.closure_expr();
        }

        // Try to detect map: expr followed by `:`
        // Save position and errors, try parsing an expression, check for `:`
        let probe_pos = self.pos;
        let probe_errors = self.errors.len();
        if let Some(_) = self.expression() {
            if self.check(&Token::Colon) {
                // It's a map — restore to probe_pos and parse properly.
                self.pos = probe_pos;
                self.errors.truncate(probe_errors);
                return self.map_literal(start);
            }
        }

        // Otherwise it's a closure body (no params) — discard probe errors.
        self.pos = probe_pos;
        self.errors.truncate(probe_errors);
        self.parse_closure_body(start, Vec::new())
    }

    fn closure_expr(&mut self) -> Option<Spanned<Expr>> {
        let start = self.current_span().start;
        self.expect(&Token::LeftBrace, "expected '{'");
        self.skip_newlines();

        let mut params = Vec::new();
        if self.match_token(&Token::Pipe) {
            if !self.check(&Token::Pipe) {
                loop {
                    self.skip_newlines();
                    if !self.check(&Token::Ident) {
                        break;
                    }
                    self.advance();
                    let text = self.previous().text.clone();
                    let span = self.previous_span();
                    let sym = self.intern(&text);
                    params.push((sym, span));
                    if !self.match_token(&Token::Comma) {
                        break;
                    }
                }
            }
            self.expect(&Token::Pipe, "expected '|' after parameters");
            self.skip_newlines();
        }

        self.parse_closure_body(start, params)
    }

    fn parse_closure_body(
        &mut self,
        start: usize,
        params: Vec<Spanned<SymbolId>>,
    ) -> Option<Spanned<Expr>> {
        self.skip_newlines();

        // Single-expression closure vs multi-statement
        if self.check(&Token::RightBrace) {
            self.advance();
            let end = self.previous_span().end;
            return Some((
                Expr::Closure {
                    params,
                    body: Box::new((Stmt::Block(Vec::new()), start..end)),
                },
                start..end,
            ));
        }

        // Try single expression (if next token after expr is `}`)
        let saved = self.pos;
        let saved_errors = self.errors.len();
        if let Some(expr) = self.expression() {
            self.skip_newlines();
            if self.check(&Token::RightBrace) {
                self.advance();
                let end = self.previous_span().end;
                let expr_span = expr.1.clone();
                return Some((
                    Expr::Closure {
                        params,
                        body: Box::new((Stmt::Expr(expr), expr_span)),
                    },
                    start..end,
                ));
            }
        }

        // Multi-statement body — discard errors from failed single-expr probe.
        self.pos = saved;
        self.errors.truncate(saved_errors);
        let mut stmts = Vec::new();
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            self.skip_newlines();
            if self.check(&Token::RightBrace) {
                break;
            }
            if let Some(stmt) = self.declaration() {
                stmts.push(stmt);
            } else {
                self.advance();
            }
            self.skip_newlines();
        }
        self.expect(&Token::RightBrace, "expected '}'");
        let end = self.previous_span().end;
        let body_span = start..end;
        Some((
            Expr::Closure {
                params,
                body: Box::new((Stmt::Block(stmts), body_span)),
            },
            start..end,
        ))
    }

    fn map_literal(&mut self, start: usize) -> Option<Spanned<Expr>> {
        let mut entries = Vec::new();
        if !self.check(&Token::RightBrace) {
            loop {
                self.skip_newlines();
                let key = self.expression()?;
                self.expect(&Token::Colon, "expected ':' after map key");
                self.skip_newlines();
                let value = self.expression()?;
                entries.push((key, value));
                self.skip_newlines();
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        self.skip_newlines();
        self.expect(&Token::RightBrace, "expected '}'");
        let end = self.previous_span().end;
        Some((Expr::MapLiteral(entries), start..end))
    }
}

/// Convenience: lex and parse source code in one call.
pub fn parse(source: &str) -> ParseResult {
    let (lexemes, lex_errors) = crate::parse::lexer::lex(source);
    let interner = Interner::new();
    let mut parser = Parser::new(lexemes, interner);
    parser.errors.extend(lex_errors);
    parser.parse()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_expr(source: &str) -> (Expr, Vec<Diagnostic>) {
        let result = parse(source);
        if result.module.is_empty() {
            return (Expr::Null, result.errors);
        }
        match &result.module[0].0 {
            Stmt::Expr(e) => (e.0.clone(), result.errors),
            other => panic!("expected Expr statement, got {:?}", other),
        }
    }

    fn parse_ok(source: &str) -> Module {
        let result = parse(source);
        assert!(
            result.errors.is_empty(),
            "unexpected errors: {:?}",
            result.errors
        );
        result.module
    }

    fn parse_one_stmt(source: &str) -> Stmt {
        let module = parse_ok(source);
        assert_eq!(module.len(), 1, "expected 1 statement, got {}", module.len());
        module[0].0.clone()
    }

    // -- Expression tests ---------------------------------------------------

    #[test]
    fn test_parse_number() {
        let (expr, errors) = parse_expr("42");
        assert!(errors.is_empty());
        assert_eq!(expr, Expr::Num(42.0));
    }

    #[test]
    fn test_parse_hex_number() {
        let (expr, errors) = parse_expr("0xff");
        assert!(errors.is_empty());
        assert_eq!(expr, Expr::Num(255.0));
    }

    #[test]
    fn test_parse_string() {
        let (expr, errors) = parse_expr(r#""hello""#);
        assert!(errors.is_empty());
        assert_eq!(expr, Expr::Str("hello".into()));
    }

    #[test]
    fn test_parse_interpolation() {
        let (expr, errors) = parse_expr(r#""a %(1 + 2) b""#);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        match expr {
            Expr::Interpolation(parts) => {
                assert!(parts.len() >= 2);
            }
            _ => panic!("expected Interpolation, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_bool_null() {
        assert_eq!(parse_expr("true").0, Expr::Bool(true));
        assert_eq!(parse_expr("false").0, Expr::Bool(false));
        assert_eq!(parse_expr("null").0, Expr::Null);
    }

    #[test]
    fn test_parse_this() {
        assert_eq!(parse_expr("this").0, Expr::This);
    }

    #[test]
    fn test_parse_identifiers() {
        match parse_expr("foo").0 {
            Expr::Ident(_) => {}
            other => panic!("expected Ident, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_fields() {
        match parse_expr("_field").0 {
            Expr::Field(_) => {}
            other => panic!("expected Field, got {:?}", other),
        }
        match parse_expr("__static").0 {
            Expr::StaticField(_) => {}
            other => panic!("expected StaticField, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_unary() {
        let (expr, errors) = parse_expr("-x");
        assert!(errors.is_empty());
        match expr {
            Expr::UnaryOp { op, .. } => assert_eq!(op, UnaryOp::Neg),
            _ => panic!("expected UnaryOp"),
        }

        let (expr, _) = parse_expr("!x");
        match expr {
            Expr::UnaryOp { op, .. } => assert_eq!(op, UnaryOp::Not),
            _ => panic!("expected UnaryOp"),
        }
    }

    #[test]
    fn test_parse_binary_precedence() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let (expr, errors) = parse_expr("1 + 2 * 3");
        assert!(errors.is_empty());
        match expr {
            Expr::BinaryOp { op, right, .. } => {
                assert_eq!(op, BinaryOp::Add);
                match &right.0 {
                    Expr::BinaryOp { op, .. } => assert_eq!(*op, BinaryOp::Mul),
                    _ => panic!("expected Mul on right"),
                }
            }
            _ => panic!("expected BinaryOp"),
        }
    }

    #[test]
    fn test_parse_comparison() {
        let (expr, errors) = parse_expr("a < b");
        assert!(errors.is_empty());
        match expr {
            Expr::BinaryOp { op, .. } => assert_eq!(op, BinaryOp::Lt),
            _ => panic!("expected BinaryOp"),
        }
    }

    #[test]
    fn test_parse_logical() {
        let (expr, errors) = parse_expr("a || b && c");
        assert!(errors.is_empty());
        match expr {
            Expr::LogicalOp { op, right, .. } => {
                assert_eq!(op, LogicalOp::Or);
                match &right.0 {
                    Expr::LogicalOp { op, .. } => assert_eq!(*op, LogicalOp::And),
                    _ => panic!("expected And on right"),
                }
            }
            _ => panic!("expected LogicalOp"),
        }
    }

    #[test]
    fn test_parse_is() {
        let (expr, errors) = parse_expr("obj is Num");
        assert!(errors.is_empty());
        match expr {
            Expr::Is { .. } => {}
            _ => panic!("expected Is"),
        }
    }

    #[test]
    fn test_parse_assignment() {
        let (expr, errors) = parse_expr("x = 5");
        assert!(errors.is_empty());
        match expr {
            Expr::Assign { .. } => {}
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn test_parse_compound_assign() {
        let (expr, errors) = parse_expr("x += 1");
        assert!(errors.is_empty());
        match expr {
            Expr::CompoundAssign { op, .. } => assert_eq!(op, BinaryOp::Add),
            _ => panic!("expected CompoundAssign"),
        }
    }

    #[test]
    fn test_parse_conditional() {
        let (expr, errors) = parse_expr("a ? b : c");
        assert!(errors.is_empty());
        match expr {
            Expr::Conditional { .. } => {}
            _ => panic!("expected Conditional"),
        }
    }

    #[test]
    fn test_parse_range() {
        let (expr, errors) = parse_expr("1..10");
        assert!(errors.is_empty());
        match expr {
            Expr::Range { inclusive, .. } => assert!(inclusive),
            _ => panic!("expected Range"),
        }

        let (expr, errors) = parse_expr("0...count");
        assert!(errors.is_empty());
        match expr {
            Expr::Range { inclusive, .. } => assert!(!inclusive),
            _ => panic!("expected Range"),
        }
    }

    #[test]
    fn test_parse_call() {
        let (expr, errors) = parse_expr("obj.method(a, b)");
        assert!(errors.is_empty());
        match expr {
            Expr::Call { args, receiver, .. } => {
                assert!(receiver.is_some());
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn test_parse_subscript() {
        let (expr, errors) = parse_expr("list[0]");
        assert!(errors.is_empty());
        match expr {
            Expr::Subscript { args, .. } => assert_eq!(args.len(), 1),
            _ => panic!("expected Subscript"),
        }
    }

    #[test]
    fn test_parse_subscript_setter() {
        let (expr, errors) = parse_expr("list[0] = val");
        assert!(errors.is_empty());
        match expr {
            Expr::SubscriptSet { .. } => {}
            _ => panic!("expected SubscriptSet"),
        }
    }

    #[test]
    fn test_parse_super_call() {
        let (expr, errors) = parse_expr("super.method(x)");
        assert!(errors.is_empty());
        match expr {
            Expr::SuperCall { method, args } => {
                assert!(method.is_some());
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected SuperCall"),
        }
    }

    #[test]
    fn test_parse_super_constructor() {
        let (expr, errors) = parse_expr("super(x)");
        assert!(errors.is_empty());
        match expr {
            Expr::SuperCall { method, args } => {
                assert!(method.is_none());
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected SuperCall"),
        }
    }

    #[test]
    fn test_parse_list_literal() {
        let (expr, errors) = parse_expr("[1, 2, 3]");
        assert!(errors.is_empty());
        match expr {
            Expr::ListLiteral(elems) => assert_eq!(elems.len(), 3),
            _ => panic!("expected ListLiteral"),
        }
    }

    #[test]
    fn test_parse_map_literal() {
        let result = parse(r#"{"a": 1, "b": 2}"#);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module[0].0 {
            Stmt::Expr((Expr::MapLiteral(entries), _)) => assert_eq!(entries.len(), 2),
            other => panic!("expected MapLiteral, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_closure() {
        let (expr, errors) = parse_expr("Fn.new { |x| x + 1 }");
        assert!(errors.is_empty(), "errors: {:?}", errors);
        // Fn.new is a call with a block arg
        match expr {
            Expr::Call { block_arg, .. } => assert!(block_arg.is_some()),
            _ => panic!("expected Call with block_arg"),
        }
    }

    #[test]
    fn test_parse_closure_no_params() {
        let result = parse(r#"Fn.new { "hello" }"#);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_parse_grouping() {
        let (expr, errors) = parse_expr("(1 + 2) * 3");
        assert!(errors.is_empty());
        match expr {
            Expr::BinaryOp { op, left, .. } => {
                assert_eq!(op, BinaryOp::Mul);
                match &left.0 {
                    Expr::BinaryOp { op, .. } => assert_eq!(*op, BinaryOp::Add),
                    _ => panic!("expected Add in group"),
                }
            }
            _ => panic!("expected BinaryOp"),
        }
    }

    #[test]
    fn test_parse_chained_calls() {
        let (expr, errors) = parse_expr("a.b.c(1).d");
        assert!(errors.is_empty());
        // Should be a nested chain of calls
        match expr {
            Expr::Call { receiver: Some(r), .. } => match &r.0 {
                Expr::Call { receiver: Some(r2), .. } => match &r2.0 {
                    Expr::Call { .. } => {}
                    _ => panic!("expected inner Call"),
                },
                _ => panic!("expected Call chain"),
            },
            _ => panic!("expected Call"),
        }
    }

    // -- Statement tests ----------------------------------------------------

    #[test]
    fn test_parse_var_decl() {
        let stmt = parse_one_stmt("var x = 42");
        match stmt {
            Stmt::Var { initializer, .. } => assert!(initializer.is_some()),
            _ => panic!("expected Var"),
        }
    }

    #[test]
    fn test_parse_var_no_init() {
        let stmt = parse_one_stmt("var y");
        match stmt {
            Stmt::Var { initializer, .. } => assert!(initializer.is_none()),
            _ => panic!("expected Var"),
        }
    }

    #[test]
    fn test_parse_if() {
        let stmt = parse_one_stmt("if (x) y");
        match stmt {
            Stmt::If { else_branch, .. } => assert!(else_branch.is_none()),
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn test_parse_if_else() {
        let stmt = parse_one_stmt("if (x) y else z");
        match stmt {
            Stmt::If { else_branch, .. } => assert!(else_branch.is_some()),
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn test_parse_while() {
        let stmt = parse_one_stmt("while (true) x");
        match stmt {
            Stmt::While { .. } => {}
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn test_parse_for() {
        let module = parse_ok("for (x in list) x");
        match &module[0].0 {
            Stmt::For { .. } => {}
            _ => panic!("expected For"),
        }
    }

    #[test]
    fn test_parse_block() {
        let result = parse("{\n  var x = 1\n  x\n}");
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_parse_break_continue() {
        assert!(matches!(parse_one_stmt("break"), Stmt::Break));
        assert!(matches!(parse_one_stmt("continue"), Stmt::Continue));
    }

    #[test]
    fn test_parse_return() {
        match parse_one_stmt("return") {
            Stmt::Return(None) => {}
            _ => panic!("expected Return(None)"),
        }
        match parse_one_stmt("return 42") {
            Stmt::Return(Some(_)) => {}
            _ => panic!("expected Return(Some)"),
        }
    }

    #[test]
    fn test_parse_newline_separation() {
        let module = parse_ok("x\ny");
        assert_eq!(module.len(), 2);
    }

    // -- Class tests --------------------------------------------------------

    #[test]
    fn test_parse_class_empty() {
        let stmt = parse_one_stmt("class Foo {}");
        match stmt {
            Stmt::Class(c) => assert!(c.methods.is_empty()),
            _ => panic!("expected Class"),
        }
    }

    #[test]
    fn test_parse_class_inherit() {
        let stmt = parse_one_stmt("class Foo is Bar {}");
        match stmt {
            Stmt::Class(c) => assert!(c.superclass.is_some()),
            _ => panic!("expected Class"),
        }
    }

    #[test]
    fn test_parse_class_constructor() {
        let result = parse("class Foo {\n  construct new() {}\n}");
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module[0].0 {
            Stmt::Class(c) => {
                assert_eq!(c.methods.len(), 1);
                match &c.methods[0].0.signature {
                    MethodSig::Construct { .. } => {}
                    _ => panic!("expected Construct"),
                }
            }
            _ => panic!("expected Class"),
        }
    }

    #[test]
    fn test_parse_class_methods() {
        let source = r#"class Foo {
  getter { _x }
  method(a) { a }
  setter=(v) { _x = v }
  +(other) { }
  [i] { }
}"#;
        let result = parse(source);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module[0].0 {
            Stmt::Class(c) => assert_eq!(c.methods.len(), 5),
            _ => panic!("expected Class"),
        }
    }

    #[test]
    fn test_parse_class_static() {
        let result = parse("class Foo {\n  static bar() {}\n}");
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module[0].0 {
            Stmt::Class(c) => assert!(c.methods[0].0.is_static),
            _ => panic!("expected Class"),
        }
    }

    #[test]
    fn test_parse_class_foreign() {
        let result = parse("foreign class Foo {\n  foreign bar()\n}");
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        match &result.module[0].0 {
            Stmt::Class(c) => {
                assert!(c.is_foreign);
                assert!(c.methods[0].0.is_foreign);
                assert!(c.methods[0].0.body.is_none());
            }
            _ => panic!("expected Class"),
        }
    }

    // -- Import tests -------------------------------------------------------

    #[test]
    fn test_parse_import_bare() {
        let stmt = parse_one_stmt(r#"import "module""#);
        match stmt {
            Stmt::Import { names, .. } => assert!(names.is_empty()),
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn test_parse_import_for() {
        let stmt = parse_one_stmt(r#"import "module" for Name"#);
        match stmt {
            Stmt::Import { names, .. } => {
                assert_eq!(names.len(), 1);
                assert!(names[0].alias.is_none());
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn test_parse_import_as() {
        let stmt = parse_one_stmt(r#"import "module" for Name as Alias"#);
        match stmt {
            Stmt::Import { names, .. } => {
                assert_eq!(names.len(), 1);
                assert!(names[0].alias.is_some());
            }
            _ => panic!("expected Import"),
        }
    }

    // -- Error recovery test ------------------------------------------------

    #[test]
    fn test_parse_error_recovery() {
        let result = parse("var = 1\nvar y = 2");
        // Should produce errors but not panic, and recover to parse y
        assert!(!result.errors.is_empty());
    }

    // -- Complete program test ----------------------------------------------

    #[test]
    fn test_parse_complete_program() {
        let source = r#"
class Greeter {
  construct new(name) {
    _name = name
  }
  greet() {
    System.print("Hello, %(_name)!")
  }
}

var g = Greeter.new("world")
g.greet()
"#;
        let result = parse(source);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.module.len() >= 3); // class + var + call
    }
}
