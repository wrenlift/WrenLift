// Copyright 2021 Zenturi Software Co.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use wren_parser::{Parser};
use scanner::{TokenType, Token};

#[derive(Debug, Clone)]
pub struct Compiler<'a> {
    parser: Parser<'a>,
    scope_depth: usize,
    parent: Option<Compiler<'a>>,
}

impl<'a> Compiler<'a> {
    fn new(_parser: &'a mut Parser, _parent:Option<Compiler<'a>>) -> Self {
        let mut scopeDepth = -1;
        if let Some(p) = _parent {
            scopeDepth = 0;
        }
        Self {
            parser: _parser,
            scope_depth: scopeDepth,
            parent: _parent,
        }
    }

    fn error(&self, message: &'a str) {

    }

    /// Returns the type of the current token.
    fn peek(&self) -> Option<TokenType> {
        if let Some(tok) = self.parser.current{
            tok.token_type
        }
        None
    }

    /// Returns the type of the next token.
    fn peek_next(&self) -> Option<TokenType> {
        if let Some(tok) = self.parser.next {
            tok.token_type
        }
        None
    }

    /// Consumes the current token if its type is [expected]. Returns true if a
    /// token was consumed.
    fn match(&self, expected: TokenType) -> bool {
            let Some(t) =  self.peek() {
                if t != expected {
                    false
                }

                self.parser.next_token();
                true
            }
            false
    }

    /// Consumes the current token. Emits an error if its type is not [expected].
    fn consume(&self, expected: TokenType, error_message: &'a str) {
        self.parser.next_token();
        if let Some(prev) = self.parser.previous {
            if prev.token_type != expected {
                self.error(error_message);

                // If the next token is the one we want, assume the current one is just a
                // spurious error and discard it to minimize the number of cascaded errors.
                if let Some(cur) = self.parser.current {
                    if cur == expected {
                        self.parser.next_token();
                    }
                }
            }
        }
    }

    /// Matches one or more newlines. Returns true if at least one was found.
    fn match_line(&self) -> bool {
        if !self.match(TokenType::Line) false

        while self.match(TokenType::Line) {

        }

        true
    }

    /// Discards any newlines starting at the current token.
    fn ignore_new_lines(&self) {
        self.match_line();
    }

    /// Consumes the current token. Emits an error if it is not a newline. Then
    /// discards any duplicate newlines following it.
    fn consume_line(&self, error_message: &'a str) {
        self.consume(TokenType::Line, error_message);
        self.ignore_new_lines();
    }

    fn allow_before_dot(&self) {
        match self.peek() {
            Some(TokenType::Line) => {
                match self.peek_next() {
                    Some(TokenType::Dot) => {
                        self.parser.next_token();
                    }
                    None => {}
                }
            }
            None => {}
        }
    }

    /// Starts a new local block scope.
    fn push_scope(&mut self) {
        self.scope_depth += 1;
    }
}