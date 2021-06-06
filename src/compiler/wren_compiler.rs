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

// use crate::compiler::wren_parser::{Parser};
use crate::compiler::scanner::{Tokens, TokenType, Lexer, GrammarFn, GrammarRule, Precedence};
use core::mem::transmute;
// use std::cell::Cell;


pub struct Compiler<'a> {
    pub tokens: Vec<Tokens<'a>>,
    lexer: Lexer<'a>,
}

impl<'a> Compiler<'a> {
    pub fn new(source: &'a str) -> Self {
       let ret =  Compiler {
            tokens:   Vec::new(),
            lexer: Lexer::new(source),
        };

        ret
    }

    fn error(&self, message: String){
        println!("{}", message);
    }

    unsafe fn next(&mut self) -> bool {
        self.lexer.next_token()
    }

    /// Returns the type of the current token.
    fn peek(&self) -> Option<TokenType<'a>> {
        if let Some(Tokens::Token(token_type, _, _)) =  self.lexer.current  {
            return Some(token_type);
        }

        None
    }
    /// Consumes the current token if its type is [expected]. Returns true if a
    /// token was consumed.
    fn match_token(&mut self, expected: TokenType<'a>) -> bool {
        if let Some(token_type) = self.peek() {
            if token_type != expected {
                return false;
            }
        }
       unsafe { self.lexer.next_token(); }
        return true;
    }

    /// Consumes the current token. Emits an error if its type is not [expected].
    unsafe fn consume(&mut self, expected: TokenType<'a>, error_message: String) {
        self.next();
        if let Some(Tokens::Token(token_type, _, _)) = self.lexer.previous {
            if token_type != expected {
                self.error(error_message);

                // If the next token is the one we want, assume the current one is just a
                // spurious error and discard it to minimize the number of cascaded errors.
                if let Some(Tokens::Token(token_type, _, _)) = self.lexer.current {
                    if token_type == expected {
                        self.next();
                    }
                }
            }
        }
    }

    /// Matches one or more newlines. Returns true if at least one was found.
    fn match_line(&mut self) -> bool {
        if self.match_token(TokenType::Line(GrammarRule::unused())) {
            return false;
        }

        while self.match_token(TokenType::Line(GrammarRule::unused())) {}
        
        true
    }


    /// Discards any new lines starting at the current token.
    fn ignore_new_lines(&mut self){
        self.match_line();
    }

    /// Consumes the current token. Emits an error if it is not a new line. Then
    /// discards any duplicate newlines following it.
    unsafe fn consume_line(&mut self, error_message: String) {
        self.consume(TokenType::Line(GrammarRule::unused()), error_message);
        self.ignore_new_lines();
    }

    unsafe fn parse_precedence(&mut self, precedence:Precedence) {
        self.next();

        let mut can_assign = false;

        if let Some(Tokens::Token(token_type, _, _)) = self.lexer.previous {
            let mut _p = token_type.get_rule().prefix;
            match _p {
                Some(prefix) => {
                    // Track if the precendence of the surrounding expression is low enough to
                    // allow an assignment inside this one. We can't compile an assignment like
                    // a normal expression because it requires us to handle the LHS specially --
                    // it needs to be an lvalue, not an rvalue. So, for each of the kinds of
                    // expressions that are valid lvalues -- names, subscripts, fields, etc. --
                    // we pass in whether or not it appears in a context loose enough to allow
                    // "=". If so, it will parse the "=" itself and handle it appropriately.
                    can_assign = precedence <= Precedence::PrecConditional;
                    
                    prefix(transmute::<&mut Self, &'a mut Self>(self), can_assign);
                }
                None => {
                    self.error("Expected expression.".to_owned());
                    return;
                }
            }
        }
		
        
        if let Some(Tokens::Token(token_type, _, _)) = self.lexer.current {
            while precedence <= token_type.get_rule().precedence.unwrap() {
                self.lexer.next_token();

                if let Some(Tokens::Token(token_type, _, _)) = self.lexer.previous {
                    let infix = token_type.get_rule().infix;
                    if let Some(_infix) = infix {
                        _infix(transmute::<&mut Self, &'a mut Self>(self), can_assign);
                    }
                }
            }
        }
        
    }
}

// impl<'a> Compiler<'a> {
//     pub fn new(_parser: Parser<'static>, _parent:Option<Box<Compiler<'a>>>) -> Self {
//         let mut scopeDepth = -1 as i32;
//         match _parent {
//             Some(_) => scopeDepth = 0,
//             None => {}
//         }
        
//         let ret = Self {
//             parser: _parser,
//             scope_depth: scopeDepth,
//             parent: if let Some(p) = _parent { Some(p) } else {None},
//         };

//         ret
//     }

//     pub fn print(&'a self) {
//         println!("{:#?}", &self.parser.tokens);
//     }

//     fn error(&self, message: &str) {

//     }

//     /// Returns the type of the current token.
//     fn peek(&'a mut self) -> Option<TokenType> {
//         if let Some(tok) = self.parser.current.get() {
//             tok.token_type
//         } else {
//             None
//         }

//     }
//     fn as_ref(&'a self) -> &'a Compiler<'a>  {
//         let ret = Cell::new(self);
//         ret.get()
//     }

//     fn next_token(&'static self) {
//         self.parser.next_token();
//     }

//     /// Returns the type of the next token.
//     fn peek_next(&'a mut self) -> Option<TokenType> {
//         if let Some(tok) = self.parser.next.get() {
//             tok.token_type
//         } else {
//             None
//         }
//     }

//     /// Consumes the current token if its type is [expected]. Returns true if a
//     /// token was consumed.
//     fn match_token (&'static mut self, expected: TokenType) -> bool {
//             if let Some(t) = self.parser.current.get() {
//                 if t.token_type != Some(expected) {
//                     false
//                 } else {
//                     self.next_token();
//                     true
//                 }
//             } else {
//                 false
//             }
            
//     }

//     /// Consumes the current token. Emits an error if its type is not [expected].
//     fn consume(&'static mut self, expected: TokenType, error_message: &'a str) {
//         self.next_token();
//         if let Some(prev) = self.parser.previous.get() {
//             if prev.token_type != Some(expected) {
//                 self.error(error_message);

//                 // If the next token is the one we want, assume the current one is just a
//                 // spurious error and discard it to minimize the number of cascaded errors.
//                 if let Some(cur) = self.parser.current.get() {
//                     if cur.token_type == Some(expected) {
//                         self.parser.next_token();
//                     }
//                 }
//             }
//         }
//     }

//     /// Matches one or more newlines. Returns true if at least one was found.
//     fn match_line(&'static self) -> bool {
//         if !self.match_token(TokenType::Line) { false} else {
//             while self.match_token(TokenType::Line) {

//             }
    
//             true
//         } 
//     }

//     /// Discards any newlines starting at the current token.
//     // fn ignore_new_lines(&'static self) {
//     //     self.match_line();
//     // }

//     /// Consumes the current token. Emits an error if it is not a newline. Then
//     /// discards any duplicate newlines following it.
//     fn consume_line(&'a self, error_message: &'a str) {
//         self.consume(TokenType::Line, error_message);
//         self.match_line(); //ignore line
//     }

//     fn allow_before_dot(&'a mut self) {
//         match self.peek() {
//             Some(TokenType::Line) => {
//                 match self.peek_next() {
//                     Some(TokenType::Dot) => {
//                         self.parser.next_token();
//                     }
//                     None => {}
//                     _ => {}
//                 }
//             }
//             None => {}
//             _ => {}
//         }
//     }

//     /// Starts a new local block scope.
//     fn push_scope(&mut self) {
//         self.scope_depth += 1;
//     }
// }