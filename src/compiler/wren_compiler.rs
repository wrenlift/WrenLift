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
use crate::compiler::scanner::{Tokens, TokenType, Lexer, GrammarFn, GrammarRule, Precedence, KEYWORDS, GRAMMAR_RULES};
use crate::compiler::grammar::{Grammar};
use core::mem::transmute;
// use std::cell::Cell;


pub const MAX_VARIABLE_NAME: usize = 64;

pub struct Compiler<'a> {
    pub tokens: Vec<Tokens<'a>>,
    lexer: Lexer<'a>,
    pub scope_depth: i32,
}

impl<'a> Compiler<'a> {
    pub fn new(source: &'a str) -> Self {
       let ret =  Compiler {
            tokens:   Vec::new(),
            lexer: Lexer::new(source),
            scope_depth: -1
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

    pub fn expression(&mut self)  {
        unsafe { self.parse_precedence(Precedence::PrecLowest);}
    }

    /// Compiles a "definition". These are the statements that bind new variables.
    /// They can only appear at the top level of a block and are prohibited in places
    /// like the non-curly body of an if or while.
    pub unsafe fn definition(&'a mut self) {
        let class_type = transmute::<TokenType<'static>, TokenType<'a>>(KEYWORDS[1]);
        let foreign_type = transmute::<TokenType<'static>, TokenType<'a>>(KEYWORDS[6]);
        let import_type = transmute::<TokenType<'static>, TokenType<'a>>(KEYWORDS[8]);
        let var_type = transmute::<TokenType<'static>, TokenType<'a>>(KEYWORDS[17]);
        if self.match_token(class_type) {
            self.classDefinition(false);
        } else if self.match_token(foreign_type) {
            self.consume(class_type, String::from("Expect 'class' after 'foreign'."));
            self.classDefinition(true);
        } else if self.match_token(import_type) {
            self.import();
        } else if self.match_token(var_type) {
            self.variableDefinition();
        } else {
            self.statement();
        }
    }

    /// Compiles a class definition. Assumes the "class" token has already been
    /// consumed (along with a possibly preceding "foreign" token).
    pub fn classDefinition(&self, is_foreign:bool) {
        // 1. Create a variable to store the class in.
        // Todo
        // 2. Create shared class name value
        // Todo
        // 3. Create class name string to track method duplicates
        // Todo
        // 4. Make a string constant for the name.
        // Todo
        // 5. Load the superclass (if there is one) OR Implicitly inherit from Object (if there's no superclass)
    }

    /// Compiles a "var" variable definition statement.
    pub unsafe fn variableDefinition(&'a mut self) {
        // Grab its name, but don't declare it yet. A (local) variable shouldn't be
        // in scope in its own initializer.
        let name_type = transmute::<TokenType<'static>, TokenType<'a>>(GRAMMAR_RULES[35]);
        self.consume(name_type, String::from("Expect variable name."));
        let name_token = self.lexer.previous;

        let assign_type = transmute::<TokenType<'static>, TokenType<'a>>(GRAMMAR_RULES[26]);

        // Compile the initializer.
        if self.match_token(assign_type) {
            self.ignore_new_lines();
            self.expression();
        } else {
            // Default initialize it to null.
            Grammar::Null(transmute::<&mut Compiler<'a>, &'a  mut Compiler<'a>>(self), false);
        }
        // Now put it in scope.
        let symbol = self.declare_variable(name_token);
        self.define_variable(symbol);
    }   

    pub fn import(&self){}

    pub fn statement(&self){

    }

    /// Stores a variable with the previously defined symbol in the current scope.
    pub fn define_variable(&self, symbol: i32) {
        // Store the variable. If it's a local, the result of the initializer is
		// in the correct slot on the stack already so we're done.
        if self.scope_depth >= 0 {
            return;
        }

        // It's a module-level variable, so store the value in the module slot and
        // then discard the temporary for the initializer.
        // STORE_MODULE_VAR(symbol)
        // POP STACK
    }

    pub unsafe fn declare_variable(&self, token:Option<Tokens<'a>>) -> i32 {
        let mut _token = self.lexer.previous.unwrap();
        if let Some(tok) = token {
            _token = tok;
        }
        let mut length = 0;

        match _token {
            Tokens::Token(_, s, _) => {
                length = s.len();
            }
        }
        
        if length > MAX_VARIABLE_NAME {
			self.error(String::from("Variable name cannot be longer than ${MAX_VARIABLE_NAME} characters."));
		}

        // add local variable to stack

        0
    }

    fn push_scope(&mut self){
        self.scope_depth += 1;
    }
}