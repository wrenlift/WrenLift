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

use std::iter::Peekable;
use std::str::Chars;
use std::char;
// use alloc::vec::Vec;
use std::str;
use crate::compiler::utils;
use std::cell::Cell;

use crate::compiler::scanner::{Token, TokenType, Keywords, SourceBuffer};

pub const MAX_INTERPOLATION_NESTING: usize = 8;


fn print_error(line: usize, label: &str, formatted_str: String){
    println!("line {} : {} {}", line, label, formatted_str);
}

pub fn lex_error(current_line:usize, message: String) {
    print_error(current_line, "Error", message);
}



// #[derive(Debug, Clone)]
pub struct Parser <'a>{
    pub source: SourceBuffer<'a>,
    pub has_error: bool,
    /// Whether compile errors should be printed to stderr or discarded.
    pub print_errors: bool,
    /// If subsequent newline tokens should be discarded.
    pub skip_new_lines: bool,
    pub current_line: usize,

    /// The most recently consumed/advanced token.
    pub previous: Option<Token<'a>>,
    /// The most recently lexed token.
    pub current: Option<Token<'a>>,
    pub next: Option<Token<'a>>,
    /// The current character being lexed in [source].
    pub current_char: char,
    pub current_offset: usize,
    pub token_offset: usize,
    pub tokens: Vec<Token<'a>>,
    pub parens: Vec<u32>,
    pub num_parens: usize,
    pub module_name: &'a str,
    /// The beginning of the currently-being-lexed token in [source].
    pub token_start:  &'a str,
}


impl<'a> Parser<'a> {
    pub fn new(_source: &'a str, modulename: Option<&'a str>) -> Self {
        let mut _module_name: &'a str = "<unknown>";
        if let Some(name) = modulename {
            _module_name = name;
        }

        let toks:Vec<Token<'a>> = Vec::new();
      
    
        Self {
            source: SourceBuffer::new(_source),
            module_name: _module_name,
            current: None,
            next: None,
            has_error: false,
            print_errors: false,
            skip_new_lines: false,
            current_line: 1,
            previous: None,
            token_start: _source,
            current_offset: 0,
            current_char: '\0',
            token_offset: 0,
            parens: Vec::new(),
            tokens: toks,
            num_parens: 0,
        
        }
    }

    pub fn as_mut(&'a mut self) -> &'a mut Self {
        self
    }

    

    /// Sets the parser's current token to the given [type] and current character
    /// range.
    pub fn make_token(&self, t:TokenType) {
        let off: &'a str = &self.token_start[self.source.current_offset..self.token_offset];
        let tok:Token<'a> = Token {
            token_type: Some(t),
            start: off,
            length: self.token_start.len(),
            line: self.source.current_line,
        };
        let mut _cur = self.current;
        _cur = Some(tok);
        
       // self.tokens.push(tok);

        // Make line tokens appear on the line containing the "\n".
        if TokenType::Line == t {
            self.current.unwrap().line -= 1;
        }
    }

    /// If the current character is [c], then consumes it and makes a token of type
    /// [two]. Otherwise makes a token of type [one].
    fn two_char_token(&'a self, c: char, two: TokenType, one: TokenType) {
        if self.source.match_char(c) {
            self.make_token(two);
            self.source.next_char();
        } else {
            self.make_token(one);
        }
    }



   

    pub fn make_number(&'a mut self) {
        self.make_token(TokenType::Number);
    }

    

    /// Finishes lexing an identifier. Handles reserved words.
    fn read_name(&'a self, tokenType: TokenType, first_char: char) {
        let mut _tokenType = tokenType;
        if let Some(c) = self.source.peek_char() {
            while is_name(c) || is_digit(c) {
                self.source.next_char();
            }
        }

        let mut i = 0;
        loop {
            match Keywords[i].identifier {
                Some(_) => {
                    let kwd = &Keywords[i];
                    let word = str::from_utf8(&(self.token_start.as_bytes()[self.current_offset..self.token_offset]));

                    if let Some(w) = kwd.identifier {
                        if w == String::from(word.unwrap()) {
                            _tokenType = Keywords[i].tokenType;
                            break;
                        }
                    }
                }
                None => {break;}
            }

            i += 1;
        }

        self.make_token(_tokenType);
    }

 

    /// Lex the next token and store it in [parser.next].
    pub fn next_token(&'a mut self) {
        self.previous = self.current.clone();
        self.current = self.next.clone();

        // If we are out of tokens, don't try to tokenize any more. We *do* still
        // copy the TOKEN_EOF to previous so that code that expects it to be consumed
        // will still work.
        if let Some(tok) = self.next {
            if tok.token_type == Some(TokenType::Eof) {
                return;
            }
        }
        if let Some(tok) = self.current {
            if tok.token_type == Some(TokenType::Eof) {
                return;
            }
        }

        if let Some(_c) =  self.source.peek_char() {
            while _c != '\0' {
                self.token_offset = self.current_offset;

                match self.source.next_char() {
                    Some('(') => {
                        // If we are inside an interpolated expression, count the unmatched "(".
                        if self.num_parens > 0 {
                            self.parens[(self.num_parens - 1) as usize] += 1;
                            self.make_token(TokenType::LeftParen);
                            return;
                        }
                    }
                    Some(')') => {
                        // If we are inside an interpolated expression, count the ")".
                        self.parens[(self.num_parens - 1) as usize] -= 1;
                        if self.num_parens > 0 && self.parens[(self.num_parens - 1) as usize] == 0 {
                            // This is the final ")", so the interpolation expression has ended.
                            // This ")" now begins the next section of the template string.
                            self.num_parens -= 1;
                            self.source.read_string(*self);
                            return;
                        }
                        self.make_token(TokenType::RightParen);
                        return;
                    }
                    Some('[') => {self.make_token(TokenType::LeftBracket); return;}
                    Some(']') => {self.make_token(TokenType::RightBracket); return;}
                    Some('{') => {self.make_token(TokenType::LeftBrace); return;}
                    Some('}') => {self.make_token(TokenType::RightBrace); return;}
                    Some(':') => {self.make_token(TokenType::Colon); return;}
                    Some(',') => {self.make_token(TokenType::Comma); return;}
                    Some('*') => {self.make_token(TokenType::Star); return;}
                    Some('%') => {self.make_token(TokenType::Percent); return;}
                    Some('#') => {
                        // Ignore shebang on the first line.
                        if self.source.current_line == 1 && self.source.peek_char() == Some('!') && self.source.peek_next_char() == Some('/') {
                            self.source.skip_line_comment(); 
                            continue;
                        } else {
                            self.make_token(TokenType::Hash);
                            return;
                        }
                    }
                    Some('^') => {self.make_token(TokenType::Caret); return;}
                    Some('+') => {self.make_token(TokenType::Plus); return;}
                    Some('-') => {self.make_token(TokenType::Minus); return;}
                    Some('~') => {self.make_token(TokenType::Tilde); return;}
                    Some('?') => {self.make_token(TokenType::Question); return;}

                    Some('|') => {self.two_char_token('|', TokenType::PipePipe, TokenType::Pipe); return;}
                    Some('&') => {self.two_char_token('&', TokenType::AmpAmp, TokenType::Amp); return;}
                    Some('=') => {self.two_char_token('=', TokenType::EqEq, TokenType::Eq); return;}
                    Some('!') => {self.two_char_token('=', TokenType::BangEq, TokenType::Bang); return;}
                    

                    Some('.') => {

                        if self.source.match_char('.') {
                            self.two_char_token('.', TokenType::DotDotDot, TokenType::DotDot); return;
                        }
                        self.make_token(TokenType::Dot);
                        return;
                    } 

                    Some('/') => {
                        if self.source.match_char('/') {
                            self.source.skip_line_comment();
                            continue;
                        }
                        if self.source.match_char('*') {
                            self.source.skip_block_comment();
                            continue;
                        }

                        self.make_token(TokenType::Slash);
                        return;
                    }

                    Some('<') => {
                        if self.source.match_char('<') {
                            self.make_token(TokenType::LtLt);
                        }
                        else if self.source.match_char('=') {
                            self.two_char_token('=', TokenType::LtEq, TokenType::Lt);
                        }
                        return;
                    }

                    Some('\n') => {
                        self.make_token(TokenType::Line);
                        return;
                    }
 
                    Some(' ') | Some('\r') | Some('\t') => {
                        loop {
                            if let Some(space) = self.source.peek_char() {
                                if space == ' ' || space == '\r' || space == '\t' {
                                    self.source.next_char();
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } 
                    }
                    Some('"') => {
                        if let Some(s) = self.source.peek_char() {
                            if let Some(ss) = self.source.peek_next_char() {
                                if s == '"' && ss == '"' {
                                    self.source.read_raw_string();
                                    return;
                                }
                            }
                        }
                        self.source.read_string(*self);
                        return;
                    }
                    Some('_') => {
                        match self.source.peek_char() {
                            Some(n) => {
                                if n == '_'  {
                                    self.read_name(TokenType::StaticField, n);
                                } else {
                                    self.read_name(TokenType::Field, n);
                                }
                            }
                            None => {}
                        }

                        return;
                    }

                    Some('0') => {
                        if let Some(hx) = self.source.peek_char() {
                            if hx == 'x' {
                                self.source.read_hex_number(&mut self);
                                return;
                            }
                        }
                        self.source.read_number(&mut self);
                        return;
                    }

                    other => {
                        if let Some(x) = other {
                            if is_name(x) {
                                self.read_name(TokenType::Name, x);
                            } else if is_digit(x) {
                                self.source.read_number(&mut self);
                            } else {
                                if x as u8 >= 32 && x as u8 <= 126 {
                                    lex_error(self.current_line, format!("Invalid character '{}'", x));
                                } else {
                                    // Don't show non-ASCII values since we didn't UTF-8 decode the
                                    // bytes. Since there are no non-ASCII byte values that are
                                    // meaningful code units in Wren, the lexer works on raw bytes,
                                    // even though the source code and console output are UTF-8.
                                    lex_error(self.current_line, format!("Invalid byte 0x{}", x as u8));
                                }
                                if let Some(tok) = self.next {
                                    let mut token = tok;
                                    token.token_type = Some(TokenType::Error);
                                    token.length = 0;

                                    self.next = Some(token);
                                }
                            }
                            return;
                        }
                    }

                }
            }
        }

        // If we get here, we're out of source, so just make EOF tokens.
        // if let Some(t) = self.source.nth(self.current_offset) {
        //     let ref _tok = self.token_start;
        //     *_tok = t.to_string().as_str();
        // }

        self.make_token(TokenType::Eof); 
        
    }

    /// Returns the type of the current token.
    pub fn peek(&self) -> Option<TokenType> {
        if let Some(tok) = self.current {
            tok.token_type
        } else {
            None
        }
    }

    /// Returns the type of the next token.
    pub fn peek_next(&self) -> Option<TokenType> {
        if let Some(tok) = self.next {
            tok.token_type
        } else {
            None
        }
    }

    /// Consumes the current token if its type is [expected]. Returns true if a
    /// token was consumed.
    pub fn match_token (&'a mut self, expected: TokenType) -> bool {
        if let Some(t) = self.current {
            if t.token_type != Some(expected) {
                false
            } else {
                self.next_token();
                true
            }
        } else {
            false
        }
        
    }

    /// Consumes the current token. Emits an error if its type is not [expected].
    pub fn consume(&'a self, expected: TokenType, error_message: &str) {
        self.next_token();
        if let Some(prev) = self.previous {
            if prev.token_type != Some(expected) {
                self.error(error_message);

                // If the next token is the one we want, assume the current one is just a
                // spurious error and discard it to minimize the number of cascaded errors.
                if let Some(cur) = self.current {
                    if cur.token_type == Some(expected) {
                        self.next_token();
                    }
                }
            }
        }
    }

    /// Matches one or more newlines. Returns true if at least one was found.
    pub fn match_line(&'a mut self) -> bool {
        if !self.match_token(TokenType::Line) { false} else {
            while self.match_token(TokenType::Line) {

            }
    
            true
        } 
    }

    /// Discards any newlines starting at the current token.
    pub fn ignore_new_lines(&'a self) {
        self.match_line();
    }

    /// Consumes the current token. Emits an error if it is not a newline. Then
    /// discards any duplicate newlines following it.
    pub fn consume_line(&'a  self, error_message: &'a str) {
        self.consume(TokenType::Line, error_message);
        self.match_line(); //ignore line
    }


    pub fn allow_before_dot(&'a mut self) {
        match self.peek() {
            Some(TokenType::Line) => {
                match self.peek_next() {
                    Some(TokenType::Dot) => {
                        self.next_token();
                    }
                    None => {}
                    _ => {}
                }
            }
            None => {}
            _ => {}
        }
    }

    pub fn print(&'a self) {
        println!("{:#?}", &self.tokens);
    }

    fn error(&self, message: &str) {

    }
}

/// Returns true if [c] is a valid (non-initial) identifier character.
pub fn is_name(c:char) -> bool {
    c.is_alphabetic() || c == '_'
}

pub fn is_digit(c:char) -> bool {
    c.is_numeric()
}