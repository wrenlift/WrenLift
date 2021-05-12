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
use alloc::vec::Vec;
use std::str;
use utils;

use scanner::{Token, TokenType, Keywords};

pub const MAX_INTERPOLATION_NESTING: i8 = 8;

#[derive(Debug, Clone)]
pub struct Parser<'a> {
    source: Peekable<Chars<'a>>,
    has_error: bool,
    /// Whether compile errors should be printed to stderr or discarded.
    print_errors: bool,
    /// If subsequent newline tokens should be discarded.
    skip_new_lines: bool,
    current_line: u32,

    /// The most recently consumed/advanced token.
    previous: Option<Token>,
    /// The most recently lexed token.
    current: Option<Token>,
    next: Option<Token>,
    /// The current character being lexed in [source].
    current_char: char,
    current_offset: u32,
    token_offset: u32,
    
    parens: Vec<u32>,
    num_parens: u32,
    module_name: &'a str,
    /// The beginning of the currently-being-lexed token in [source].
    token_start: &'a str,


    tokens: Vec<Token>,
}


impl<'a> Parser<'a> {
    pub fn new(_source: &'a str, modulename: Option<&'a str>) -> Self {
        let _module_name = "<unknown>".to_string();
        if let Some(name) = modulename {
            _module_name = name.to_string();
        }

      
    
        Self {
            source: _source.chars().peekable(),
            module_name: _module_name,
            current: Some(Token::new()),
            has_error: false,
            print_errors: false,
            skip_new_lines: false,
            current_line: 1,
            previous: None,
            token_start: _source,
            current_offset: 0,
            current_char: 0,
            token_offset: 0,
            parens: Vec::new(),
            tokens: Vec::new(),
            num_parens: 0,
        }
    }

    fn print_error(&self, line: usize, label: &'a str, formatted_str: String){
        println!("line {} : {} {}", line, label, formatted_str);
    }

    fn lex_error(&self, message: String) {
        self.print_error(self.current_line, 'Error', message);
    }

    /// Returns the current character the parser is sitting on.
    fn peek_char(&self) -> Option<&'a char> {
        self.source.peek()
    }

    fn peek_next_char(&self) -> Option<&'a char> {
        if let Some(c) = self.peek_char() {
            if c == &'\0' {
                return Some(c)
            }
        }

        self.source.next()
    }

    /// Advances the parser forward one character.
    fn next_char(&mut self) -> Option<&'a char> {
        let c = self.peek_char();
        self.current_offset += 1;
        if let Some(&"\n") = c {
            self.current_line += 1;
        }
        self.source.advance_by(self.current_offset);
        self.source.next()
    }

    /// If the current character is [c], consumes it and returns `true`.
    fn match_char(&self, c: char) -> bool {
        if let Some(&c) = self.peek_char() {
            false
        } else {
            self.next_char();
            true
        }
    }

    /// Sets the parser's current token to the given [type] and current character
    /// range.
    fn make_token(&mut self, t:TokenType) {
        let tok = Token {
            token_type: t,
            start: self.token_start[self.current_offset..self.token_offset],
            length: self.token_start.len(),
            line: self.current_line,
        }
        self.current = Some(tok);
        
        self.tokens.push(tok);

        // Make line tokens appear on the line containing the "\n".
        if TokenType::Line == t {
            self.current.line -= 1;
        }
    }

    /// If the current character is [c], then consumes it and makes a token of type
    /// [two]. Otherwise makes a token of type [one].
    fn two_char_token(&self, c: char, two: TokenType, one: TokenType) {
        if self.match_char(c) {
            self.make_token(two);
        } else {
            self.make_token(one);
        }
    }

    /// Skips the rest of the current line.
    fn skip_line_comment(&self) {
        while self.peek_char() != Some(&'\n') && self.peek_char() != Some(&'\0') {
            self.next_char();
        }
    }
    /// Skips the rest of a block comment.
    fn skip_block_comment(&self) {
        let mut nesting = 1;
        while nesting > 0 {
            if let Some(c) = self.peek_char() {
                if *c == 0x05 {
                    return;
                }
                self.lex_error("Unterminated block comment.".to_owned());
            }

            if let Some(&'/') = self.peek_char() && Some(&'*') = self.peek_next_char() {
                self.next_char();
                self.next_char();
                nesting += 1;
                continue;
            }

            if let Some(&'*') = self.peek_char() && Some(&'/') = self.peek_next_char() {
                self.next_char();
                self.next_char();
                nesting -= 1;
                continue;
            }

            /// Regular comment character.
            self.next_char();
        }
    }

    /// Reads the next character, which should be a hex digit (0-9, a-f, or A-F) and
    /// returns its numeric value. If the character isn't a hex digit, returns -1.
    fn read_hex_digit(&mut self) -> i8 {
        let c = self.next_char();
        if let Some(_c) =  c {
            if *_c >= '0' && _c <= '9' {
                *_c - '0'; 
            }
            if *_c >= 'a' && _c <= 'f' {
                *_c - 'a' + 10; 
            }
            if *_c >= 'A' && _c <= 'F' {
                *_c - 'A' + 10; 
            }
        }
        // Don't consume it if it isn't expected. Keeps us from reading past the end
		// of an unterminated string.
        self.current_offset -= 1; 
        return -1;
    }

    fn read_hex_number(self) {
        // Skip past the `x` used to denote a hexadecimal literal.
        self.next_char();
        // Iterate over all the valid hexadecimal digits found.
        while self.read_hex_digit() != -1 {
            continue;
        }

        self.make_number();
    }

    fn make_number(&self) {
        self.make_token(TokenType::Number);
    }

    fn read_number(&self){
        if let Some(c) = self.peek_char() {
            while is_digit(*c) {
                self.next_char();
            }
        }
        
        // See if it has a floating point. Make sure there is a digit after the "."
		// so we don't get confused by method calls on number literals.
        if let Some(c) = self.peek_char() {
            if let Some(cc) = self.peek_next_char() {
                if *c == '.' && is_digit(*cc) {
                    self.next_char();
                    while is_digit(*cc) {
                        self.next_char();
                    }
                }
            }
        }

        if let Some(cc) = self.peek_next_char() {
            // See if the number is in scientific notation.
            if self.match_char('e') && is_digit(*cc) {
                // Allow a single positive/negative exponent symbol.
                if !self.match_char('+') {
                    self.match_char('-');
                }
                if let Some(c) = self.peek_char() {
                    if is_digit(*c) {
                        self.lex_error("Unterminated scientific notation.".to_owned());
                    }

                    while is_digit(*c) {
                        self.next_char();
                    }
                }
                
            }
        }


        self.make_number();   
    }

    /// Finishes lexing an identifier. Handles reserved words.
    fn read_name(&self, tokenType: &mut TokenType, first_char: char) {
        if let Some(c) = self.peek_char() {
            while is_name(*c) || is_digit(c) {
                self.next_char();
            }
        }

        let i = 0;
        while true {
            match Keywords[i].identifier {
                Some(s) => {
                    let kwd = Keywords[i];
                    let word = self.token_start[self.current_offset..self.token_offfset];

                    if word == kwd.identifier {
                        tokenType = Keywords[i].tokenType;
                        break;
                    }
                },
                None => break;
            }

            i += 1;
        }

        self.make_token(tokenType);
    }

    /// Reads [digits] hex digits in a string literal and returns their number value.
    fn read_hex_escape(&mut self, digits: usize, description: &'a str) -> u8 {
        let mut _value = 0;
        for i in 0..digits {
            if let Some(c) = self.peek_char() {
                if *c == '"' || *c == "\0" {
                    self.lex_error(format!("Incomplete {} escape sequence.", description));
                    // Don't consume it if it isn't expected. Keeps us from reading past the
                    // end of an unterminated string.
                    self.current_offset -= 1; 
                    break;
                }
            }
        
            let digit = self.read_hex_digit();
            if i == -1 {
                self.lex_error(format!("Invalid {} escape sequence.", description));
                break;
            }
            _value = (_value * 16) | digit;
        }

        _value as u8
    }

    /// Reads a hex digit Unicode escape sequence in a string literal.
    fn read_unicode_escape(&self, string: &mut String, length: usize) {
        let mut _value = self.read_hex_escape(length, "Unicode");

        // Grow the buffer enough for the encoded result.
        let num_bytes = utils::wren_utf8_encode_num_bytes(_value);
        if num_bytes != 0 {
            // let mut v: Vec<u8> = Vec::new();
            for i in 0...num_bytes {
                string.push(0);
            }
             //= std::str::from_utf8_unchecked(&v);
        }
    }

    // Finishes lexing a string literal.
    fn read_string(&mut self) {
        let mut string = String::new();
        let mut tokenType = TokenType::String;

        while true {
            if let Some(c) = self.next_char() {
                if *c == '"' {
                    break;
                }

                if *c == '\0' {
                    self.lex_error("Unterminated string.".to_owned());
                    // Don't consume it if it isn't expected. Keeps us from reading past the
                    // end of an unterminated string.
                    self.current_offset -= 1;
                    break;
                }

                if *c == '%' {
                    if self.num_parens < MAX_INTERPOLATION_NESTING {
                        // TODO: Allow format string.
                        if let Some(_c) = self.next_char() {
                            if *_c != '(' {
                                self.lex_error("Expect '(' after '%%'".to_owned());
                            }
                        }
                        let idx = self.num_parens + 1;
                        self.parens[idx] = 1;
                        tokenType = TokenType::Interpolation;
                        break; 
                    }

                    self.lex_error(format!("Interpolation may only nest {} levels deep.", MAX_INTERPOLATION_NESTING));
                }
               
                if *c == '\\' {
                    match self.next_char() {
                        Some(&'"') => string.push('"');
                        Some(&'\\') => string.push('\\');
                        Some(&'%') => string.push('%');
                        Some(&'0') => string.push('\0');
                        Some(&'a') => string.push('\a');
                        Some(&'b') => string.push('\b');
                        Some(&'e') => string.push('\33');
                        Some(&'f') => string.push('\f');
                        Some(&'n') => string.push('\n');
                        Some(&'r') => string.push('\r');
                        Some(&'t') => string.push('\t');
                        Some(&'u') => self.read_unicode_escape(&string, 4);
                        Some(&'U') => self.read_unicode_escape(&string, 8);
                        Some(&'v') => string.push('\v');
                        Some(&'x') => {
                            string.push(self.read_hex_escape(2, "byte") as char);
                        }
                        None => {
                            if let Some(cc) = self.source.nth[self.current_offset - 1] {
                                self.lex_error(format!("Invalid escape character '{}'.", cc));
                            }
                            
                        }
                    }
                } else {
                    string.push(c);
                }
            }
        }

        self.make_token(tokenType);
    }

    fn read_raw_string(&mut self) {
        let mut string = String::new();
        let mut tokenType = TokenType::String;

        //consume the second and third "
        self.next_char();
        self.next_char();

        let mut skip_start = 0 as i32;
        let mut first_new_line = -1 as i32;


        let mut skip_end = -1 as i32;
        let mut last_new_line = -1 as i32;

        loop {
            let mut c = self.next_char();
            let mut c1 = self.peek_char();
            let mut c2 = self.peek_next_char();

            match c {
                Some(&'\n') => {
                    lastNewline = string.len();
                    skip_end = last_new_line;
                    first_new_line = if first_new_line == -1 { string.len() } else  { first_new_line };
                }
                None => {}
            }

            if c == Some(&'"') && c1 == Some(&'"') && c2 == Some(&'"') {
               break;
            }

            let mut is_white_space: bool = c == Some(&' ') || c == Some(&'\t');
            skip_end = if c == Some(&'\n') || is_white_space  { skip_end } else {-1};
            // If we haven't seen a newline or other character yet, 
            // and still seeing whitespace, count the characters 
            // as skippable till we know otherwise

            let mut skippable: bool = skip_start != -1 && is_white_space && first_new_line == -1;
            skip_start = if skippable {string.len() + 1} else {skip_start};

            // We've counted leading whitespace till we hit something else, 
            // but it's not a newline, so we reset skipStart since we need these characters
            if first_new_line == -1 && !is_white_space && c != Some(&'\n') {
                skip_start = -1;
            }

            if c == Some(&'\0') && c1 == Some(&'\0') && c2 == Some(&'\0') {
                self.lex_error("Unterminated raw string.".to_owned());
                // Don't consume it if it isn't expected. Keeps us from reading past the
                // end of an unterminated string.
                self.current_offset -= 1;
                break;
            }
            string.push(c.unwrap());
        }

        //consume the second and third "
        self.next_char();
        self.next_char();
    }

    /// Lex the next token and store it in [parser.next].
    fn next_token(&mut self) {
        self.previous = self.current;
        self.current = self.next;

        // If we are out of tokens, don't try to tokenize any more. We *do* still
        // copy the TOKEN_EOF to previous so that code that expects it to be consumed
        // will still work.
        if let Some(tok) = self.next {
            if tok.token_type == TokenType::Eof {
                return
            }
        }
        if let Some(tok) = self.current {
            if tok.token_type == TokenType::Eof {
                return
            }
        }

        if let Some(_c) =  self.peek_char() {
            while *_c != '\0' {
                self.token_offset = self.current_offset;

                match self.next_char() {
                    Some(&'(') => {
                        // If we are inside an interpolated expression, count the unmatched "(".
                        if self.num_parens > 0 {
                            self.parens[self.num_parens - 1] += 1;
                            self.make_token(TokenType::LeftParen);
                            return;
                        }
                    }
                    Some(&')') => {
                        // If we are inside an interpolated expression, count the ")".
                        self.parens[self.num_parens - 1] -= 1;
                        if self.num_parens > 0 && self.parens[self.num_parens - 1] == 0 {
                            // This is the final ")", so the interpolation expression has ended.
                            // This ")" now begins the next section of the template string.
                            self.num_parens -= 1;
                            self.read_string();
                            return;
                        }
                        self.make_token(TokenType::RightParen);
                        return;
                    }
                    Some(&'[') => {self.make_token(TokenType::LeftBracket); return;}
                    Some(&']') => {self.make_token(TokenType::RightBracket); return;}
                    Some(&'{') => {self.make_token(TokenType::LeftBrace); return;}
                    Some(&'}') => {self.make_token(TokenType::RightBrace); return;}
                    Some(&':') => {self.make_token(TokenType::Colon); return;}
                    Some(&',') => {self.make_token(TokenType::Comma); return;}
                    Some(&'*') => {self.make_token(TokenType::Star); return;}
                    Some(&'%') => {self.make_token(TokenType::Percent); return;}
                    Some(&'#') => {
                        // Ignore shebang on the first line.
                        if self.current_line == 1 && self.peek_char() == Some(&'!') && self.peek_next_char() == Some(&'/') {
                            self.skip_line_comment(); 
                            continue;
                        } else {
                            self.make_token(TokenType::Hash);
                            return;
                        }
                    }
                    Some(&'^') => {self.make_token(TokenType::Caret); return;}
                    Some(&'+') => {self.make_token(TokenType::Plus); return;}
                    Some(&'-') => {self.make_token(TokenType::Minus); return;}
                    Some(&'~') => {self.make_token(TokenType::Tilde); return;}
                    Some(&'?') => {self.make_token(TokenType::Question); return;}

                    Some(&'|') => {self.two_char_token('|', TokenType::PipePipe, TokenType::Pipe); return;}
                    Some(&'&') => {self.two_char_token('&', TokenType::AmpAmp, TokenType::Amp); return;}
                    Some(&'=') => {self.two_char_token('=', TokenType::EqEq, TokenType::Eq); return;}
                    Some(&'!') => {self.two_char_token('=', TokenType::BangEq, TokenType::Bang); return;}
                    

                    Some(&'.') => {

                        if self.match_char('.') {
                            self.two_char_token('.', TokenType::DotDotDot, TokenType::DotDot) return;
                        }
                        self.make_token(TokenType::Dot);
                        return;
                    } 

                    Some(&'/') => {
                        if self.match_char('/') {
                            self.skip_line_comment();
                            continue;
                        }
                        if self.match_char('*') {
                            self.skip_block_comment();
                            continue;
                        }

                        self.make_token(TokenType::Slash);
                        return;
                    }

                    Some(&'<') => {
                        if self.match_char('<') {
                            self.make_token(TokenType::LtLt);
                        }
                        else self.match_char('=') {
                            self.two_char_token('=', TokenType::LtEq, TokenType::Lt);
                        }
                        return;
                    }

                    Some(&'\n') => {
                        self.make_token(TokenType::Line);
                        return;
                    }
 
                    Some(&' ') | Some(&'\r') | Some(&'\t') => {
                        loop {
                            if let Some(space) = self.peek_char() {
                                if *space == ' ' || *space == '\r' || *space == '\t' {
                                    self.next_char();
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } 
                    }
                    Some(&'"') => {
                        if let Some(s) = self.peek_char() {
                            if let Some(ss) = self.peek_next_char() {
                                if *s == '"' && *ss == '"' {
                                    self.read_raw_string();
                                    return;
                                }
                            }
                        }
                        self.read_string();
                        return;
                    }
                    Some(&'_') => {
                        match self.peek_char() {
                            Some(n) => {
                                if *n == '_'  {
                                    self.read_name(TokenType::StaticField, *n);
                                } else {
                                    self.read_name(TokenType::Field, *n);
                                }
                            }
                            None => {}
                        }

                        return;
                    }

                    Some(&'0') => {
                        if let Some(hx) == self.peek_char() {
                            if *hx == "x" {
                                self.read_hex_number();
                                return;
                            }
                        }
                        self.read_number();
                        return;
                    }

                    other => {
                        if let Some(x) == other {
                            if is_name(*x) {
                                self.read_name(TokenType::Name, *x);
                            } else if is_digit(*x) {
                                self.read_number();
                            } else {
                                if *x >= 32 && *x <= 126 {
                                    self.lex_error(format!("Invalid character '{}'", *x));
                                } else {
                                    // Don't show non-ASCII values since we didn't UTF-8 decode the
                                    // bytes. Since there are no non-ASCII byte values that are
                                    // meaningful code units in Wren, the lexer works on raw bytes,
                                    // even though the source code and console output are UTF-8.
                                    self.lex_error(format!("Invalid byte 0x{}", *x as u8));
                                }
                                if let Some(tok) = self.next {
                                    let mut token = tok;
                                    token.token_type = TokenType::Error;
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
        self.token_start = self.source.nth(self.current_offset);
        self.make_token(TokenType::Eof); 
    }
}

/// Returns true if [c] is a valid (non-initial) identifier character.
fn is_name(c:char) -> bool {
    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
}

fn is_digit(c:char) -> bool {
    c >= '0' && c <= '9'
}