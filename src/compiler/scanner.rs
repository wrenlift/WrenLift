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
// use crate::lazy_static;
use crate::compiler::wren_parser::{Parser, lex_error, is_digit, is_name, MAX_INTERPOLATION_NESTING};
use crate::compiler::grammar;
use std::fmt::{Display, Formatter, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::iter::Peekable;
use std::str::Chars;
use std::char;
use crate::compiler::utils::{ wren_utf8_encode_num_bytes };


const MAX_PARAMETERS: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precedence {
    PrecNone = 0,
    PrecLowest,
    PrecAssignment,
    PrecConditional,
    PrecLogicalOr,
    PrecLogicalAnd,
    PrecEquality,
    PrecIs,
    PrecComparison,
    PrecBitwiseOr,
    PrecBitwiseXor,
    PrecBitwiseAnd,
    PrecBitwiseShift,
    PrecRange,
    PrecTerm,
    PrecFactor,
    PrecUnary,
    PrecCall,
    PrecPrimary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    LeftParen(GrammarRule),
    RightParen(GrammarRule),
    LeftBracket(GrammarRule),
    RightBracket(GrammarRule),
    LeftBrace(GrammarRule),
    RightBrace( GrammarRule),
    Colon(GrammarRule),
    Dot(GrammarRule),
    DotDot(GrammarRule),
    DotDotDot(GrammarRule),
    Comma(GrammarRule),
    Star(GrammarRule),
    Slash(GrammarRule),
    Percent(GrammarRule),
    Plus(GrammarRule),
    Minus(GrammarRule),
    LtLt(GrammarRule),
    GtGt(GrammarRule),
    Pipe(GrammarRule),
    PipePipe(GrammarRule),
    Caret(GrammarRule),
    Amp(GrammarRule),
    AmpAmp(GrammarRule),
    Bang(GrammarRule),
    Tilde(GrammarRule),
    Question(GrammarRule),
    Eq(GrammarRule),
    Lt(GrammarRule),
    Gt(GrammarRule),
    LtEq(GrammarRule),
    GtEq(GrammarRule),
    EqEq(GrammarRule),
    BangEq(GrammarRule),
    Hash(GrammarRule),
  
    Break(&'static str, GrammarRule),
    Class(&'static str, GrammarRule),
    Construct(&'static str, GrammarRule),
    Else(&'static str, GrammarRule),
    False(&'static str, GrammarRule),
    For(&'static str, GrammarRule),
    Foreign(&'static str, GrammarRule),
    If(&'static str, GrammarRule),
    Import(&'static str, GrammarRule),
    In(&'static str, GrammarRule),
    Is(&'static str, GrammarRule),
    Null(&'static str, GrammarRule),
    Return(&'static str, GrammarRule),
    Static(&'static str, GrammarRule),
    Super(&'static str, GrammarRule),
    This(&'static str, GrammarRule),
    True(&'static str, GrammarRule),
    Var(&'static str, GrammarRule),
    While(&'static str, GrammarRule),
  
    Field(GrammarRule),
    StaticField(GrammarRule),
    Name(GrammarRule),
    Number(GrammarRule),
    
    // A string literal without any interpolation or the last section of a
    // string following the last interpolated expression.
    String(GrammarRule),
    
    // A portion of a string literal preceding an interpolated expression. This
    // string:
    //
    //     "a %(b) c %(d) e"
    //
    // is tokenized to:
    //
    //     TokenType::Interpolation "a "
    //     TokenType::Name           b
    //     TokenType::Interpolation " c "
    //     TokenType::Name         d
    //     TokenType::String       " e"
    Interpolation(GrammarRule),
  
    Line(GrammarRule),
  
    Error(GrammarRule),
    Eof(GrammarRule),
}

impl TokenType {
    fn get_kwd(&self) -> &'static str {
        match self {
            TokenType::Eof(_) => {
                return "";
            },
            TokenType::Break(s,_) 
            | TokenType::Class(s,_)
            | TokenType::Construct(s,_)
            | TokenType::Else(s,_)
            | TokenType::False(s,_)
            | TokenType::True(s,_)
            | TokenType::For(s,_)
            | TokenType::Foreign(s,_)
            | TokenType::While(s,_)
            | TokenType::If(s,_)
            | TokenType::Import(s,_)
            | TokenType::Static(s,_)
            | TokenType::Super(s,_)
            | TokenType::In(s,_)
            | TokenType::Is(s,_)
            | TokenType::This(s,_)
            | TokenType::Null(s,_)
            | TokenType::Var(s,_)
             => {
                s
            }
            _ => {
               return  "";
            }
        }
    }
}


type GrammarFn = fn(parser: Parser, canAssign: bool);
type SignatureFn = fn(parser: Parser, signature: &'static mut Signature<'static>) -> Option<&'static mut Signature<'static>>;




#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarRule {
    prefix: Option<GrammarFn>,
    infix: Option<GrammarFn>,
    method: Option<SignatureFn>,
    precedence: Option<Precedence>,
    name: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub enum Tokens<'a> {
    Token(
        TokenType,
        &'a str,
        usize
    )
}



lazy_static! {
pub static ref KEYWORDS: [TokenType; 20] = [
    TokenType::Break("break", GrammarRule::unused()),
    TokenType::Class("class", GrammarRule::unused()),
    TokenType::Construct("construct", GrammarRule{ name: None, prefix: None, infix: None,  method: Some(grammar::ConstructorSignature), precedence: Some(Precedence::PrecNone) }),
    TokenType::Else("else", GrammarRule::unused()),
    TokenType::False("false", GrammarRule::prefix(Some(grammar::Boolean))),
    TokenType::For("for", GrammarRule::unused()),
    TokenType::Foreign("foreign", GrammarRule::unused()),
    TokenType::If("if", GrammarRule::unused()),
    TokenType::Import("import", GrammarRule::unused()),
    TokenType::In("in", GrammarRule::unused()),
    TokenType::Is("is", GrammarRule::infix_operator(Some(Precedence::PrecIs), "is")),
    TokenType::Null("null", GrammarRule::prefix(Some(grammar::Null))),
    TokenType::Return("return", GrammarRule::unused()),   
    TokenType::Static("static", GrammarRule::unused()),  
    TokenType::Super("super", GrammarRule::prefix(Some(grammar::Super))),    
    TokenType::This("this", GrammarRule::prefix(Some(grammar::This))),     
    TokenType::True("true", GrammarRule::prefix(Some(grammar::Boolean))),      
    TokenType::Var("var", GrammarRule::unused()),       
    TokenType::While("while",GrammarRule::unused()),
    TokenType::Eof(GrammarRule::unused()),
];
pub static ref GRAMMAR_RULES: [TokenType; 63] = [
    TokenType::LeftParen(GrammarRule::prefix(Some(grammar::Grouping))),
    TokenType::RightParen(GrammarRule::unused()),
    TokenType::LeftBracket(GrammarRule { name: None, prefix: Some(grammar::List), infix: Some(grammar::Subscript), method: Some(grammar::SubscriptSignature), precedence: Some(Precedence::PrecCall) }),
    TokenType::RightBracket(GrammarRule::unused()),
    TokenType::LeftBrace(GrammarRule::prefix(Some(grammar::Map))),
    TokenType::RightBrace(GrammarRule::unused()),
    TokenType::Colon(GrammarRule::unused()),
    TokenType::Dot(GrammarRule::infix(Some(Precedence::PrecCall), Some(grammar::Call))),
    TokenType::DotDot(GrammarRule::infix_operator(Some(Precedence::PrecRange), "..")),
    TokenType::DotDotDot(GrammarRule::infix_operator(Some(Precedence::PrecRange), "...")),
    TokenType::Comma(GrammarRule::unused()),
    TokenType::Star(GrammarRule::infix_operator(Some(Precedence::PrecFactor), "*")),
    TokenType::Slash(GrammarRule::infix_operator(Some(Precedence::PrecFactor), "/")),
    TokenType::Percent(GrammarRule::infix_operator(Some(Precedence::PrecFactor), "%")),
    TokenType::Plus(GrammarRule::infix_operator(Some(Precedence::PrecTerm), "+")),
    TokenType::Minus(GrammarRule::operator("-")),
    TokenType::LtLt(GrammarRule::infix_operator(Some(Precedence::PrecBitwiseShift), "<<")),
    TokenType::GtGt(GrammarRule::infix_operator(Some(Precedence::PrecBitwiseShift), ">>")),
    TokenType::Pipe(GrammarRule::infix_operator(Some(Precedence::PrecBitwiseOr), "|")),
    TokenType::PipePipe(GrammarRule::infix(Some(Precedence::PrecLogicalOr), Some(grammar::Or))),
    TokenType::Caret(GrammarRule::infix_operator(Some(Precedence::PrecBitwiseXor), "^")),
    TokenType::Amp(GrammarRule::infix_operator(Some(Precedence::PrecBitwiseAnd), "&")),
    TokenType::AmpAmp(GrammarRule::infix(Some(Precedence::PrecLogicalAnd), Some(grammar::And))),
    TokenType::Bang(GrammarRule::prefix_operator("!")),
    TokenType::Tilde(GrammarRule::prefix_operator("~")),
    TokenType::Question(GrammarRule::infix(Some(Precedence::PrecAssignment), Some(grammar::Conditional))),
    TokenType::Eq(GrammarRule::unused()),
    TokenType::Lt(GrammarRule::infix_operator(Some(Precedence::PrecComparison), "<")),
    TokenType::Gt(GrammarRule::infix_operator(Some(Precedence::PrecComparison), ">")),
    TokenType::LtEq(GrammarRule::infix_operator(Some(Precedence::PrecComparison), "<=")),
    TokenType::GtEq(GrammarRule::infix_operator(Some(Precedence::PrecComparison), "=>")),
    TokenType::EqEq(GrammarRule::infix_operator(Some(Precedence::PrecEquality), "==")),
    TokenType::BangEq(GrammarRule::infix_operator(Some(Precedence::PrecEquality), "!=")),
    TokenType::Field(GrammarRule::prefix(Some(grammar::Field))),
    TokenType::StaticField(GrammarRule::prefix(Some(grammar::StaticField))),
    TokenType::Name(GrammarRule{ name: None, infix: None, prefix: Some(grammar::Name), method: Some(grammar::NamedSignature), precedence: Some(Precedence::PrecNone) }),
    TokenType::Number(GrammarRule::prefix(Some(grammar::Literal))),
    TokenType::String(GrammarRule::prefix(Some(grammar::Literal))),
    TokenType::Interpolation(GrammarRule::prefix(Some(grammar::StringInterpolation))),
    TokenType::Line(GrammarRule::unused()),
    TokenType::Error(GrammarRule::unused()),
    TokenType::Eof(GrammarRule::unused()),
    TokenType::Break("break", GrammarRule::unused()),
    TokenType::Class("class", GrammarRule::unused()),
    TokenType::Construct("construct", GrammarRule{ name: None, prefix: None, infix: None,  method: Some(grammar::ConstructorSignature), precedence: Some(Precedence::PrecNone) }),
    TokenType::Else("else", GrammarRule::unused()),
    TokenType::False("false", GrammarRule::prefix(Some(grammar::Boolean))),
    TokenType::For("for", GrammarRule::unused()),
    TokenType::Foreign("foreign", GrammarRule::unused()),
    TokenType::If("if", GrammarRule::unused()),
    TokenType::Import("import", GrammarRule::unused()),
    TokenType::In("in", GrammarRule::unused()),
    TokenType::Is("is", GrammarRule::infix_operator(Some(Precedence::PrecIs), "is")),
    TokenType::Null("null", GrammarRule::prefix(Some(grammar::Null))),
    TokenType::Return("return", GrammarRule::unused()),   
    TokenType::Static("static", GrammarRule::unused()),  
    TokenType::Super("super", GrammarRule::prefix(Some(grammar::Super))),    
    TokenType::This("this", GrammarRule::prefix(Some(grammar::This))),     
    TokenType::True("true", GrammarRule::prefix(Some(grammar::Boolean))),      
    TokenType::Var("var", GrammarRule::unused()),       
    TokenType::While("while",GrammarRule::unused()),
    TokenType::Eof(GrammarRule::unused()),
    TokenType::Hash(GrammarRule::unused()),
];
}

pub struct Lexer<'a> {
    input: std::iter::Peekable<std::str::CharIndices<'a>>,
    input_copy: &'a str,
    pub current: Option<Tokens<'a>>,
    previous: Option<Tokens<'a>>,
    next: Option<Tokens<'a>>,
    num_parens: usize,
    parens:Vec<usize>,
    token_offset: usize,
    char_offset: usize,
    current_line: usize,
}


impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            input: source.char_indices().peekable(),
            input_copy: source,
            current: None,
            previous: None,
            next: None,
            num_parens: 0,
            parens: Vec::new(),
            token_offset: 0,
            char_offset: 0,
            current_line: 0,
        }
    }

    fn next(&mut self) -> Option<(usize, char)> {
        if let Some(&(_, c)) =  self.input.peek() {
            if c == '\n' {
                self.current_line += 1;
                println!("current_line {}", self.current_line);
            }
            if let Some((pos, next)) = self.input.next() {
                self.char_offset = pos + 1;
                return Some((pos, next));
            }
        }
        None
    }

    fn peek(&mut self) -> Option<&(usize, char)> {
        self.input.peek()
    }

    fn peek_next(&mut self) -> Option<(usize, char)> {
        if let Some(&(p, c)) = self.peek() {
            if c == '\0' {
                return Some((p, '\0'));
            }
        }

        self.input_copy.char_indices().nth(self.char_offset + 1)
    }

    fn skip_line_comment(&mut self) {
        if let Some(&(_, c1)) = self.peek() {
            if c1 != '\n' && c1 != '\0' {
                self.next();
            }
        }
    }

    fn skip_block_comment(&mut self) {
        let mut nesting = 1;

        while nesting > 0 {
            if let Some(&(_, c1)) = self.peek() {
               
                if c1 == '\0' {
                    lex_error(self.current_line, String::from("Unterminated block comment."));
                    return;
                }

                if let Some((_, c2)) = self.peek_next() {
                    if c1 == '/' &&  c2 == '*' {
                        self.next();
                        self.char_offset +=1;
                        self.next();
                        self.char_offset +=1;
                        nesting += 1;
                        continue;
                    }
                }

                if let Some((_, c2)) = self.peek_next() {
                    if c1 == '*' &&  c2 == '/' {
                        self.next();
                        self.char_offset +=1;
                        self.next();
                        self.char_offset +=1;
                        nesting -= 1;
                        continue;
                    }
                }
                
                // Regular comment character.
                self.next();
            }
        }
    }

    fn make_token(&mut self, tok:TokenType) {

        let _tok = Tokens::Token(
            tok,
            &self.input_copy[self.token_offset..self.char_offset],
            self.current_line,
        );
        self.current = Some(_tok);
        

        // Make line tokens appear on the line containing the "\n".
        if let TokenType::Line(_) = tok {
            match self.next {
                Some(Tokens::Token(token_type, chars, line)) => {
                    self.next = Some(Tokens::Token(
                        token_type,
                        chars,
                        line - 1
                    ));
                }
                None => {}
            }
        }
    }


    /// Reads the next character, which should be a hex digit (0-9, a-f, or A-F) and
    /// returns its numeric value. If the character isn't a hex digit, returns -1.
    fn read_hex_digit(&mut self) -> i32 {
        if let Some(&(_, _c)) = self.peek() {
            
            if '\n' == _c {
                self.current_line += 1;
            }
        
            if _c >= '0' && _c <= '9' {
                self.next();
               return (_c as u8 - b'0') as i32;
            }
            if _c >= 'a' && _c <= 'f' {
                self.next();
                return (_c as u8 - b'a' + 10) as i32;
            }
            if _c >= 'A' && _c <= 'F' {
                self.next();
                return (_c as u8 - b'A' + 10) as i32;
            }
        }
    
        // Don't consume it if it isn't expected. Keeps us from reading past the end
		// of an unterminated string.
        self.char_offset -= 1; 
        -1
    }

    pub fn read_hex_number(&mut self) {
        // Skip past the `x` used to denote a hexadecimal literal.
        if let Some((_, _)) = self.next() {
            self.current_line += 1;
        }
        
        // Iterate over all the valid hexadecimal digits found.
        while self.read_hex_digit() != -1 {
            continue;
        }

        self.make_number();
    }

    fn match_char(&mut self, c:char) -> bool {
        match self.peek() {
            Some(&(_, _c)) => {
                if c != _c {
                    return false;
                } else {
                    self.next();
                    return true;
                } 
            },
            None => {
                return false;
            }
        }
    }

    fn make_number(&mut self) {

        // Todo: store number value

        self.make_token(GRAMMAR_RULES[36]);
    }

    fn two_char_token(&mut self, c: char, two: TokenType, one: TokenType){
        if self.match_char(c) {
            self.make_token(two);
        } else  {
            self.make_token(one);
        }
    }

    /// Finishes lexing an identifier. Handles reserved words.
    fn read_name(&mut self, token_type: TokenType, first_char: char) {
        let mut _token_type = token_type;
        loop {
            if let Some(&(_, c)) = self.peek() {
                if is_name(c) || is_digit(c) {
                    self.next();
                } else {
                    break;
                }
            }
        }
        

        
   
        for kwd in KEYWORDS.into_iter() {
            let _word = std::str::from_utf8(&self.input_copy[self.token_offset..self.char_offset].as_bytes());
            let w = kwd.get_kwd();
            
            if w == String::from(_word.unwrap()) {
                _token_type = *kwd;
                break;
            }
        }

      
        self.make_token(_token_type);
    }

    fn read_number(&mut self) {
        loop {
            if let Some(&(_, ch)) = self.peek() {
                if is_digit(ch) {
                    self.next();
                } else {
                    break;
                }
            } 
        }
       
        // See if it has a floating point. Make sure there is a digit after the "."
        // so we don't get confused by method calls on number literals.
        if let Some(&(_, ch)) = self.peek() {
            if let Some((_, ch2)) = self.peek_next() {
                if ch == '.' && is_digit(ch2) {
                    self.next();
                    loop {
                        if let Some(&(_, ch)) = self.peek() {
                            if is_digit(ch) {
                                self.next();
                            } else {
                                break;
                            }
                        } 
                    }
                    
                }
            }
        }
        // // See if the number is in scientific notation. 
        if self.match_char('e') || self.match_char('E') {
            // Allow a single positive/negative exponent symbol.
            if !self.match_char('+') {
                self.match_char('-');
            } 
            if let Some(&(_, ch)) = self.peek() {
                if !is_digit(ch) {
                    lex_error(self.current_line,"Unterminated scientific notation.".to_owned());
                }
            }
            
            loop {
                if let Some(&(_, ch)) = self.peek() {
                    if is_digit(ch) {
                        self.next();
                    } else {
                        break;
                    }
                } 
            }
            
        }
        self.make_number();
    }

    fn read_hex_escape(&mut self, digits: usize, description: String) -> u8 {
        let mut value  = 0;
        for _i in 0..digits {
            if let Some(&(_, ch)) = self.peek() {
                if ch == '"' || ch == '\0' {
                    lex_error(self.current_line, format!("Incomplete {} scape sequence.", description));
                    self.char_offset -= 1;
                    break;
                }
            }
             
            let digit = self.read_hex_digit();
            if digit == -1 {
                lex_error(self.current_line, format!("Invalid {} escape sequence.", description));
                break;
            }

            value = (value * 16) | digit;
        }
        value as u8
    }

    /// Reads a hex digit Unicode escape sequence in a string literal.
    fn read_unicode_escape(& mut self, string: &mut String, length: usize) {
        let mut _value = self.read_hex_escape(length, String::from("Unicode"));

        // Grow the buffer enough for the encoded result.
        let num_bytes = wren_utf8_encode_num_bytes(_value as i32);
        if num_bytes != 0 {
            // let mut v: Vec<u8> = Vec::new();
            for _ in 0..num_bytes {
                string.push(char::from_u32(0).unwrap());
            }
             //= std::str::from_utf8_unchecked(&v);
        }
    }

    // Finishes lexing a string literal.
    fn read_string(&mut self) {
        let mut string = String::new();
        let mut token_type = GRAMMAR_RULES[37];
        loop {
            if let Some((_, c)) = self.next() {
                if c == '"' {
                    break;
                }
                if c == '\0' {
                    lex_error(self.current_line, String::from("Unterminated string."));
                    // Don't consume it if it isn't expected. Keeps us from reading past the
				    // end of an unterminated string.
                    self.char_offset -= 1;
                    break;
                }
                if c == '%' {
                    if self.num_parens < MAX_INTERPOLATION_NESTING {
                        // TODO: Allow format string.
                        if let Some((_, _c)) = self.next() {
                            
                            if _c != '(' {
                                lex_error(self.current_line, String::from("Expect '(' after '%'."));
                            }
                            self.num_parens += 1;
                            self.parens[self.num_parens] = 1;
                            token_type = GRAMMAR_RULES[38];
                            break;
                        }
                    }
                    lex_error(self.current_line, format!("Interpolation may only nest {} levels deep.", MAX_INTERPOLATION_NESTING));
                }

                if c == '\\' {
                    if let Some((_, ch)) = self.next() {
                        
                        match ch {
                            '"' => {string.push('"'); break;}
                            '\\' => {string.push('\\'); break;}
                            '%' => {string.push('%'); break;}
                            '0' => {string.push('\0'); break;}
                            'a' => {string.push_str("\x07"); break;}
                            'b' => {string.push_str("\x08"); break;}
                            'e' => {string.push_str("\x1B"); break;}
                            'f' => {string.push_str("\x0C"); break;}
                            'n' => {string.push('\n'); break;}
                            'r' => {string.push('\r'); break;}
                            't' => {string.push('\t'); break;}
                            'u' => {self.read_unicode_escape(&mut string, 4); break;}
                            'U' => {self.read_unicode_escape(&mut string, 8); break;}
                            'v' => {string.push_str("\x0B"); break;}
                            'x' => {string.push(self.read_hex_escape(2, String::from("byte")) as char); break;}
                            _ => {lex_error(self.current_line, format!("invalid escape character {}", self.input.nth(self.char_offset-1).unwrap().1)); break;}
                        }
                    }
                } {
                    string.push(c);
                }
            } else {
                break;
            }
            
        }
        string.clear();
        // Todo: store string literal
        self.make_token(token_type);

    }

    pub fn next_token(&mut self) -> bool {
        self.previous = self.current.clone();
        self.current = self.next.clone();

        if let Some(Tokens::Token(token_type, chars, line)) = self.next.clone() {
            match token_type {
                TokenType::Eof(_) => {
                    return true;
                }
                _ => {}
            }
        }

        if let Some(Tokens::Token(token_type, chars, line)) = self.current.clone() {
            match token_type {
                TokenType::Eof(_) => {
                    return true;
                }
                _ => {}
            }
        }

        loop {
            match self.peek() {
                Some(&(_pos, ch)) => {
                    self.token_offset = self.char_offset;
                        let c = self.next();
                        match c {
                            Some((_, '(')) => {
                                // If we are inside an interpolated expression, count the unmatched "(".
                                if self.num_parens > 0 {
                                    self.parens[(self.num_parens - 1) as usize] += 1;
                                }
                                self.make_token(GRAMMAR_RULES[0]);
                                return true;
                            },
                            Some((_, ')')) => {
                                // If we are inside an interpolated expression, count the ")".
                                if self.num_parens > 0 {
                                    self.parens[(self.num_parens - 1) as usize] -= 1;
                                    if self.parens[(self.num_parens - 1) as usize] == 0 {
                                        // This is the final ")", so the interpolation expression has ended.
                                        // This ")" now begins the next section of the template string.
                                        self.num_parens -= 1;
                                        self.read_string();
                                        return true;
                                    }
                                }
                                self.make_token(GRAMMAR_RULES[1]);
                                return true;
                            },
                            Some((_, '[')) => {
                                self.make_token(GRAMMAR_RULES[2]);
                                return true;
                            }
                            Some((_, ']')) => {
                                self.make_token(GRAMMAR_RULES[3]);
                                return true;
                            }
                            Some((_, '{')) => {
                                self.make_token(GRAMMAR_RULES[4]);
                                return true;
                            }
                            Some((_, '}')) => {
                                self.make_token(GRAMMAR_RULES[5]);
                                return true;
                            }
                            Some((_, ':')) => {
                                self.make_token(GRAMMAR_RULES[6]);
                                return true;
                            }
                            Some((_, ',')) => {
                                self.make_token(GRAMMAR_RULES[10]);
                                return true;
                            }
                            Some((_, '*')) => {
                                self.make_token(GRAMMAR_RULES[11]);
                                return true;
                            }
                            Some((_, '%')) => {
                                self.make_token(GRAMMAR_RULES[13]);
                                return true;
                            }
                            Some((_, '#')) => {
                                // Ignore shebang on the first line.
                                if let Some(&(_, ch)) = self.peek() {
                                    if let Some((_, ch2)) = self.peek_next() {
                                        if self.current_line == 1 && ch == '!' && ch2 == '/' {
                                            self.skip_line_comment();
                                            break;
                                        }
                                    }
                                }
                                self.make_token(GRAMMAR_RULES[62]);
                                return true;
                            }
                            Some((_, '+')) => {
                                self.make_token(GRAMMAR_RULES[14]);
                                return true;
                            }
                            Some((_, '-')) => {
                                self.make_token(GRAMMAR_RULES[15]);
                                return true;
                            }
                            Some((_, '^')) => {
                                self.make_token(GRAMMAR_RULES[20]);
                                return true;
                            }
                            Some((_, '~')) => {
                                self.make_token(GRAMMAR_RULES[24]);
                                return true;
                            }
                            Some((_, '?')) => {
                                self.make_token(GRAMMAR_RULES[25]);
                                return true;
                            }
                            Some((_, '|')) => {
                                self.two_char_token('|', GRAMMAR_RULES[19], GRAMMAR_RULES[18]);
                                return true;
                            }
                            Some((_, '&')) => {
                                self.two_char_token('&', GRAMMAR_RULES[22], GRAMMAR_RULES[21]);
                                return true;
                            }
                            Some((_, '!')) => {
                                // println!("{}", ch);
                                self.two_char_token('!', GRAMMAR_RULES[32], GRAMMAR_RULES[23]);
                                return true;
                            }
                            Some((_, '=')) => {
                                self.two_char_token('=', GRAMMAR_RULES[31], GRAMMAR_RULES[26]);
                                return true;
                            }
                            Some((_, '.')) => {
                                if self.match_char('.') {
                                    self.two_char_token('.', GRAMMAR_RULES[9], GRAMMAR_RULES[8]);
                                    return true;
                                }
                                self.make_token(GRAMMAR_RULES[7]);
                                
                                return true;
                            }
                            Some((_, '/')) => {
                                if self.match_char('/') {
                                    self.skip_line_comment();
                                    continue;
                                }
                                if self.match_char('*') {
                                    self.skip_block_comment();
                                    continue;
                                }
                                self.make_token(GRAMMAR_RULES[12]);
                                return true;
                            }
                            Some((_, '<')) => {
                                if self.match_char('<') {
                                    self.make_token(GRAMMAR_RULES[16]);
                                } else {
                                    self.two_char_token('=', GRAMMAR_RULES[29], GRAMMAR_RULES[27])
                                }

                                return true;
                            }
                            Some((_, '>')) => {
                                if self.match_char('>') {
                                    self.make_token(GRAMMAR_RULES[17]);
                                } else {
                                    self.two_char_token('=', GRAMMAR_RULES[30], GRAMMAR_RULES[28])
                                }

                                return true;
                            }
                            Some((_, '\n')) => {
                                self.make_token(GRAMMAR_RULES[39]);
                                return true;
                            }
                            Some((_, ' ')) | Some((_, '\r')) | Some((_, '\t'))  => {
                                'inner1: loop {
                                    if let Some(&(_, ch)) = self.peek() {
                                        if ch == ' ' || ch == '\r' || ch == '\t'{
                                            self.next();
                                        } else {
                                            break 'inner1;
                                        }
                                    }
                                }
                                continue;
                            }
                            Some((_, '"')) => {
                                if let Some(&(_, ch1)) = self.peek() {
                                    if let Some((_, ch2)) = self.peek_next() {
                                        if ch1 == '"' && ch2 == '"'{
                                            return true;
                                        } 
                                    }
                                }
                                self.read_string();
                                return true;
                            }
                            Some((_, '_')) => {
                                if let Some(&(_, ch)) = self.peek() {
                                    self.read_name(if ch == '_' {GRAMMAR_RULES[34]} else {GRAMMAR_RULES[33]}, ch);
                                }
                                return true;
                            }
                            Some((_, '0')) => {
                                if let Some(&(_, ch)) = self.peek() {
                                    if ch == 'x' {
                                        self.read_hex_number();
                                        return true;
                                    }
                                }
                                self.read_number();
                                return true;
                            }
                            _ => {
                                match ch {
                                    '0'..='9' |
                                    'a'..='z' | 'A'..='Z' => {
                                        if is_name(ch) {
                                            self.read_name(GRAMMAR_RULES[35], ch);
                                        } else if is_digit(ch) {
                                            self.read_number();
                                        } 
                                        return true;
                                    }
                                    _ => {
                                        if (ch as u8) >= 32 && (ch as u8) <= 126 {
                                            lex_error(self.current_line, format!("Invalid character {}.", ch));
                                        } else {
                                            // Don't show non-ASCII values since we didn't UTF-8 decode the
                                            // bytes. Since there are no non-ASCII byte values that are
                                            // meaningful code units in Wren, the lexer works on raw bytes,
                                            // even though the source code and console output are UTF-8.
                                            lex_error(self.current_line, format!("Invalid character {:#x}.", ch as u8));
                                        }
                                   
                                        self.current = Some(Tokens::Token(
                                            GRAMMAR_RULES[40],
                                            "",
                                            self.current_line
                                        ));
                                        return true;
                                    }
                                }
                            
                                
                            }
                        }
                }
                None => {
                  return false;  
                }
                
            }
        }

        // If we get here, we're out of source, so just make EOF tokens.
        self.token_offset = self.char_offset;
        self.make_token(GRAMMAR_RULES[61]);
        return false;
    }

}

#[derive(Debug, Clone)]
pub enum SignatureType {
    /// A name followed by a (possibly empty) parenthesized parameter list. Also
    /// used for binary operators.
    Method = 0,
    /// Just a name. Also used for unary operators.
    Getter,
    /// A name followed by "=".
    Setter,
    /// A square bracketed parameter list.
    Subscript,
    /// A square bracketed parameter list followed by "=".
    SubscriptSetter,
    /// A constructor initializer function. This has a distinct signature to
    /// prevent it from being invoked directly outside of the constructor on the
    /// metaclass.
    Initializer,

}

#[derive(Debug, Clone)]
pub struct Signature<'a>{
    pub sigType: SignatureType,
    pub length: usize,
    pub name: &'a str,
    pub arity: u32
}

impl<'a> Signature<'a> {
    fn parameterList(&self, numParams: u32, leftBracket: &'a str, rightBracket: &'a str) -> String {
        let mut i = 0 as u32;
        let mut ret  = String::from(leftBracket);
        while i < numParams && i < MAX_PARAMETERS {
            if i > 0 {
                ret.push_str(",");
            }
            ret.push_str("_");
            i += 1;
        }
        ret.push_str(rightBracket);
        ret
    }
}

impl<'a> Display for Signature<'a> {
    
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut name = self.name.to_owned();
        
        match self.sigType {
            SignatureType::Method => name.push_str(self.parameterList(self.arity, "(", ")").as_str()),
            SignatureType::Getter => {},
            SignatureType::Setter => {
                name.push_str("=");
                name.push_str(self.parameterList(self.arity, "(", ")").as_str());
            } 
            SignatureType::Subscript => {
                name.push_str("");
                name.push_str(self.parameterList(self.arity, "[", "]").as_str());
            }
            SignatureType::SubscriptSetter => {
                name.push_str(self.parameterList(self.arity - 1, "[", "]").as_str());
                name.push_str("=");
                name.push_str(self.parameterList(self.arity, "(", ")").as_str());
            }
            SignatureType::Initializer => {
                name.push_str(format!("init {}", name.as_str()).as_str());
                name.push_str(self.parameterList(self.arity, "(", ")").as_str());
            }
        }
        write!(f, "{}", name)
    }
}





impl GrammarRule {
    #[inline]
    pub fn unused() -> Self {
        Self { 
            precedence: Some(Precedence::PrecNone),
            infix: None,
            method: None,
            name: None,
            prefix: None,
        }
    }

    #[inline]
    pub fn prefix(fun: Option<GrammarFn>) -> Self {
        Self {
            prefix: fun,
            method: None,
            infix: None,
            name: None,
            precedence: Some(Precedence::PrecNone),
        }
    }

    #[inline]
    pub fn infix(pre: Option<Precedence>, fun: Option<GrammarFn>) -> Self {
        Self {
            infix: fun,
            precedence: pre,
            prefix: None,
            name: None, 
            method: None,
        }
    }

    #[inline]
    pub fn infix_operator(pre: Option<Precedence>, name: &'static str) -> Self {
        Self {
            name: Some(name),
            infix: Some(grammar::InfixOp),
            method: Some(grammar::InfixSignature),
            prefix: None,
            precedence: pre,
        }
    }

    #[inline]
    pub fn prefix_operator(name: &'static str) -> Self {
        Self {
            name: Some(name),
            prefix: Some(grammar::UnaryOp),
            method: Some(grammar::UnarySignature),
            precedence: Some(Precedence::PrecNone),
            infix: None,
        }
    }

    #[inline]
    pub fn operator(name: &'static str) -> Self {
        Self {
            name: Some(name),
            prefix: Some(grammar::UnaryOp),
            infix: Some(grammar::InfixOp),
            method: Some(grammar::MixedSignature),
            precedence: Some(Precedence::PrecTerm),
        }
    }

}




// impl<'a> SourceBuffer<'a> {
//     pub fn new(_source: &'a str) -> Self {
//         Self {
//             source: _source.chars().peekable(),
//             current_offset: 0,
//             current_line: 0,
//         }
//     }

//     pub fn peek_char(&'a mut self) -> Option<char> {
//         self.source.peek().cloned()
//     }

//     pub fn peek_next_char(&mut self) -> Option<char> {
//         if let Some(c) = self.source.peek().cloned() {
//             if c == '\0' {
//                 return Some(c);
//             }
//         }
    
//         self.source.next()
//     }

//     pub fn nth(&mut self,i:usize) -> Option<char> {
//         self.source.nth(i)
//     }

//     pub fn next_char(&'a mut self) -> Option<char> {
//         let c = self.source.peek().cloned();
//         self.current_offset += 1;
//         if let Some('\n') = c {
//             self.current_line += 1;
//         }
        
//         self.source.advance_by(self.current_offset as usize);
//         self.source.next()
//     }

//     /// If the current character is [c], consumes it and returns `true`.
//     pub fn match_char(&'a mut self, c: char) -> bool {
//         if let Some(_c) = self.source.peek().cloned() {
//             if c != _c {
//                 false
//             } else {
//                 self.next_char();
//                 true
//             }
//         }  else {
//             false
//         }
        
//     }

//     /// Skips the rest of the current line.
//     pub fn skip_line_comment(&'a mut self) {
//         while self.peek_char() != Some('\n') && self.peek_next_char() != Some('\0') {
//             self.next_char();
//         }
//     }

//     /// Skips the rest of a block comment.
//     pub fn skip_block_comment(&'a mut self) {
//             let mut nesting = 1;
//             while nesting > 0 {
//                 if let Some(c) = self.source.peek().cloned() {
//                     if c == '\0' {
//                         return;
//                     }
//                     lex_error(self.current_line, "Unterminated block comment.".to_owned());
//                 }
    
//                 if let Some('/') = self.source.peek().cloned() {
//                     if let Some('*') = self.peek_next_char() {
//                         self.next_char();
//                         self.next_char();
//                         nesting += 1;
//                         continue;
//                     }
//                 }
    
//                 if let Some('*') = self.source.peek().cloned() {
//                     if let Some('/') = self.peek_next_char() {
//                         self.next_char();
//                         self.next_char();
//                         nesting -= 1;
//                         continue;
//                     }
//                 }
    
//                 // Regular comment character.
//                 self.next_char();
//             }
//     }

//     /// Reads the next character, which should be a hex digit (0-9, a-f, or A-F) and
//     /// returns its numeric value. If the character isn't a hex digit, returns -1.
//     fn read_hex_digit(&'a mut self) -> i8 {
//         let _c = self.peek_char();
//         self.current_offset += 1;
//         if let Some('\n') = _c {
//             self.current_line += 1;
//         }
        
//         self.source.advance_by(self.current_offset as usize);
//         let c = self.source.next();

//         if let Some(_c) =  c {
//             if _c >= '0' && _c <= '9' {
//                 _c as u8 - b'0'; 
//             }
//             if _c >= 'a' && _c <= 'f' {
//                 _c as u8 - b'a' + 10; 
//             }
//             if _c >= 'A' && _c <= 'F' {
//                 _c as u8 - b'A' + 10; 
//             }
//         }
//         // Don't consume it if it isn't expected. Keeps us from reading past the end
// 		// of an unterminated string.
//         self.current_offset -= 1; 
//         return -1;
//     }

//     pub fn read_hex_number(&'a self, parser: &'a mut Parser<'a>) {
//         // Skip past the `x` used to denote a hexadecimal literal.
//         self.next_char();
//         // Iterate over all the valid hexadecimal digits found.
//         while self.read_hex_digit() != -1 {
//             continue;
//         }

//         parser.make_number();
//     }

//     pub fn read_number(&'a self, parser:&'a mut Parser<'a>){
//         if let Some(c) = self.peek_char() {
//             while is_digit(c) {
//                 self.next_char();
//             }
//         }
        
//         // See if it has a floating point. Make sure there is a digit after the "."
// 		// so we don't get confused by method calls on number literals.
//         if let Some(c) = self.peek_char() {
//             if let Some(cc) = self.peek_next_char() {
//                 if c == '.' && is_digit(cc) {
//                     self.next_char();
//                     while is_digit(cc) {
//                         self.next_char();
//                     }
//                 }
//             }
//         }

//         if let Some(cc) = self.peek_next_char() {
//             // See if the number is in scientific notation.
//             if self.match_char('e') && is_digit(cc) {
//                 // Allow a single positive/negative exponent symbol.
//                 if !self.match_char('+') {
//                     self.match_char('-');
//                 }
//                 if let Some(c) = self.peek_char() {
//                     if is_digit(c) {
//                         lex_error(self.current_line, "Unterminated scientific notation.".to_owned());
//                     }

//                     while is_digit(c) {
//                         self.next_char();
//                     }
//                 }
                
//             }
//         }


//         parser.make_number();   
//     }

//     /// Reads [digits] hex digits in a string literal and returns their number value.
//     pub fn read_hex_escape(&'a self, digits: usize, description: &str) -> u8 {
//         let mut _value = 0;
//         for _ in 0..digits {
//             if let Some(c) = self.peek_char() {
//                 if c == '"' || c == '\0' {
//                     lex_error(self.current_line, format!("Incomplete {} escape sequence.", description));
//                     // Don't consume it if it isn't expected. Keeps us from reading past the
//                     // end of an unterminated string.
//                     self.current_offset -= 1; 
//                     break;
//                 }
//             }
        
//             let digit = self.read_hex_digit();
//             if digit == -1 {
//                 lex_error(self.current_line,format!("Invalid {} escape sequence.", description));
//                 break;
//             }
//             _value = (_value * 16) | digit;
//         }

//         _value as u8
//     }

//    // Finishes lexing a string literal.
//    pub fn read_string(&'a mut self, parser: Parser<'a>) {
//         let mut string = String::new();
//         let mut tokenType = TokenType::String;

//         loop {
//             if let Some(c) = self.next_char() {
//                 if c == '"' {
//                     break;
//                 }

//                 if c == '\0' {
//                     lex_error(self.current_line,"Unterminated string.".to_owned());
//                     // Don't consume it if it isn't expected. Keeps us from reading past the
//                     // end of an unterminated string.
//                     self.current_offset -= 1;
//                     break;
//                 }

//                 if c == '%' {
//                     if parser.num_parens < MAX_INTERPOLATION_NESTING {
//                         // TODO: Allow format string.
//                         if let Some(_c) = self.next_char() {
//                             if _c != '(' {
//                                 lex_error(self.current_line,"Expect '(' after '%%'".to_owned());
//                             }
//                         }
//                         let idx = (parser.num_parens + 1) as usize;
//                         parser.parens[idx] = 1;
//                         tokenType = TokenType::Interpolation;
//                         break; 
//                     }

//                     lex_error(self.current_line, format!("Interpolation may only nest {} levels deep.", MAX_INTERPOLATION_NESTING));
//                 }

                
            
//                 if c == '\\' {
//                     match self.next_char() {
//                         Some('"') => {string.push('"');}
//                         Some('\\') => {string.push('\\');}
//                         Some('%') => {string.push('%');}
//                         Some('0') => {string.push('\0');}
//                         Some('a') => {string.push_str("\x07");}
//                         Some('b') => {string.push_str("\x08");}
//                         Some('e') => {string.push_str("\x1B");}
//                         Some('f') => {string.push_str("\x0C");}
//                         Some('n') => {string.push('\n');}
//                         Some('r') => {string.push('\r');}
//                         Some('t') => {string.push('\t');}
//                         Some('u') => {self.read_unicode_escape(&mut string, 4);}
//                         Some('U') => {self.read_unicode_escape(&mut string, 8);}
//                         Some('v') => {string.push_str("\x0B");}
//                         Some('x') => {
//                             string.push(self.read_hex_escape(2, "byte") as char);
//                         }
//                         None => {
//                             if let Some(cc) = self.nth(self.current_offset - 1) {
//                                 lex_error(self.current_line, format!("Invalid escape character '{}'.", cc));
//                             }
                            
//                         }
//                         _ => {}
//                     }
//                 } else {
//                     string.push(c);
//                 }
//             }
//         }

//         parser.make_token(tokenType);
//     }

//     /// Reads a hex digit Unicode escape sequence in a string literal.
//     fn read_unicode_escape(&'a self, string: &mut String, length: usize) {
//         let mut _value = self.read_hex_escape(length, "Unicode");

//         // Grow the buffer enough for the encoded result.
//         let num_bytes = wren_utf8_encode_num_bytes(_value);
//         if num_bytes != 0 {
//             // let mut v: Vec<u8> = Vec::new();
//             for _ in 0..num_bytes {
//                 string.push(char::from_u32(0).unwrap());
//             }
//              //= std::str::from_utf8_unchecked(&v);
//         }
//     }

//     pub fn read_raw_string(&'a mut self) {
//         let mut string = String::new();
//         let tokenType = TokenType::String;

//         //consume the second and third "
//         self.next_char();
//         self.next_char();

//         let mut skip_start = 0 as i32;
//         let mut first_new_line = -1 as i32;


//         let mut skip_end = -1 as i32;
//         let mut last_new_line = -1 as i32;

//         loop {
//             let c = self.next_char();
//             let c1 = self.peek_char();
//             let c2 = self.peek_next_char();

//             match c {
//                 Some('\n') => {
//                     last_new_line = string.len() as i32;
//                     skip_end = last_new_line;
//                     first_new_line = if first_new_line == -1 { string.len() as i32 } else  { first_new_line };
//                 }
//                 None => {}
//                 _ => {}
//             }

//             if c == Some('"') && c1 == Some('"') && c2 == Some('"') {
//                break;
//             }

//             let is_white_space: bool = c == Some(' ') || c == Some('\t');
//             skip_end = if c == Some('\n') || is_white_space  { skip_end } else {-1};
//             // If we haven't seen a newline or other character yet, 
//             // and still seeing whitespace, count the characters 
//             // as skippable till we know otherwise

//             let skippable: bool = skip_start != -1 && is_white_space && first_new_line == -1;
//             skip_start = if skippable {string.len() as i32 + 1} else {skip_start};

//             // We've counted leading whitespace till we hit something else, 
//             // but it's not a newline, so we reset skipStart since we need these characters
//             if first_new_line == -1 && !is_white_space && c != Some('\n') {
//                 skip_start = -1;
//             }

//             if c == Some('\0') && c1 == Some('\0') && c2 == Some('\0') {
//                 lex_error(self.current_line, "Unterminated raw string.".to_owned());
//                 // Don't consume it if it isn't expected. Keeps us from reading past the
//                 // end of an unterminated string.
//                 self.current_offset -= 1;
//                 break;
//             }
//             string.push(c.unwrap());
//         }

//         //consume the second and third "
//         self.next_char();
//         self.next_char();
//     }
// }






