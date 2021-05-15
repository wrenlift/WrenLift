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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone, Copy)]
pub enum TokenType {
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Dot,
    DotDot,
    DotDotDot,
    Comma,
    Star,
    Slash,
    Percent,
    Plus,
    Minus,
    LtLt,
    GtGt,
    Pipe,
    PipePipe,
    Caret,
    Amp,
    AmpAmp,
    Bang,
    Tilde,
    Question,
    Eq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    EqEq,
    BangEq,
    Hash,
  
    Break,
    Class,
    Construct,
    Else,
    False,
    For,
    Foreign,
    If,
    Import,
    In,
    Is,
    Null,
    Return,
    Static,
    Super,
    This,
    True,
    Var,
    While,
  
    Field,
    StaticField,
    Name,
    Number,
    
    // A string literal without any interpolation or the last section of a
    // string following the last interpolated expression.
    String,
    
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
    Interpolation,
  
    Line,
  
    Error,
    Eof,
}

impl Eq for TokenType {}
impl PartialEq for TokenType {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}
impl Hash for TokenType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // self.hash(state);
        // self.hash(state);
    }
}


#[derive(Debug, Clone, Copy)]
pub struct Token<'a> {
    pub token_type: Option<TokenType>,
    pub start: &'a str,
    pub length: usize,
    pub line: usize
}

impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{:#?}", self)
    }
}

impl<'a> Token<'a> {
    pub fn new() -> Self {
        Self {
            length: 0,
            line: 0,
            token_type: None,
            start: ""
        }
    }
}

#[derive(Debug, Clone)]
pub struct Keyword {
    pub identifier:Option<&'static str>,
    pub length:usize,
    pub tokenType:TokenType
}

impl<'a> Display for Keyword {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{:#?}", self)
    }
}

pub const Keywords: [Keyword; 20] = [
    Keyword{identifier: Some("break"),     length: 5, tokenType: TokenType::Break},
    Keyword{identifier: Some("class"),     length: 5, tokenType:  TokenType::Class},
    Keyword{identifier: Some("construct"), length: 9, tokenType:  TokenType::Construct},
    Keyword{identifier: Some("else"),      length: 4, tokenType:  TokenType::Else},
    Keyword{identifier: Some("false"),     length: 5, tokenType:  TokenType::False},
    Keyword{identifier: Some("for"),       length: 3, tokenType:  TokenType::For},
    Keyword{identifier: Some("foreign"),   length: 7, tokenType:  TokenType::Foreign},
    Keyword{identifier: Some("if"),        length: 2, tokenType:  TokenType::If},
    Keyword{identifier: Some("import"),    length: 6, tokenType:  TokenType::Import},
    Keyword{identifier: Some("in"),        length: 2, tokenType:  TokenType::In},
    Keyword{identifier: Some("is"),        length: 2, tokenType:  TokenType::Is},
    Keyword{identifier: Some("null"),      length: 4, tokenType:  TokenType::Null},
    Keyword{identifier: Some("return"),    length: 6, tokenType:  TokenType::Return},
    Keyword{identifier: Some("static"),    length: 6, tokenType:  TokenType::Static},
    Keyword{identifier: Some("super"),     length: 5, tokenType:  TokenType::Super},
    Keyword{identifier: Some("this"),      length: 4, tokenType:  TokenType::This},
    Keyword{identifier: Some("true"),      length: 4, tokenType:  TokenType::True},
    Keyword{identifier: Some("var"),       length: 3, tokenType:  TokenType::Var},
    Keyword{identifier: Some("while"),     length: 5, tokenType:  TokenType::While},
    Keyword{identifier: None,        length: 0, tokenType: TokenType::Eof },
];


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


type GrammarFn = fn(parser: Parser, canAssign: bool);
type SignatureFn = fn(parser: Parser, signature: &'static mut Signature<'static>) -> Option<&'static mut Signature<'static>>;


#[derive(Clone)]
pub struct GrammarRule {
    prefix: Option<GrammarFn>,
    infix: Option<GrammarFn>,
    method: Option<SignatureFn>,
    precedence: Option<Precedence>,
    name: Option<&'static str>,
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

lazy_static! {
    pub static ref GrammarRules: HashMap<TokenType, GrammarRule> = [
        (TokenType::LeftParen, GrammarRule::prefix(Some(grammar::Grouping))),
        (TokenType::RightParen, GrammarRule::unused()),
        (TokenType::LeftBracket, GrammarRule { name: None, prefix: Some(grammar::List), infix: Some(grammar::Subscript), method: Some(grammar::SubscriptSignature), precedence: Some(Precedence::PrecCall) }),
        (TokenType::RightBracket, GrammarRule::unused()),
        (TokenType::LeftBrace, GrammarRule::prefix(Some(grammar::Map))),
        (TokenType::RightBrace, GrammarRule::unused()),
        (TokenType::Colon, GrammarRule::unused()),
        (TokenType::Dot, GrammarRule::infix(Some(Precedence::PrecCall), Some(grammar::Call))),
        (TokenType::DotDot, GrammarRule::infix_operator(Some(Precedence::PrecRange), "..")),
        (TokenType::DotDotDot, GrammarRule::infix_operator(Some(Precedence::PrecRange), "...")),
        (TokenType::Comma, GrammarRule::unused()),
        (TokenType::Star, GrammarRule::infix_operator(Some(Precedence::PrecFactor), "*")),
        (TokenType::Slash, GrammarRule::infix_operator(Some(Precedence::PrecFactor), "/")),
        (TokenType::Percent, GrammarRule::infix_operator(Some(Precedence::PrecFactor), "%")),
        (TokenType::Plus, GrammarRule::infix_operator(Some(Precedence::PrecTerm), "+")),
        (TokenType::Minus, GrammarRule::operator("-")),
        (TokenType::LtLt, GrammarRule::infix_operator(Some(Precedence::PrecBitwiseShift), "<<")),
        (TokenType::GtGt, GrammarRule::infix_operator(Some(Precedence::PrecBitwiseShift), ">>")),
        (TokenType::Pipe, GrammarRule::infix_operator(Some(Precedence::PrecBitwiseOr), "|")),
        (TokenType::PipePipe, GrammarRule::infix(Some(Precedence::PrecLogicalOr), Some(grammar::Or))),
        (TokenType::Caret, GrammarRule::infix_operator(Some(Precedence::PrecBitwiseXor), "^")),
        (TokenType::Amp, GrammarRule::infix_operator(Some(Precedence::PrecBitwiseAnd), "&")),
        (TokenType::AmpAmp, GrammarRule::infix(Some(Precedence::PrecLogicalAnd), Some(grammar::And))),
        (TokenType::Bang, GrammarRule::prefix_operator("!")),
        (TokenType::Tilde, GrammarRule::prefix_operator("~")),
        (TokenType::Question, GrammarRule::infix(Some(Precedence::PrecAssignment), Some(grammar::Conditional))),
        (TokenType::Eq, GrammarRule::unused()),
        (TokenType::Lt, GrammarRule::infix_operator(Some(Precedence::PrecComparison), "<")),
        (TokenType::Gt, GrammarRule::infix_operator(Some(Precedence::PrecComparison), ">")),
        (TokenType::LtEq, GrammarRule::infix_operator(Some(Precedence::PrecComparison), "<=")),
        (TokenType::GtEq, GrammarRule::infix_operator(Some(Precedence::PrecComparison), "=>")),
        (TokenType::EqEq, GrammarRule::infix_operator(Some(Precedence::PrecEquality), "==")),
        (TokenType::BangEq, GrammarRule::infix_operator(Some(Precedence::PrecEquality), "!=")),
        (TokenType::Break, GrammarRule::unused()),
        (TokenType::Class, GrammarRule::unused()),
        (TokenType::Construct, GrammarRule{ name: None, prefix: None, infix: None,  method: Some(grammar::ConstructorSignature), precedence: Some(Precedence::PrecNone) }),
        (TokenType::Else, GrammarRule::unused()),
        (TokenType::False, GrammarRule::prefix(Some(grammar::Boolean))),
        (TokenType::For, GrammarRule::unused()),
        (TokenType::Foreign, GrammarRule::unused()),
        (TokenType::If, GrammarRule::unused()),
        (TokenType::Import, GrammarRule::unused()),
        (TokenType::In, GrammarRule::unused()),
        (TokenType::Is, GrammarRule::infix_operator(Some(Precedence::PrecIs), "is")),
        (TokenType::Null, GrammarRule::prefix(Some(grammar::Null))),
        (TokenType::Return, GrammarRule::unused()),
        (TokenType::Static, GrammarRule::unused()),
        (TokenType::Super, GrammarRule::prefix(Some(grammar::Super))),
        (TokenType::This, GrammarRule::prefix(Some(grammar::This))),
        (TokenType::True, GrammarRule::prefix(Some(grammar::Boolean))),
        (TokenType::Var, GrammarRule::unused()),
        (TokenType::While, GrammarRule::unused()),
        (TokenType::Field, GrammarRule::prefix(Some(grammar::Field))),
        (TokenType::StaticField, GrammarRule::prefix(Some(grammar::StaticField))),
        (TokenType::Name, GrammarRule{ name: None, infix: None, prefix: Some(grammar::Name), method: Some(grammar::NamedSignature), precedence: Some(Precedence::PrecNone) }),
        (TokenType::Number, GrammarRule::prefix(Some(grammar::Literal))),
        (TokenType::String, GrammarRule::prefix(Some(grammar::Literal))),
        (TokenType::Interpolation, GrammarRule::prefix(Some(grammar::StringInterpolation))),
        (TokenType::Line, GrammarRule::unused()),
        (TokenType::Error, GrammarRule::unused()),
        (TokenType::Eof, GrammarRule::unused()),
    ].iter().cloned().collect();
}






pub struct SourceBuffer<'a> {
    pub source: Peekable<Chars<'a>>,
    pub current_offset: usize,
    pub current_line: usize,

}


impl<'a> SourceBuffer<'a> {
    pub fn new(_source: &'a str) -> Self {
        Self {
            source: _source.chars().peekable(),
            current_offset: 0,
            current_line: 0,
        }
    }

    pub fn peek_char(&'a mut self) -> Option<char> {
        self.source.peek().cloned()
    }

    pub fn peek_next_char(&mut self) -> Option<char> {
        if let Some(c) = self.source.peek().cloned() {
            if c == '\0' {
                return Some(c);
            }
        }
    
        self.source.next()
    }

    pub fn nth(&mut self,i:usize) -> Option<char> {
        self.source.nth(i)
    }

    pub fn next_char(&'a mut self) -> Option<char> {
        let c = self.source.peek().cloned();
        self.current_offset += 1;
        if let Some('\n') = c {
            self.current_line += 1;
        }
        
        self.source.advance_by(self.current_offset as usize);
        self.source.next()
    }

    /// If the current character is [c], consumes it and returns `true`.
    pub fn match_char(&'a mut self, c: char) -> bool {
        if let Some(_c) = self.source.peek().cloned() {
            if c != _c {
                false
            } else {
                self.next_char();
                true
            }
        }  else {
            false
        }
        
    }

    /// Skips the rest of the current line.
    pub fn skip_line_comment(&'a mut self) {
        while self.peek_char() != Some('\n') && self.peek_next_char() != Some('\0') {
            self.next_char();
        }
    }

    /// Skips the rest of a block comment.
    pub fn skip_block_comment(&'a mut self) {
            let mut nesting = 1;
            while nesting > 0 {
                if let Some(c) = self.source.peek().cloned() {
                    if c == '\0' {
                        return;
                    }
                    lex_error(self.current_line, "Unterminated block comment.".to_owned());
                }
    
                if let Some('/') = self.source.peek().cloned() {
                    if let Some('*') = self.peek_next_char() {
                        self.next_char();
                        self.next_char();
                        nesting += 1;
                        continue;
                    }
                }
    
                if let Some('*') = self.source.peek().cloned() {
                    if let Some('/') = self.peek_next_char() {
                        self.next_char();
                        self.next_char();
                        nesting -= 1;
                        continue;
                    }
                }
    
                // Regular comment character.
                self.next_char();
            }
    }

    /// Reads the next character, which should be a hex digit (0-9, a-f, or A-F) and
    /// returns its numeric value. If the character isn't a hex digit, returns -1.
    fn read_hex_digit(&'a mut self) -> i8 {
        let _c = self.peek_char();
        self.current_offset += 1;
        if let Some('\n') = _c {
            self.current_line += 1;
        }
        
        self.source.advance_by(self.current_offset as usize);
        let c = self.source.next();

        if let Some(_c) =  c {
            if _c >= '0' && _c <= '9' {
                _c as u8 - b'0'; 
            }
            if _c >= 'a' && _c <= 'f' {
                _c as u8 - b'a' + 10; 
            }
            if _c >= 'A' && _c <= 'F' {
                _c as u8 - b'A' + 10; 
            }
        }
        // Don't consume it if it isn't expected. Keeps us from reading past the end
		// of an unterminated string.
        self.current_offset -= 1; 
        return -1;
    }

    pub fn read_hex_number(&'a self, parser: &'a mut Parser<'a>) {
        // Skip past the `x` used to denote a hexadecimal literal.
        self.next_char();
        // Iterate over all the valid hexadecimal digits found.
        while self.read_hex_digit() != -1 {
            continue;
        }

        parser.make_number();
    }

    pub fn read_number(&'a self, parser:&'a mut Parser<'a>){
        if let Some(c) = self.peek_char() {
            while is_digit(c) {
                self.next_char();
            }
        }
        
        // See if it has a floating point. Make sure there is a digit after the "."
		// so we don't get confused by method calls on number literals.
        if let Some(c) = self.peek_char() {
            if let Some(cc) = self.peek_next_char() {
                if c == '.' && is_digit(cc) {
                    self.next_char();
                    while is_digit(cc) {
                        self.next_char();
                    }
                }
            }
        }

        if let Some(cc) = self.peek_next_char() {
            // See if the number is in scientific notation.
            if self.match_char('e') && is_digit(cc) {
                // Allow a single positive/negative exponent symbol.
                if !self.match_char('+') {
                    self.match_char('-');
                }
                if let Some(c) = self.peek_char() {
                    if is_digit(c) {
                        lex_error(self.current_line, "Unterminated scientific notation.".to_owned());
                    }

                    while is_digit(c) {
                        self.next_char();
                    }
                }
                
            }
        }


        parser.make_number();   
    }

    /// Reads [digits] hex digits in a string literal and returns their number value.
    pub fn read_hex_escape(&'a self, digits: usize, description: &str) -> u8 {
        let mut _value = 0;
        for _ in 0..digits {
            if let Some(c) = self.peek_char() {
                if c == '"' || c == '\0' {
                    lex_error(self.current_line, format!("Incomplete {} escape sequence.", description));
                    // Don't consume it if it isn't expected. Keeps us from reading past the
                    // end of an unterminated string.
                    self.current_offset -= 1; 
                    break;
                }
            }
        
            let digit = self.read_hex_digit();
            if digit == -1 {
                lex_error(self.current_line,format!("Invalid {} escape sequence.", description));
                break;
            }
            _value = (_value * 16) | digit;
        }

        _value as u8
    }

   // Finishes lexing a string literal.
   pub fn read_string(&'a mut self, parser: Parser<'a>) {
        let mut string = String::new();
        let mut tokenType = TokenType::String;

        loop {
            if let Some(c) = self.next_char() {
                if c == '"' {
                    break;
                }

                if c == '\0' {
                    lex_error(self.current_line,"Unterminated string.".to_owned());
                    // Don't consume it if it isn't expected. Keeps us from reading past the
                    // end of an unterminated string.
                    self.current_offset -= 1;
                    break;
                }

                if c == '%' {
                    if parser.num_parens < MAX_INTERPOLATION_NESTING {
                        // TODO: Allow format string.
                        if let Some(_c) = self.next_char() {
                            if _c != '(' {
                                lex_error(self.current_line,"Expect '(' after '%%'".to_owned());
                            }
                        }
                        let idx = (parser.num_parens + 1) as usize;
                        parser.parens[idx] = 1;
                        tokenType = TokenType::Interpolation;
                        break; 
                    }

                    lex_error(self.current_line, format!("Interpolation may only nest {} levels deep.", MAX_INTERPOLATION_NESTING));
                }

                
            
                if c == '\\' {
                    match self.next_char() {
                        Some('"') => {string.push('"');}
                        Some('\\') => {string.push('\\');}
                        Some('%') => {string.push('%');}
                        Some('0') => {string.push('\0');}
                        Some('a') => {string.push_str("\x07");}
                        Some('b') => {string.push_str("\x08");}
                        Some('e') => {string.push_str("\x1B");}
                        Some('f') => {string.push_str("\x0C");}
                        Some('n') => {string.push('\n');}
                        Some('r') => {string.push('\r');}
                        Some('t') => {string.push('\t');}
                        Some('u') => {self.read_unicode_escape(&mut string, 4);}
                        Some('U') => {self.read_unicode_escape(&mut string, 8);}
                        Some('v') => {string.push_str("\x0B");}
                        Some('x') => {
                            string.push(self.read_hex_escape(2, "byte") as char);
                        }
                        None => {
                            if let Some(cc) = self.nth(self.current_offset - 1) {
                                lex_error(self.current_line, format!("Invalid escape character '{}'.", cc));
                            }
                            
                        }
                        _ => {}
                    }
                } else {
                    string.push(c);
                }
            }
        }

        parser.make_token(tokenType);
    }

    /// Reads a hex digit Unicode escape sequence in a string literal.
    fn read_unicode_escape(&'a self, string: &mut String, length: usize) {
        let mut _value = self.read_hex_escape(length, "Unicode");

        // Grow the buffer enough for the encoded result.
        let num_bytes = wren_utf8_encode_num_bytes(_value);
        if num_bytes != 0 {
            // let mut v: Vec<u8> = Vec::new();
            for _ in 0..num_bytes {
                string.push(char::from_u32(0).unwrap());
            }
             //= std::str::from_utf8_unchecked(&v);
        }
    }

    pub fn read_raw_string(&'a mut self) {
        let mut string = String::new();
        let tokenType = TokenType::String;

        //consume the second and third "
        self.next_char();
        self.next_char();

        let mut skip_start = 0 as i32;
        let mut first_new_line = -1 as i32;


        let mut skip_end = -1 as i32;
        let mut last_new_line = -1 as i32;

        loop {
            let c = self.next_char();
            let c1 = self.peek_char();
            let c2 = self.peek_next_char();

            match c {
                Some('\n') => {
                    last_new_line = string.len() as i32;
                    skip_end = last_new_line;
                    first_new_line = if first_new_line == -1 { string.len() as i32 } else  { first_new_line };
                }
                None => {}
                _ => {}
            }

            if c == Some('"') && c1 == Some('"') && c2 == Some('"') {
               break;
            }

            let is_white_space: bool = c == Some(' ') || c == Some('\t');
            skip_end = if c == Some('\n') || is_white_space  { skip_end } else {-1};
            // If we haven't seen a newline or other character yet, 
            // and still seeing whitespace, count the characters 
            // as skippable till we know otherwise

            let skippable: bool = skip_start != -1 && is_white_space && first_new_line == -1;
            skip_start = if skippable {string.len() as i32 + 1} else {skip_start};

            // We've counted leading whitespace till we hit something else, 
            // but it's not a newline, so we reset skipStart since we need these characters
            if first_new_line == -1 && !is_white_space && c != Some('\n') {
                skip_start = -1;
            }

            if c == Some('\0') && c1 == Some('\0') && c2 == Some('\0') {
                lex_error(self.current_line, "Unterminated raw string.".to_owned());
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
}






