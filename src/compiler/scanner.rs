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

use compiler::{Compiler};
use grammar;
use std::fmt;
use alloc::vec::Vec;
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub enum TokenType {
    LeftParen,
    RightParen,
    LeftBracket,
    RighBracket,
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


#[derive(Debug, Clone)]
pub struct Token<'a> {
    token_type: Option<TokenType>,
    start: &'a str,
    length: usize,
    line: usize
}

impl<'a> Display for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#?}", self.0)
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
pub struct Keyword<'a> {
    identifier:Option<&'a str>,
    length:usize,
    tokenType:TokenType
}

impl<'a> Display for Keyword<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#?}", self.0)
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
    Keyword{identifier: None,        length: 0, tokenType: TokenType::Eof) },
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
pub struct Signature{
    sigType: SignatureType,
    length: usize,
    name: &str,
    arity: i32
}

impl Display for Signature {
    fn parameterList(numParams: u32, leftBracket: &str, rightBracket: &str) -> &str {
        let i = 0;
        let ret = leftBracket.to_owned();
        while i < numParams && i < MAX_PARAMETERS {
            if i > 0 {
                ret.push_str(",");
            }
            ret.push_str("_");
            i += 1;
        }
        ret.push_str(rightBracket);
        ret.as_ref()
    }
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = self.name.to_owned();
        
        match self.sigType {
            SignatureType::Method => name.push_str(parameterList(self.arity, "(", ")")),
            SignatureType::Getter => {},
            SignatureType::Setter => {
                name.push_str("=");
                name.push_str(parameterList(self.arity, "(", ")"));
            } 
            SignatureType::Subscript => {
                name.push_str("");
                name.push_str(parameterList(self.arity, "[", "]"));
            }
            SignatureType::SubscriptSetter => {
                name.push_str(parameterList(self.arity - 1, "[", "]"));
                name.push_str("=");
                name.push_str(parameterList(self.arity, "(", ")"));
            }
            SignatureType::Initializer => {
                name.push_str(format!("init {}", name));
                name.push_str(parameterList(self.arity, "(", ")"));
            }
        }
        write!(f, "{}", self.0)
    }
}


type GrammarFn = fn(compiler: Compiler, canAssign: bool);
type SignatureFn = fn(compiler: Compiler, signature: &mut Signature) -> Option<Signature>;


#[derive(Debug, Clone)]
pub struct GrammarRule<'a> {
    prefix: Option<GrammarFn>,
    infix: Option<GrammarFn>,
    method: Option<SignatureFn>,
    precedence: Option<Precedence>,
    name: Option<&'a str>,
}


impl<'a> GrammarRule<'a> {
    #[inline]
    pub fn unused() -> Self {
        Self { precedence: Some(Precedence::PrecNone)};
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
    pub fn infix_operator(pre: Option<Precedence>, name: &'a str) -> Self {
        Self {
            name: Some(name),
            infix: Some(grammar::InfixOp),
            method: Some(grammar::InfixSignature),
            prefix: None,
            precedence: pre,
        }
    }

    #[inline]
    pub fn prefix_operator(name: &'a str) -> Self {
        Self {
            name: Some(name),
            prefix: Some(grammar::UnaryOp),
            method: Some(grammar::UnarySignature),
            precedence: Some(Precedence::PrecNone),
            infix: None,
        }
    }

    #[inline]
    pub fn operator(name: &'a str) -> Self {
        Self {
            name: Some(name),
            prefix: Some(grammar::UnaryOp),
            infix: Some(grammar::InfixOp),
            method: Some(grammar::MixedSignature),
            precedence: Some(Precedence::PrecTerm),
        }
    }

}


pub const GrammarRules: HashMap<TokenType, GrammarRule> = [
    (TokenType::LeftParen, GrammarRule::prefix(Some(grammar::Grouping))),
    (TokenType::RightParen, GrammarRule::unused()),
    (TokenType::LeftBracket, GrammarRule{ prefix: Some(grammar::List), infix: Some(grammar::Subscript), method: Some(grammar::SubscriptSignature), precedence: Some(Precedence::PrecCall) }),
    (TokenType::RighBracket, GrammarRule::unused()),
    (TokenType::LeftBrace, GrammarRule::prefix(Some(grammar::Map))),
    (TokenType::RightBrace, GrammarRule::unused()),
    (TokenType::Colon, GrammarRule::unused()),
    (TokenType::Dot, GrammarRule::infix(Some(Precedence::PrecCall), Some(grammar::Call))),
    (TokenType::DotDot, GrammarRule::infix_operator(Some(Precedence::PrecRange), "..")()),
    (TokenType::DotDotDot, GrammarRule::infix_operator(Some(Precedence::PrecRange), "...")()),
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
    (TokenType::Construct, GrammarRule{ method: Some(grammar::ConstructorSignature), precedence: Some(Precedence::PrecNone) }),
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
    (TokenType::Name, GrammarRule{ prefix: Some(grammar::Name), method: grammar::NamedSignature, precedence: Some(Precedence::PrecNone) }),
    (TokenType::Number, GrammarRule::prefix(Some(grammar::Literal))),
    (TokenType::String, GrammarRule::prefix(Some(grammar::Literal))),
    (TokenType::Interpolation, GrammarRule::prefix(Some(grammar::StringInterpolation))),
    (TokenType::Line, GrammarRule::unused()),
    (TokenType::Error, GrammarRule::unused()),
    (TokenType::Eof, GrammarRule::unused()),
].iter().cloned().collect();












