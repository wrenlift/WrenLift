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

use compiler::Compiler;
use scanner::{Signature, SignatureType, Precedence, Token, TokenType};

pub fn InfixOp(compiler: Compiler, canAssign: bool){
    // let rule = compiler.getRule(compiler.parser.previous.type);

    // // An infix operator cannot end an expression.
    // compiler.ignoreNewlines();

    // // Compile the right-hand side.
    // compiler.parsePrecedence(rule.precedence + 1);

    // // Call the operator method on the left-hand side.
    // let signature: Signature = {
    //     name: rule.name,
    //     length: rule.name.length,
    //     sigType: SignatureType::Method,
    //     arity: 1
    // };
    // compiler.callSignature(CODE_CALL_0, signature);
}

/// Compiles a method signature for an infix operator.
pub fn InfixSignature(compiler: Compiler,  signature: &mut Signature) -> Option<Signature> {
    // Add the RHS parameter.
    signature.sigType = SignatureType::Method;
    signature.arity = 1;
    // Parse the parameter name.
    // compiler.consume(TokenType::LeftParen, "Expect '(' after operator name.");
    // compiler.declareNamedVariable();
    // compiler.consume(TokenType::RightParen, "Expect ')' after parameter name.");
    Some(signature)
}

/// Unary operators like `-foo`.
pub fn UnaryOp(compiler: Compiler, canAssign: bool) {
    // let rule = compiler.getRule(compiler.parser.previous.type);
    // compiler.ignoreNewlines();
    // // Compile the argument.
    // compiler.parsePrecedence((Precedence::PrecUnary + 1));
    // // Call the operator method on the left-hand side.
    // compiler.callMethod(0, rule.name);
}

/// Compiles a method signature for an unary operator (i.e. "!").
pub fn UnarySignature(compiler: Compiler,  signature: &mut Signature)  -> Option<Signature> {
    // Do nothing. The name is already complete.
	signature.sigType = SignatureType::Getter;
	Some(signature)
}

/// Compiles a method signature for an operator that can either be unary or
/// infix (i.e. "-").
pub fn MixedSignature(compiler: Compiler, signature: &mut Signature) -> Option<Signature> {
    signature.sigType = SignatureType::Getter;
    // If there is a parameter, it's an infix operator, otherwise it's unary.
    // if (compiler.match(TokenType::LeftParen)) {
    //     // Add the RHS parameter.
    //     signature.sigType = SignatureType::Method;
    //     signature.arity = 1;

    //     // Parse the parameter name.
    //     compiler.declareNamedVariable();
    //     compiler.consume(TokenType::RightParen, "Expect ')' after parameter name.");
    // }
    Some(signature)
}

fn maybeSetter(compiler: Compiler, signature:  &mut Signature) -> Option<Signature> {
    None
}


pub fn Grouping(compiler: Compiler, canAssign: bool){
    // compiler.expression();
    // compiler.consume(TokenType::RightParen, "Expect ')' after expression.");
}

pub fn List(compiler: Compiler, canAssign: bool){}

pub fn SubscriptSignature(compiler: Compiler,  signature: &mut Signature) -> Option<Signature> {
        signature.sigType = SignatureType::Subscript;

		// The signature currently has "[" as its name since that was the token that
		// matched it. Clear that out.
		signature.length = 0;

		// Parse the parameters inside the subscript.
		signature = finishParameterList(compiler, signature);
		// compiler.consume(TokenType::RighBracket, "Expect ']' after parameters.");

		maybeSetter(compiler, signature);
		signature;
}


pub fn ConstructorSignature(compiler: Compiler, signature: &mut  Signature) -> Option<Signature> {
    Some(signature)
}


fn finishParameterList(compiler: Compiler,  signature: &mut Signature) -> Option<Signature> {
    Some(signature)
}


pub fn Subscript(compiler: Compiler, canAssign: bool){}

pub fn Map(compiler: Compiler, canAssign: bool){}

pub fn Call(compiler: Compiler, canAssign: bool){}

pub fn Or(compiler: Compiler, canAssign: bool){}

pub fn And(compiler: Compiler, canAssign: bool){}

pub fn Conditional(compiler: Compiler, canAssign: bool){}

pub fn Boolean(compiler: Compiler, canAssign: bool){}

pub fn Null(compiler: Compiler, canAssign: bool){}

pub fn Super(compiler: Compiler, canAssign: bool){}

pub fn This(compiler: Compiler, canAssign: bool){}

pub fn Field(compiler: Compiler, canAssign: bool) {}

pub fn StaticField(compiler: Compiler, canAssign: bool) {}

pub fn Name(compiler: Compiler, canAssign: bool) {}


pub fn NamedSignature(compiler: Compiler, signature: &mut Signature) -> Option<Signature> {
    Some(signature)
}

pub fn Literal(compiler: Compiler, canAssign: bool) {}

pub fn StringInterpolation(compiler: Compiler, canAssign: bool) {}