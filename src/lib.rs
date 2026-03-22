#![feature(thread_local)]

pub mod ast;
pub mod capi;
pub mod codegen;
pub mod diagnostics;
pub mod intern;
pub mod mir;
pub mod parse;
pub mod runtime;
pub mod sema;
