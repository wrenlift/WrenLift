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

#![feature(iter_advance_by)]
#[macro_use]
extern crate lazy_static;

use std::cell::Cell;

mod compiler;

// use compiler::wren_compiler::Compiler;
use compiler::wren_parser::Parser;


fn build<'a> () {
    let source = "var isDone = false || true System.print(1 != 2 ? \"math is sane\" : \"math is not sane!\")";
    let mut _parser = Parser::new(source, Some("main"));
    _parser.next_token();
    // let _compiler = Compiler::new(_parser, None);
    //compiler::wren_parser::next_token(.as_mut());

    // _compiler.print();
}

 
fn main(){
    build();
}