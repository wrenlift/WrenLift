use wren_lift::runtime::vm::VM;
use wren_lift::runtime::engine::InterpretResult;

fn run(src: &str) -> (InterpretResult, String) {
    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let r = vm.interpret("main", src);
    let o = vm.take_output();
    (r, o)
}

fn dump_mir(src: &str) {
    use wren_lift::sema::resolve::resolve_with_prelude;
    use wren_lift::mir::builder::lower_module;

    let parse_result = wren_lift::parse::parser::parse(src);
    let mut interner = parse_result.interner;

    let core_names = [
        "Object", "Class", "Bool", "Num", "String", "List", "Map",
        "Range", "Null", "Fn", "Fiber", "System", "Sequence",
    ];
    let prelude: Vec<wren_lift::intern::SymbolId> =
        core_names.iter().map(|n| interner.intern(n)).collect();
    let resolve_result = resolve_with_prelude(&parse_result.module, &interner, &prelude);
    let module_mir = lower_module(&parse_result.module, &mut interner, &resolve_result);

    for class in &module_mir.classes {
        for method in &class.methods {
            println!("=== {} :: {} ===", class.name, method.signature);
            println!("{}", method.mir.dump());
        }
    }
}

fn main() {
    // Test: matched var with &&
    let src = r#"class SM {
    construct new() {
        _state = "idle"
        _log = []
    }
    state { _state }
    transition(event) {
        var matched = false
        if (_state == "idle" && event == "start") {
            _state = "running"
            matched = true
        }
        if (!matched && _state == "running" && event == "pause") {
            _state = "paused"
            matched = true
        }
        if (matched) {
            _log.add(_state)
        } else {
            _log.add("invalid: " + _state + " + " + event)
        }
    }
    log { _log }
}

var sm = SM.new()
sm.transition("start")
sm.transition("pause")
System.print(sm.state)"#;

    dump_mir(src);
    let (r, o) = run(src);
    println!("result => {:?} | {:?}", r, o.trim());
}
