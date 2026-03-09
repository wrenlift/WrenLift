use wren_lift::runtime::engine::InterpretResult;
use wren_lift::runtime::vm::VM;

fn run(src: &str) -> (InterpretResult, String) {
    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let r = vm.interpret("main", src);
    let o = vm.take_output();
    (r, o)
}

fn dump_mir(src: &str) {
    use wren_lift::mir::builder::lower_module;
    use wren_lift::sema::resolve::resolve_with_prelude;

    let parse_result = wren_lift::parse::parser::parse(src);
    let mut interner = parse_result.interner;
    let core_names = [
        "Object", "Class", "Bool", "Num", "String", "List", "Map", "Range", "Null", "Fn", "Fiber",
        "System", "Sequence",
    ];
    let prelude: Vec<wren_lift::intern::SymbolId> =
        core_names.iter().map(|n| interner.intern(n)).collect();
    let resolve_result = resolve_with_prelude(&parse_result.module, &interner, &prelude);

    println!("=== RESOLUTIONS ===");
    for (span_start, resolved) in &resolve_result.resolutions {
        println!("  span {} => {:?}", span_start, resolved);
    }

    let module_mir = lower_module(&parse_result.module, &mut interner, &resolve_result);

    println!("=== TOP-LEVEL MIR ===");
    println!("{}", module_mir.top_level.dump());
}

fn main() {
    dump_mir(
        r#"var x = "outer"
{
    var x = "inner"
    System.print(x)
}
System.print(x)"#,
    );

    println!("\n=== EXECUTION ===");
    let (r, o) = run(r#"var x = "outer"
{
    var x = "inner"
    System.print(x)
}
System.print(x)"#);
    println!("{:?} | {:?}", r, o.trim());
}
