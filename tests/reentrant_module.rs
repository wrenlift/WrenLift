//! Loading a class from a host method must preserve the caller's module slots.
use wren_lift::runtime::{
    engine::{ExecutionMode, InterpretResult},
    object::ObjClass,
    value::Value,
    vm::{VM, VMConfig},
};

#[test]
fn host_can_install_modules_during_an_interpreted_expression() {
    fn install(vm: &mut VM, context: usize, _: &[Value]) -> Value {
        let blob = unsafe { &*(context as *const Vec<u8>) };
        for i in 0..64 {
            assert_eq!(
                vm.interpret_bytecode(&format!("lazy-{i}"), blob),
                InterpretResult::Success
            );
        }
        Value::num(7.0)
    }
    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    });
    vm.output_buffer = Some(String::new());
    let blob = vm.compile_source_to_blob("class Payload {}\n").unwrap();
    assert_eq!(
        vm.interpret("host", "class Host {}\n"),
        InterpretResult::Success
    );
    let class = vm
        .find_imported_var_from("Host", "host")
        .unwrap()
        .as_object()
        .unwrap() as *mut ObjClass;
    let signature = vm.interner.intern("static:install()");
    unsafe { (*class).bind_host(signature, install, &blob as *const Vec<u8> as usize) };
    assert_eq!(
        vm.interpret(
            "main",
            r#"
import "host" for Host
var before = 42
var after = Host.install()
System.print(before)
System.print(after)
before = after + 1
System.print(before)
"#
        ),
        InterpretResult::Success
    );
    assert_eq!(vm.take_output(), "42\n7\n8\n");
}
