use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn fn_new(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // Validate that args[1] is a closure/fn, return it.
    // For now just return args[1].
    args[1]
}

fn fn_arity(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Extract ObjClosure from receiver, get function, return arity as Num.
    // For now return 0 as placeholder.
    Value::num(0.0)
}

fn fn_to_string(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string("<fn>".to_string())
}

fn fn_call_stub(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Fn.call not yet implemented.".to_string());
    Value::null()
}

pub fn bind(vm: &mut VM) {
    let class = vm.fn_class;

    vm.primitive_static(class, "new(_)", fn_new);
    vm.primitive(class, "arity", fn_arity);
    vm.primitive(class, "toString", fn_to_string);

    // call() through call(_,_,...,_) with 0 to 16 arguments.
    let call_signatures: [&str; 17] = [
        "call()",
        "call(_)",
        "call(_,_)",
        "call(_,_,_)",
        "call(_,_,_,_)",
        "call(_,_,_,_,_)",
        "call(_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)",
        "call(_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_)",
    ];

    for sig in &call_signatures {
        vm.primitive(class, sig, fn_call_stub);
    }
}
