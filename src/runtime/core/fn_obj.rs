use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn fn_new(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // Validate that args[1] is a closure/fn, return it.
    // For now just return args[1].
    args[1]
}

fn fn_arity(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(ptr) = args[0].as_object() {
        let header = ptr as *const crate::runtime::object::ObjHeader;
        unsafe {
            if (*header).obj_type == crate::runtime::object::ObjType::Closure {
                let closure = ptr as *const crate::runtime::object::ObjClosure;
                let func = (*closure).function;
                return Value::num((*func).arity as f64);
            }
        }
    }
    Value::num(0.0)
}

fn fn_to_string(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string("<fn>".to_string())
}

fn fn_call_stub(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.runtime_error("Receiver is not a function.".to_string());
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
        // Mark this symbol for O(1) "is call?" checks in the dispatch hot path.
        let sym = vm.interner.intern(sig);
        vm.mark_call_symbol(sym);
    }

    // Also mark the bare "call" getter (used as `fn.call` without parens in Wren).
    let call_bare = vm.interner.intern("call");
    vm.mark_call_symbol(call_bare);
}
