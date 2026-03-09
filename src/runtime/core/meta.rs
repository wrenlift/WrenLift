/// Optional `meta` module — provides the `Meta` class for meta-programming.
///
/// Loaded on-demand when `import "meta" for Meta` is encountered.
///
/// ## Methods
/// - `Meta.getModuleVariables(module)` — list all module-level variable names
/// - `Meta.eval(source)` — compile and execute Wren source in current module
/// - `Meta.compile(source)` — compile source into a closure (not executed)
/// - `Meta.compileExpression(expr)` — compile an expression into a closure
use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

/// Meta.getModuleVariables(module) → List of Strings
fn meta_get_module_variables(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(module_name) = super::validate_string(ctx, args[1], "Module") else {
        return Value::null();
    };
    match ctx.get_module_variable_names(&module_name) {
        Some(names) => {
            let elements: Vec<Value> = names.into_iter().map(|n| ctx.alloc_string(n)).collect();
            ctx.alloc_list(elements)
        }
        None => {
            ctx.runtime_error(format!("Unknown module '{}'.", module_name));
            Value::null()
        }
    }
}

/// Meta.eval(source) — compile and execute source
///
/// Note: Full eval requires compiling Wren source at runtime and executing
/// it in the current module's scope. This is a simplified version that
/// interprets the source in a fresh "__eval__" module context.
fn meta_eval(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(source) = super::validate_string(ctx, args[1], "Source") else {
        return Value::null();
    };
    // We store the source string; the VM will handle compilation
    // by intercepting this in the interpreter loop.
    // For now, write a diagnostic — full eval requires VM-level support.
    let _ = source;
    ctx.runtime_error("Meta.eval() is not yet supported.".into());
    Value::null()
}

/// Meta.compile(source) — compile source into a Fn closure
fn meta_compile(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(_source) = super::validate_string(ctx, args[1], "Source") else {
        return Value::null();
    };
    ctx.runtime_error("Meta.compile() is not yet supported.".into());
    Value::null()
}

/// Meta.compileExpression(expr) — compile expression into a Fn closure
fn meta_compile_expression(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(_expr) = super::validate_string(ctx, args[1], "Source") else {
        return Value::null();
    };
    ctx.runtime_error("Meta.compileExpression() is not yet supported.".into());
    Value::null()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("Meta", vm.object_class);

    vm.primitive_static(class, "getModuleVariables(_)", meta_get_module_variables);
    vm.primitive_static(class, "eval(_)", meta_eval);
    vm.primitive_static(class, "compile(_)", meta_compile);
    vm.primitive_static(class, "compileExpression(_)", meta_compile_expression);

    class
}
