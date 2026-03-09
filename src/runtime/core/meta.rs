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

/// Meta.eval(source) — compile and execute source in the calling module scope
fn meta_eval(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(source) = super::validate_string(ctx, args[1], "Source") else {
        return Value::null();
    };
    if ctx.meta_eval("main", &source) {
        Value::null()
    } else {
        // Error already reported by the compilation/execution pipeline
        Value::null()
    }
}

/// Meta.compile(source) — compile source into a Fn closure
fn meta_compile(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(source) = super::validate_string(ctx, args[1], "Source") else {
        return Value::null();
    };
    ctx.meta_compile(&source).unwrap_or(Value::null())
}

/// Meta.compileExpression(expr) — compile expression into a Fn closure
fn meta_compile_expression(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(expr) = super::validate_string(ctx, args[1], "Source") else {
        return Value::null();
    };
    ctx.meta_compile_expression(&expr).unwrap_or(Value::null())
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
