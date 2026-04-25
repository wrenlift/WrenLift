/// `hatch` module — in-process module reloading for dev-mode frameworks.
///
/// Exposed as `import "hatch" for Hatch`.
///
/// ## Methods
/// - `Hatch.reload(name)`         — re-parse + re-install the named module.
///                                   Returns true on success, false on failure.
///                                   Classes declared in the module are mutated
///                                   in place (same `ObjClass` pointer), so
///                                   instances that predate the reload continue
///                                   to dispatch to the new method bodies.
/// - `Hatch.moduleMtime(path)`    — file mtime in seconds-since-epoch, or null
///                                   when the path can't be stat'd.
/// - `Hatch.modulePath(name)`     — canonical on-disk path for a loaded module,
///                                   or null for built-in / unloaded modules.
/// - `Hatch.loadedModules`        — list of module names currently registered.
use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;

fn hatch_reload(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(name) = super::validate_string(ctx, args[1], "Module name") else {
        return Value::bool(false);
    };
    match ctx.reload_module(&name) {
        Ok(()) => Value::bool(true),
        Err(msg) => {
            ctx.runtime_error(msg);
            Value::bool(false)
        }
    }
}

fn hatch_module_mtime(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    match std::fs::metadata(&path).and_then(|m| m.modified()) {
        Ok(mtime) => {
            let secs = mtime
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            Value::num(secs)
        }
        Err(_) => Value::null(),
    }
}

fn hatch_module_path(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(name) = super::validate_string(ctx, args[1], "Module name") else {
        return Value::null();
    };
    match ctx.module_canonical_path(&name) {
        Some(path) => ctx.alloc_string(path),
        None => Value::null(),
    }
}

fn hatch_loaded_modules(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let names = ctx.loaded_module_names();
    let elements: Vec<Value> = names.into_iter().map(|n| ctx.alloc_string(n)).collect();
    ctx.alloc_list(elements)
}

/// `Hatch.onReload(fn)` — register a Wren `Fn` invoked with the
/// reloaded module's canonical path after every successful module
/// reload. Frameworks use this to re-run setup that registers state
/// into pre-existing instances (e.g. routes into an `App` router).
fn hatch_on_reload(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !is_closure(args[1]) {
        ctx.runtime_error("Hatch.onReload: argument must be a function.".to_string());
        return Value::null();
    }
    ctx.register_reload_callback(args[1]);
    Value::null()
}

/// `Hatch.beforeReload(fn)` — register a callback that fires with
/// the module path BEFORE each reload re-runs top-level. Used by
/// frameworks to flush state the reloaded module owns (e.g. clear
/// the route table so a re-running top-level can re-register fresh
/// `app.get/post(...)` calls without doubling up).
fn hatch_before_reload(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !is_closure(args[1]) {
        ctx.runtime_error("Hatch.beforeReload: argument must be a function.".to_string());
        return Value::null();
    }
    ctx.register_before_reload_callback(args[1]);
    Value::null()
}

fn is_closure(v: Value) -> bool {
    if !v.is_object() {
        return false;
    }
    let header = unsafe { &*(v.as_object().unwrap() as *const crate::runtime::object::ObjHeader) };
    header.obj_type == crate::runtime::object::ObjType::Closure
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(vm: &mut crate::runtime::vm::VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("Hatch", vm.object_class);

    vm.primitive_static(class, "reload(_)", hatch_reload);
    vm.primitive_static(class, "moduleMtime(_)", hatch_module_mtime);
    vm.primitive_static(class, "modulePath(_)", hatch_module_path);
    vm.primitive_static(class, "loadedModules", hatch_loaded_modules);
    vm.primitive_static(class, "onReload(_)", hatch_on_reload);
    vm.primitive_static(class, "beforeReload(_)", hatch_before_reload);

    class
}
