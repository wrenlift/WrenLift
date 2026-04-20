//! Optional `os` module — process-level primitives: platform
//! string, env vars, argv, exit, tty detection.
//!
//! Loaded on-demand when `import "os" for OS` is encountered.
//!
//! Everything here is a thin wrapper over `std::env` / `std::io`
//! / `std::process`. Policy (e.g. should color be enabled?) lives
//! in `@hatch:os` on top.

use std::io::IsTerminal;

use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Platform --------------------------------------------------

fn os_platform(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string(std::env::consts::OS.to_string())
}

fn os_arch(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.alloc_string(std::env::consts::ARCH.to_string())
}

// --- Environment ----------------------------------------------

/// OS.env(name) → String, or null if unset.
fn os_env(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(name) = super::validate_string(ctx, args[1], "Name") else {
        return Value::null();
    };
    match std::env::var_os(&name) {
        Some(v) => ctx.alloc_string(v.to_string_lossy().into_owned()),
        None => Value::null(),
    }
}

/// OS.setEnv(name, value). Process-wide.
fn os_set_env(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(name) = super::validate_string(ctx, args[1], "Name") else {
        return Value::null();
    };
    let Some(value) = super::validate_string(ctx, args[2], "Value") else {
        return Value::null();
    };
    // SAFETY: std::env::set_var is marked unsafe since Rust 1.74 on
    // some platforms because it isn't thread-safe w.r.t. concurrent
    // readers. wren_lift is single-threaded at the Wren level, and
    // any background JIT threads don't read env vars, so this is
    // sound in practice.
    unsafe {
        std::env::set_var(name, value);
    }
    Value::null()
}

fn os_unset_env(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(name) = super::validate_string(ctx, args[1], "Name") else {
        return Value::null();
    };
    unsafe {
        std::env::remove_var(name);
    }
    Value::null()
}

/// OS.envMap() → Map<String, String> of every env var.
fn os_env_map(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let map = ctx.alloc_map();
    let map_ptr = map.as_object().unwrap() as *mut crate::runtime::object::ObjMap;
    for (k, v) in std::env::vars() {
        let kv = ctx.alloc_string(k);
        let vv = ctx.alloc_string(v);
        unsafe { (*map_ptr).set(kv, vv) };
    }
    map
}

// --- Args -----------------------------------------------------

/// OS.args → List<String> of process argv.
fn os_args(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let elements: Vec<Value> = std::env::args()
        .map(|a| ctx.alloc_string(a))
        .collect();
    ctx.alloc_list(elements)
}

// --- Exit -----------------------------------------------------

/// OS.exit(code). Never returns.
fn os_exit(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let code = match args[1].as_num() {
        Some(n) => n as i32,
        None => {
            ctx.runtime_error("OS.exit: code must be a number.".to_string());
            return Value::null();
        }
    };
    std::process::exit(code);
}

// --- Terminal detection ---------------------------------------

/// OS.isatty(fd) — 0=stdin, 1=stdout, 2=stderr. Any other fd is
/// rejected (we don't expose an fd table to Wren yet).
fn os_is_tty(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fd = match args[1].as_num() {
        Some(n) => n as i32,
        None => {
            ctx.runtime_error("OS.isatty: fd must be 0, 1, or 2.".to_string());
            return Value::null();
        }
    };
    let is_tty = match fd {
        0 => std::io::stdin().is_terminal(),
        1 => std::io::stdout().is_terminal(),
        2 => std::io::stderr().is_terminal(),
        _ => {
            ctx.runtime_error(format!(
                "OS.isatty: only 0 (stdin), 1 (stdout), 2 (stderr) supported, got {}.",
                fd
            ));
            return Value::null();
        }
    };
    Value::bool(is_tty)
}

// --- Identity --------------------------------------------------

/// OS.username — best-effort via $USER / $USERNAME (Windows).
/// Returns null if neither is set.
fn os_username(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let name = std::env::var_os("USER")
        .or_else(|| std::env::var_os("USERNAME"));
    match name {
        Some(n) => ctx.alloc_string(n.to_string_lossy().into_owned()),
        None => Value::null(),
    }
}

// --- Registration ---------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("OS", vm.object_class);

    vm.primitive_static(class, "platform", os_platform);
    vm.primitive_static(class, "arch", os_arch);
    vm.primitive_static(class, "env(_)", os_env);
    vm.primitive_static(class, "setEnv(_,_)", os_set_env);
    vm.primitive_static(class, "unsetEnv(_)", os_unset_env);
    vm.primitive_static(class, "envMap", os_env_map);
    vm.primitive_static(class, "args", os_args);
    vm.primitive_static(class, "exit(_)", os_exit);
    vm.primitive_static(class, "isatty(_)", os_is_tty);
    vm.primitive_static(class, "username", os_username);

    class
}
