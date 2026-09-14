//! `IsolateCore`: spawn a VM on another thread, pass values by copy,
//! and channels between isolates. Handles are numbers; a package
//! wraps them in classes.

use crate::runtime::core::fiber::{deadline_arg, park_on, sched_of, sched_vm, token_arg};
use crate::runtime::isolate::{self, Receive, Xfer};
use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn export_arg(ctx: &mut dyn NativeContext, what: &str, v: Value) -> Option<Xfer> {
    match isolate::export(ctx, v) {
        Ok(x) => Some(x),
        Err(msg) => {
            ctx.runtime_error(format!("{what}: {msg}."));
            None
        }
    }
}

fn string_arg(ctx: &mut dyn NativeContext, what: &str, v: Value) -> Option<String> {
    if crate::runtime::core::is_string(v) {
        Some(crate::runtime::core::as_string(v).to_owned())
    } else {
        ctx.runtime_error(format!("{what}: expected a string."));
        None
    }
}

/// `IsolateCore.spawn(module, arg)` → id. Runs `import "module"` on a
/// new thread with `arg` (copied) as its `IsolateCore.arg`.
fn isolate_spawn(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "IsolateCore.spawn") else {
        return Value::null();
    };
    let Some(module) = string_arg(ctx, "IsolateCore.spawn", args[1]) else {
        return Value::null();
    };
    let Some(arg) = export_arg(ctx, "IsolateCore.spawn", args[2]) else {
        return Value::null();
    };
    match isolate::spawn(unsafe { &*vm }, module, arg) {
        Ok(id) => Value::num(id as f64),
        Err(msg) => {
            ctx.runtime_error(format!("IsolateCore.spawn: {msg}."));
            Value::null()
        }
    }
}

/// `IsolateCore.arg` → the value this isolate was spawned with.
fn isolate_arg(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let vm = ctx.krio_vm_raw_ptr() as *mut VM;
    if vm.is_null() {
        return Value::null();
    }
    let arg = unsafe { (*vm).isolate_arg.clone() };
    arg.map_or_else(Value::null, |x| isolate::import(ctx, &x))
}

fn isolate_of(
    ctx: &mut dyn NativeContext,
    what: &str,
    v: Value,
) -> Option<std::sync::Arc<isolate::Isolate>> {
    let id = token_arg(ctx, what, v)?;
    match isolate::isolate(id) {
        Some(i) => Some(i),
        None => {
            ctx.runtime_error(format!("{what}: unknown isolate {id}."));
            None
        }
    }
}

fn isolate_is_done(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    isolate_of(ctx, "IsolateCore.isDone", args[1])
        .map_or_else(Value::null, |i| Value::bool(i.is_done()))
}

fn isolate_error(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(i) = isolate_of(ctx, "IsolateCore.error", args[1]) else {
        return Value::null();
    };
    i.error().map_or_else(Value::null, |e| ctx.alloc_string(e))
}

/// `IsolateCore.join(id, ms)` → whether the isolate finished within
/// `ms` (null waits for it).
fn isolate_join(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "IsolateCore.join") else {
        return Value::null();
    };
    let Some(i) = isolate_of(ctx, "IsolateCore.join", args[1]) else {
        return Value::null();
    };
    let Ok(deadline) = deadline_arg(ctx, "IsolateCore.join", args[2]) else {
        return Value::null();
    };
    loop {
        let token = sched_of(vm).new_waiter();
        if i.join_with(token) {
            sched_of(vm).discard_waiter(token);
            return Value::bool(true);
        }
        match park_on(ctx, vm, token, deadline) {
            None => return Value::null(),
            Some(true) => continue,
            Some(false) => {
                i.forget_joiner(token);
                return Value::bool(i.is_done());
            }
        }
    }
}

fn isolate_cpus(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(std::thread::available_parallelism().map_or(1.0, |n| n.get() as f64))
}

// --- Channels ---

fn channel_new(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(isolate::new_channel() as f64)
}

fn channel_of(
    ctx: &mut dyn NativeContext,
    what: &str,
    v: Value,
) -> Option<std::sync::Arc<isolate::Channel>> {
    let id = token_arg(ctx, what, v)?;
    match isolate::channel(id) {
        Some(c) => Some(c),
        None => {
            ctx.runtime_error(format!("{what}: unknown channel {id}."));
            None
        }
    }
}

/// `IsolateCore.send(ch, value)` → false once the channel is closed.
fn channel_send(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(ch) = channel_of(ctx, "IsolateCore.send", args[1]) else {
        return Value::null();
    };
    let Some(x) = export_arg(ctx, "IsolateCore.send", args[2]) else {
        return Value::null();
    };
    Value::bool(ch.send(x))
}

/// `IsolateCore.receive(ch, ms)` → `[value]`, or null when nothing
/// arrived before `ms` (null waits) or the channel is closed.
fn channel_receive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "IsolateCore.receive") else {
        return Value::null();
    };
    let Some(ch) = channel_of(ctx, "IsolateCore.receive", args[1]) else {
        return Value::null();
    };
    let Ok(deadline) = deadline_arg(ctx, "IsolateCore.receive", args[2]) else {
        return Value::null();
    };
    let boxed = |ctx: &mut dyn NativeContext, x: Xfer| {
        let v = isolate::import(ctx, &x);
        ctx.alloc_list(vec![v])
    };
    loop {
        let token = sched_of(vm).new_waiter();
        match ch.wait_with(token) {
            Receive::Value(x) => {
                sched_of(vm).discard_waiter(token);
                return boxed(ctx, x);
            }
            Receive::Closed => {
                sched_of(vm).discard_waiter(token);
                return Value::null();
            }
            Receive::Parked => {}
        }
        match park_on(ctx, vm, token, deadline) {
            None => return Value::null(),
            Some(true) => continue,
            Some(false) => {
                ch.forget_receiver(token);
                return match ch.try_receive() {
                    Some(x) => boxed(ctx, x),
                    None => Value::null(),
                };
            }
        }
    }
}

fn channel_try_receive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(ch) = channel_of(ctx, "IsolateCore.tryReceive", args[1]) else {
        return Value::null();
    };
    match ch.try_receive() {
        Some(x) => {
            let v = isolate::import(ctx, &x);
            ctx.alloc_list(vec![v])
        }
        None => Value::null(),
    }
}

fn channel_close(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(ch) = channel_of(ctx, "IsolateCore.close", args[1]) {
        ch.close();
    }
    Value::null()
}

fn channel_is_closed(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    channel_of(ctx, "IsolateCore.isClosed", args[1])
        .map_or_else(Value::null, |c| Value::bool(c.is_closed()))
}

fn channel_pending(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    channel_of(ctx, "IsolateCore.pending", args[1])
        .map_or_else(Value::null, |c| Value::num(c.pending() as f64))
}

/// `IsolateCore.drop(ch)`: forget a channel, closing it.
fn channel_drop(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(id) = token_arg(ctx, "IsolateCore.drop", args[1]) {
        isolate::drop_channel(id);
    }
    Value::null()
}

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("IsolateCore", vm.object_class);

    vm.primitive_static(class, "spawn(_,_)", isolate_spawn);
    vm.primitive_static(class, "arg", isolate_arg);
    vm.primitive_static(class, "isDone(_)", isolate_is_done);
    vm.primitive_static(class, "error(_)", isolate_error);
    vm.primitive_static(class, "join(_,_)", isolate_join);
    vm.primitive_static(class, "cpus", isolate_cpus);

    vm.primitive_static(class, "channel()", channel_new);
    vm.primitive_static(class, "send(_,_)", channel_send);
    vm.primitive_static(class, "receive(_,_)", channel_receive);
    vm.primitive_static(class, "tryReceive(_)", channel_try_receive);
    vm.primitive_static(class, "close(_)", channel_close);
    vm.primitive_static(class, "isClosed(_)", channel_is_closed);
    vm.primitive_static(class, "pending(_)", channel_pending);
    vm.primitive_static(class, "drop(_)", channel_drop);

    class
}
