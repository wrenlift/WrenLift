//! `Isolate` and `Channel`, the built-in module `isolate`: spawn a VM
//! on another thread and pass values to it by copy. Both are
//! instances with one field, the runtime handle; an instance itself
//! crosses between isolates as that handle.

use crate::runtime::core::fiber::{deadline_arg, park_on, sched_vm};
use crate::runtime::core::sequence::{instance_field, set_instance_field};
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

/// The handle stored in an `Isolate` or `Channel` instance.
pub(crate) fn handle_of(receiver: Value) -> u64 {
    instance_field(receiver, 0).as_num().unwrap_or(0.0) as u64
}

/// An instance of `class` carrying `id`.
pub(crate) fn wrap_handle(ctx: &mut dyn NativeContext, class: &str, id: u64) -> Value {
    let class = ctx.lookup_class(class).expect("isolate module loaded");
    let inst = ctx.alloc_instance(class);
    set_instance_field(ctx, inst, 0, Value::num(id as f64));
    inst
}

fn string_arg(ctx: &mut dyn NativeContext, what: &str, v: Value) -> Option<String> {
    if crate::runtime::core::is_string(v) {
        Some(crate::runtime::core::as_string(v).to_owned())
    } else {
        ctx.runtime_error(format!("{what}: expected a string."));
        None
    }
}

/// `Isolate.spawn(module, arg)`: run `import "module"` on a new
/// thread with `arg` (copied) as its `Isolate.arg`.
fn isolate_spawn(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Isolate.spawn") else {
        return Value::null();
    };
    let Some(module) = string_arg(ctx, "Isolate.spawn", args[1]) else {
        return Value::null();
    };
    let arg = args.get(2).copied().unwrap_or_else(Value::null);
    let Some(arg) = export_arg(ctx, "Isolate.spawn", arg) else {
        return Value::null();
    };
    match isolate::spawn(unsafe { &*vm }, module, arg) {
        Ok(id) => wrap_handle(ctx, "Isolate", id),
        Err(msg) => {
            ctx.runtime_error(format!("Isolate.spawn: {msg}."));
            Value::null()
        }
    }
}

/// `Isolate.arg`: the value this isolate was spawned with.
fn isolate_arg(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let vm = ctx.krio_vm_raw_ptr() as *mut VM;
    if vm.is_null() {
        return Value::null();
    }
    let arg = unsafe { (&mut *vm).isolate_arg.clone() };
    arg.map_or_else(Value::null, |x| isolate::import(ctx, &x))
}

fn isolate_of(
    ctx: &mut dyn NativeContext,
    what: &str,
    v: Value,
) -> Option<std::sync::Arc<isolate::Isolate>> {
    let id = handle_of(v);
    match isolate::isolate(id) {
        Some(i) => Some(i),
        None => {
            ctx.runtime_error(format!("{what}: unknown isolate."));
            None
        }
    }
}

fn isolate_is_done(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    isolate_of(ctx, "Isolate.isDone", args[0])
        .map_or_else(Value::null, |i| Value::bool(i.is_done()))
}

fn isolate_error(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(i) = isolate_of(ctx, "Isolate.error", args[0]) else {
        return Value::null();
    };
    i.error().map_or_else(Value::null, |e| ctx.alloc_string(e))
}

/// `join()` / `join(ms)`: whether the isolate finished, waiting up
/// to `ms` (forever without).
fn isolate_join(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Isolate.join") else {
        return Value::null();
    };
    let Some(i) = isolate_of(ctx, "Isolate.join", args[0]) else {
        return Value::null();
    };
    let ms = args.get(1).copied().unwrap_or_else(Value::null);
    let Ok(deadline) = deadline_arg(ctx, "Isolate.join", ms) else {
        return Value::null();
    };
    loop {
        let token = crate::runtime::sched::new_waiter(vm);
        if i.join_with(token) {
            crate::runtime::sched::discard_waiter(vm, token);
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

fn channel_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let id = isolate::new_channel();
    wrap_handle(ctx, "Channel", id)
}

fn channel_of(
    ctx: &mut dyn NativeContext,
    what: &str,
    v: Value,
) -> Option<std::sync::Arc<isolate::Channel>> {
    let id = handle_of(v);
    match isolate::channel(id) {
        Some(c) => Some(c),
        None => {
            ctx.runtime_error(format!("{what}: the channel was dropped."));
            None
        }
    }
}

/// `send(value)`: false once the channel is closed.
fn channel_send(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(ch) = channel_of(ctx, "Channel.send", args[0]) else {
        return Value::null();
    };
    let Some(x) = export_arg(ctx, "Channel.send", args[1]) else {
        return Value::null();
    };
    Value::bool(ch.send(x))
}

/// `receive()` / `receive(ms)`: the next value, or null when the
/// channel is closed or `ms` passes first.
fn channel_receive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Channel.receive") else {
        return Value::null();
    };
    let Some(ch) = channel_of(ctx, "Channel.receive", args[0]) else {
        return Value::null();
    };
    let ms = args.get(1).copied().unwrap_or_else(Value::null);
    let Ok(deadline) = deadline_arg(ctx, "Channel.receive", ms) else {
        return Value::null();
    };
    loop {
        let token = crate::runtime::sched::new_waiter(vm);
        match ch.wait_with(token) {
            Receive::Value(x) => {
                crate::runtime::sched::discard_waiter(vm, token);
                return isolate::import(ctx, &x);
            }
            Receive::Closed => {
                crate::runtime::sched::discard_waiter(vm, token);
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
                    Some(x) => isolate::import(ctx, &x),
                    None => Value::null(),
                };
            }
        }
    }
}

fn channel_try_receive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(ch) = channel_of(ctx, "Channel.tryReceive", args[0]) else {
        return Value::null();
    };
    match ch.try_receive() {
        Some(x) => isolate::import(ctx, &x),
        None => Value::null(),
    }
}

fn channel_close(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(ch) = channel_of(ctx, "Channel.close", args[0]) {
        ch.close();
    }
    Value::null()
}

fn channel_is_closed(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    channel_of(ctx, "Channel.isClosed", args[0])
        .map_or_else(Value::null, |c| Value::bool(c.is_closed()))
}

fn channel_count(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    channel_of(ctx, "Channel.count", args[0])
        .map_or_else(Value::null, |c| Value::num(c.pending() as f64))
}

/// `drop()`: release the channel, closing it; every instance of it
/// is stale from here.
fn channel_drop(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    isolate::drop_channel(handle_of(args[0]));
    let _ = ctx;
    Value::null()
}

/// Make the `Isolate` and `Channel` classes; returns them in that order.
pub fn register(
    vm: &mut VM,
) -> (
    *mut crate::runtime::object::ObjClass,
    *mut crate::runtime::object::ObjClass,
) {
    let isolate = vm.make_class("Isolate", vm.object_class);
    let channel = vm.make_class("Channel", vm.object_class);
    unsafe {
        (*isolate).num_fields = 1;
        (*channel).num_fields = 1;
    }

    vm.primitive_static(isolate, "spawn(_)", isolate_spawn);
    vm.primitive_static(isolate, "spawn(_,_)", isolate_spawn);
    vm.primitive_static(isolate, "arg", isolate_arg);
    vm.primitive_static(isolate, "cpus", isolate_cpus);
    vm.primitive(isolate, "isDone", isolate_is_done);
    vm.primitive(isolate, "error", isolate_error);
    vm.primitive(isolate, "join()", isolate_join);
    vm.primitive(isolate, "join(_)", isolate_join);

    vm.primitive_static(channel, "new()", channel_new);
    vm.primitive(channel, "send(_)", channel_send);
    vm.primitive(channel, "receive()", channel_receive);
    vm.primitive(channel, "receive(_)", channel_receive);
    vm.primitive(channel, "tryReceive()", channel_try_receive);
    vm.primitive(channel, "close()", channel_close);
    vm.primitive(channel, "isClosed", channel_is_closed);
    vm.primitive(channel, "count", channel_count);
    vm.primitive(channel, "drop()", channel_drop);

    (isolate, channel)
}
