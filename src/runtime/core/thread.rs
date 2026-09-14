//! `Thread`, `Mutex`, `Lock` and `Deque`, the built-in module
//! `thread`: tasks on the program's worker threads and the ways they
//! wait for each other. A wait parks the task on its world, never the
//! OS thread; a release from any thread wakes it through the
//! scheduler's registry.
//!
//! The primitives keep all their state in instance fields, so the
//! collector traces what they hold: a guard word spun on with atomic
//! operations on the field, a count or a list of items, and a list of
//! waiter tokens.

use crate::runtime::core::fiber::{deadline_arg, park_on, sched_of, sched_vm};
use crate::runtime::core::sequence::{instance_field, set_instance_field};
use crate::runtime::object::{NativeContext, ObjInstance, ObjList};
use crate::runtime::sched;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// -- Fields -------------------------------------------------------------------

const GUARD: usize = 0;
/// Mutex: held flag. Lock: units available. Deque: the items.
const STATE: usize = 1;
const WAITERS: usize = 2;

fn field_ptr(receiver: Value, index: usize) -> *mut u64 {
    unsafe {
        let inst = receiver.as_object().unwrap() as *mut ObjInstance;
        (*inst).fields.add(index) as *mut u64
    }
}

/// Spin until the guard word is ours.
fn guard(receiver: Value) {
    let word = unsafe { std::sync::atomic::AtomicU64::from_ptr(field_ptr(receiver, GUARD)) };
    let free = Value::num(0.0).to_bits();
    let held = Value::num(1.0).to_bits();
    while word
        .compare_exchange_weak(
            free,
            held,
            std::sync::atomic::Ordering::Acquire,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_err()
    {
        std::hint::spin_loop();
    }
}

fn unguard(receiver: Value) {
    let word = unsafe { std::sync::atomic::AtomicU64::from_ptr(field_ptr(receiver, GUARD)) };
    word.store(
        Value::num(0.0).to_bits(),
        std::sync::atomic::Ordering::Release,
    );
}

fn num_field(receiver: Value, index: usize) -> f64 {
    instance_field(receiver, index).as_num().unwrap_or(0.0)
}

fn list_field(receiver: Value, index: usize) -> &'static mut ObjList {
    unsafe { &mut *(instance_field(receiver, index).as_object().unwrap() as *mut ObjList) }
}

fn init(ctx: &mut dyn NativeContext, class: &str, state: Value) -> Value {
    let class = ctx.lookup_class(class).expect("thread module loaded");
    let inst = ctx.alloc_instance(class);
    let waiters = ctx.alloc_list(Vec::new());
    set_instance_field(ctx, inst, GUARD, Value::num(0.0));
    set_instance_field(ctx, inst, STATE, state);
    set_instance_field(ctx, inst, WAITERS, waiters);
    inst
}

/// Pop the first waiter token, under the guard.
fn pop_waiter(receiver: Value) -> Option<u64> {
    list_field(receiver, WAITERS)
        .remove(0)
        .and_then(|t| t.as_num())
        .map(|t| t as u64)
}

fn push_waiter(receiver: Value, token: u64) {
    list_field(receiver, WAITERS).add(Value::num(token as f64));
}

/// Drop `token` from the waiters; false when a release already took it.
fn forget_waiter(receiver: Value, token: u64) -> bool {
    let list = list_field(receiver, WAITERS);
    let wanted = Value::num(token as f64).to_bits();
    match list.as_slice().iter().position(|v| v.to_bits() == wanted) {
        Some(i) => {
            list.remove(i);
            true
        }
        None => false,
    }
}

// -- Thread -------------------------------------------------------------------

/// `Thread.create(fn)`: run `fn` as a task on a worker thread.
fn thread_create(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Thread.create") else {
        return Value::null();
    };
    let is_fn = args[1].as_object().is_some_and(|p| unsafe {
        (*(p as *const crate::runtime::object::ObjHeader)).obj_type
            == crate::runtime::object::ObjType::Closure
    });
    if !is_fn {
        ctx.runtime_error("Thread.create: expected a function.".to_string());
        return Value::null();
    }
    unsafe { (*vm).create_thread(args[1]) };
    Value::null()
}

/// `Thread.current`: the fiber running this task.
fn thread_current(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let f = ctx.get_current_fiber();
    if f.is_null() {
        Value::null()
    } else {
        Value::object(f as *mut u8)
    }
}

/// `Thread.yield()`: let the other tasks of this world run.
fn thread_yield(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    if crate::runtime::core::fiber::try_krio_yield_pub(Value::null()).is_none() {
        ctx.set_fiber_action_yield(Value::null());
    }
    Value::null()
}

/// `Thread.count`: the worker threads.
fn thread_count(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let vm = ctx.krio_vm_raw_ptr() as *mut VM;
    if vm.is_null() {
        return Value::num(0.0);
    }
    let n = unsafe {
        (&*vm)
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .endpoints()
            .len()
    };
    Value::num(n as f64)
}

// -- Mutex --------------------------------------------------------------------

fn mutex_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    init(ctx, "Mutex", Value::num(0.0))
}

fn mutex_acquire(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Mutex.acquire") else {
        return Value::null();
    };
    let m = args[0];
    loop {
        guard(m);
        if num_field(m, STATE) == 0.0 {
            set_instance_field(ctx, m, STATE, Value::num(1.0));
            unguard(m);
            return Value::null();
        }
        let token = sched_of(vm).new_waiter();
        push_waiter(m, token);
        unguard(m);
        if park_on(ctx, vm, token, None).is_none() {
            return Value::null();
        }
    }
}

fn mutex_try_acquire(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let m = args[0];
    guard(m);
    let free = num_field(m, STATE) == 0.0;
    if free {
        set_instance_field(ctx, m, STATE, Value::num(1.0));
    }
    unguard(m);
    Value::bool(free)
}

fn mutex_release(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let m = args[0];
    guard(m);
    set_instance_field(ctx, m, STATE, Value::num(0.0));
    let next = pop_waiter(m);
    unguard(m);
    if let Some(token) = next {
        sched::wake(token);
    }
    Value::null()
}

// -- Lock ---------------------------------------------------------------------

fn lock_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    init(ctx, "Lock", Value::num(0.0))
}

/// `wait()` / `wait(ms)`: take a unit, waiting for a `release` to
/// give one; false when `ms` ran out first.
fn lock_wait(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Lock.wait") else {
        return Value::null();
    };
    let ms = args.get(1).copied().unwrap_or_else(Value::null);
    let Ok(deadline) = deadline_arg(ctx, "Lock.wait", ms) else {
        return Value::null();
    };
    let l = args[0];
    guard(l);
    let units = num_field(l, STATE);
    if units > 0.0 {
        set_instance_field(ctx, l, STATE, Value::num(units - 1.0));
        unguard(l);
        return Value::bool(true);
    }
    let token = sched_of(vm).new_waiter();
    push_waiter(l, token);
    unguard(l);
    match park_on(ctx, vm, token, deadline) {
        None => Value::null(),
        // A release handed this waiter its unit.
        Some(true) => Value::bool(true),
        Some(false) => {
            guard(l);
            let still_waiting = forget_waiter(l, token);
            unguard(l);
            // Not in the list any more: a release took it just as the
            // wait ran out, and the unit is ours.
            Value::bool(!still_waiting)
        }
    }
}

fn lock_release(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let l = args[0];
    guard(l);
    let next = pop_waiter(l);
    if next.is_none() {
        let units = num_field(l, STATE);
        set_instance_field(ctx, l, STATE, Value::num(units + 1.0));
    }
    unguard(l);
    if let Some(token) = next {
        // A waiter that timed out meanwhile forgets itself only under
        // the guard, so one popped here is still parked or about to
        // find its wake.
        if !sched::wake(token) {
            // Its park resolved on the timeout first: the unit goes
            // back for the next waiter.
            guard(l);
            let units = num_field(l, STATE);
            set_instance_field(ctx, l, STATE, Value::num(units + 1.0));
            unguard(l);
        }
    }
    Value::null()
}

// -- Deque --------------------------------------------------------------------

fn deque_new(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let items = ctx.alloc_list(Vec::new());
    init(ctx, "Deque", items)
}

fn deque_put(ctx: &mut dyn NativeContext, args: &[Value], front: bool) -> Value {
    let d = args[0];
    guard(d);
    let items = list_field(d, STATE);
    if front {
        items.insert(0, args[1]);
    } else {
        items.add(args[1]);
    }
    ctx.write_barrier(instance_field(d, STATE), args[1]);
    let next = pop_waiter(d);
    unguard(d);
    if let Some(token) = next {
        sched::wake(token);
    }
    Value::null()
}

fn deque_add(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    deque_put(ctx, args, false)
}

fn deque_push(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    deque_put(ctx, args, true)
}

/// `pop(block)`: the first item; with `block`, wait for one.
fn deque_pop(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(vm) = sched_vm(ctx, "Deque.pop") else {
        return Value::null();
    };
    let d = args[0];
    let block = args[1].as_bool().unwrap_or(false);
    loop {
        guard(d);
        if let Some(v) = list_field(d, STATE).remove(0) {
            unguard(d);
            return v;
        }
        if !block {
            unguard(d);
            return Value::null();
        }
        let token = sched_of(vm).new_waiter();
        push_waiter(d, token);
        unguard(d);
        if park_on(ctx, vm, token, None).is_none() {
            return Value::null();
        }
    }
}

fn deque_count(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let d = args[0];
    guard(d);
    let n = list_field(d, STATE).len();
    unguard(d);
    Value::num(n as f64)
}

/// Make the module's classes; returned in the order `Thread`, `Mutex`,
/// `Lock`, `Deque`.
pub fn register(vm: &mut VM) -> [*mut crate::runtime::object::ObjClass; 4] {
    let thread = vm.make_class("Thread", vm.object_class);
    let mutex = vm.make_class("Mutex", vm.object_class);
    let lock = vm.make_class("Lock", vm.object_class);
    let deque = vm.make_class("Deque", vm.object_class);
    unsafe {
        (*mutex).num_fields = 3;
        (*lock).num_fields = 3;
        (*deque).num_fields = 3;
    }

    vm.primitive_static(thread, "create(_)", thread_create);
    vm.primitive_static(thread, "current", thread_current);
    vm.primitive_static(thread, "yield()", thread_yield);
    vm.primitive_static(thread, "count", thread_count);

    vm.primitive_static(mutex, "new()", mutex_new);
    vm.primitive(mutex, "acquire()", mutex_acquire);
    vm.primitive(mutex, "tryAcquire()", mutex_try_acquire);
    vm.primitive(mutex, "release()", mutex_release);

    vm.primitive_static(lock, "new()", lock_new);
    vm.primitive(lock, "wait()", lock_wait);
    vm.primitive(lock, "wait(_)", lock_wait);
    vm.primitive(lock, "release()", lock_release);

    vm.primitive_static(deque, "new()", deque_new);
    vm.primitive(deque, "add(_)", deque_add);
    vm.primitive(deque, "push(_)", deque_push);
    vm.primitive(deque, "pop(_)", deque_pop);
    vm.primitive(deque, "count", deque_count);

    [thread, mutex, lock, deque]
}
