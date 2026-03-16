use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// ---------------------------------------------------------------------------
// Iterate protocol helpers
// ---------------------------------------------------------------------------

/// Drive the iterate/iteratorValue protocol, collecting all elements.
fn collect_elements(ctx: &mut dyn NativeContext, receiver: Value) -> Option<Vec<Value>> {
    let mut elements = Vec::new();
    let mut iterator = Value::null();

    loop {
        iterator = ctx.call_method_on(receiver, "iterate(_)", &[iterator])?;
        if iterator.is_falsy() || iterator.is_null() {
            break;
        }
        let value = ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator])?;
        elements.push(value);
    }
    Some(elements)
}

// ---------------------------------------------------------------------------
// Sequence instance methods
// ---------------------------------------------------------------------------

fn seq_all(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let func = args[1];
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::bool(true),
        };
        if iterator.is_falsy() || iterator.is_null() {
            return Value::bool(true);
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::bool(true),
        };
        let result = match ctx.call_method_on(func, "call(_)", &[value]) {
            Some(v) => v,
            None => return Value::bool(true),
        };
        if result.is_falsy() {
            return result;
        }
    }
}

fn seq_any(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let func = args[1];
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::bool(false),
        };
        if iterator.is_falsy() || iterator.is_null() {
            return Value::bool(false);
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::bool(false),
        };
        let result = match ctx.call_method_on(func, "call(_)", &[value]) {
            Some(v) => v,
            None => return Value::bool(false),
        };
        if !result.is_falsy() {
            return result;
        }
    }
}

fn seq_contains(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let element = args[1];
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::bool(false),
        };
        if iterator.is_falsy() || iterator.is_null() {
            return Value::bool(false);
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::bool(false),
        };
        if value.to_bits() == element.to_bits() {
            return Value::bool(true);
        }
        // Also check via == for objects
        if let Some(result) = ctx.call_method_on(element, "==(_)", &[value]) {
            if !result.is_falsy() {
                return Value::bool(true);
            }
        }
    }
}

fn seq_count(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let mut count: f64 = 0.0;
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::num(count),
        };
        if iterator.is_falsy() || iterator.is_null() {
            return Value::num(count);
        }
        count += 1.0;
    }
}

fn seq_count_where(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let func = args[1];
    let mut count: f64 = 0.0;
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::num(count),
        };
        if iterator.is_falsy() || iterator.is_null() {
            return Value::num(count);
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::num(count),
        };
        if let Some(result) = ctx.call_method_on(func, "call(_)", &[value]) {
            if !result.is_falsy() {
                count += 1.0;
            }
        }
    }
}

fn seq_each(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let func = args[1];
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::null(),
        };
        if iterator.is_falsy() || iterator.is_null() {
            return Value::null();
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return Value::null(),
        };
        ctx.call_method_on(func, "call(_)", &[value]);
    }
}

fn seq_is_empty(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let iterator = ctx.call_method_on(receiver, "iterate(_)", &[Value::null()]);
    match iterator {
        Some(v) if !v.is_falsy() => Value::bool(false),
        _ => Value::bool(true),
    }
}

fn seq_join_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    seq_join_impl(ctx, args[0], "")
}

fn seq_join_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sep = super::validate_string(ctx, args[1], "Separator");
    match sep {
        Some(s) => seq_join_impl(ctx, args[0], &s),
        None => Value::null(),
    }
}

fn seq_join_impl(ctx: &mut dyn NativeContext, receiver: Value, sep: &str) -> Value {
    let mut result = String::new();
    let mut first = true;
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => break,
        };
        if iterator.is_falsy() || iterator.is_null() {
            break;
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => break,
        };

        if !first {
            result.push_str(sep);
        }
        first = false;

        // Get toString of element
        let s = value_to_string(ctx, value);
        result.push_str(&s);
    }

    ctx.alloc_string(result)
}

fn seq_to_list(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    match collect_elements(ctx, receiver) {
        Some(elements) => ctx.alloc_list(elements),
        None => ctx.alloc_list(Vec::new()),
    }
}

fn seq_reduce_2(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let mut acc = args[1];
    let func = args[2];
    let mut iterator = Value::null();

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return acc,
        };
        if iterator.is_falsy() || iterator.is_null() {
            return acc;
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return acc,
        };
        acc = match ctx.call_method_on(func, "call(_,_)", &[acc, value]) {
            Some(v) => v,
            None => return acc,
        };
    }
}

fn seq_reduce_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let receiver = args[0];
    let func = args[1];
    let mut iterator = Value::null();

    // Get first element as seed
    iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
        Some(v) => v,
        None => {
            ctx.runtime_error("Can't reduce an empty sequence.".to_string());
            return Value::null();
        }
    };
    if iterator.is_falsy() || iterator.is_null() {
        ctx.runtime_error("Can't reduce an empty sequence.".to_string());
        return Value::null();
    }

    let mut acc = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
        Some(v) => v,
        None => return Value::null(),
    };

    loop {
        iterator = match ctx.call_method_on(receiver, "iterate(_)", &[iterator]) {
            Some(v) => v,
            None => return acc,
        };
        if iterator.is_falsy() || iterator.is_null() {
            return acc;
        }
        let value = match ctx.call_method_on(receiver, "iteratorValue(_)", &[iterator]) {
            Some(v) => v,
            None => return acc,
        };
        acc = match ctx.call_method_on(func, "call(_,_)", &[acc, value]) {
            Some(v) => v,
            None => return acc,
        };
    }
}

// ---------------------------------------------------------------------------
// Helper: convert value to string representation
// ---------------------------------------------------------------------------

pub(crate) fn value_to_string(ctx: &mut dyn NativeContext, value: Value) -> String {
    if let Some(n) = value.as_num() {
        if n == n.trunc() && !n.is_infinite() && n.abs() < 1e15 {
            return format!("{}", n as i64);
        }
        return format!("{}", n);
    }
    if value.is_null() {
        return "null".to_string();
    }
    if let Some(b) = value.as_bool() {
        return if b { "true" } else { "false" }.to_string();
    }

    // Try calling toString on the object
    if let Some(s) = ctx.call_method_on(value, "toString", &[]) {
        if super::is_string(s) {
            return super::as_string(s).to_string();
        }
    }
    "<object>".to_string()
}

// ---------------------------------------------------------------------------
// Sequence: map(_), skip(_), take(_), where(_) — return lazy wrapper instances
// ---------------------------------------------------------------------------

use crate::runtime::object::ObjInstance;

/// Helper to read an ObjInstance field from the receiver value.
pub(crate) fn instance_field(receiver: Value, index: usize) -> Value {
    unsafe {
        let ptr = receiver.as_object().unwrap();
        let inst = &*(ptr as *const ObjInstance);
        inst.get_field(index).unwrap_or(Value::null())
    }
}

/// Helper to write an ObjInstance field.
pub(crate) fn set_instance_field(
    ctx: &mut dyn NativeContext,
    receiver: Value,
    index: usize,
    value: Value,
) {
    unsafe {
        let ptr = receiver.as_object().unwrap();
        let inst = &mut *(ptr as *mut ObjInstance);
        inst.set_field(index, value);
    }
    ctx.write_barrier(receiver, value);
}

fn seq_map(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class = ctx.lookup_class("MapSequence").unwrap();
    let inst = ctx.alloc_instance(class);
    set_instance_field(ctx, inst, 0, args[0]); // _sequence
    set_instance_field(ctx, inst, 1, args[1]); // _fn
    inst
}

fn seq_skip(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(n) = args[1].as_num() {
        if n != n.trunc() || n < 0.0 {
            ctx.runtime_error("Count must be a non-negative integer.".to_string());
            return Value::null();
        }
    } else {
        ctx.runtime_error("Count must be a non-negative integer.".to_string());
        return Value::null();
    }
    let class = ctx.lookup_class("SkipSequence").unwrap();
    let inst = ctx.alloc_instance(class);
    set_instance_field(ctx, inst, 0, args[0]); // _sequence
    set_instance_field(ctx, inst, 1, args[1]); // _count
    inst
}

fn seq_take(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(n) = args[1].as_num() {
        if n != n.trunc() || n < 0.0 {
            ctx.runtime_error("Count must be a non-negative integer.".to_string());
            return Value::null();
        }
    } else {
        ctx.runtime_error("Count must be a non-negative integer.".to_string());
        return Value::null();
    }
    let class = ctx.lookup_class("TakeSequence").unwrap();
    let inst = ctx.alloc_instance(class);
    set_instance_field(ctx, inst, 0, args[0]); // _sequence
    set_instance_field(ctx, inst, 1, args[1]); // _count
    inst
}

fn seq_where(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let class = ctx.lookup_class("WhereSequence").unwrap();
    let inst = ctx.alloc_instance(class);
    set_instance_field(ctx, inst, 0, args[0]); // _sequence
    set_instance_field(ctx, inst, 1, args[1]); // _fn (predicate)
    inst
}

// ---------------------------------------------------------------------------
// MapSequence: iterate(_) and iteratorValue(_)
// Fields: [0]=sequence, [1]=fn
// ---------------------------------------------------------------------------

fn map_seq_iterate(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    ctx.call_method_on(sequence, "iterate(_)", &[args[1]])
        .unwrap_or(Value::bool(false))
}

fn map_seq_iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    let func = instance_field(args[0], 1);
    let inner_val = ctx
        .call_method_on(sequence, "iteratorValue(_)", &[args[1]])
        .unwrap_or(Value::null());
    ctx.call_method_on(func, "call(_)", &[inner_val])
        .unwrap_or(Value::null())
}

// ---------------------------------------------------------------------------
// SkipSequence: iterate(_) and iteratorValue(_)
// Fields: [0]=sequence, [1]=count
// ---------------------------------------------------------------------------

fn skip_seq_iterate(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    let iterator = args[1];

    if !iterator.is_null() && !iterator.is_falsy() {
        // Already started, just delegate.
        return ctx
            .call_method_on(sequence, "iterate(_)", &[iterator])
            .unwrap_or(Value::bool(false));
    }

    // First call: skip `count` items.
    let count = instance_field(args[0], 1).as_num().unwrap_or(0.0) as i64;
    let mut it = ctx
        .call_method_on(sequence, "iterate(_)", &[Value::null()])
        .unwrap_or(Value::bool(false));
    let mut remaining = count;
    while remaining > 0 && !it.is_falsy() && !it.is_null() {
        it = ctx
            .call_method_on(sequence, "iterate(_)", &[it])
            .unwrap_or(Value::bool(false));
        remaining -= 1;
    }
    it
}

fn skip_seq_iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    ctx.call_method_on(sequence, "iteratorValue(_)", &[args[1]])
        .unwrap_or(Value::null())
}

// ---------------------------------------------------------------------------
// TakeSequence: iterate(_) and iteratorValue(_)
// Fields: [0]=sequence, [1]=count, [2]=taken
// ---------------------------------------------------------------------------

fn take_seq_iterate(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    let max_count = instance_field(args[0], 1).as_num().unwrap_or(0.0);
    let iterator = args[1];

    let taken = if iterator.is_null() || iterator.is_falsy() {
        1.0
    } else {
        instance_field(args[0], 2).as_num().unwrap_or(0.0) + 1.0
    };

    set_instance_field(ctx, args[0], 2, Value::num(taken));

    if taken > max_count {
        return Value::null();
    }

    ctx.call_method_on(sequence, "iterate(_)", &[iterator])
        .unwrap_or(Value::bool(false))
}

fn take_seq_iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    ctx.call_method_on(sequence, "iteratorValue(_)", &[args[1]])
        .unwrap_or(Value::null())
}

// ---------------------------------------------------------------------------
// WhereSequence: iterate(_) and iteratorValue(_)
// Fields: [0]=sequence, [1]=fn
// ---------------------------------------------------------------------------

fn where_seq_iterate(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    let func = instance_field(args[0], 1);
    let mut iterator = args[1];

    loop {
        iterator = match ctx.call_method_on(sequence, "iterate(_)", &[iterator]) {
            Some(v) if !v.is_falsy() && !v.is_null() => v,
            _ => return Value::bool(false),
        };
        let value = ctx
            .call_method_on(sequence, "iteratorValue(_)", &[iterator])
            .unwrap_or(Value::null());
        let result = ctx
            .call_method_on(func, "call(_)", &[value])
            .unwrap_or(Value::bool(false));
        if !result.is_falsy() {
            return iterator;
        }
    }
}

fn where_seq_iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let sequence = instance_field(args[0], 0);
    ctx.call_method_on(sequence, "iteratorValue(_)", &[args[1]])
        .unwrap_or(Value::null())
}

// ---------------------------------------------------------------------------
// Bind
// ---------------------------------------------------------------------------

pub fn bind(vm: &mut VM) {
    let class = vm.sequence_class;

    vm.primitive(class, "all(_)", seq_all);
    vm.primitive(class, "any(_)", seq_any);
    vm.primitive(class, "contains(_)", seq_contains);
    vm.primitive(class, "count", seq_count);
    vm.primitive(class, "count(_)", seq_count_where);
    vm.primitive(class, "each(_)", seq_each);
    vm.primitive(class, "isEmpty", seq_is_empty);
    vm.primitive(class, "join()", seq_join_0);
    vm.primitive(class, "join(_)", seq_join_1);
    vm.primitive(class, "toList", seq_to_list);
    vm.primitive(class, "reduce(_)", seq_reduce_1);
    vm.primitive(class, "reduce(_,_)", seq_reduce_2);
    vm.primitive(class, "map(_)", seq_map);
    vm.primitive(class, "skip(_)", seq_skip);
    vm.primitive(class, "take(_)", seq_take);
    vm.primitive(class, "where(_)", seq_where);

    // MapSequence methods
    let map_seq = vm.map_sequence_class;
    vm.primitive(map_seq, "iterate(_)", map_seq_iterate);
    vm.primitive(map_seq, "iteratorValue(_)", map_seq_iterator_value);

    // SkipSequence methods
    let skip_seq = vm.skip_sequence_class;
    vm.primitive(skip_seq, "iterate(_)", skip_seq_iterate);
    vm.primitive(skip_seq, "iteratorValue(_)", skip_seq_iterator_value);

    // TakeSequence methods
    let take_seq = vm.take_sequence_class;
    vm.primitive(take_seq, "iterate(_)", take_seq_iterate);
    vm.primitive(take_seq, "iteratorValue(_)", take_seq_iterator_value);

    // WhereSequence methods
    let where_seq = vm.where_sequence_class;
    vm.primitive(where_seq, "iterate(_)", where_seq_iterate);
    vm.primitive(where_seq, "iteratorValue(_)", where_seq_iterator_value);
}
