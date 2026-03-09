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
}
