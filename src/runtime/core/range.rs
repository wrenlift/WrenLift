use crate::runtime::object::{NativeContext, ObjRange};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn receiver_range(args: &[Value]) -> &ObjRange {
    unsafe { &*(args[0].as_object().unwrap() as *const ObjRange) }
}

fn range_from(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    Value::num(range.from)
}

fn range_to(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    Value::num(range.to)
}

fn range_min(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    Value::num(f64::min(range.from, range.to))
}

fn range_max(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    Value::num(f64::max(range.from, range.to))
}

fn range_is_inclusive(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    Value::bool(range.is_inclusive)
}

fn range_iterate(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    match range_iterate_value(range, args[1]) {
        Some(value) => value,
        None => {
            ctx.runtime_error("Iterator must be a number.".to_string());
            Value::null()
        }
    }
}

/// The side-effect-free part of Range.iterate(_), shared with compiled
/// sends so both paths agree on ascending, descending and empty ranges.
#[inline(always)]
pub(crate) fn range_iterate_value(range: &ObjRange, iterator: Value) -> Option<Value> {
    // If the iterator is null, we return the start of the range.
    if iterator.is_null() {
        // Empty range check: exclusive range where from == to has no elements.
        if !range.is_inclusive && range.from == range.to {
            return Some(Value::bool(false));
        }
        return Some(Value::num(range.from));
    }

    let current = iterator.as_num()?;

    // Determine step direction.
    let value = if range.from < range.to {
        let next = current + 1.0;
        if range.is_inclusive {
            if next > range.to {
                Value::bool(false)
            } else {
                Value::num(next)
            }
        } else if next >= range.to {
            Value::bool(false)
        } else {
            Value::num(next)
        }
    } else if range.from > range.to {
        let next = current - 1.0;
        if range.is_inclusive {
            if next < range.to {
                Value::bool(false)
            } else {
                Value::num(next)
            }
        } else if next <= range.to {
            Value::bool(false)
        } else {
            Value::num(next)
        }
    } else {
        // from == to, inclusive range with single element; already yielded.
        Value::bool(false)
    };
    Some(value)
}

fn range_iterator_value(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    args[1]
}

fn range_to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let range = receiver_range(args);
    let dots = if range.is_inclusive { ".." } else { "..." };
    let s = format!("{}{}{}", range.from, dots, range.to);
    ctx.alloc_string(s)
}

pub fn bind(vm: &mut VM) {
    let class = vm.range_class;
    vm.primitive(class, "from", range_from);
    vm.primitive(class, "to", range_to);
    vm.primitive(class, "min", range_min);
    vm.primitive(class, "max", range_max);
    vm.primitive(class, "isInclusive", range_is_inclusive);
    vm.primitive(class, "iterate(_)", range_iterate);
    vm.primitive(class, "iteratorValue(_)", range_iterator_value);
    vm.primitive(class, "toString", range_to_string);
}
