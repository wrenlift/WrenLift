use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn receiver_str(args: &[Value]) -> &str {
    super::as_string(args[0])
}

// ---------------------------------------------------------------------------
// Static methods
// ---------------------------------------------------------------------------

fn from_code_point(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let code_point = match super::validate_int(ctx, args[1], "Code point") {
        Some(v) => v as u32,
        None => return Value::null(),
    };

    match char::from_u32(code_point) {
        Some(c) => {
            let mut buf = String::new();
            buf.push(c);
            ctx.alloc_string(buf)
        }
        None => {
            ctx.runtime_error(format!("Invalid code point {}.", code_point));
            Value::null()
        }
    }
}

fn from_byte(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let byte = match super::validate_int(ctx, args[1], "Byte") {
        Some(v) => v,
        None => return Value::null(),
    };

    if byte < 0 || byte > 255 {
        ctx.runtime_error(format!("Byte must be in range 0..255, was {}.", byte));
        return Value::null();
    }

    let s = String::from(byte as u8 as char);
    ctx.alloc_string(s)
}

// ---------------------------------------------------------------------------
// Instance methods
// ---------------------------------------------------------------------------

fn plus(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !super::is_string(args[1]) {
        ctx.runtime_error("Right operand must be a string.".to_string());
        return Value::null();
    }

    let left = receiver_str(args);
    let right = super::as_string(args[1]);
    let mut result = String::with_capacity(left.len() + right.len());
    result.push_str(left);
    result.push_str(right);
    ctx.alloc_string(result)
}

fn subscript(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);

    if !args[1].is_num() {
        ctx.runtime_error("Subscript must be a number.".to_string());
        return Value::null();
    }

    let mut index = args[1].as_num().unwrap() as i64;
    let char_count = s.chars().count() as i64;

    if index < 0 {
        index += char_count;
    }

    if index < 0 || index >= char_count {
        ctx.runtime_error(format!("String index {} out of bounds.", args[1].as_num().unwrap() as i64));
        return Value::null();
    }

    match s.chars().nth(index as usize) {
        Some(c) => {
            let mut buf = String::new();
            buf.push(c);
            ctx.alloc_string(buf)
        }
        None => {
            ctx.runtime_error(format!("String index {} out of bounds.", index));
            Value::null()
        }
    }
}

fn byte_at(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);
    let index = match super::validate_index(ctx, args[1], s.len(), "Index") {
        Some(i) => i,
        None => return Value::null(),
    };

    Value::num(s.as_bytes()[index] as f64)
}

fn byte_count(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);
    Value::num(s.len() as f64)
}

fn code_point_at(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);
    let index = match super::validate_index(ctx, args[1], s.len(), "Index") {
        Some(i) => i,
        None => return Value::null(),
    };

    // Get the char starting at the given byte index.
    match s[index..].chars().next() {
        Some(c) => Value::num(c as u32 as f64),
        None => {
            ctx.runtime_error(format!("Index {} is not a valid byte index.", index));
            Value::null()
        }
    }
}

fn contains(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !super::is_string(args[1]) {
        ctx.runtime_error("Argument must be a string.".to_string());
        return Value::null();
    }

    let haystack = receiver_str(args);
    let needle = super::as_string(args[1]);
    Value::bool(haystack.contains(needle))
}

fn ends_with(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !super::is_string(args[1]) {
        ctx.runtime_error("Argument must be a string.".to_string());
        return Value::null();
    }

    let s = receiver_str(args);
    let suffix = super::as_string(args[1]);
    Value::bool(s.ends_with(suffix))
}

fn index_of_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !super::is_string(args[1]) {
        ctx.runtime_error("Argument must be a string.".to_string());
        return Value::null();
    }

    let haystack = receiver_str(args);
    let needle = super::as_string(args[1]);

    match haystack.find(needle) {
        Some(pos) => Value::num(pos as f64),
        None => Value::num(-1.0),
    }
}

fn index_of_2(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !super::is_string(args[1]) {
        ctx.runtime_error("First argument must be a string.".to_string());
        return Value::null();
    }

    let start = match super::validate_int(ctx, args[2], "Start") {
        Some(v) => v as usize,
        None => return Value::null(),
    };

    let haystack = receiver_str(args);
    let needle = super::as_string(args[1]);

    if start > haystack.len() {
        return Value::num(-1.0);
    }

    match haystack[start..].find(needle) {
        Some(pos) => Value::num((start + pos) as f64),
        None => Value::num(-1.0),
    }
}

fn iterate(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);

    if s.is_empty() {
        return Value::bool(false);
    }

    // If the iterator is null, start at 0.
    if args[1].is_null() {
        return Value::num(0.0);
    }

    let index = match super::validate_int(ctx, args[1], "Iterator") {
        Some(v) => v as usize,
        None => return Value::null(),
    };

    // Advance past the current character to the next char boundary.
    if index >= s.len() {
        return Value::bool(false);
    }

    // Find the length of the UTF-8 character at `index`.
    let remaining = &s[index..];
    let current_char_len = match remaining.chars().next() {
        Some(c) => c.len_utf8(),
        None => return Value::bool(false),
    };

    let next_index = index + current_char_len;
    if next_index >= s.len() {
        Value::bool(false)
    } else {
        Value::num(next_index as f64)
    }
}

fn iterate_byte(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);

    if s.is_empty() {
        return Value::bool(false);
    }

    // If the iterator is null, start at 0.
    if args[1].is_null() {
        return Value::num(0.0);
    }

    let index = match super::validate_int(ctx, args[1], "Iterator") {
        Some(v) => v as usize,
        None => return Value::null(),
    };

    let next = index + 1;
    if next >= s.len() {
        Value::bool(false)
    } else {
        Value::num(next as f64)
    }
}

fn iterator_value(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = receiver_str(args);

    let index = match super::validate_int(ctx, args[1], "Iterator") {
        Some(v) => v as usize,
        None => return Value::null(),
    };

    if index >= s.len() {
        ctx.runtime_error(format!("Iterator value {} out of bounds.", index));
        return Value::null();
    }

    match s[index..].chars().next() {
        Some(c) => {
            let mut buf = String::new();
            buf.push(c);
            ctx.alloc_string(buf)
        }
        None => {
            ctx.runtime_error(format!("Invalid byte index {}.", index));
            Value::null()
        }
    }
}

fn count(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::num(receiver_str(args).chars().count() as f64)
}

fn is_empty(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    Value::bool(receiver_str(args).is_empty())
}

fn split(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let delimiter = match super::validate_string(ctx, args[1], "Delimiter") {
        Some(d) => d,
        None => return Value::null(),
    };

    if delimiter.is_empty() {
        ctx.runtime_error("Delimiter cannot be empty.".to_string());
        return Value::null();
    }

    let s = receiver_str(args);
    let elements: Vec<Value> = s
        .split(delimiter.as_str())
        .map(|part| ctx.alloc_string(part.to_string()))
        .collect();
    ctx.alloc_list(elements)
}

fn replace(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let from = match super::validate_string(ctx, args[1], "From") {
        Some(s) => s,
        None => return Value::null(),
    };
    let to = match super::validate_string(ctx, args[2], "To") {
        Some(s) => s,
        None => return Value::null(),
    };

    let s = receiver_str(args);
    let result = s.replace(from.as_str(), to.as_str());
    ctx.alloc_string(result)
}

fn trim(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    ctx.alloc_string(receiver_str(args).trim().to_string())
}

fn trim_start(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    ctx.alloc_string(receiver_str(args).trim_start().to_string())
}

fn trim_end(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    ctx.alloc_string(receiver_str(args).trim_end().to_string())
}

fn multiply(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = match super::validate_int(ctx, args[1], "Count") {
        Some(v) => v,
        None => return Value::null(),
    };

    if n < 0 {
        ctx.runtime_error("Count must be non-negative.".to_string());
        return Value::null();
    }

    let s = receiver_str(args);
    let result = s.repeat(n as usize);
    ctx.alloc_string(result)
}

fn starts_with(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !super::is_string(args[1]) {
        ctx.runtime_error("Argument must be a string.".to_string());
        return Value::null();
    }

    let s = receiver_str(args);
    let prefix = super::as_string(args[1]);
    Value::bool(s.starts_with(prefix))
}

fn to_string(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    args[0]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn bind(vm: &mut VM) {
    let class = vm.string_class;

    // Static methods
    vm.primitive_static(class, "fromCodePoint(_)", from_code_point);
    vm.primitive_static(class, "fromByte(_)", from_byte);

    // Instance getters
    vm.primitive(class, "byteCount_", byte_count);
    vm.primitive(class, "count", count);
    vm.primitive(class, "isEmpty", is_empty);
    vm.primitive(class, "toString", to_string);
    vm.primitive(class, "trim()", trim);
    vm.primitive(class, "trimStart()", trim_start);
    vm.primitive(class, "trimEnd()", trim_end);

    // Subscript
    vm.primitive(class, "[_]", subscript);

    // Binary / single-arg methods
    vm.primitive(class, "+(_)", plus);
    vm.primitive(class, "*(_)", multiply);
    vm.primitive(class, "contains(_)", contains);
    vm.primitive(class, "endsWith(_)", ends_with);
    vm.primitive(class, "indexOf(_)", index_of_1);
    vm.primitive(class, "indexOf(_,_)", index_of_2);
    vm.primitive(class, "split(_)", split);
    vm.primitive(class, "startsWith(_)", starts_with);
    vm.primitive(class, "replace(_,_)", replace);

    // Byte access
    vm.primitive(class, "byteAt_(_)", byte_at);
    vm.primitive(class, "codePointAt_(_)", code_point_at);

    // Iterators
    vm.primitive(class, "iterate(_)", iterate);
    vm.primitive(class, "iterateByte_(_)", iterate_byte);
    vm.primitive(class, "iteratorValue(_)", iterator_value);
}
