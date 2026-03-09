use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

pub fn bind(vm: &mut VM) {
    // Static methods
    vm.primitive_static(vm.num_class, "infinity", num_infinity);
    vm.primitive_static(vm.num_class, "nan", num_nan);
    vm.primitive_static(vm.num_class, "pi", num_pi);
    vm.primitive_static(vm.num_class, "tau", num_tau);
    vm.primitive_static(vm.num_class, "largest", num_largest);
    vm.primitive_static(vm.num_class, "smallest", num_smallest);
    vm.primitive_static(vm.num_class, "maxSafeInteger", num_max_safe_integer);
    vm.primitive_static(vm.num_class, "minSafeInteger", num_min_safe_integer);
    vm.primitive_static(vm.num_class, "fromString(_)", num_from_string);

    // Arithmetic operators
    vm.primitive(vm.num_class, "+(_)", num_add);
    vm.primitive(vm.num_class, "-(_)", num_sub);
    vm.primitive(vm.num_class, "*(_)", num_mul);
    vm.primitive(vm.num_class, "/(_)", num_div);
    vm.primitive(vm.num_class, "%(_)", num_mod);

    // Comparison operators
    vm.primitive(vm.num_class, "<(_)", num_lt);
    vm.primitive(vm.num_class, ">(_)", num_gt);
    vm.primitive(vm.num_class, "<=(_)", num_lte);
    vm.primitive(vm.num_class, ">=(_)", num_gte);
    vm.primitive(vm.num_class, "==(_)", num_eq);
    vm.primitive(vm.num_class, "!=(_)", num_neq);

    // Bitwise operators
    vm.primitive(vm.num_class, "&(_)", num_bitwise_and);
    vm.primitive(vm.num_class, "|(_)", num_bitwise_or);
    vm.primitive(vm.num_class, "^(_)", num_bitwise_xor);
    vm.primitive(vm.num_class, "<<(_)", num_left_shift);
    vm.primitive(vm.num_class, ">>(_)", num_right_shift);
    vm.primitive(vm.num_class, "~", num_bitwise_not);

    // Unary math methods
    vm.primitive(vm.num_class, "abs", num_abs);
    vm.primitive(vm.num_class, "acos", num_acos);
    vm.primitive(vm.num_class, "asin", num_asin);
    vm.primitive(vm.num_class, "atan", num_atan_getter);
    vm.primitive(vm.num_class, "cbrt", num_cbrt);
    vm.primitive(vm.num_class, "ceil", num_ceil);
    vm.primitive(vm.num_class, "cos", num_cos);
    vm.primitive(vm.num_class, "floor", num_floor);
    vm.primitive(vm.num_class, "-", num_negate);
    vm.primitive(vm.num_class, "round", num_round);
    vm.primitive(vm.num_class, "sin", num_sin);
    vm.primitive(vm.num_class, "sqrt", num_sqrt);
    vm.primitive(vm.num_class, "tan", num_tan);
    vm.primitive(vm.num_class, "log", num_log);
    vm.primitive(vm.num_class, "log2", num_log2);
    vm.primitive(vm.num_class, "exp", num_exp);

    // Other instance methods
    vm.primitive(vm.num_class, "atan(_)", num_atan2);
    vm.primitive(vm.num_class, "min(_)", num_min);
    vm.primitive(vm.num_class, "max(_)", num_max);
    vm.primitive(vm.num_class, "clamp(_,_)", num_clamp);
    vm.primitive(vm.num_class, "pow(_)", num_pow);
    vm.primitive(vm.num_class, "fraction", num_fraction);
    vm.primitive(vm.num_class, "isInfinity", num_is_infinity);
    vm.primitive(vm.num_class, "isInteger", num_is_integer);
    vm.primitive(vm.num_class, "isNan", num_is_nan);
    vm.primitive(vm.num_class, "sign", num_sign);
    vm.primitive(vm.num_class, "toString", num_to_string);
    vm.primitive(vm.num_class, "truncate", num_truncate);
    vm.primitive(vm.num_class, "..(_)", num_range_inclusive);
    vm.primitive(vm.num_class, "...(_)", num_range_exclusive);
}

// Static methods

fn num_infinity(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(f64::INFINITY)
}

fn num_nan(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(f64::NAN)
}

fn num_pi(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(std::f64::consts::PI)
}

fn num_tau(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(std::f64::consts::TAU)
}

fn num_largest(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(f64::MAX)
}

fn num_smallest(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(f64::MIN_POSITIVE)
}

fn num_max_safe_integer(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(9007199254740991.0)
}

fn num_min_safe_integer(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    Value::num(-9007199254740991.0)
}

fn num_from_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let s = match super::validate_string(ctx, args[1], "Argument") {
        Some(v) => v,
        None => return Value::null(),
    };
    match s.parse::<f64>() {
        Ok(n) => Value::num(n),
        Err(_) => Value::null(),
    }
}

// Arithmetic operators

fn num_add(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left + right)
}

fn num_sub(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left - right)
}

fn num_mul(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left * right)
}

fn num_div(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left / right)
}

fn num_mod(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left % right)
}

// Comparison operators

fn num_lt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::bool(left < right)
}

fn num_gt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::bool(left > right)
}

fn num_lte(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::bool(left <= right)
}

fn num_gte(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::bool(left >= right)
}

fn num_eq(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_num() {
        return Value::bool(false);
    }
    let left = args[0].as_num().unwrap();
    let right = args[1].as_num().unwrap();
    Value::bool(left == right)
}

fn num_neq(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !args[1].is_num() {
        return Value::bool(true);
    }
    let left = args[0].as_num().unwrap();
    let right = args[1].as_num().unwrap();
    Value::bool(left != right)
}

// Bitwise operators

fn num_bitwise_and(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap() as u32;
    let right = right as u32;
    Value::num((left & right) as f64)
}

fn num_bitwise_or(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap() as u32;
    let right = right as u32;
    Value::num((left | right) as f64)
}

fn num_bitwise_xor(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap() as u32;
    let right = right as u32;
    Value::num((left ^ right) as f64)
}

fn num_left_shift(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap() as u32;
    let right = right as u32;
    Value::num((left << right) as f64)
}

fn num_right_shift(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap() as u32;
    let right = right as u32;
    Value::num((left >> right) as f64)
}

fn num_bitwise_not(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap() as u32;
    Value::num((!n) as f64)
}

// Unary math methods

fn num_abs(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.abs())
}

fn num_acos(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.acos())
}

fn num_asin(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.asin())
}

fn num_atan_getter(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.atan())
}

fn num_cbrt(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.cbrt())
}

fn num_ceil(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.ceil())
}

fn num_cos(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.cos())
}

fn num_floor(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.floor())
}

fn num_negate(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(-n)
}

fn num_round(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.round())
}

fn num_sin(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.sin())
}

fn num_sqrt(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.sqrt())
}

fn num_tan(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.tan())
}

fn num_log(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.ln())
}

fn num_log2(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.log2())
}

fn num_exp(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.exp())
}

// Other instance methods

fn num_atan2(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left.atan2(right))
}

fn num_min(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left.min(right))
}

fn num_max(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left.max(right))
}

fn num_clamp(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let min = match super::validate_num(ctx, args[1], "Min") {
        Some(v) => v,
        None => return Value::null(),
    };
    let max = match super::validate_num(ctx, args[2], "Max") {
        Some(v) => v,
        None => return Value::null(),
    };
    let n = args[0].as_num().unwrap();
    Value::num(n.clamp(min, max))
}

fn num_pow(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    Value::num(left.powf(right))
}

fn num_fraction(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.fract())
}

fn num_is_infinity(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::bool(n.is_infinite())
}

fn num_is_integer(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::bool(n.trunc() == n && !n.is_infinite())
}

fn num_is_nan(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::bool(n.is_nan())
}

fn num_sign(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    let sign = if n > 0.0 {
        1.0
    } else if n < 0.0 {
        -1.0
    } else {
        0.0
    };
    Value::num(sign)
}

fn num_to_string(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    let s = if n == n.trunc() && !n.is_infinite() && !n.is_nan() {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    };
    ctx.alloc_string(s)
}

fn num_truncate(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = args[0].as_num().unwrap();
    Value::num(n.trunc())
}

fn num_range_inclusive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    ctx.alloc_range(left, right, true)
}

fn num_range_exclusive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let right = match super::validate_num(ctx, args[1], "Right operand") {
        Some(v) => v,
        None => return Value::null(),
    };
    let left = args[0].as_num().unwrap();
    ctx.alloc_range(left, right, false)
}
