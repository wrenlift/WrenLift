/// Runtime functions callable from JIT-compiled code.
///
/// These are `extern "C"` functions that the JIT emits `CallRuntime` for.
/// The `resolve` function maps a runtime function name to its address so
/// the codegen can patch `CallRuntime` → `CallInd` with the actual pointer.
///
/// Convention:
/// - All values are passed/returned as NaN-boxed `u64`
/// - The VM context pointer is passed in a dedicated register (set up by prologue)
/// - x86_64: System V ABI (RDI, RSI, RDX, RCX, R8, R9 → RAX)
/// - aarch64: AAPCS64 (X0-X7 → X0)

use crate::runtime::value::Value;

// ---------------------------------------------------------------------------
// Runtime function implementations
// ---------------------------------------------------------------------------

/// Get a module variable by slot index.
/// Args: vm_ctx (implicit), slot (u64)
/// Returns: NaN-boxed Value
pub extern "C" fn wren_get_module_var(_slot: u64) -> u64 {
    // Stub — requires VM context pointer to access module vars.
    Value::null().to_bits()
}

/// Call a method on a receiver.
/// Args: vm_ctx (implicit), receiver (u64), method_sym (u64), argc (u64), arg0..argN
/// Returns: NaN-boxed Value
pub extern "C" fn wren_call(_receiver: u64, _method: u64) -> u64 {
    Value::null().to_bits()
}

/// Allocate a new list with given elements.
pub extern "C" fn wren_make_list() -> u64 {
    Value::null().to_bits()
}

/// Allocate a new map.
pub extern "C" fn wren_make_map() -> u64 {
    Value::null().to_bits()
}

/// Allocate a new range.
pub extern "C" fn wren_make_range(_from: u64, _to: u64, _inclusive: u64) -> u64 {
    Value::null().to_bits()
}

/// Allocate a closure.
pub extern "C" fn wren_make_closure(_fn_id: u64) -> u64 {
    Value::null().to_bits()
}

/// Concatenate two strings.
pub extern "C" fn wren_string_concat(_a: u64, _b: u64) -> u64 {
    Value::null().to_bits()
}

/// Convert a value to its string representation.
pub extern "C" fn wren_to_string(_val: u64) -> u64 {
    Value::null().to_bits()
}

/// Type check: is value an instance of class?
pub extern "C" fn wren_is_type(_val: u64, _class: u64) -> u64 {
    Value::bool(false).to_bits()
}

/// Subscript get (list[idx] or map[key]).
pub extern "C" fn wren_subscript_get(_receiver: u64, _index: u64) -> u64 {
    Value::null().to_bits()
}

/// Subscript set (list[idx] = val or map[key] = val).
pub extern "C" fn wren_subscript_set(_receiver: u64, _index: u64, _value: u64) -> u64 {
    Value::null().to_bits()
}

/// Super call dispatch.
pub extern "C" fn wren_super_call(_receiver: u64, _method: u64) -> u64 {
    Value::null().to_bits()
}

// ---------------------------------------------------------------------------
// Boxed NaN-boxed arithmetic runtime functions
// ---------------------------------------------------------------------------

/// Helper: unbox a NaN-boxed u64 as f64.
#[inline(always)]
fn unbox_num(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// Helper: box an f64 as NaN-boxed u64.
#[inline(always)]
fn box_num(n: f64) -> u64 {
    Value::num(n).to_bits()
}

pub extern "C" fn wren_num_add(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) + unbox_num(b))
}

pub extern "C" fn wren_num_sub(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) - unbox_num(b))
}

pub extern "C" fn wren_num_mul(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) * unbox_num(b))
}

pub extern "C" fn wren_num_div(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) / unbox_num(b))
}

pub extern "C" fn wren_num_mod(a: u64, b: u64) -> u64 {
    box_num(unbox_num(a) % unbox_num(b))
}

pub extern "C" fn wren_num_neg(a: u64) -> u64 {
    box_num(-unbox_num(a))
}

// ---------------------------------------------------------------------------
// Boxed comparison runtime functions
// ---------------------------------------------------------------------------

pub extern "C" fn wren_cmp_lt(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) < unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_gt(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) > unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_le(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) <= unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_ge(a: u64, b: u64) -> u64 {
    Value::bool(unbox_num(a) >= unbox_num(b)).to_bits()
}

pub extern "C" fn wren_cmp_eq(a: u64, b: u64) -> u64 {
    Value::bool(a == b).to_bits()
}

pub extern "C" fn wren_cmp_ne(a: u64, b: u64) -> u64 {
    Value::bool(a != b).to_bits()
}

// ---------------------------------------------------------------------------
// Boxed logical
// ---------------------------------------------------------------------------

pub extern "C" fn wren_not(a: u64) -> u64 {
    Value::bool(Value::from_bits(a).is_falsy()).to_bits()
}

// ---------------------------------------------------------------------------
// Upvalue, guard, and misc runtime functions
// ---------------------------------------------------------------------------

pub extern "C" fn wren_get_upvalue(_closure: u64, _index: u64) -> u64 {
    Value::null().to_bits()
}

pub extern "C" fn wren_set_upvalue(_closure: u64, _index: u64, _value: u64) -> u64 {
    Value::null().to_bits()
}

pub extern "C" fn wren_set_module_var(_slot: u64, _value: u64) -> u64 {
    Value::null().to_bits()
}

pub extern "C" fn wren_guard_class(_value: u64, _class: u64) -> u64 {
    // Guard pass — no-op if type matches, trap on mismatch.
    // Stub: always passes.
    _value
}

pub extern "C" fn wren_is_truthy(value: u64) -> u64 {
    let v = Value::from_bits(value);
    Value::bool(!v.is_falsy()).to_bits()
}

// ---------------------------------------------------------------------------
// Address resolution
// ---------------------------------------------------------------------------

/// Resolve a runtime function name to its address.
/// Returns `None` if the name is unknown.
pub fn resolve(name: &str) -> Option<usize> {
    match name {
        "wren_get_module_var" => Some(wren_get_module_var as usize),
        "wren_call" => Some(wren_call as usize),
        "wren_make_list" => Some(wren_make_list as usize),
        "wren_make_map" => Some(wren_make_map as usize),
        "wren_make_range" => Some(wren_make_range as usize),
        "wren_make_closure" => Some(wren_make_closure as usize),
        "wren_string_concat" => Some(wren_string_concat as usize),
        "wren_to_string" => Some(wren_to_string as usize),
        "wren_is_type" => Some(wren_is_type as usize),
        "wren_subscript_get" => Some(wren_subscript_get as usize),
        "wren_subscript_set" => Some(wren_subscript_set as usize),
        "wren_super_call" => Some(wren_super_call as usize),
        // Boxed arithmetic
        "wren_num_add" => Some(wren_num_add as usize),
        "wren_num_sub" => Some(wren_num_sub as usize),
        "wren_num_mul" => Some(wren_num_mul as usize),
        "wren_num_div" => Some(wren_num_div as usize),
        "wren_num_mod" => Some(wren_num_mod as usize),
        "wren_num_neg" => Some(wren_num_neg as usize),
        // Boxed comparisons
        "wren_cmp_lt" => Some(wren_cmp_lt as usize),
        "wren_cmp_gt" => Some(wren_cmp_gt as usize),
        "wren_cmp_le" => Some(wren_cmp_le as usize),
        "wren_cmp_ge" => Some(wren_cmp_ge as usize),
        "wren_cmp_eq" => Some(wren_cmp_eq as usize),
        "wren_cmp_ne" => Some(wren_cmp_ne as usize),
        // Logical
        "wren_not" => Some(wren_not as usize),
        // Upvalue, guard, misc
        "wren_get_upvalue" => Some(wren_get_upvalue as usize),
        "wren_set_upvalue" => Some(wren_set_upvalue as usize),
        "wren_set_module_var" => Some(wren_set_module_var as usize),
        "wren_guard_class" => Some(wren_guard_class as usize),
        "wren_is_truthy" => Some(wren_is_truthy as usize),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CallRuntime → CallInd lowering pass
// ---------------------------------------------------------------------------

use crate::codegen::{MachInst, MachFunc};

/// Lower all `CallRuntime` instructions to `LoadImm` + `CallInd`.
/// This must run after MIR lowering but before register allocation.
pub fn link_runtime_calls(mf: &mut MachFunc) {
    let mut i = 0;
    while i < mf.insts.len() {
        if let MachInst::CallRuntime { name, args, ret } = &mf.insts[i] {
            let name = *name;
            let args = args.clone();
            let ret = *ret;

            if let Some(addr) = resolve(name) {
                // Replace CallRuntime with LoadImm + CallInd
                let addr_reg = mf.new_gp();

                let load = MachInst::LoadImm { dst: addr_reg, bits: addr as u64 };
                let call = MachInst::CallInd { target: addr_reg };

                // Replace the CallRuntime with LoadImm, insert CallInd after
                mf.insts[i] = load;
                mf.insts.insert(i + 1, call);

                let _ = (args, ret); // consumed by regalloc liveness
                i += 2;
            } else {
                // Unknown runtime function — leave as error for backend
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_functions() {
        assert!(resolve("wren_get_module_var").is_some());
        assert!(resolve("wren_call").is_some());
        assert!(resolve("wren_make_list").is_some());
        assert!(resolve("wren_subscript_get").is_some());
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve("nonexistent_fn").is_none());
    }

    #[test]
    fn function_pointers_are_nonzero() {
        assert_ne!(resolve("wren_get_module_var").unwrap(), 0);
        assert_ne!(resolve("wren_call").unwrap(), 0);
    }
}
