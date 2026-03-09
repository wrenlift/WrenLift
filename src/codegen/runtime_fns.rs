/// Runtime functions callable from JIT-compiled code.
///
/// These are `extern "C"` functions that the JIT emits `CallRuntime` for.
/// The `resolve` function maps a runtime function name to its address so
/// the codegen can patch `CallRuntime` → `CallInd` with the actual pointer.
///
/// Convention:
/// - All values are passed/returned as NaN-boxed `u64`
/// - VM context is accessed via thread-local `JIT_CONTEXT`
/// - x86_64: System V ABI (RDI, RSI, RDX, RCX, R8, R9 → RAX)
/// - aarch64: AAPCS64 (X0-X7 → X0)

use crate::runtime::value::Value;
use crate::runtime::object::{
    MapKey, Method, ObjClass, ObjClosure, ObjHeader, ObjList, ObjMap,
    ObjString, ObjType,
};

// ---------------------------------------------------------------------------
// JIT context — thread-local VM state for runtime functions
// ---------------------------------------------------------------------------

/// Context passed to JIT-compiled code via thread-local storage.
/// Set by the interpreter before calling native code, read by runtime functions.
#[repr(C)]
pub struct JitContext {
    /// Pointer to module variable storage (Vec<Value> data pointer).
    pub module_vars: *mut u64,
    /// Number of module variables.
    pub module_var_count: u32,
    /// Raw pointer to the VM (for method dispatch, allocation, etc.).
    pub vm: *mut u8,
    /// Module name pointer (C string, for engine lookups).
    pub module_name: *const u8,
    /// Module name length.
    pub module_name_len: u32,
}

unsafe impl Send for JitContext {}

impl Default for JitContext {
    fn default() -> Self {
        Self {
            module_vars: std::ptr::null_mut(),
            module_var_count: 0,
            vm: std::ptr::null_mut(),
            module_name: std::ptr::null(),
            module_name_len: 0,
        }
    }
}

thread_local! {
    /// Thread-local JIT context — set before calling compiled code.
    pub static JIT_CONTEXT: std::cell::RefCell<JitContext> = std::cell::RefCell::new(JitContext::default());
}

/// Set up the JIT context for the current thread before calling compiled code.
pub fn set_jit_context(ctx: JitContext) {
    JIT_CONTEXT.with(|c| {
        *c.borrow_mut() = ctx;
    });
}

/// Access the JIT context. Returns None if not set (vm is null).
fn with_context<T>(f: impl FnOnce(&JitContext) -> T) -> Option<T> {
    JIT_CONTEXT.with(|c| {
        let ctx = c.borrow();
        if ctx.vm.is_null() {
            None
        } else {
            Some(f(&*ctx))
        }
    })
}

/// Get the VM pointer from the JIT context.
/// # Safety
/// Caller must ensure the VM pointer is valid.
unsafe fn vm_ref() -> Option<&'static mut crate::runtime::vm::VM> {
    JIT_CONTEXT.with(|c| {
        let ctx = c.borrow();
        if ctx.vm.is_null() {
            None
        } else {
            Some(&mut *(ctx.vm as *mut crate::runtime::vm::VM))
        }
    })
}

/// Get the module name from the JIT context.
pub fn module_name() -> String {
    JIT_CONTEXT.with(|c| {
        let ctx = c.borrow();
        if ctx.module_name.is_null() || ctx.module_name_len == 0 {
            String::new()
        } else {
            unsafe {
                let slice = std::slice::from_raw_parts(ctx.module_name, ctx.module_name_len as usize);
                String::from_utf8_lossy(slice).into_owned()
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Runtime function implementations
// ---------------------------------------------------------------------------

/// Get a module variable by slot index.
pub extern "C" fn wren_get_module_var(slot: u64) -> u64 {
    with_context(|ctx| {
        let idx = slot as usize;
        if idx < ctx.module_var_count as usize {
            unsafe { *ctx.module_vars.add(idx) }
        } else {
            Value::null().to_bits()
        }
    }).unwrap_or(Value::null().to_bits())
}

/// Set a module variable by slot index.
pub extern "C" fn wren_set_module_var(slot: u64, value: u64) -> u64 {
    with_context(|ctx| {
        let idx = slot as usize;
        if idx < ctx.module_var_count as usize {
            unsafe { *ctx.module_vars.add(idx) = value; }
        }
    });
    value
}

/// Call a method on a receiver.
/// Args: receiver (NaN-boxed), method_sym (u64 raw SymbolId)
pub extern "C" fn wren_call(receiver: u64, method: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let method_sym = crate::intern::SymbolId::from_raw(method as u32);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let class = vm.class_of(recv);
    let method_entry = unsafe { (*class).find_method(method_sym).cloned() };

    match method_entry {
        Some(Method::Native(native_fn)) => {
            let args = &[recv];
            native_fn(vm, args).to_bits()
        }
        Some(Method::Closure(_closure_ptr)) => {
            // Closure dispatch requires pushing a MIR frame — complex.
            // For now, fall back to null. Full closure dispatch from JIT
            // requires re-entering the interpreter.
            Value::null().to_bits()
        }
        Some(Method::Constructor(_closure_ptr)) => {
            Value::null().to_bits()
        }
        None => {
            // Try static method dispatch
            if class == vm.class_class {
                let method_name = vm.interner.resolve(method_sym).to_string();
                let static_name = format!("static:{}", method_name);
                let static_sym = vm.interner.intern(&static_name);
                if let Some(recv_ptr) = recv.as_object() {
                    let recv_class_ptr = recv_ptr as *mut ObjClass;
                    let static_entry = unsafe { (*recv_class_ptr).find_method(static_sym).cloned() };
                    match static_entry {
                        Some(Method::Native(native_fn)) => {
                            let args = &[recv];
                            native_fn(vm, args).to_bits()
                        }
                        _ => Value::null().to_bits(),
                    }
                } else {
                    Value::null().to_bits()
                }
            } else {
                Value::null().to_bits()
            }
        }
    }
}

/// Super call dispatch.
pub extern "C" fn wren_super_call(receiver: u64, method: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let method_sym = crate::intern::SymbolId::from_raw(method as u32);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let class = vm.class_of(recv);
    let superclass = unsafe { (*class).superclass };
    if superclass.is_null() {
        return Value::null().to_bits();
    }

    let method_entry = unsafe { (*superclass).find_method(method_sym).cloned() };
    match method_entry {
        Some(Method::Native(native_fn)) => {
            let args = &[recv];
            native_fn(vm, args).to_bits()
        }
        _ => Value::null().to_bits(),
    }
}

/// Allocate a new empty list.
pub extern "C" fn wren_make_list() -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let list_ptr = vm.gc.alloc_list();
    unsafe { (*list_ptr).header.class = vm.list_class; }
    Value::object(list_ptr as *mut u8).to_bits()
}

/// Allocate a new empty map.
pub extern "C" fn wren_make_map() -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let map_ptr = vm.gc.alloc_map();
    unsafe { (*map_ptr).header.class = vm.map_class; }
    Value::object(map_ptr as *mut u8).to_bits()
}

/// Allocate a new range.
pub extern "C" fn wren_make_range(from: u64, to: u64, inclusive: u64) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let from_val = f64::from_bits(from);
    let to_val = f64::from_bits(to);
    let is_inclusive = inclusive != 0;

    let range_ptr = vm.gc.alloc_range(from_val, to_val, is_inclusive);
    unsafe { (*range_ptr).header.class = vm.range_class; }
    Value::object(range_ptr as *mut u8).to_bits()
}

/// Allocate a closure from a function ID.
pub extern "C" fn wren_make_closure(fn_id: u64) -> u64 {
    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let func_id = fn_id as u32;
    let name_sym = vm.interner.intern("<closure>");
    let fn_ptr = vm.gc.alloc_fn(name_sym, 0, 0, func_id);
    unsafe { (*fn_ptr).header.class = vm.fn_class; }

    let closure_ptr = vm.gc.alloc_closure(fn_ptr);
    unsafe { (*closure_ptr).header.class = vm.fn_class; }
    Value::object(closure_ptr as *mut u8).to_bits()
}

/// Concatenate two strings.
pub extern "C" fn wren_string_concat(a: u64, b: u64) -> u64 {
    let va = Value::from_bits(a);
    let vb = Value::from_bits(b);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let sa = crate::runtime::vm_interp::value_to_string(vm, va);
    let sb = crate::runtime::vm_interp::value_to_string(vm, vb);
    let concatenated = format!("{}{}", sa, sb);
    vm.new_string(concatenated).to_bits()
}

/// Convert a value to its string representation.
pub extern "C" fn wren_to_string(val: u64) -> u64 {
    let v = Value::from_bits(val);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(v) => v,
        None => return Value::null().to_bits(),
    };

    let s = crate::runtime::vm_interp::value_to_string(vm, v);
    vm.new_string(s).to_bits()
}

/// Type check: is value an instance of class?
/// class_sym is a SymbolId identifying the class name.
pub extern "C" fn wren_is_type(val: u64, class_sym: u64) -> u64 {
    let v = Value::from_bits(val);
    let target_sym = crate::intern::SymbolId::from_raw(class_sym as u32);

    let vm = unsafe { vm_ref() };
    let vm = match vm {
        Some(vm) => vm,
        None => return Value::bool(false).to_bits(),
    };

    // Walk the class hierarchy
    let mut class = vm.class_of(v);
    loop {
        if class.is_null() { break; }
        let class_name = unsafe { (*class).name };
        if class_name == target_sym {
            return Value::bool(true).to_bits();
        }
        class = unsafe { (*class).superclass };
    }
    Value::bool(false).to_bits()
}

/// Subscript get (list[idx] or map[key]).
pub extern "C" fn wren_subscript_get(receiver: u64, index: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let idx = Value::from_bits(index);

    if recv.is_object() {
        let ptr = recv.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        let obj_type = unsafe { (*header).obj_type };

        match obj_type {
            ObjType::List => {
                let list = ptr as *const ObjList;
                if let Some(n) = idx.as_num() {
                    let i = n as usize;
                    let count = unsafe { (*list).count as usize };
                    if i < count {
                        if let Some(val) = unsafe { (*list).get(i) } {
                            return val.to_bits();
                        }
                    }
                }
            }
            ObjType::Map => {
                let map = ptr as *const ObjMap;
                let map_key = MapKey::new(idx);
                if let Some(val) = unsafe { (*map).entries.get(&map_key) } {
                    return val.to_bits();
                }
            }
            ObjType::String => {
                let string = ptr as *const ObjString;
                if let Some(n) = idx.as_num() {
                    let i = n as usize;
                    let s = unsafe { (*string).as_str() };
                    if let Some(ch) = s.chars().nth(i) {
                        let vm = unsafe { vm_ref() };
                        if let Some(vm) = vm {
                            return vm.new_string(ch.to_string()).to_bits();
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Value::null().to_bits()
}

/// Subscript set (list[idx] = val or map[key] = val).
pub extern "C" fn wren_subscript_set(receiver: u64, index: u64, value: u64) -> u64 {
    let recv = Value::from_bits(receiver);
    let idx = Value::from_bits(index);

    if recv.is_object() {
        let ptr = recv.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        let obj_type = unsafe { (*header).obj_type };

        match obj_type {
            ObjType::List => {
                let list = ptr as *mut ObjList;
                if let Some(n) = idx.as_num() {
                    let i = n as usize;
                    let count = unsafe { (*list).count as usize };
                    if i < count {
                        unsafe { (*list).set(i, Value::from_bits(value)); }
                        return value;
                    }
                }
            }
            ObjType::Map => {
                let map = ptr as *mut ObjMap;
                let map_key = MapKey::new(idx);
                unsafe { (*map).entries.insert(map_key, Value::from_bits(value)); }
                return value;
            }
            _ => {}
        }
    }
    value
}

/// Get an upvalue from the current closure.
pub extern "C" fn wren_get_upvalue(closure: u64, index: u64) -> u64 {
    let closure_val = Value::from_bits(closure);
    if !closure_val.is_object() {
        return Value::null().to_bits();
    }
    let ptr = closure_val.as_object().unwrap() as *mut ObjClosure;
    let idx = index as usize;
    unsafe {
        let upvalues = &(*ptr).upvalues;
        if idx < upvalues.len() {
            let upvalue = upvalues[idx];
            if !upvalue.is_null() {
                return (*upvalue).get().to_bits();
            }
        }
    }
    Value::null().to_bits()
}

/// Set an upvalue in the current closure.
pub extern "C" fn wren_set_upvalue(closure: u64, index: u64, value: u64) -> u64 {
    let closure_val = Value::from_bits(closure);
    if !closure_val.is_object() {
        return Value::null().to_bits();
    }
    let ptr = closure_val.as_object().unwrap() as *mut ObjClosure;
    let idx = index as usize;
    unsafe {
        let upvalues = &(*ptr).upvalues;
        if idx < upvalues.len() {
            let upvalue = upvalues[idx];
            if !upvalue.is_null() {
                (*upvalue).set(Value::from_bits(value));
            }
        }
    }
    value
}

/// Guard: check that value is an instance of the expected class.
/// Returns the value if check passes, traps otherwise.
pub extern "C" fn wren_guard_class(value: u64, class: u64) -> u64 {
    // For now, always pass the guard. A proper implementation would
    // check the class hierarchy and deoptimize on mismatch.
    let _ = class;
    value
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

pub extern "C" fn wren_is_truthy(value: u64) -> u64 {
    let v = Value::from_bits(value);
    Value::bool(!v.is_falsy()).to_bits()
}

// ---------------------------------------------------------------------------
// ABI physical register mappings
// ---------------------------------------------------------------------------

/// ABI argument registers for each target (physical register indices).
fn abi_regs(target: super::Target) -> (&'static [u32], u32, u32, u32) {
    // Returns (arg_regs, ret_reg, call_scratch, copy_scratch)
    match target {
        super::Target::Aarch64 => (
            &[0, 1, 2, 3, 4, 5],  // X0-X5
            0,                      // return in X0
            16,                     // X16 (IP0) for function pointer
            17,                     // X17 (IP1) for cycle breaking
        ),
        super::Target::X86_64 => (
            &[7, 6, 2, 1, 8, 9],  // RDI, RSI, RDX, RCX, R8, R9
            0,                      // return in RAX
            11,                     // R11 for function pointer
            11,                     // R11 for cycle breaking (safe: copy resolves before ptr load)
        ),
        super::Target::Wasm => (&[], 0, 0, 0),
    }
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
// CallRuntime → ABI setup + CallInd lowering pass
// ---------------------------------------------------------------------------

use crate::codegen::{MachInst, MachFunc, VReg};

/// Lower all `CallRuntime` instructions to proper ABI calling sequences.
///
/// This pass MUST run AFTER register allocation and sentinel fixup, when all
/// vregs are physical register indices. It uses parallel copy resolution to
/// avoid the classic register-clobbering problem with sequential moves.
///
/// Sequence per call:
/// 1. Parallel copy: move arguments from their physical registers to ABI arg registers
/// 2. Load function address into scratch register
/// 3. Indirect call via scratch
/// 4. Move return value from ABI return register to destination
pub fn link_runtime_calls(mf: &mut MachFunc, target: super::Target) {
    let (abi_args, abi_ret, call_scratch, copy_scratch) = abi_regs(target);
    if abi_args.is_empty() { return; } // Wasm: no runtime calls

    let mut i = 0;
    while i < mf.insts.len() {
        if let MachInst::CallRuntime { name, args, ret } = &mf.insts[i] {
            let name = *name;
            let args = args.clone();
            let ret = *ret;

            if let Some(addr) = resolve(name) {
                let mut new_insts: Vec<MachInst> = Vec::new();

                // 1. Build (src_phys, dst_abi) move pairs and resolve in parallel
                let moves: Vec<(u32, u32)> = args.iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx < abi_args.len())
                    .map(|(idx, vreg)| (vreg.index, abi_args[idx]))
                    .collect();

                for (src, dst) in resolve_parallel_copy(&moves, copy_scratch) {
                    new_insts.push(MachInst::Mov {
                        dst: VReg::gp(dst),
                        src: VReg::gp(src),
                    });
                }

                // 2. Load function address into scratch register
                new_insts.push(MachInst::LoadImm {
                    dst: VReg::gp(call_scratch),
                    bits: addr as u64,
                });

                // 3. Indirect call
                new_insts.push(MachInst::CallInd {
                    target: VReg::gp(call_scratch),
                });

                // 4. Move return value from ABI return register
                if let Some(ret_vreg) = ret {
                    if ret_vreg.index != abi_ret {
                        new_insts.push(MachInst::Mov {
                            dst: ret_vreg,
                            src: VReg::gp(abi_ret),
                        });
                    }
                }

                let seq_len = new_insts.len();
                mf.insts.splice(i..=i, new_insts);
                i += seq_len;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
}

/// Resolve a set of parallel register moves so no source is clobbered before
/// it is read. Uses a scratch register to break cycles.
///
/// Input: slice of (src_phys, dst_phys) pairs.
/// Output: ordered sequence of (src, dst) moves that is safe to execute sequentially.
fn resolve_parallel_copy(moves: &[(u32, u32)], scratch: u32) -> Vec<(u32, u32)> {
    // Filter identity moves (src == dst).
    let mut remaining: Vec<(u32, u32)> = moves.iter()
        .filter(|(s, d)| s != d)
        .cloned()
        .collect();

    if remaining.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();

    // Phase 1: Emit moves whose destination is NOT a source of any other move.
    // These are safe because writing the destination won't clobber anything.
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < remaining.len() {
            let (_, dst) = remaining[i];
            let dst_is_source = remaining.iter()
                .enumerate()
                .any(|(j, (s, _))| j != i && *s == dst);
            if !dst_is_source {
                result.push(remaining.remove(i));
                changed = true;
            } else {
                i += 1;
            }
        }
    }

    // Phase 2: Only cycles remain. Break each cycle using the scratch register.
    while !remaining.is_empty() {
        let (first_src, first_dst) = remaining.remove(0);
        // Save first source to scratch so its register can be overwritten.
        result.push((first_src, scratch));

        // Follow the cycle chain.
        let mut current = first_dst;
        while let Some(idx) = remaining.iter().position(|(s, _)| *s == current) {
            let (_, next_dst) = remaining.remove(idx);
            result.push((current, next_dst));
            current = next_dst;
        }

        // Close the cycle: move from scratch to the first destination.
        result.push((scratch, first_dst));
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    #[test]
    fn test_jit_context_thread_local() {
        // Default context has null VM
        assert!(with_context(|_| ()).is_none());

        // Set a context
        let mut dummy_vm = [0u8; 1024];
        set_jit_context(JitContext {
            module_vars: std::ptr::null_mut(),
            module_var_count: 0,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
        });
        assert!(with_context(|_| ()).is_some());

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_get_module_var_with_context() {
        let mut vars: Vec<u64> = vec![
            Value::num(42.0).to_bits(),
            Value::bool(true).to_bits(),
            Value::null().to_bits(),
        ];
        let mut dummy_vm = [0u8; 4096];

        set_jit_context(JitContext {
            module_vars: vars.as_mut_ptr(),
            module_var_count: 3,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
        });

        // Read first module var (num 42)
        let result = wren_get_module_var(0);
        assert_eq!(result, Value::num(42.0).to_bits());

        // Read second (bool true)
        let result = wren_get_module_var(1);
        assert_eq!(result, Value::bool(true).to_bits());

        // Out of bounds
        let result = wren_get_module_var(99);
        assert_eq!(result, Value::null().to_bits());

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_set_module_var_with_context() {
        let mut vars: Vec<u64> = vec![Value::null().to_bits(); 3];
        let mut dummy_vm = [0u8; 4096];

        set_jit_context(JitContext {
            module_vars: vars.as_mut_ptr(),
            module_var_count: 3,
            vm: dummy_vm.as_mut_ptr(),
            module_name: std::ptr::null(),
            module_name_len: 0,
        });

        let new_val = Value::num(99.0).to_bits();
        wren_set_module_var(1, new_val);

        // Verify the module var was updated
        assert_eq!(vars[1], new_val);

        // Clean up
        set_jit_context(JitContext::default());
    }

    #[test]
    fn test_link_inserts_abi_moves_aarch64() {
        // Simulate post-regalloc state: args already in physical registers.
        let mut mf = MachFunc::new("test".into());
        // Pretend regalloc assigned arg0 to X3, arg1 to X5, ret to X3.
        let arg0 = VReg::gp(3);
        let arg1 = VReg::gp(5);
        let ret = VReg::gp(3);

        mf.emit(MachInst::LoadImm { dst: arg0, bits: 10 });
        mf.emit(MachInst::LoadImm { dst: arg1, bits: 20 });
        mf.emit(MachInst::CallRuntime {
            name: "wren_num_add",
            args: vec![arg0, arg1],
            ret: Some(ret),
        });
        mf.emit(MachInst::Ret);

        let before_len = mf.insts.len();
        link_runtime_calls(&mut mf, crate::codegen::Target::Aarch64);

        // CallRuntime replaced: 2 arg movs + LoadImm + CallInd + ret mov = 5 new
        assert_eq!(mf.insts.len(), before_len + 4);

        // Verify moves target ABI physical registers (X0=0, X1=1)
        let has_x0_move = mf.insts.iter().any(|inst| {
            matches!(inst, MachInst::Mov { dst, .. } if dst.index == 0)
        });
        assert!(has_x0_move, "should have move to X0");

        let has_x1_move = mf.insts.iter().any(|inst| {
            matches!(inst, MachInst::Mov { dst, .. } if dst.index == 1)
        });
        assert!(has_x1_move, "should have move to X1");
    }

    #[test]
    fn test_parallel_copy_no_cycle() {
        // (3→0, 5→1) — no conflicts
        let result = resolve_parallel_copy(&[(3, 0), (5, 1)], 17);
        assert_eq!(result.len(), 2);
        // Both moves should appear, order doesn't matter for correctness
        assert!(result.contains(&(3, 0)));
        assert!(result.contains(&(5, 1)));
    }

    #[test]
    fn test_parallel_copy_swap_cycle() {
        // (0→1, 1→0) — classic swap cycle
        let result = resolve_parallel_copy(&[(0, 1), (1, 0)], 17);
        // Should use scratch: 3 moves total
        assert_eq!(result.len(), 3, "swap should produce 3 moves: {:?}", result);
        assert!(result.iter().any(|(_, d)| *d == 17), "should use scratch as dst");
        assert!(result.iter().any(|(s, _)| *s == 17), "should use scratch as src");
    }

    #[test]
    fn test_parallel_copy_identity_filtered() {
        // (0→0, 1→1) — all identity, nothing to do
        let result = resolve_parallel_copy(&[(0, 0), (1, 1)], 17);
        assert!(result.is_empty());
    }

    #[test]
    fn test_boxed_arithmetic_still_works() {
        let a = Value::num(10.0).to_bits();
        let b = Value::num(3.0).to_bits();
        assert_eq!(wren_num_add(a, b), Value::num(13.0).to_bits());
        assert_eq!(wren_num_sub(a, b), Value::num(7.0).to_bits());
        assert_eq!(wren_num_mul(a, b), Value::num(30.0).to_bits());
        assert_eq!(wren_cmp_lt(a, b), Value::bool(false).to_bits());
        assert_eq!(wren_cmp_gt(a, b), Value::bool(true).to_bits());
        assert_eq!(wren_cmp_eq(a, a), Value::bool(true).to_bits());
    }
}
