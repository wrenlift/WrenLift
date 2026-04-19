/// Core library for the Wren language.
///
/// Implements all built-in classes and their native methods, following
/// the standard Wren specification from wren-lang/wren.
mod bool;
mod cls;
mod fiber;
mod fn_obj;
mod list;
mod map;
pub mod meta;
mod null;
mod num;
mod obj;
pub mod random;
mod range;
mod sequence;
mod string;
mod system;

use super::vm::VM;

/// Initialize all core classes and bind their native methods.
///
/// Bootstrap order follows standard Wren:
/// 1. Object + Class (mutual dependency, special-cased)
/// 2. All other core classes
/// 3. Bind native methods to each class
pub fn initialize(vm: &mut VM) {
    // 1. Bootstrap Object and Class (chicken-and-egg).
    //    Object has no superclass. Class is the class of all classes.
    let object_class = vm.make_class("Object", std::ptr::null_mut());
    vm.object_class = object_class;

    let class_class = vm.make_class("Class", object_class);
    vm.class_class = class_class;

    // Wire up: Object's class is Class, Class's class is Class.
    unsafe {
        (*object_class).header.class = class_class;
        (*class_class).header.class = class_class;
    }

    // 2. Create remaining core classes.
    //    Sequence is created first so String/List/Map/Range can inherit from it.
    vm.bool_class = vm.make_class("Bool", object_class);
    vm.num_class = vm.make_class("Num", object_class);
    vm.null_class = vm.make_class("Null", object_class);
    vm.fn_class = vm.make_class("Fn", object_class);
    vm.fiber_class = vm.make_class("Fiber", object_class);
    vm.system_class = vm.make_class("System", object_class);

    // Sequence is the base class for iterable types.
    vm.sequence_class = vm.make_class("Sequence", object_class);

    // Collection classes inherit from Sequence (user classes can too).
    vm.string_class = vm.make_class("String", vm.sequence_class);
    vm.list_class = vm.make_class("List", vm.sequence_class);
    vm.map_class = vm.make_class("Map", vm.sequence_class);
    vm.range_class = vm.make_class("Range", vm.sequence_class);

    // Wrapper sequence classes (lazy iterators).
    let seq = vm.sequence_class;
    vm.map_sequence_class = vm.make_class("MapSequence", seq);
    unsafe {
        (*vm.map_sequence_class).num_fields = 2;
    } // [0]=sequence, [1]=fn
    vm.skip_sequence_class = vm.make_class("SkipSequence", seq);
    unsafe {
        (*vm.skip_sequence_class).num_fields = 2;
    } // [0]=sequence, [1]=count
    vm.take_sequence_class = vm.make_class("TakeSequence", seq);
    unsafe {
        (*vm.take_sequence_class).num_fields = 3;
    } // [0]=sequence, [1]=count, [2]=taken
    vm.where_sequence_class = vm.make_class("WhereSequence", seq);
    unsafe {
        (*vm.where_sequence_class).num_fields = 2;
    } // [0]=sequence, [1]=fn

    // String sub-sequences
    vm.string_byte_seq_class = vm.make_class("StringByteSequence", seq);
    unsafe {
        (*vm.string_byte_seq_class).num_fields = 1;
    } // [0]=string
    vm.string_code_point_seq_class = vm.make_class("StringCodePointSequence", seq);
    unsafe {
        (*vm.string_code_point_seq_class).num_fields = 1;
    } // [0]=string

    // MapEntry helper
    let obj = vm.object_class;
    vm.map_entry_class = vm.make_class("MapEntry", obj);
    unsafe {
        (*vm.map_entry_class).num_fields = 2;
    } // [0]=key, [1]=value

    // 3. Bind native methods to each class.
    //    Sequence methods are bound first, then subclass-specific methods
    //    override as needed (e.g., List has a native `count` that's faster
    //    than Sequence's iterate-based one).
    obj::bind(vm);
    cls::bind(vm);
    self::bool::bind(vm);
    null::bind(vm);
    num::bind(vm);
    sequence::bind(vm);
    string::bind(vm);
    list::bind(vm);
    map::bind(vm);
    range::bind(vm);
    fn_obj::bind(vm);
    fiber::bind(vm);
    system::bind(vm);

    // Re-propagate inherited methods so every core class sees its
    // ancestors' fully-populated method tables.
    //
    // `make_class` clones the superclass's method table at creation
    // time, but during bootstrap the superclass's table is still
    // empty — it only gets populated by the `*::bind` calls above.
    // Without this fixup, `[1, 2] == [3]` would throw "List does
    // not implement '==(_)'" because `Object.==(_)` never trickles
    // down to `List`.
    propagate_inherited_methods(vm);
}

/// Walk the core class list and ensure every subclass's method
/// table reflects its superclass's final state. A method is only
/// copied down when the subclass's own slot is empty, so overrides
/// defined in `list::bind` (etc.) stay put.
fn propagate_inherited_methods(vm: &mut super::vm::VM) {
    let classes = [
        vm.object_class,
        vm.class_class,
        vm.bool_class,
        vm.num_class,
        vm.null_class,
        vm.fn_class,
        vm.fiber_class,
        vm.system_class,
        vm.sequence_class,
        vm.string_class,
        vm.list_class,
        vm.map_class,
        vm.range_class,
        vm.map_sequence_class,
        vm.skip_sequence_class,
        vm.take_sequence_class,
        vm.where_sequence_class,
        vm.string_byte_seq_class,
        vm.string_code_point_seq_class,
        vm.map_entry_class,
    ];
    for &cls in &classes {
        if cls.is_null() {
            continue;
        }
        unsafe {
            let superclass = (*cls).superclass;
            if superclass.is_null() {
                continue;
            }
            let parent_methods_ptr = &raw const (*superclass).methods;
            let parent_methods = (&*parent_methods_ptr).clone();
            let cls_methods_ptr = &raw mut (*cls).methods;
            let cls_methods = &mut *cls_methods_ptr;
            for (idx, method) in parent_methods.iter().enumerate() {
                if let Some(m) = method {
                    if idx >= cls_methods.len() {
                        cls_methods.resize(idx + 1, None);
                    }
                    if cls_methods[idx].is_none() {
                        cls_methods[idx] = Some(*m);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers for core primitives
// ---------------------------------------------------------------------------

use super::object::{NativeContext, ObjHeader, ObjString, ObjType};
use super::value::Value;

/// Validate that a value is a Num, returning the f64 or signaling a runtime error.
pub fn validate_num(ctx: &mut dyn NativeContext, value: Value, arg_name: &str) -> Option<f64> {
    match value.as_num() {
        Some(n) => Some(n),
        None => {
            ctx.runtime_error(format!("{} must be a number.", arg_name));
            None
        }
    }
}

/// Validate that a value is an integer Num.
pub fn validate_int(ctx: &mut dyn NativeContext, value: Value, arg_name: &str) -> Option<i64> {
    match value.as_num() {
        Some(n) if n == n.trunc() && !n.is_infinite() => Some(n as i64),
        Some(_) => {
            ctx.runtime_error(format!("{} must be an integer.", arg_name));
            None
        }
        None => {
            ctx.runtime_error(format!("{} must be a number.", arg_name));
            None
        }
    }
}

/// Validate that a value is a String, returning a reference to the string data.
pub fn validate_string(
    ctx: &mut dyn NativeContext,
    value: Value,
    arg_name: &str,
) -> Option<String> {
    if !value.is_object() {
        ctx.runtime_error(format!("{} must be a string.", arg_name));
        return None;
    }
    let ptr = value.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::String {
            ctx.runtime_error(format!("{} must be a string.", arg_name));
            return None;
        }
        let obj = &*(ptr as *const ObjString);
        Some(obj.value.clone())
    }
}

/// Get string data from a Value that is known to be a string.
pub fn as_string(value: Value) -> &'static str {
    unsafe {
        let ptr = value.as_object().unwrap();
        let obj = &*(ptr as *const ObjString);
        // SAFETY: The string lives on the GC heap and won't move while we hold a reference.
        &obj.value
    }
}

/// Validate a list index, handling negative indices (Wren convention).
pub fn validate_index(
    ctx: &mut dyn NativeContext,
    value: Value,
    count: usize,
    arg_name: &str,
) -> Option<usize> {
    let raw = validate_int(ctx, value, arg_name)?;
    let index = if raw < 0 { raw + count as i64 } else { raw };
    if index < 0 || index as usize >= count {
        ctx.runtime_error(format!("{} out of bounds.", arg_name));
        return None;
    }
    Some(index as usize)
}

/// Check if a Value is an ObjString.
pub fn is_string(value: Value) -> bool {
    if !value.is_object() {
        return false;
    }
    let ptr = value.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe { (*header).obj_type == ObjType::String }
}

/// Check if a Value is an ObjList.
pub fn is_list(value: Value) -> bool {
    if !value.is_object() {
        return false;
    }
    let ptr = value.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe { (*header).obj_type == ObjType::List }
}
