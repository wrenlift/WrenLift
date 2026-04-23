#![allow(clippy::not_unsafe_ptr_arg_deref)]
/// C embedding API for WrenLift.
///
/// Provides a wren.h-compatible interface for embedding WrenLift in C/C++
/// applications. All functions use `extern "C"` linkage and `#[no_mangle]`
/// for direct FFI consumption.
///
/// # Usage from C
///
/// ```c
/// #include "wrenlift.h"
///
/// static void write_fn(WrenVM* vm, const char* text) {
///     printf("%s", text);
/// }
///
/// int main() {
///     WrenConfiguration config;
///     wrenInitConfiguration(&config);
///     config.writeFn = write_fn;
///
///     WrenVM* vm = wrenNewVM(&config);
///     wrenInterpret(vm, "main", "System.print(\"Hello!\")");
///     wrenFreeVM(vm);
/// }
/// ```
use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::ptr;

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::{VMConfig, VM};

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

/// Opaque VM handle for C consumers.
pub type WrenVM = VM;

/// A persistent handle to a Wren value (prevents GC collection).
#[repr(C)]
pub struct WrenHandle {
    index: usize,
    /// Method signature for call handles (empty for value handles).
    signature: String,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrenErrorType {
    CompileError = 0,
    RuntimeError = 1,
    StackTrace = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrenInterpretResult {
    Success = 0,
    CompileError = 1,
    RuntimeError = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrenType {
    Bool = 0,
    Num = 1,
    Foreign = 2,
    List = 3,
    Map = 4,
    Null = 5,
    String = 6,
    Unknown = 7,
}

// ---------------------------------------------------------------------------
// Callback types (C function pointers)
// ---------------------------------------------------------------------------

pub type WrenWriteFn = extern "C" fn(vm: *mut WrenVM, text: *const c_char);
pub type WrenErrorFn = extern "C" fn(
    vm: *mut WrenVM,
    error_type: WrenErrorType,
    module: *const c_char,
    line: c_int,
    message: *const c_char,
);
pub type WrenForeignMethodFn = extern "C" fn(vm: *mut WrenVM);
pub type WrenFinalizerFn = extern "C" fn(data: *mut c_void);
pub type WrenResolveModuleFn =
    extern "C" fn(vm: *mut WrenVM, importer: *const c_char, name: *const c_char) -> *const c_char;
pub type WrenLoadModuleFn =
    extern "C" fn(vm: *mut WrenVM, name: *const c_char) -> WrenLoadModuleResult;
pub type WrenBindForeignMethodFn = extern "C" fn(
    vm: *mut WrenVM,
    module: *const c_char,
    class_name: *const c_char,
    is_static: bool,
    signature: *const c_char,
) -> Option<WrenForeignMethodFn>;
pub type WrenBindForeignClassFn = extern "C" fn(
    vm: *mut WrenVM,
    module: *const c_char,
    class_name: *const c_char,
) -> WrenForeignClassMethods;

// ---------------------------------------------------------------------------
// Configuration and result structs
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct WrenLoadModuleResult {
    pub source: *const c_char,
    pub on_complete: Option<extern "C" fn(*mut WrenVM, *const c_char, WrenLoadModuleResult)>,
    pub user_data: *mut c_void,
}

#[repr(C)]
pub struct WrenForeignClassMethods {
    pub allocate: Option<WrenForeignMethodFn>,
    pub finalize: Option<WrenFinalizerFn>,
}

#[repr(C)]
pub struct WrenConfiguration {
    pub reallocate_fn: *mut c_void, // Not used, kept for ABI compat
    pub resolve_module_fn: Option<WrenResolveModuleFn>,
    pub load_module_fn: Option<WrenLoadModuleFn>,
    pub bind_foreign_method_fn: Option<WrenBindForeignMethodFn>,
    pub bind_foreign_class_fn: Option<WrenBindForeignClassFn>,
    pub write_fn: Option<WrenWriteFn>,
    pub error_fn: Option<WrenErrorFn>,
    pub initial_heap_size: usize,
    pub min_heap_size: usize,
    pub heap_growth_percent: c_int,
    pub user_data: *mut c_void,
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

pub const WREN_VERSION_MAJOR: c_int = 0;
pub const WREN_VERSION_MINOR: c_int = 5;
pub const WREN_VERSION_PATCH: c_int = 0;

#[no_mangle]
pub extern "C" fn wrenGetVersionNumber() -> c_int {
    WREN_VERSION_MAJOR * 1_000_000 + WREN_VERSION_MINOR * 1_000 + WREN_VERSION_PATCH
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenInitConfiguration(config: *mut WrenConfiguration) {
    if config.is_null() {
        return;
    }
    unsafe {
        *config = WrenConfiguration {
            reallocate_fn: ptr::null_mut(),
            resolve_module_fn: None,
            load_module_fn: None,
            bind_foreign_method_fn: None,
            bind_foreign_class_fn: None,
            write_fn: None,
            error_fn: None,
            initial_heap_size: 10 * 1024 * 1024,
            min_heap_size: 1024 * 1024,
            heap_growth_percent: 50,
            user_data: ptr::null_mut(),
        };
    }
}

// ---------------------------------------------------------------------------
// VM lifecycle
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenNewVM(config: *const WrenConfiguration) -> *mut WrenVM {
    let vm_config = if config.is_null() {
        VMConfig::default()
    } else {
        unsafe { build_vm_config(&*config) }
    };

    let vm = Box::new(VM::new(vm_config));
    Box::into_raw(vm)
}

#[no_mangle]
pub extern "C" fn wrenFreeVM(vm: *mut WrenVM) {
    if !vm.is_null() {
        unsafe {
            drop(Box::from_raw(vm));
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenCollectGarbage(vm: *mut WrenVM) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).collect_garbage();
    }
}

// ---------------------------------------------------------------------------
// Interpret
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenInterpret(
    vm: *mut WrenVM,
    module: *const c_char,
    source: *const c_char,
) -> WrenInterpretResult {
    if vm.is_null() || module.is_null() || source.is_null() {
        return WrenInterpretResult::RuntimeError;
    }
    let module_name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("main");
    let source_str = unsafe { CStr::from_ptr(source) }.to_str().unwrap_or("");
    let vm_ref = unsafe { &mut *vm };
    use crate::runtime::engine::InterpretResult;
    match vm_ref.interpret(module_name, source_str) {
        InterpretResult::Success => WrenInterpretResult::Success,
        InterpretResult::CompileError => WrenInterpretResult::CompileError,
        InterpretResult::RuntimeError => WrenInterpretResult::RuntimeError,
    }
}

// ---------------------------------------------------------------------------
// Call handles
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenMakeCallHandle(vm: *mut WrenVM, signature: *const c_char) -> *mut WrenHandle {
    if vm.is_null() || signature.is_null() {
        return ptr::null_mut();
    }
    let sig = unsafe { CStr::from_ptr(signature) }
        .to_str()
        .unwrap_or("")
        .to_string();
    Box::into_raw(Box::new(WrenHandle {
        index: 0,
        signature: sig,
    }))
}

#[no_mangle]
pub extern "C" fn wrenCall(vm: *mut WrenVM, method: *mut WrenHandle) -> WrenInterpretResult {
    if vm.is_null() || method.is_null() {
        return WrenInterpretResult::RuntimeError;
    }
    let handle = unsafe { &*method };
    let vm_ref = unsafe { &mut *vm };

    // Receiver is in slot 0, args in slots 1..N
    let receiver = vm_ref.get_slot(0);

    // Count args from signature (number of underscores)
    let num_args = handle.signature.chars().filter(|&c| c == '_').count();
    let mut args = Vec::with_capacity(num_args);
    for i in 1..=num_args {
        args.push(vm_ref.get_slot(i));
    }

    // call_method_on handles class receivers with static: prefix fallback
    let result = vm_ref.call_method_on(receiver, &handle.signature, &args);

    match result {
        Some(value) => {
            vm_ref.set_slot(0, value);
            WrenInterpretResult::Success
        }
        None => WrenInterpretResult::RuntimeError,
    }
}

#[no_mangle]
pub extern "C" fn wrenReleaseHandle(vm: *mut WrenVM, handle: *mut WrenHandle) {
    if vm.is_null() || handle.is_null() {
        return;
    }
    unsafe {
        let h = Box::from_raw(handle);
        (*vm).release_handle(h.index);
    }
}

// ---------------------------------------------------------------------------
// Slot API — reading values
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetSlotCount(vm: *mut WrenVM) -> c_int {
    if vm.is_null() {
        return 0;
    }
    unsafe { (*vm).api_stack.len() as c_int }
}

#[no_mangle]
pub extern "C" fn wrenEnsureSlots(vm: *mut WrenVM, num_slots: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        let vm = &mut *vm;
        while vm.api_stack.len() < num_slots as usize {
            vm.api_stack.push(Value::null());
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotType(vm: *mut WrenVM, slot: c_int) -> WrenType {
    if vm.is_null() {
        return WrenType::Unknown;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if val.is_bool() {
        WrenType::Bool
    } else if val.is_num() {
        WrenType::Num
    } else if val.is_null() {
        WrenType::Null
    } else if val.is_object() {
        let ptr = val.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        unsafe {
            match (*header).obj_type {
                ObjType::String => WrenType::String,
                ObjType::List => WrenType::List,
                ObjType::Map => WrenType::Map,
                ObjType::Foreign => WrenType::Foreign,
                _ => WrenType::Unknown,
            }
        }
    } else {
        WrenType::Unknown
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotBool(vm: *mut WrenVM, slot: c_int) -> bool {
    if vm.is_null() {
        return false;
    }
    unsafe { (*vm).get_slot(slot as usize).as_bool().unwrap_or(false) }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotDouble(vm: *mut WrenVM, slot: c_int) -> c_double {
    if vm.is_null() {
        return 0.0;
    }
    unsafe { (*vm).get_slot(slot as usize).as_num().unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotString(vm: *mut WrenVM, slot: c_int) -> *const c_char {
    if vm.is_null() {
        return ptr::null();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return ptr::null();
    }
    let ptr = val.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::String {
            return ptr::null();
        }
        let obj = &*(ptr as *const ObjString);
        obj.as_c_str()
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotBytes(
    vm: *mut WrenVM,
    slot: c_int,
    length: *mut c_int,
) -> *const c_char {
    if vm.is_null() {
        return ptr::null();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return ptr::null();
    }
    let ptr = val.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::String {
            return ptr::null();
        }
        let obj = &*(ptr as *const ObjString);
        if !length.is_null() {
            *length = obj.value.len() as c_int;
        }
        obj.value.as_ptr() as *const c_char
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotForeign(vm: *mut WrenVM, slot: c_int) -> *mut c_void {
    if vm.is_null() {
        return ptr::null_mut();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return ptr::null_mut();
    }
    // Return pointer to the foreign object's data.
    val.as_object().unwrap() as *mut c_void
}

#[no_mangle]
pub extern "C" fn wrenGetSlotHandle(vm: *mut WrenVM, slot: c_int) -> *mut WrenHandle {
    if vm.is_null() {
        return ptr::null_mut();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    let idx = unsafe { (*vm).make_handle(val) };
    Box::into_raw(Box::new(WrenHandle {
        index: idx,
        signature: String::new(),
    }))
}

// ---------------------------------------------------------------------------
// Slot API — writing values
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenSetSlotBool(vm: *mut WrenVM, slot: c_int, value: bool) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).set_slot(slot as usize, Value::bool(value));
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotDouble(vm: *mut WrenVM, slot: c_int, value: c_double) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).set_slot(slot as usize, Value::num(value));
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNull(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).set_slot(slot as usize, Value::null());
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotString(vm: *mut WrenVM, slot: c_int, text: *const c_char) {
    if vm.is_null() || text.is_null() {
        return;
    }
    let s = unsafe { CStr::from_ptr(text) }
        .to_str()
        .unwrap_or("")
        .to_string();
    unsafe {
        let val = (*vm).new_string(s);
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotBytes(
    vm: *mut WrenVM,
    slot: c_int,
    bytes: *const c_char,
    length: usize,
) {
    if vm.is_null() || bytes.is_null() {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes as *const u8, length) };
    let s = String::from_utf8_lossy(slice).into_owned();
    unsafe {
        let val = (*vm).new_string(s);
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotHandle(vm: *mut WrenVM, slot: c_int, handle: *mut WrenHandle) {
    if vm.is_null() || handle.is_null() {
        return;
    }
    unsafe {
        let h = &*handle;
        let vm = &mut *vm;
        if h.index < vm.handles.len() {
            let val = vm.handles[h.index].value;
            vm.set_slot(slot as usize, val);
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNewList(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        let val = (*vm).new_list(vec![]);
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNewMap(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        let val = (*vm).new_map();
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNewForeign(
    vm: *mut WrenVM,
    slot: c_int,
    _class_slot: c_int,
    size: usize,
) -> *mut c_void {
    if vm.is_null() {
        return ptr::null_mut();
    }
    let data = vec![0u8; size];
    unsafe {
        let obj = (*vm).gc.alloc_foreign(data);
        let val = Value::object(obj as *mut u8);
        (*vm).set_slot(slot as usize, val);
        (*obj).data.as_mut_ptr() as *mut c_void
    }
}

// ---------------------------------------------------------------------------
// List operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetListCount(vm: *mut WrenVM, slot: c_int) -> c_int {
    if vm.is_null() {
        return 0;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return 0;
    }
    let ptr = val.as_object().unwrap();
    unsafe {
        let list = &*(ptr as *const ObjList);
        list.len() as c_int
    }
}

#[no_mangle]
pub extern "C" fn wrenGetListElement(
    vm: *mut WrenVM,
    list_slot: c_int,
    index: c_int,
    element_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let list_val = unsafe { (*vm).get_slot(list_slot as usize) };
    if !list_val.is_object() {
        return;
    }
    let ptr = list_val.as_object().unwrap();
    unsafe {
        let list = &*(ptr as *const ObjList);
        let idx = if index < 0 {
            (list.len() as i32 + index) as usize
        } else {
            index as usize
        };
        if let Some(elem) = list.get(idx) {
            (*vm).set_slot(element_slot as usize, elem);
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenSetListElement(
    vm: *mut WrenVM,
    list_slot: c_int,
    index: c_int,
    element_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let list_val = unsafe { (*vm).get_slot(list_slot as usize) };
    let elem = unsafe { (*vm).get_slot(element_slot as usize) };
    if !list_val.is_object() {
        return;
    }
    let ptr = list_val.as_object().unwrap();
    unsafe {
        let list = &mut *(ptr as *mut ObjList);
        let idx = if index < 0 {
            (list.len() as i32 + index) as usize
        } else {
            index as usize
        };
        list.set(idx, elem);
    }
}

#[no_mangle]
pub extern "C" fn wrenInsertInList(
    vm: *mut WrenVM,
    list_slot: c_int,
    index: c_int,
    element_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let list_val = unsafe { (*vm).get_slot(list_slot as usize) };
    let elem = unsafe { (*vm).get_slot(element_slot as usize) };
    if !list_val.is_object() {
        return;
    }
    let ptr = list_val.as_object().unwrap();
    unsafe {
        let list = &mut *(ptr as *mut ObjList);
        let idx = if index < 0 {
            let i = list.len() as i32 + 1 + index;
            if i < 0 {
                0usize
            } else {
                i as usize
            }
        } else {
            index as usize
        };
        let idx = idx.min(list.len());
        list.insert(idx, elem);
    }
}

// ---------------------------------------------------------------------------
// Map operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetMapCount(vm: *mut WrenVM, slot: c_int) -> c_int {
    if vm.is_null() {
        return 0;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return 0;
    }
    let ptr = val.as_object().unwrap();
    unsafe {
        let map = &*(ptr as *const ObjMap);
        map.entries.len() as c_int
    }
}

#[no_mangle]
pub extern "C" fn wrenGetMapContainsKey(vm: *mut WrenVM, map_slot: c_int, key_slot: c_int) -> bool {
    if vm.is_null() {
        return false;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    if !map_val.is_object() {
        return false;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &*(ptr as *const ObjMap);
        map.contains(key)
    }
}

#[no_mangle]
pub extern "C" fn wrenGetMapValue(
    vm: *mut WrenVM,
    map_slot: c_int,
    key_slot: c_int,
    value_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    if !map_val.is_object() {
        return;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &*(ptr as *const ObjMap);
        let val = map.get(key).unwrap_or(Value::null());
        (*vm).set_slot(value_slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetMapValue(
    vm: *mut WrenVM,
    map_slot: c_int,
    key_slot: c_int,
    value_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    let val = unsafe { (*vm).get_slot(value_slot as usize) };
    if !map_val.is_object() {
        return;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &mut *(ptr as *mut ObjMap);
        map.set(key, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenRemoveMapValue(
    vm: *mut WrenVM,
    map_slot: c_int,
    key_slot: c_int,
    removed_value_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    if !map_val.is_object() {
        return;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &mut *(ptr as *mut ObjMap);
        let removed = map.remove(key).unwrap_or(Value::null());
        (*vm).set_slot(removed_value_slot as usize, removed);
    }
}

// ---------------------------------------------------------------------------
// Module / variable lookup
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenHasModule(vm: *mut WrenVM, module: *const c_char) -> bool {
    if vm.is_null() || module.is_null() {
        return false;
    }
    let name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("");
    unsafe { (*vm).engine.modules.contains_key(name) }
}

#[no_mangle]
pub extern "C" fn wrenHasVariable(
    vm: *mut WrenVM,
    module: *const c_char,
    name: *const c_char,
) -> bool {
    if vm.is_null() || module.is_null() || name.is_null() {
        return false;
    }
    let mod_name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("");
    let var_name = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("");
    let vm_ref = unsafe { &*vm };
    if let Some(entry) = vm_ref.engine.modules.get(mod_name) {
        entry.var_names.iter().any(|n| n == var_name)
    } else {
        false
    }
}

#[no_mangle]
pub extern "C" fn wrenGetVariable(
    vm: *mut WrenVM,
    module: *const c_char,
    name: *const c_char,
    slot: c_int,
) {
    if vm.is_null() || module.is_null() || name.is_null() {
        return;
    }
    let mod_name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("");
    let var_name = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("");
    let vm_ref = unsafe { &mut *vm };
    if let Some(entry) = vm_ref.engine.modules.get(mod_name) {
        if let Some(idx) = entry.var_names.iter().position(|n| n == var_name) {
            let value = entry.vars[idx];
            vm_ref.set_slot(slot as usize, value);
        }
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenAbortFiber(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    let msg = if val.is_object() {
        let ptr = val.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        unsafe {
            if (*header).obj_type == ObjType::String {
                let obj = &*(ptr as *const ObjString);
                obj.value.clone()
            } else {
                "Runtime error.".to_string()
            }
        }
    } else {
        "Runtime error.".to_string()
    };
    unsafe {
        (*vm).runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// User data
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetUserData(vm: *mut WrenVM) -> *mut c_void {
    if vm.is_null() {
        return ptr::null_mut();
    }
    unsafe { (*vm).user_data }
}

#[no_mangle]
pub extern "C" fn wrenSetUserData(vm: *mut WrenVM, user_data: *mut c_void) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).user_data = user_data;
    }
}

// ---------------------------------------------------------------------------
// Hatch package loading
//
// Mirrors the Rust-side `hatch_runner::HatchRunner` API, but keeps
// every function as a plain `extern "C"` so C / C++ / Swift / FFI
// consumers can call them directly. Return values are 0 on success,
// non-zero on error — matching the Unix-style idiom and letting C
// callers chain with `if (wrenInstallHatchFile(...)) return 1;`.
// ---------------------------------------------------------------------------

/// Install a pre-loaded `.hatch` byte buffer into the VM. Returns
/// 0 on success, 1 on compile error, 2 on runtime error, -1 on
/// null-pointer inputs. Useful for embedders that ship packages
/// baked into the binary via `include_bytes!` / a resource file.
#[no_mangle]
pub extern "C" fn wrenInstallHatchBytes(
    vm: *mut WrenVM,
    bytes: *const u8,
    len: usize,
) -> c_int {
    if vm.is_null() || bytes.is_null() {
        return -1;
    }
    let vm_ref = unsafe { &mut *vm };
    let slice = unsafe { std::slice::from_raw_parts(bytes, len) };
    use crate::runtime::engine::InterpretResult;
    match vm_ref.install_hatch_modules(slice) {
        InterpretResult::Success => 0,
        InterpretResult::CompileError => 1,
        InterpretResult::RuntimeError => 2,
    }
}

/// Install a `.hatch` file from disk. Returns 0 on success, 1 on
/// compile error, 2 on runtime error, 3 if the file can't be read,
/// -1 on null-pointer inputs.
#[no_mangle]
pub extern "C" fn wrenInstallHatchFile(
    vm: *mut WrenVM,
    path: *const c_char,
) -> c_int {
    if vm.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(_) => return 3,
    };
    wrenInstallHatchBytes(vm, bytes.as_ptr(), bytes.len())
}

/// Append a directory to the native-library search path. The VM
/// consults this when resolving `#!native` dylib references in
/// imported classes. Per-hatch `native_search_paths` are already
/// picked up automatically; this is for embedder overrides.
/// Returns 0 on success, -1 on null-pointer inputs.
#[no_mangle]
pub extern "C" fn wrenAddNativeSearchPath(
    vm: *mut WrenVM,
    path: *const c_char,
) -> c_int {
    if vm.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let vm_ref = unsafe { &mut *vm };
    let pb = std::path::PathBuf::from(path_str);
    if !vm_ref.native_search_paths.contains(&pb) {
        vm_ref.native_search_paths.push(pb);
    }
    0
}

/// Scan a directory for `.hatch` files and install every one of
/// them. The name embedded in each hatchfile is what later code
/// imports as. Returns the number of artifacts installed on
/// success, or a negative error code (-1 null / -2 read-dir /
/// -3 install failed partway through).
#[no_mangle]
pub extern "C" fn wrenInstallHatchDir(
    vm: *mut WrenVM,
    path: *const c_char,
) -> c_int {
    if vm.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let entries = match std::fs::read_dir(path_str) {
        Ok(e) => e,
        Err(_) => return -2,
    };
    let mut installed = 0;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some("hatch") {
            continue;
        }
        let bytes = match std::fs::read(&p) {
            Ok(b) => b,
            Err(_) => return -3,
        };
        let rc = wrenInstallHatchBytes(vm, bytes.as_ptr(), bytes.len());
        if rc != 0 {
            return -3;
        }
        installed += 1;
    }
    installed as c_int
}

// ---------------------------------------------------------------------------
// Internal: convert C config to Rust config
// ---------------------------------------------------------------------------

unsafe fn build_vm_config(c_config: &WrenConfiguration) -> VMConfig {
    let mut config = VMConfig::default();

    if let Some(write_fn) = c_config.write_fn {
        // Store the C function pointer and wrap it.
        // We need a stable VM pointer for the callback — this is set post-creation.
        config.write_fn = Some(Box::new(move |text: &str| {
            if let Ok(cstr) = CString::new(text) {
                write_fn(ptr::null_mut(), cstr.as_ptr());
            }
        }));
    }

    if c_config.initial_heap_size > 0 {
        config.initial_heap_size = c_config.initial_heap_size;
    }
    if c_config.min_heap_size > 0 {
        config.min_heap_size = c_config.min_heap_size;
    }
    if c_config.heap_growth_percent > 0 {
        config.heap_growth_percent = c_config.heap_growth_percent as u32;
    }

    config
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = wrenGetVersionNumber();
        assert!(v > 0);
        assert_eq!(v, 5 * 1_000);
    }

    #[test]
    fn test_vm_lifecycle() {
        let mut config = std::mem::MaybeUninit::<WrenConfiguration>::uninit();
        wrenInitConfiguration(config.as_mut_ptr());
        let config = unsafe { config.assume_init() };

        let vm = wrenNewVM(&config as *const WrenConfiguration);
        assert!(!vm.is_null());

        wrenCollectGarbage(vm);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_bool() {
        let vm = wrenNewVM(ptr::null());
        assert!(!vm.is_null());

        wrenEnsureSlots(vm, 3);
        assert!(wrenGetSlotCount(vm) >= 3);

        wrenSetSlotBool(vm, 0, true);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Bool);
        assert!(wrenGetSlotBool(vm, 0));

        wrenSetSlotBool(vm, 0, false);
        assert!(!wrenGetSlotBool(vm, 0));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_double() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        wrenSetSlotDouble(vm, 0, 1.234);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Num);
        assert!((wrenGetSlotDouble(vm, 0) - 1.234).abs() < 1e-10);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_null() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        wrenSetSlotNull(vm, 0);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Null);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_string() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        let text = CString::new("hello").unwrap();
        wrenSetSlotString(vm, 0, text.as_ptr());
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::String);

        let got = wrenGetSlotString(vm, 0);
        assert!(!got.is_null());
        let got_str = unsafe { CStr::from_ptr(got) }.to_str().unwrap();
        assert_eq!(got_str, "hello");

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_list() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 3);

        wrenSetSlotNewList(vm, 0);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::List);
        assert_eq!(wrenGetListCount(vm, 0), 0);

        // Insert element
        wrenSetSlotDouble(vm, 1, 42.0);
        wrenInsertInList(vm, 0, -1, 1);
        assert_eq!(wrenGetListCount(vm, 0), 1);

        // Read element
        wrenGetListElement(vm, 0, 0, 2);
        assert_eq!(wrenGetSlotDouble(vm, 2), 42.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_map() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 4);

        wrenSetSlotNewMap(vm, 0);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Map);
        assert_eq!(wrenGetMapCount(vm, 0), 0);

        // Set key-value
        let key = CString::new("x").unwrap();
        wrenSetSlotString(vm, 1, key.as_ptr());
        wrenSetSlotDouble(vm, 2, 99.0);
        wrenSetMapValue(vm, 0, 1, 2);
        assert_eq!(wrenGetMapCount(vm, 0), 1);

        // Check containsKey
        assert!(wrenGetMapContainsKey(vm, 0, 1));

        // Get value
        wrenGetMapValue(vm, 0, 1, 3);
        assert_eq!(wrenGetSlotDouble(vm, 3), 99.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_handle_lifecycle() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        wrenSetSlotDouble(vm, 0, 42.0);
        let handle = wrenGetSlotHandle(vm, 0);
        assert!(!handle.is_null());

        // Use handle to restore value
        wrenSetSlotNull(vm, 0);
        wrenSetSlotHandle(vm, 0, handle);
        assert_eq!(wrenGetSlotDouble(vm, 0), 42.0);

        wrenReleaseHandle(vm, handle);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_user_data() {
        let vm = wrenNewVM(ptr::null());

        assert!(wrenGetUserData(vm).is_null());

        let data: u64 = 0xDEADBEEF;
        wrenSetUserData(vm, &data as *const u64 as *mut c_void);
        let got = wrenGetUserData(vm) as *const u64;
        assert_eq!(unsafe { *got }, 0xDEADBEEF);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_interpret_runs_code() {
        let vm = wrenNewVM(ptr::null());
        assert!(!vm.is_null());
        let vm_ref = unsafe { &mut *vm };
        vm_ref.output_buffer = Some(String::new());

        let module = CString::new("test").unwrap();
        let source = CString::new("System.print(\"hello from capi\")").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        let output = vm_ref.take_output();
        assert_eq!(output, "hello from capi\n");

        wrenFreeVM(vm);
    }

    #[test]
    fn test_interpret_compile_error() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        let source = CString::new("class {").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::CompileError);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_has_variable() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        let source = CString::new("var Foo = 42").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        let var_name = CString::new("Foo").unwrap();
        assert!(wrenHasVariable(vm, module.as_ptr(), var_name.as_ptr()));

        let missing = CString::new("Bar").unwrap();
        assert!(!wrenHasVariable(vm, module.as_ptr(), missing.as_ptr()));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_get_variable() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        let source = CString::new("var MyVal = 99").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        wrenEnsureSlots(vm, 1);
        let var_name = CString::new("MyVal").unwrap();
        wrenGetVariable(vm, module.as_ptr(), var_name.as_ptr(), 0);
        assert_eq!(wrenGetSlotDouble(vm, 0), 99.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_call_method() {
        let vm = wrenNewVM(ptr::null());
        let vm_ref = unsafe { &mut *vm };
        vm_ref.output_buffer = Some(String::new());

        // Define a class with static methods
        let module = CString::new("test").unwrap();
        let source = CString::new(
            "class Greeter {\n  static double(x) { return x * 2 }\n  static greet(name) { return \"hello \" + name }\n}",
        )
        .unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        // Test 1: numeric static method
        wrenEnsureSlots(vm, 2);
        let class_name = CString::new("Greeter").unwrap();
        wrenGetVariable(vm, module.as_ptr(), class_name.as_ptr(), 0);
        wrenSetSlotDouble(vm, 1, 21.0);

        let sig = CString::new("double(_)").unwrap();
        let handle = wrenMakeCallHandle(vm, sig.as_ptr());
        assert!(!handle.is_null());

        let call_result = wrenCall(vm, handle);
        assert_eq!(call_result, WrenInterpretResult::Success);
        assert_eq!(wrenGetSlotDouble(vm, 0), 42.0);
        wrenReleaseHandle(vm, handle);

        // Test 2: string static method — slots must survive across calls
        wrenEnsureSlots(vm, 2);
        wrenGetVariable(vm, module.as_ptr(), class_name.as_ptr(), 0);
        let arg = CString::new("world").unwrap();
        wrenSetSlotString(vm, 1, arg.as_ptr());

        let sig2 = CString::new("greet(_)").unwrap();
        let handle2 = wrenMakeCallHandle(vm, sig2.as_ptr());
        assert!(!handle2.is_null());

        let call_result2 = wrenCall(vm, handle2);
        assert_eq!(call_result2, WrenInterpretResult::Success);

        let result_str = wrenGetSlotString(vm, 0);
        assert!(!result_str.is_null());
        let got = unsafe { CStr::from_ptr(result_str) }.to_str().unwrap();
        assert_eq!(got, "hello world");

        wrenReleaseHandle(vm, handle2);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_has_module() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("mymod").unwrap();
        let source = CString::new("var X = 1").unwrap();

        // Module doesn't exist yet
        assert!(!wrenHasModule(vm, module.as_ptr()));

        wrenInterpret(vm, module.as_ptr(), source.as_ptr());

        // Now it should exist
        assert!(wrenHasModule(vm, module.as_ptr()));

        let bogus = CString::new("nonexistent").unwrap();
        assert!(!wrenHasModule(vm, bogus.as_ptr()));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_list_set_element() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 3);

        wrenSetSlotNewList(vm, 0);
        wrenSetSlotDouble(vm, 1, 10.0);
        wrenInsertInList(vm, 0, -1, 1);
        wrenSetSlotDouble(vm, 1, 20.0);
        wrenInsertInList(vm, 0, -1, 1);
        assert_eq!(wrenGetListCount(vm, 0), 2);

        // Overwrite first element
        wrenSetSlotDouble(vm, 1, 99.0);
        wrenSetListElement(vm, 0, 0, 1);

        wrenGetListElement(vm, 0, 0, 2);
        assert_eq!(wrenGetSlotDouble(vm, 2), 99.0);

        // Second element unchanged
        wrenGetListElement(vm, 0, 1, 2);
        assert_eq!(wrenGetSlotDouble(vm, 2), 20.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_map_remove() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 4);

        wrenSetSlotNewMap(vm, 0);

        let key = CString::new("k").unwrap();
        wrenSetSlotString(vm, 1, key.as_ptr());
        wrenSetSlotDouble(vm, 2, 7.0);
        wrenSetMapValue(vm, 0, 1, 2);
        assert_eq!(wrenGetMapCount(vm, 0), 1);

        // Remove by key
        wrenRemoveMapValue(vm, 0, 1, 3);
        assert_eq!(wrenGetMapCount(vm, 0), 0);
        assert_eq!(wrenGetSlotDouble(vm, 3), 7.0); // removed value returned

        wrenFreeVM(vm);
    }

    #[test]
    fn test_interpret_runtime_error() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        // Fiber.abort triggers a runtime error
        let source = CString::new("Fiber.abort(\"boom\")").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::RuntimeError);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_multiple_modules() {
        let vm = wrenNewVM(ptr::null());
        let vm_ref = unsafe { &mut *vm };
        vm_ref.output_buffer = Some(String::new());

        // First module
        let mod_a = CString::new("mod_a").unwrap();
        let src_a = CString::new("var A = 100").unwrap();
        assert_eq!(
            wrenInterpret(vm, mod_a.as_ptr(), src_a.as_ptr()),
            WrenInterpretResult::Success
        );

        // Second module
        let mod_b = CString::new("mod_b").unwrap();
        let src_b = CString::new("var B = 200").unwrap();
        assert_eq!(
            wrenInterpret(vm, mod_b.as_ptr(), src_b.as_ptr()),
            WrenInterpretResult::Success
        );

        // Both modules exist
        assert!(wrenHasModule(vm, mod_a.as_ptr()));
        assert!(wrenHasModule(vm, mod_b.as_ptr()));

        // Variables are isolated per module
        wrenEnsureSlots(vm, 1);
        let var_a = CString::new("A").unwrap();
        let var_b = CString::new("B").unwrap();
        wrenGetVariable(vm, mod_a.as_ptr(), var_a.as_ptr(), 0);
        assert_eq!(wrenGetSlotDouble(vm, 0), 100.0);
        wrenGetVariable(vm, mod_b.as_ptr(), var_b.as_ptr(), 0);
        assert_eq!(wrenGetSlotDouble(vm, 0), 200.0);

        // Cross-module: mod_a doesn't have B
        assert!(!wrenHasVariable(vm, mod_a.as_ptr(), var_b.as_ptr()));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_null_safety() {
        // All functions should handle null pointers gracefully
        assert_eq!(
            wrenInterpret(ptr::null_mut(), ptr::null(), ptr::null()),
            WrenInterpretResult::RuntimeError
        );
        assert!(wrenMakeCallHandle(ptr::null_mut(), ptr::null()).is_null());
        assert_eq!(
            wrenCall(ptr::null_mut(), ptr::null_mut()),
            WrenInterpretResult::RuntimeError
        );
        wrenFreeVM(ptr::null_mut()); // should not crash
        wrenCollectGarbage(ptr::null_mut()); // should not crash
        assert!(!wrenHasModule(ptr::null_mut(), ptr::null()));
        assert!(!wrenHasVariable(ptr::null_mut(), ptr::null(), ptr::null()));
        // Hatch-loader surface: every function returns -1 on null
        // rather than dereferencing, so the bindings are safe to
        // call from a freshly-zeroed C config.
        assert_eq!(wrenInstallHatchBytes(ptr::null_mut(), ptr::null(), 0), -1);
        assert_eq!(wrenInstallHatchFile(ptr::null_mut(), ptr::null()), -1);
        assert_eq!(wrenAddNativeSearchPath(ptr::null_mut(), ptr::null()), -1);
        assert_eq!(wrenInstallHatchDir(ptr::null_mut(), ptr::null()), -1);
    }
}
