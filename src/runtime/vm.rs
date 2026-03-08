/// The Wren virtual machine.
///
/// Owns all runtime state: GC heap, interner, core classes, fibers,
/// module registry, and configuration callbacks.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use crate::intern::{Interner, SymbolId};
use super::gc::Gc;
use super::object::*;
use super::value::Value;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Callback for System.print output.
pub type WriteFn = Box<dyn Fn(&str)>;

/// Callback for error reporting.
pub type ErrorFn = Box<dyn Fn(ErrorKind, &str, i32, &str)>;

/// Callback to resolve a module name relative to an importer.
pub type ResolveModuleFn = Box<dyn Fn(&str, &str) -> Option<String>>;

/// Callback to load a module's source code.
pub type LoadModuleFn = Box<dyn Fn(&str) -> Option<String>>;

/// Callback to bind a foreign method.
pub type BindForeignMethodFn =
    Box<dyn Fn(&str, &str, bool, &str) -> Option<NativeFn>>;

/// Callback to bind a foreign class (allocate + optional finalize).
pub type BindForeignClassFn =
    Box<dyn Fn(&str, &str) -> Option<ForeignClassMethods>>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrorKind {
    Compile,
    Runtime,
    StackTrace,
}

/// Foreign class method pair.
pub struct ForeignClassMethods {
    pub allocate: NativeFn,
    pub finalize: Option<fn(*mut u8)>,
}

/// VM configuration. Set before creating the VM.
pub struct VMConfig {
    pub write_fn: Option<WriteFn>,
    pub error_fn: Option<ErrorFn>,
    pub resolve_module_fn: Option<ResolveModuleFn>,
    pub load_module_fn: Option<LoadModuleFn>,
    pub bind_foreign_method_fn: Option<BindForeignMethodFn>,
    pub bind_foreign_class_fn: Option<BindForeignClassFn>,
    pub initial_heap_size: usize,
    pub min_heap_size: usize,
    pub heap_growth_percent: u32,
}

impl Default for VMConfig {
    fn default() -> Self {
        Self {
            write_fn: None,
            error_fn: None,
            resolve_module_fn: None,
            load_module_fn: None,
            bind_foreign_method_fn: None,
            bind_foreign_class_fn: None,
            initial_heap_size: 10 * 1024 * 1024,
            min_heap_size: 1024 * 1024,
            heap_growth_percent: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// Handle (persistent reference, prevents GC collection)
// ---------------------------------------------------------------------------

/// A persistent handle to a Wren value, preventing GC collection.
pub struct WrenHandle {
    pub value: Value,
}

// ---------------------------------------------------------------------------
// VM
// ---------------------------------------------------------------------------

pub struct VM {
    // -- Memory --
    pub gc: Gc,
    pub interner: Interner,

    // -- Core classes --
    pub object_class: *mut ObjClass,
    pub class_class: *mut ObjClass,
    pub bool_class: *mut ObjClass,
    pub num_class: *mut ObjClass,
    pub string_class: *mut ObjClass,
    pub list_class: *mut ObjClass,
    pub map_class: *mut ObjClass,
    pub range_class: *mut ObjClass,
    pub null_class: *mut ObjClass,
    pub fn_class: *mut ObjClass,
    pub fiber_class: *mut ObjClass,
    pub system_class: *mut ObjClass,
    pub sequence_class: *mut ObjClass,

    // -- Execution state --
    pub fiber: *mut ObjFiber,
    pub modules: HashMap<String, *mut ObjModule>,

    // -- API --
    pub api_stack: Vec<Value>,
    pub handles: Vec<WrenHandle>,
    pub user_data: *mut c_void,

    // -- Configuration --
    pub config: VMConfig,

    // -- Error state (set by primitives) --
    pub has_error: bool,
}

impl VM {
    /// Create a new VM with the given configuration.
    pub fn new(config: VMConfig) -> Self {
        let mut vm = Self {
            gc: Gc::new(),
            interner: Interner::new(),

            object_class: ptr::null_mut(),
            class_class: ptr::null_mut(),
            bool_class: ptr::null_mut(),
            num_class: ptr::null_mut(),
            string_class: ptr::null_mut(),
            list_class: ptr::null_mut(),
            map_class: ptr::null_mut(),
            range_class: ptr::null_mut(),
            null_class: ptr::null_mut(),
            fn_class: ptr::null_mut(),
            fiber_class: ptr::null_mut(),
            system_class: ptr::null_mut(),
            sequence_class: ptr::null_mut(),

            fiber: ptr::null_mut(),
            modules: HashMap::new(),

            api_stack: vec![Value::null(); 16],
            handles: Vec::new(),
            user_data: ptr::null_mut(),

            config,
            has_error: false,
        };

        // Bootstrap core classes.
        super::core::initialize(&mut vm);

        vm
    }

    /// Create a new VM with default configuration.
    pub fn new_default() -> Self {
        Self::new(VMConfig::default())
    }

    // -- Helpers for core library primitives --

    /// Allocate a GC-managed string and return it as a Value.
    pub fn new_string(&mut self, s: String) -> Value {
        let obj = self.gc.alloc_string(s);
        unsafe { (*obj).header.class = self.string_class; }
        Value::object(obj as *mut u8)
    }

    /// Allocate a GC-managed list and return it as a Value.
    pub fn new_list(&mut self, elements: Vec<Value>) -> Value {
        let obj = self.gc.alloc_list();
        unsafe {
            (*obj).header.class = self.list_class;
            (*obj).elements = elements;
        }
        Value::object(obj as *mut u8)
    }

    /// Allocate a GC-managed range and return it as a Value.
    pub fn new_range(&mut self, from: f64, to: f64, inclusive: bool) -> Value {
        let obj = self.gc.alloc_range(from, to, inclusive);
        unsafe { (*obj).header.class = self.range_class; }
        Value::object(obj as *mut u8)
    }

    /// Allocate a GC-managed map and return it as a Value.
    pub fn new_map(&mut self) -> Value {
        let obj = self.gc.alloc_map();
        unsafe { (*obj).header.class = self.map_class; }
        Value::object(obj as *mut u8)
    }

    /// Create a core class with the given name and optional superclass.
    pub fn make_class(&mut self, name: &str, superclass: *mut ObjClass) -> *mut ObjClass {
        let sym = self.interner.intern(name);
        let class = self.gc.alloc_class(sym, superclass);
        unsafe {
            (*class).header.class = self.class_class;
        }
        class
    }

    /// Bind a native primitive to a class by method signature string.
    pub fn primitive(&mut self, class: *mut ObjClass, signature: &str, func: NativeFn) {
        let sym = self.interner.intern(signature);
        unsafe {
            (*class).bind_native(sym, func);
        }
    }

    /// Bind a native primitive to the class's metaclass (static method).
    /// For simplicity, we store static methods directly on the class.
    pub fn primitive_static(&mut self, class: *mut ObjClass, signature: &str, func: NativeFn) {
        // In Wren, static methods are on the metaclass. We store them
        // on the class itself with a "static:" prefix for now.
        let sig = format!("static:{}", signature);
        let sym = self.interner.intern(&sig);
        unsafe {
            (*class).bind_native(sym, func);
        }
    }

    /// Look up a static method on a class.
    pub fn find_static(&self, class: *mut ObjClass, signature: &str) -> Option<&Method> {
        let sig = format!("static:{}", signature);
        let sym = self.interner.lookup(&sig)?;
        unsafe { (*class).find_method(sym) }
    }

    /// Get the class for a value (using the built-in class hierarchy).
    pub fn class_of(&self, value: Value) -> *mut ObjClass {
        if value.is_num() {
            self.num_class
        } else if value.is_bool() {
            self.bool_class
        } else if value.is_null() {
            self.null_class
        } else if value.is_object() {
            let ptr = value.as_object().unwrap();
            let header = ptr as *const ObjHeader;
            unsafe {
                let cls = (*header).class;
                if cls.is_null() {
                    self.object_class
                } else {
                    cls
                }
            }
        } else {
            self.object_class
        }
    }

    /// Get the class name for a value.
    pub fn class_name_of(&self, value: Value) -> String {
        let class = self.class_of(value);
        if class.is_null() {
            return "Object".to_string();
        }
        unsafe {
            self.interner.resolve((*class).name).to_string()
        }
    }

    /// Write text via the configured write callback (or stdout).
    pub fn write(&self, text: &str) {
        if let Some(ref f) = self.config.write_fn {
            f(text);
        } else {
            print!("{}", text);
        }
    }

    /// Report a runtime error.
    pub fn report_error(&self, msg: &str) {
        if let Some(ref f) = self.config.error_fn {
            f(ErrorKind::Runtime, "", 0, msg);
        } else {
            eprintln!("Runtime error: {}", msg);
        }
    }

    /// Create a persistent handle that prevents a value from being GC'd.
    pub fn make_handle(&mut self, value: Value) -> usize {
        let idx = self.handles.len();
        self.handles.push(WrenHandle { value });
        idx
    }

    /// Release a handle.
    pub fn release_handle(&mut self, idx: usize) {
        if idx < self.handles.len() {
            self.handles[idx].value = Value::null();
        }
    }

    /// Run the garbage collector.
    pub fn collect_garbage(&mut self) {
        let mut roots: Vec<Value> = Vec::new();

        // API stack roots
        roots.extend_from_slice(&self.api_stack);

        // Handle roots
        for h in &self.handles {
            roots.push(h.value);
        }

        self.gc.collect(&mut roots);
    }
}

// ---------------------------------------------------------------------------
// NativeContext implementation for VM
// ---------------------------------------------------------------------------

impl NativeContext for VM {
    fn get_slot(&self, index: usize) -> Value {
        if index < self.api_stack.len() {
            self.api_stack[index]
        } else {
            Value::null()
        }
    }

    fn set_slot(&mut self, index: usize, value: Value) {
        while self.api_stack.len() <= index {
            self.api_stack.push(Value::null());
        }
        self.api_stack[index] = value;
    }

    fn slot_count(&self) -> usize {
        self.api_stack.len()
    }

    fn alloc_string(&mut self, s: String) -> Value {
        self.new_string(s)
    }

    fn alloc_list(&mut self, elements: Vec<Value>) -> Value {
        self.new_list(elements)
    }

    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> Value {
        self.new_range(from, to, inclusive)
    }

    fn alloc_map(&mut self) -> Value {
        self.new_map()
    }

    fn runtime_error(&mut self, msg: String) {
        self.has_error = true;
        self.report_error(&msg);
    }

    fn has_error(&self) -> bool {
        self.has_error
    }

    fn get_class_of(&self, value: Value) -> *mut ObjClass {
        self.class_of(value)
    }

    fn get_class_name_of(&self, value: Value) -> String {
        self.class_name_of(value)
    }

    fn intern(&mut self, s: &str) -> SymbolId {
        self.interner.intern(s)
    }

    fn resolve_symbol(&self, id: SymbolId) -> &str {
        self.interner.resolve(id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn call_primitive(vm: &mut VM, class: *mut ObjClass, sig: &str, args: &[Value]) -> Value {
        let sym = vm.interner.intern(sig);
        let method = unsafe { (*class).find_method(sym).cloned() };
        match method {
            Some(Method::Native(func)) => func(vm, args),
            _ => panic!("Method '{}' not found", sig),
        }
    }

    fn call_static(vm: &mut VM, class: *mut ObjClass, sig: &str, args: &[Value]) -> Value {
        let full = format!("static:{}", sig);
        let sym = vm.interner.intern(&full);
        let method = unsafe { (*class).find_method(sym).cloned() };
        match method {
            Some(Method::Native(func)) => func(vm, args),
            _ => panic!("Static method '{}' not found", sig),
        }
    }

    // -- VM creation --

    #[test]
    fn test_vm_creation() {
        let vm = VM::new_default();
        assert!(!vm.object_class.is_null());
        assert!(!vm.class_class.is_null());
        assert!(!vm.num_class.is_null());
        assert!(!vm.string_class.is_null());
        assert!(!vm.bool_class.is_null());
        assert!(!vm.null_class.is_null());
        assert!(!vm.list_class.is_null());
        assert!(!vm.map_class.is_null());
        assert!(!vm.range_class.is_null());
        assert!(!vm.fn_class.is_null());
        assert!(!vm.fiber_class.is_null());
        assert!(!vm.system_class.is_null());
    }

    #[test]
    fn test_class_hierarchy() {
        let vm = VM::new_default();
        // All classes should have Class as their class.
        unsafe {
            assert_eq!((*vm.object_class).header.class, vm.class_class);
            assert_eq!((*vm.num_class).header.class, vm.class_class);
            assert_eq!((*vm.string_class).header.class, vm.class_class);
            // Num, String, etc. should inherit from Object.
            assert_eq!((*vm.num_class).superclass, vm.object_class);
            assert_eq!((*vm.string_class).superclass, vm.object_class);
        }
    }

    // -- Num primitives --

    #[test]
    fn test_num_arithmetic() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;
        let a = Value::num(10.0);
        let b = Value::num(3.0);

        let sum = call_primitive(&mut vm, cls, "+(_)", &[a, b]);
        assert_eq!(sum.as_num().unwrap(), 13.0);

        let diff = call_primitive(&mut vm, cls, "-(_)", &[a, b]);
        assert_eq!(diff.as_num().unwrap(), 7.0);

        let prod = call_primitive(&mut vm, cls, "*(_)", &[a, b]);
        assert_eq!(prod.as_num().unwrap(), 30.0);

        let quot = call_primitive(&mut vm, cls, "/(_)", &[a, b]);
        assert!((quot.as_num().unwrap() - 10.0 / 3.0).abs() < 1e-10);

        let rem = call_primitive(&mut vm, cls, "%(_)", &[a, b]);
        assert_eq!(rem.as_num().unwrap(), 1.0);
    }

    #[test]
    fn test_num_comparison() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let t = call_primitive(&mut vm, cls, "<(_)", &[Value::num(1.0), Value::num(2.0)]);
        assert_eq!(t.as_bool().unwrap(), true);

        let f = call_primitive(&mut vm, cls, ">(_)", &[Value::num(1.0), Value::num(2.0)]);
        assert_eq!(f.as_bool().unwrap(), false);

        let eq = call_primitive(&mut vm, cls, "==(_)", &[Value::num(5.0), Value::num(5.0)]);
        assert_eq!(eq.as_bool().unwrap(), true);

        let neq = call_primitive(&mut vm, cls, "!=(_)", &[Value::num(5.0), Value::num(3.0)]);
        assert_eq!(neq.as_bool().unwrap(), true);
    }

    #[test]
    fn test_num_math() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let abs = call_primitive(&mut vm, cls, "abs", &[Value::num(-5.0)]);
        assert_eq!(abs.as_num().unwrap(), 5.0);

        let ceil = call_primitive(&mut vm, cls, "ceil", &[Value::num(2.3)]);
        assert_eq!(ceil.as_num().unwrap(), 3.0);

        let floor = call_primitive(&mut vm, cls, "floor", &[Value::num(2.7)]);
        assert_eq!(floor.as_num().unwrap(), 2.0);

        let sqrt = call_primitive(&mut vm, cls, "sqrt", &[Value::num(9.0)]);
        assert_eq!(sqrt.as_num().unwrap(), 3.0);
    }

    #[test]
    fn test_num_static() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let pi = call_static(&mut vm, cls, "pi", &[Value::null()]);
        assert!((pi.as_num().unwrap() - std::f64::consts::PI).abs() < 1e-10);

        let inf = call_static(&mut vm, cls, "infinity", &[Value::null()]);
        assert!(inf.as_num().unwrap().is_infinite());
    }

    #[test]
    fn test_num_to_string() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let s = call_primitive(&mut vm, cls, "toString", &[Value::num(42.0)]);
        assert!(s.is_object());
        let text = super::super::core::as_string(s);
        assert_eq!(text, "42");
    }

    #[test]
    fn test_num_range() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let range = call_primitive(&mut vm, cls, "..(_)", &[Value::num(1.0), Value::num(5.0)]);
        assert!(range.is_object());
    }

    // -- Bool/Null primitives --

    #[test]
    fn test_bool_not() {
        let mut vm = VM::new_default();
        let cls = vm.bool_class;

        let r = call_primitive(&mut vm, cls, "!", &[Value::bool(true)]);
        assert_eq!(r.as_bool().unwrap(), false);

        let r2 = call_primitive(&mut vm, cls, "!", &[Value::bool(false)]);
        assert_eq!(r2.as_bool().unwrap(), true);
    }

    #[test]
    fn test_null_not() {
        let mut vm = VM::new_default();
        let cls = vm.null_class;

        let r = call_primitive(&mut vm, cls, "!", &[Value::null()]);
        assert_eq!(r.as_bool().unwrap(), true);
    }

    #[test]
    fn test_bool_to_string() {
        let mut vm = VM::new_default();
        let cls = vm.bool_class;

        let s = call_primitive(&mut vm, cls, "toString", &[Value::bool(true)]);
        assert_eq!(super::super::core::as_string(s), "true");

        let s2 = call_primitive(&mut vm, cls, "toString", &[Value::bool(false)]);
        assert_eq!(super::super::core::as_string(s2), "false");
    }

    // -- String primitives --

    #[test]
    fn test_string_contains() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let hello = vm.new_string("hello world".to_string());
        let sub = vm.new_string("world".to_string());
        let r = call_primitive(&mut vm, cls, "contains(_)", &[hello, sub]);
        assert_eq!(r.as_bool().unwrap(), true);

        let miss = vm.new_string("xyz".to_string());
        let r2 = call_primitive(&mut vm, cls, "contains(_)", &[hello, miss]);
        assert_eq!(r2.as_bool().unwrap(), false);
    }

    #[test]
    fn test_string_plus() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let a = vm.new_string("hello ".to_string());
        let b = vm.new_string("world".to_string());
        let r = call_primitive(&mut vm, cls, "+(_)", &[a, b]);
        assert_eq!(super::super::core::as_string(r), "hello world");
    }

    #[test]
    fn test_string_byte_count() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let s = vm.new_string("hello".to_string());
        let r = call_primitive(&mut vm, cls, "byteCount_", &[s]);
        assert_eq!(r.as_num().unwrap(), 5.0);
    }

    // -- List primitives --

    #[test]
    fn test_list_add_and_count() {
        let mut vm = VM::new_default();
        let cls = vm.list_class;

        let list = vm.new_list(vec![]);
        call_primitive(&mut vm, cls, "add(_)", &[list, Value::num(42.0)]);
        call_primitive(&mut vm, cls, "add(_)", &[list, Value::num(99.0)]);

        let count = call_primitive(&mut vm, cls, "count", &[list]);
        assert_eq!(count.as_num().unwrap(), 2.0);

        let elem = call_primitive(&mut vm, cls, "[_]", &[list, Value::num(0.0)]);
        assert_eq!(elem.as_num().unwrap(), 42.0);
    }

    #[test]
    fn test_list_remove_at() {
        let mut vm = VM::new_default();
        let cls = vm.list_class;

        let list = vm.new_list(vec![Value::num(10.0), Value::num(20.0), Value::num(30.0)]);
        let removed = call_primitive(&mut vm, cls, "removeAt(_)", &[list, Value::num(1.0)]);
        assert_eq!(removed.as_num().unwrap(), 20.0);

        let count = call_primitive(&mut vm, cls, "count", &[list]);
        assert_eq!(count.as_num().unwrap(), 2.0);
    }

    // -- Map primitives --

    #[test]
    fn test_map_set_and_get() {
        let mut vm = VM::new_default();
        let cls = vm.map_class;

        let map = vm.new_map();
        let key = vm.new_string("name".to_string());
        let val = vm.new_string("Wren".to_string());

        call_primitive(&mut vm, cls, "[_]=(_)", &[map, key, val]);
        let got = call_primitive(&mut vm, cls, "[_]", &[map, key]);
        assert_eq!(super::super::core::as_string(got), "Wren");

        let count = call_primitive(&mut vm, cls, "count", &[map]);
        assert_eq!(count.as_num().unwrap(), 1.0);
    }

    #[test]
    fn test_map_contains_key() {
        let mut vm = VM::new_default();
        let cls = vm.map_class;

        let map = vm.new_map();
        let key = vm.new_string("x".to_string());
        call_primitive(&mut vm, cls, "[_]=(_)", &[map, key, Value::num(1.0)]);

        let has = call_primitive(&mut vm, cls, "containsKey(_)", &[map, key]);
        assert_eq!(has.as_bool().unwrap(), true);

        let miss = vm.new_string("y".to_string());
        let no = call_primitive(&mut vm, cls, "containsKey(_)", &[map, miss]);
        assert_eq!(no.as_bool().unwrap(), false);
    }

    // -- Range primitives --

    #[test]
    fn test_range_properties() {
        let mut vm = VM::new_default();
        let cls = vm.range_class;

        let range = vm.new_range(1.0, 5.0, true);
        let from = call_primitive(&mut vm, cls, "from", &[range]);
        assert_eq!(from.as_num().unwrap(), 1.0);

        let to = call_primitive(&mut vm, cls, "to", &[range]);
        assert_eq!(to.as_num().unwrap(), 5.0);

        let incl = call_primitive(&mut vm, cls, "isInclusive", &[range]);
        assert_eq!(incl.as_bool().unwrap(), true);
    }

    // -- Object primitives --

    #[test]
    fn test_object_equality() {
        let mut vm = VM::new_default();
        let cls = vm.object_class;

        let a = Value::num(5.0);
        let b = Value::num(5.0);
        let eq = call_primitive(&mut vm, cls, "==(_)", &[a, b]);
        assert_eq!(eq.as_bool().unwrap(), true);

        let neq = call_primitive(&mut vm, cls, "!=(_)", &[a, Value::num(3.0)]);
        assert_eq!(neq.as_bool().unwrap(), true);
    }

    #[test]
    fn test_object_not() {
        let mut vm = VM::new_default();
        let cls = vm.object_class;

        // Object.! always returns false (all objects are truthy)
        let r = call_primitive(&mut vm, cls, "!", &[Value::num(42.0)]);
        assert_eq!(r.as_bool().unwrap(), false);
    }

    // -- Class primitives --

    #[test]
    fn test_class_name() {
        let mut vm = VM::new_default();
        let cls = vm.class_class;

        let num_cls_val = Value::object(vm.num_class as *mut u8);
        let name = call_primitive(&mut vm, cls, "name", &[num_cls_val]);
        assert_eq!(super::super::core::as_string(name), "Num");
    }

    // -- System primitives --

    #[test]
    fn test_system_clock() {
        let mut vm = VM::new_default();
        let cls = vm.system_class;

        let t = call_static(&mut vm, cls, "clock", &[Value::null()]);
        assert!(t.as_num().unwrap() > 0.0);
    }

    // -- class_of --

    #[test]
    fn test_class_of() {
        let vm = VM::new_default();
        assert_eq!(vm.class_of(Value::num(1.0)), vm.num_class);
        assert_eq!(vm.class_of(Value::bool(true)), vm.bool_class);
        assert_eq!(vm.class_of(Value::null()), vm.null_class);
    }
}
