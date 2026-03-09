/// The Wren virtual machine.
///
/// Owns all runtime state: GC heap, interner, core classes, fibers,
/// module registry, and configuration callbacks.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use crate::intern::{Interner, SymbolId};
use super::engine::{ExecutionEngine, ExecutionMode, InterpretResult};
use super::gc::Gc;
use super::object::*;
use super::value::Value;

/// Action requested by a fiber native method, handled by the interpreter loop.
#[derive(Debug)]
pub enum FiberAction {
    /// Switch to target fiber, pass value when it yields/completes.
    Call { target: *mut ObjFiber, value: Value },
    /// Yield from current fiber, return value to caller.
    Yield { value: Value },
    /// Transfer to target fiber (no caller chain).
    Transfer { target: *mut ObjFiber, value: Value },
}

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
    /// Execution mode: Interpreter, Tiered (default), or Jit.
    pub execution_mode: ExecutionMode,
    /// Call count threshold before JIT compilation in Tiered mode.
    pub jit_threshold: u32,
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
            execution_mode: ExecutionMode::default(),
            jit_threshold: 100,
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

    // -- Execution engine (tiered runtime) --
    pub engine: ExecutionEngine,

    // -- API --
    pub api_stack: Vec<Value>,
    pub handles: Vec<WrenHandle>,
    pub user_data: *mut c_void,

    // -- Configuration --
    pub config: VMConfig,

    // -- Error state (set by primitives) --
    pub has_error: bool,

    // -- Output capture (for testing; None = print to stdout) --
    pub output_buffer: Option<String>,

    // -- Fiber switching (set by fiber natives, consumed by interpreter) --
    pub pending_fiber_action: Option<FiberAction>,
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

            engine: ExecutionEngine::new(config.execution_mode),

            api_stack: vec![Value::null(); 16],
            handles: Vec::new(),
            user_data: ptr::null_mut(),

            config,
            has_error: false,
            output_buffer: None,
            pending_fiber_action: None,
        };

        // Bootstrap core classes.
        super::core::initialize(&mut vm);

        vm
    }

    /// Create a new VM with default configuration.
    pub fn new_default() -> Self {
        Self::new(VMConfig::default())
    }

    /// Compile and execute Wren source code.
    ///
    /// Pipeline: lex → parse → sema → lower to MIR → optimize → execute.
    /// Execution happens inside a Fiber context via run_fiber().
    pub fn interpret(&mut self, module_name: &str, source: &str) -> InterpretResult {
        use std::collections::HashMap as HMap;
        use crate::diagnostics::Severity;
        use crate::mir::BlockId;
        use crate::mir::opt::{
            self, constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize,
            licm::Licm, sra::Sra, MirPass,
        };
        use crate::parse::parser;
        use crate::sema;

        // 1. Parse
        let parse_result = parser::parse(source);
        if parse_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &parse_result.errors {
                err.eprint(source);
            }
            return InterpretResult::CompileError;
        }

        // 2. Use the parse interner, merge afterward
        let mut interner = parse_result.interner;

        // 3. Semantic analysis — register core class names as prelude
        let core_names = [
            "Object", "Class", "Bool", "Num", "String", "List", "Map",
            "Range", "Null", "Fn", "Fiber", "System", "Sequence",
        ];
        let prelude: Vec<crate::intern::SymbolId> =
            core_names.iter().map(|n| interner.intern(n)).collect();
        let resolve_result =
            sema::resolve::resolve_with_prelude(&parse_result.module, &interner, &prelude);
        if resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &resolve_result.errors {
                err.eprint(source);
            }
            return InterpretResult::CompileError;
        }

        // 3b. Process imports — recursively interpret imported modules
        for stmt in &parse_result.module {
            if let crate::ast::Stmt::Import { module: mod_path, names: _ } = &stmt.0 {
                let imported_module = mod_path.0.clone();

                // Skip if already loaded
                if !self.engine.modules.contains_key(&imported_module) {
                    // Resolve module source via config callback
                    let source_opt = self.config.load_module_fn.as_ref().and_then(|load_fn| {
                        load_fn(&imported_module)
                    });
                    if let Some(mod_source) = source_opt {
                        let result = self.interpret(&imported_module, &mod_source);
                        if result != InterpretResult::Success {
                            return result;
                        }
                    } else {
                        self.report_error(&format!("Could not load module '{}'", imported_module));
                        return InterpretResult::CompileError;
                    }
                }

                // Copy imported names from the imported module's vars to this module's
                // resolve_result.module_vars. We'll look them up by name after MIR
                // compilation when building the module var storage (step 8).
                // For now, just ensure the imported module exists.
            }
        }

        // 4. Lower to MIR
        let mut module_mir =
            crate::mir::builder::lower_module(&parse_result.module, &mut interner, &resolve_result);

        // 5. Optimize top-level function
        let constfold = ConstFold;
        let dce = Dce;
        let cse = Cse;
        let type_spec = TypeSpecialize::with_math(&interner);
        let licm = Licm;
        let sra = Sra;
        let passes: Vec<&dyn MirPass> = vec![
            &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
        ];
        opt::run_to_fixpoint(&mut module_mir.top_level, &passes, 10);
        // Also optimize method bodies
        for class in &mut module_mir.classes {
            for method in &mut class.methods {
                opt::run_to_fixpoint(&mut method.mir, &passes, 10);
            }
        }
        // Optimize closure bodies
        for closure in &mut module_mir.closures {
            opt::run_to_fixpoint(closure, &passes, 10);
        }

        // 6. Remap symbols: the parse interner and VM interner have different
        // indices for the same strings. Build a mapping and rewrite the MIR.
        let mut sym_map: Vec<crate::intern::SymbolId> = Vec::with_capacity(interner.len());
        for i in 0..interner.len() {
            let old_sym = crate::intern::SymbolId::from_raw(i as u32);
            let s = interner.resolve(old_sym);
            let new_sym = self.interner.intern(s);
            sym_map.push(new_sym);
        }
        module_mir.top_level.remap_symbols(|old| sym_map[old.index() as usize]);
        for class in &mut module_mir.classes {
            class.name = sym_map[class.name.index() as usize];
            if let Some(ref mut sup) = class.superclass {
                *sup = sym_map[sup.index() as usize];
            }
            for method in &mut class.methods {
                method.mir.remap_symbols(|old| sym_map[old.index() as usize]);
            }
        }
        for closure in &mut module_mir.closures {
            closure.remap_symbols(|old| sym_map[old.index() as usize]);
        }

        // 7a. Register closure functions with the engine first, so we can
        // patch MakeClosure fn_ids to use actual engine FuncIds.
        let mut closure_func_ids: Vec<u32> = Vec::new();
        for closure in module_mir.closures.drain(..) {
            let fid = self.engine.register_function(closure);
            closure_func_ids.push(fid.0);
        }

        // Patch MakeClosure instructions in top-level and method bodies
        patch_closure_ids(&mut module_mir.top_level, &closure_func_ids);
        for class in &mut module_mir.classes {
            for method in &mut class.methods {
                patch_closure_ids(&mut method.mir, &closure_func_ids);
            }
        }

        // 7b. Register top-level function with the engine
        let func_id = self.engine.register_function(module_mir.top_level);

        // 8. Create module var storage, pre-populated with core class values
        // and imported module vars.
        let module_key = module_name.to_string();
        let mut module_vars = Vec::with_capacity(resolve_result.module_vars.len());
        for &parse_sym in &resolve_result.module_vars {
            let name = interner.resolve(parse_sym);
            if let Some(value) = self.core_class_value(name) {
                module_vars.push(value);
            } else if let Some(value) = self.find_imported_var(name) {
                module_vars.push(value);
            } else {
                module_vars.push(Value::null());
            }
        }
        self.engine.modules.insert(
            module_key.clone(),
            super::engine::ModuleEntry {
                top_level: func_id,
                vars: module_vars,
            },
        );

        // 8b. Create user-defined classes and bind their methods.
        for class_mir in module_mir.classes {
            // Find module var slot for this class
            let class_name_str = self.interner.resolve(class_mir.name).to_string();
            let slot = resolve_result.module_vars.iter().position(|&sym| {
                interner.resolve(sym) == class_name_str
            });

            // Resolve superclass (check core classes, then module vars)
            let superclass = class_mir.superclass
                .and_then(|sup_sym| {
                    let sup_name = self.interner.resolve(sup_sym).to_string();
                    // Try core classes first
                    if let Some(v) = self.core_class_value(&sup_name) {
                        return v.as_object().map(|p| p as *mut ObjClass);
                    }
                    // Try module vars (user-defined classes created earlier)
                    if let Some(entry) = self.engine.modules.get(&module_key) {
                        for &var_val in &entry.vars {
                            if var_val.is_object() {
                                let ptr = var_val.as_object().unwrap() as *mut ObjClass;
                                let header = ptr as *const ObjHeader;
                                if unsafe { (*header).obj_type } == ObjType::Class {
                                    let name = unsafe { self.interner.resolve((*ptr).name).to_string() };
                                    if name == sup_name {
                                        return Some(ptr);
                                    }
                                }
                            }
                        }
                    }
                    None
                })
                .unwrap_or(self.object_class);

            // Create the class
            let class_ptr = self.gc.alloc_class(class_mir.name, superclass);
            unsafe {
                (*class_ptr).header.class = self.class_class;
                // Total fields = own fields + inherited fields from superclass chain
                let inherited_fields = if !superclass.is_null() {
                    (*superclass).num_fields
                } else {
                    0
                };
                (*class_ptr).num_fields = class_mir.num_fields + inherited_fields;
            }

            // Register each method's MIR and bind to the class
            for method_mir in class_mir.methods {
                let method_func_id = self.engine.register_function(method_mir.mir);

                // Create ObjFn + ObjClosure for the method
                let sig_sym = self.interner.intern(&method_mir.signature);
                let fn_ptr = self.gc.alloc_fn(sig_sym, 0, 0, method_func_id.0 as u32);
                unsafe {
                    (*fn_ptr).header.class = self.fn_class;
                }

                let closure_ptr = self.gc.alloc_closure(fn_ptr);
                unsafe {
                    (*closure_ptr).header.class = self.fn_class;
                }

                let sig_sym = self.interner.intern(&method_mir.signature);
                let bind_sym = if method_mir.is_static || method_mir.is_constructor {
                    let static_sig = format!("static:{}", method_mir.signature);
                    self.interner.intern(&static_sig)
                } else {
                    sig_sym
                };

                unsafe {
                    let method = if method_mir.is_constructor {
                        Method::Constructor(closure_ptr)
                    } else {
                        Method::Closure(closure_ptr)
                    };
                    (*class_ptr).methods.insert(bind_sym, method);
                }
            }

            // Store class in module vars
            if let Some(idx) = slot {
                let class_val = Value::object(class_ptr as *mut u8);
                if let Some(entry) = self.engine.modules.get_mut(&module_key) {
                    while entry.vars.len() <= idx {
                        entry.vars.push(Value::null());
                    }
                    entry.vars[idx] = class_val;
                }
            }
        }

        // 9. Create a fiber and push the initial call frame
        let fiber = self.gc.alloc_fiber();
        unsafe {
            (*fiber).header.class = self.fiber_class;
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                values: HMap::new(),
                module_name: module_key,
                return_dst: None,
                closure: None, defining_class: None,
            });
        }

        // Set as active fiber (save previous)
        let prev_fiber = self.fiber;
        self.fiber = fiber;

        // 10. Run the fiber
        let result = super::vm_interp::run_fiber(self);

        // Restore previous fiber
        self.fiber = prev_fiber;

        match result {
            Ok(_) => InterpretResult::Success,
            Err(e) => {
                self.report_error(&e.to_string());
                InterpretResult::RuntimeError
            }
        }
    }

    // -- Helpers for core library primitives --

    /// Allocate a GC-managed string and return it as a Value.
    pub fn new_string(&mut self, s: String) -> Value {
        let obj = self.gc.alloc_string(s);
        unsafe { (*obj).header.class = self.string_class; }
        Value::object(obj as *mut u8)
    }

    /// Write to output (captured buffer if set, otherwise stdout).
    pub fn vm_write(&mut self, s: &str) {
        if let Some(ref mut buf) = self.output_buffer {
            buf.push_str(s);
        } else {
            print!("{}", s);
        }
    }

    /// Take captured output (for tests).
    pub fn take_output(&mut self) -> String {
        self.output_buffer.take().unwrap_or_default()
    }

    /// Allocate a GC-managed list and return it as a Value.
    pub fn new_list(&mut self, elements: Vec<Value>) -> Value {
        let obj = self.gc.alloc_list();
        unsafe {
            (*obj).header.class = self.list_class;
            for elem in elements {
                (*obj).add(elem);
            }
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

    /// Map a core class name to its Value (pointer to ObjClass).
    /// Search all loaded modules for a variable with the given name.
    /// Used to resolve imported names across module boundaries.
    fn find_imported_var(&self, name: &str) -> Option<Value> {
        let target_sym = self.interner.lookup(name)?;
        for entry in self.engine.modules.values() {
            for &var_val in &entry.vars {
                if var_val.is_object() {
                    if let Some(ptr) = var_val.as_object() {
                        let header = ptr as *const ObjHeader;
                        if unsafe { (*header).obj_type } == ObjType::Class {
                            let cls = ptr as *mut ObjClass;
                            if unsafe { (*cls).name } == target_sym {
                                return Some(var_val);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    pub fn core_class_value(&self, name: &str) -> Option<Value> {
        let class_ptr = match name {
            "Object" => self.object_class,
            "Class" => self.class_class,
            "Bool" => self.bool_class,
            "Num" => self.num_class,
            "String" => self.string_class,
            "List" => self.list_class,
            "Map" => self.map_class,
            "Range" => self.range_class,
            "Null" => self.null_class,
            "Fn" => self.fn_class,
            "Fiber" => self.fiber_class,
            "System" => self.system_class,
            "Sequence" => self.sequence_class,
            _ => return None,
        };
        Some(Value::object(class_ptr as *mut u8))
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

    fn write_output(&mut self, s: &str) {
        self.vm_write(s);
    }

    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        self.gc.alloc_fiber()
    }

    fn get_fiber_class(&self) -> *mut ObjClass {
        self.fiber_class
    }

    fn get_fn_class(&self) -> *mut ObjClass {
        self.fn_class
    }

    fn set_fiber_action_call(&mut self, target: *mut ObjFiber, value: Value) {
        self.pending_fiber_action = Some(FiberAction::Call { target, value });
    }

    fn set_fiber_action_yield(&mut self, value: Value) {
        self.pending_fiber_action = Some(FiberAction::Yield { value });
    }

    fn set_fiber_action_transfer(&mut self, target: *mut ObjFiber, value: Value) {
        self.pending_fiber_action = Some(FiberAction::Transfer { target, value });
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Patch MakeClosure fn_id indices to actual engine FuncIds.
fn patch_closure_ids(func: &mut crate::mir::MirFunction, closure_func_ids: &[u32]) {
    for block in &mut func.blocks {
        for (_, inst) in &mut block.instructions {
            if let crate::mir::Instruction::MakeClosure { fn_id, .. } = inst {
                if let Some(&actual_id) = closure_func_ids.get(*fn_id as usize) {
                    *fn_id = actual_id;
                }
            }
        }
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

    /// Create a VM with output capture enabled. Run source, return (result, output).
    fn run_and_capture(source: &str) -> (InterpretResult, String) {
        let mut vm = VM::new_default();
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", source);
        let output = vm.take_output();
        (result, output)
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

    // -- interpret (integration) --

    #[test]
    fn test_interpret_system_print() {
        let (result, output) = run_and_capture("System.print(\"hello\")");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "hello\n");
    }

    #[test]
    fn test_interpret_arithmetic() {
        let (result, output) = run_and_capture("System.print(1 + 2)");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n");
    }

    #[test]
    fn test_interpret_module_vars() {
        let (result, output) = run_and_capture("var x = 10\nSystem.print(x)");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "10\n");
    }

    #[test]
    fn test_interpret_is_type() {
        let (result, output) = run_and_capture("System.print(42 is Num)");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "true\n");
    }

    #[test]
    fn test_interpret_class_construct() {
        let (result, _) = run_and_capture("class Foo {\n  construct new() {}\n}\nvar f = Foo.new()");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
    }

    #[test]
    fn test_interpret_class_with_fields() {
        let (result, output) = run_and_capture(r#"
class Point {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  y { _y }
}
var p = Point.new(3, 4)
System.print(p.x)
System.print(p.y)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n4\n");
    }

    #[test]
    fn test_interpret_instance_method() {
        let (result, output) = run_and_capture(r#"
class Greeter {
  construct new(name) {
    _name = name
  }
  greet() { _name }
}
var g = Greeter.new("Alice")
System.print(g.greet())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "Alice\n");
    }

    #[test]
    fn test_interpret_named_constructor() {
        let (result, output) = run_and_capture(r#"
class Foo {
  construct create(x) {
    _x = x
  }
  x { _x }
}
var f = Foo.create(42)
System.print(f.x)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_fn_call() {
        let (result, output) = run_and_capture(r#"
var fn = Fn.new {
  System.print(42)
}
fn.call()
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_fn_call_with_args() {
        let (result, output) = run_and_capture(r#"
var add = Fn.new {|a, b|
  System.print(a + b)
}
add.call(3, 4)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "7\n");
    }

    #[test]
    fn test_interpret_this_access() {
        let (result, output) = run_and_capture(r#"
class Counter {
  construct new(n) {
    _n = n
  }
  inc() { _n = _n + 1 }
  value { _n }
}
var c = Counter.new(0)
c.inc()
c.inc()
c.inc()
System.print(c.value)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n");
    }

    #[test]
    fn test_interpret_inheritance() {
        // Test method inheritance (child calls parent method)
        let (result, output) = run_and_capture(r#"
class Animal {
  construct new(name) {
    _name = name
  }
  name { _name }
}
class Dog is Animal {
  construct new(name) {
    _name = name
  }
  bark() { "woof" }
}
var d = Dog.new("Rex")
System.print(d.name)
System.print(d.bark())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "Rex\nwoof\n");
    }

    #[test]
    fn test_interpret_operator_overload() {
        let (result, output) = run_and_capture(r#"
class Vec2 {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  y { _y }
  +(other) { Vec2.new(_x + other.x, _y + other.y) }
  toString { "(" + _x.toString + ", " + _y.toString + ")" }
}
var a = Vec2.new(1, 2)
var b = Vec2.new(3, 4)
var c = a + b
System.print(c.x)
System.print(c.y)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "4\n6\n");
    }

    #[test]
    fn test_interpret_prefix_operator() {
        let (result, output) = run_and_capture(r#"
class Num2 {
  construct new(n) { _n = n }
  value { _n }
  -() { Num2.new(-_n) }
}
var a = Num2.new(5)
var b = -a
System.print(b.value)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "-5\n");
    }

    #[test]
    fn test_interpret_closure_capture() {
        let (result, output) = run_and_capture(r#"
var x = 10
var f = Fn.new { x }
System.print(f.call())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "10\n");
    }

    #[test]
    fn test_interpret_closure_capture_with_args() {
        let (result, output) = run_and_capture(r#"
var x = 10
var f = Fn.new {|y| x + y }
System.print(f.call(5))
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "15\n");
    }

    #[test]
    fn test_interpret_closure_capture_multiple() {
        let (result, output) = run_and_capture(r#"
var a = 3
var b = 7
var f = Fn.new { a + b }
System.print(f.call())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "10\n");
    }

    #[test]
    fn test_interpret_closure_capture_in_block() {
        let (result, output) = run_and_capture(r#"
var result = null
{
  var x = 42
  var f = Fn.new { x }
  result = f.call()
}
System.print(result)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_closure_nested_capture() {
        let (result, output) = run_and_capture(r#"
var x = 100
var outer = Fn.new {
  Fn.new { x }
}
var inner = outer.call()
System.print(inner.call())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "100\n");
    }

    #[test]
    fn test_interpret_super_call() {
        let (result, output) = run_and_capture(r#"
class Animal {
  construct new(name) { _name = name }
  speak() { _name }
}
class Dog is Animal {
  construct new(name) { super(name) }
  speak() { super.speak() + " says woof" }
}
var d = Dog.new("Rex")
System.print(d.speak())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "Rex says woof\n");
    }

    #[test]
    fn test_interpret_super_getter() {
        let (result, output) = run_and_capture(r#"
class Base {
  construct new(v) { _v = v }
  value { _v }
}
class Child is Base {
  construct new(v) { super(v) }
  value { super.value + 10 }
}
System.print(Child.new(5).value)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "15\n");
    }

    #[test]
    fn test_interpret_fiber_new_call() {
        let (result, output) = run_and_capture(r#"
var fiber = Fiber.new {
  System.print("inside fiber")
}
fiber.call()
System.print("after call")
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "inside fiber\nafter call\n");
    }

    #[test]
    fn test_interpret_fiber_yield() {
        let (result, output) = run_and_capture(r#"
var fiber = Fiber.new {
  System.print("before yield")
  Fiber.yield()
  System.print("after yield")
}
fiber.call()
System.print("yielded")
fiber.call()
System.print("done")
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "before yield\nyielded\nafter yield\ndone\n");
    }

    #[test]
    fn test_interpret_fiber_yield_value() {
        let (result, output) = run_and_capture(r#"
var fiber = Fiber.new {
  Fiber.yield(42)
}
var result = fiber.call()
System.print(result)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_fiber_is_done() {
        let (result, output) = run_and_capture(r#"
var fiber = Fiber.new {
  Fiber.yield()
}
System.print(fiber.isDone)
fiber.call()
System.print(fiber.isDone)
fiber.call()
System.print(fiber.isDone)
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "false\nfalse\ntrue\n");
    }

    #[test]
    fn test_interpret_fiber_pass_value_on_resume() {
        let (result, output) = run_and_capture(r#"
var fiber = Fiber.new {
  var got = Fiber.yield()
  System.print(got)
}
fiber.call()
fiber.call("hello from caller")
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "hello from caller\n");
    }

    #[test]
    fn test_interpret_fiber_multiple_yields() {
        let (result, output) = run_and_capture(r#"
var fiber = Fiber.new {
  Fiber.yield(1)
  Fiber.yield(2)
  Fiber.yield(3)
}
System.print(fiber.call())
System.print(fiber.call())
System.print(fiber.call())
"#);
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "1\n2\n3\n");
    }

    #[test]
    fn test_interpret_while_loop() {
        let (result, output) = run_and_capture(r#"
var i = 0
while (i < 3) {
  System.print(i)
  i = i + 1
}
"#);
        assert!(matches!(result, InterpretResult::Success), "while loop failed: {:?}", result);
        assert_eq!(output, "0\n1\n2\n");
    }

    #[test]
    fn test_interpret_list_subscript() {
        let (result, output) = run_and_capture(r#"
var list = [10, 20, 30]
System.print(list[0])
System.print(list[2])
list[1] = 99
System.print(list[1])
"#);
        assert!(matches!(result, InterpretResult::Success), "list subscript failed: {:?}", result);
        assert_eq!(output, "10\n30\n99\n");
    }

    #[test]
    fn test_interpret_for_in_list() {
        let (result, output) = run_and_capture(r#"
var list = [10, 20, 30]
for (x in list) {
  System.print(x)
}
"#);
        assert!(matches!(result, InterpretResult::Success), "for-in list failed: {:?}", result);
        assert_eq!(output, "10\n20\n30\n");
    }

    #[test]
    fn test_interpret_for_in_range() {
        let (result, output) = run_and_capture(r#"
for (i in 1..4) {
  System.print(i)
}
"#);
        assert!(matches!(result, InterpretResult::Success), "for-in range failed: {:?}", result);
        assert_eq!(output, "1\n2\n3\n4\n");
    }

    #[test]
    fn test_interpret_string_plus() {
        let (result, output) = run_and_capture(r#"
var a = "hello"
var b = " world"
System.print(a + b)
"#);
        assert!(matches!(result, InterpretResult::Success), "string concat failed: {:?}", result);
        assert_eq!(output, "hello world\n");
    }

    #[test]
    fn test_interpret_static_method_with_args() {
        let mut vm = VM::new(VMConfig::default());
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", r#"
class Foo {
  static double(n) { n + n }
}
System.print(Foo.double(21))
"#);
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "static method with args failed: {:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_import_module() {
        let mut config = VMConfig::default();
        config.load_module_fn = Some(Box::new(|name: &str| -> Option<String> {
            if name == "math_utils" {
                Some(r#"
class MathUtils {
  static greet() { "hello from import" }
  static double(n) { n + n }
}
"#.to_string())
            } else {
                None
            }
        }));
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", r#"
import "math_utils" for MathUtils
System.print(MathUtils.greet())
System.print(MathUtils.double(21))
"#);
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "hello from import\n42\n");
    }
}
