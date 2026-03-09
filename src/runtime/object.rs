/// Heap-allocated object types for the Wren runtime.
///
/// Every heap object starts with an `ObjHeader` that carries:
/// - A type tag for runtime dispatch
/// - A GC mark/color byte for the garbage collector
/// - An intrusive linked-list pointer for the GC's all-objects chain
/// - A class pointer for method dispatch
///
/// Objects are always accessed through `*mut ObjHeader` (or a typed wrapper
/// pointer). The NaN-boxed `Value` stores the lower 48 bits of this pointer.
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use super::value::Value;
use crate::intern::SymbolId;
use crate::mir::BlockId;
use crate::runtime::engine::FuncId;

// ---------------------------------------------------------------------------
// Object type tag
// ---------------------------------------------------------------------------

/// Discriminator for heap object types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjType {
    String,
    List,
    Map,
    Range,
    Fn,
    Closure,
    Upvalue,
    Fiber,
    Class,
    Instance,
    Foreign,
    Module,
}

// ---------------------------------------------------------------------------
// Object header (common prefix)
// ---------------------------------------------------------------------------

/// Common header for all heap-allocated objects.
///
/// Every `Obj*` struct must begin with this header so the GC can walk the
/// object graph uniformly. The `next` pointer forms an intrusive linked list
/// of all live objects for the GC sweep phase.
#[repr(C)]
pub struct ObjHeader {
    /// What kind of object this is.
    pub obj_type: ObjType, // offset 0, u8
    /// GC mark bit / tri-color byte. 0 = white, 1 = gray, 2 = black.
    pub gc_mark: u8, // offset 1
    /// GC generation. 0 = young (nursery), 1 = old.
    pub generation: u8, // offset 2
    // 5 bytes padding (implicit)
    /// Intrusive linked list of all heap objects (for GC sweep).
    pub next: *mut ObjHeader, // offset 8
    /// The class of this object (for method dispatch). Null for meta-objects.
    pub class: *mut ObjClass, // offset 16
                              // total: 24 bytes
}

impl ObjHeader {
    pub fn new(obj_type: ObjType) -> Self {
        Self {
            obj_type,
            gc_mark: 0,
            generation: 0,
            next: std::ptr::null_mut(),
            class: std::ptr::null_mut(),
        }
    }
}

impl fmt::Debug for ObjHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjHeader({:?})", self.obj_type)
    }
}

// ---------------------------------------------------------------------------
// String
// ---------------------------------------------------------------------------

/// A heap-allocated immutable string.
///
/// Strings in Wren are immutable value types. We store a precomputed hash
/// for O(1) map lookups and deduplication.
#[repr(C)]
pub struct ObjString {
    pub header: ObjHeader,
    /// Precomputed FNV-1a hash.
    pub hash: u64,
    /// The string data.
    pub value: String,
}

impl ObjString {
    pub fn new(s: String) -> Self {
        let hash = fnv1a_hash(s.as_bytes());
        Self {
            header: ObjHeader::new(ObjType::String),
            hash,
            value: s,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }
}

impl fmt::Debug for ObjString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjString({:?})", self.value)
    }
}

/// FNV-1a hash for strings.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

/// A growable array with JIT-friendly raw buffer layout.
#[repr(C)]
pub struct ObjList {
    pub header: ObjHeader, // offset 0, 24 bytes
    pub count: u32,        // offset 24
    pub capacity: u32,     // offset 28
    pub elements: *mut Value, // offset 32, heap-allocated buffer
                           // total: 40 bytes
}

const LIST_INITIAL_CAPACITY: u32 = 8;

impl Default for ObjList {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjList {
    pub fn new() -> Self {
        Self {
            header: ObjHeader::new(ObjType::List),
            count: 0,
            capacity: 0,
            elements: std::ptr::null_mut(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1) as u32;
        let layout = std::alloc::Layout::array::<Value>(cap as usize).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut Value };
        Self {
            header: ObjHeader::new(ObjType::List),
            count: 0,
            capacity: cap,
            elements: ptr,
        }
    }

    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: usize) -> Option<Value> {
        if index < self.count as usize {
            Some(unsafe { *self.elements.add(index) })
        } else {
            None
        }
    }

    pub fn set(&mut self, index: usize, value: Value) {
        if index < self.count as usize {
            unsafe {
                self.elements.add(index).write(value);
            }
        }
    }

    pub fn add(&mut self, value: Value) {
        self.ensure_capacity(self.count + 1);
        unsafe {
            self.elements.add(self.count as usize).write(value);
        }
        self.count += 1;
    }

    pub fn insert(&mut self, index: usize, value: Value) {
        let count = self.count as usize;
        if index > count {
            return;
        }
        self.ensure_capacity(self.count + 1);
        if index < count {
            unsafe {
                std::ptr::copy(
                    self.elements.add(index),
                    self.elements.add(index + 1),
                    count - index,
                );
            }
        }
        unsafe {
            self.elements.add(index).write(value);
        }
        self.count += 1;
    }

    pub fn remove(&mut self, index: usize) -> Option<Value> {
        let count = self.count as usize;
        if index >= count {
            return None;
        }
        let val = unsafe { self.elements.add(index).read() };
        if index < count - 1 {
            unsafe {
                std::ptr::copy(
                    self.elements.add(index + 1),
                    self.elements.add(index),
                    count - index - 1,
                );
            }
        }
        self.count -= 1;
        Some(val)
    }

    pub fn clear(&mut self) {
        self.count = 0;
    }

    pub fn as_slice(&self) -> &[Value] {
        if self.elements.is_null() || self.count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.elements, self.count as usize) }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [Value] {
        if self.elements.is_null() || self.count == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.elements, self.count as usize) }
        }
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        let count = self.count as usize;
        if a < count && b < count {
            unsafe {
                let va = self.elements.add(a).read();
                let vb = self.elements.add(b).read();
                self.elements.add(a).write(vb);
                self.elements.add(b).write(va);
            }
        }
    }

    /// Build a list from a Vec of values.
    pub fn from_elements(elems: Vec<Value>) -> Self {
        let mut list = Self::with_capacity(elems.len());
        for v in elems {
            list.add(v);
        }
        list
    }

    fn ensure_capacity(&mut self, needed: u32) {
        if needed <= self.capacity {
            return;
        }
        let new_cap = if self.capacity == 0 {
            LIST_INITIAL_CAPACITY.max(needed)
        } else {
            (self.capacity * 2).max(needed)
        };
        let new_layout = std::alloc::Layout::array::<Value>(new_cap as usize).unwrap();
        let new_ptr = if self.elements.is_null() {
            unsafe { std::alloc::alloc(new_layout) as *mut Value }
        } else {
            let old_layout = std::alloc::Layout::array::<Value>(self.capacity as usize).unwrap();
            unsafe {
                std::alloc::realloc(self.elements as *mut u8, old_layout, new_layout.size())
                    as *mut Value
            }
        };
        self.elements = new_ptr;
        self.capacity = new_cap;
    }
}

impl Drop for ObjList {
    fn drop(&mut self) {
        if !self.elements.is_null() && self.capacity > 0 {
            let layout = std::alloc::Layout::array::<Value>(self.capacity as usize).unwrap();
            unsafe {
                std::alloc::dealloc(self.elements as *mut u8, layout);
            }
        }
    }
}

impl fmt::Debug for ObjList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slice = if self.elements.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.elements, self.count as usize) }
        };
        write!(f, "ObjList({:?})", slice)
    }
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

/// Key wrapper that enables Value to be used as a HashMap key.
///
/// Uses the raw u64 bits for hashing (which is correct because our
/// equality semantics match: bitwise equal non-nums are equal, and
/// for nums we use the bits directly which gives consistent hashing
/// with IEEE equality for non-NaN values).
#[derive(Clone, Copy)]
pub struct MapKey(pub Value);

impl MapKey {
    pub fn new(v: Value) -> Self {
        MapKey(v)
    }

    pub fn value(self) -> Value {
        self.0
    }
}

impl PartialEq for MapKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.equals(other.0)
    }
}

impl Eq for MapKey {}

impl Hash for MapKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let v = self.0;
        if v.is_object() {
            if let Some(ptr) = v.as_object() {
                let header = ptr as *const ObjHeader;
                unsafe {
                    if (*header).obj_type == ObjType::String {
                        // Hash by string content so equal strings have equal hashes
                        let s = &*(ptr as *const ObjString);
                        0xFFu8.hash(state); // discriminant tag for strings
                        s.value.hash(state);
                        return;
                    }
                }
            }
        }
        // For non-string values, hash by raw bits (identity)
        v.to_bits().hash(state);
    }
}

impl fmt::Debug for MapKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapKey({:?})", self.0)
    }
}

/// A hash map from Value → Value.
#[repr(C)]
pub struct ObjMap {
    pub header: ObjHeader,
    pub entries: HashMap<MapKey, Value>,
}

impl Default for ObjMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjMap {
    pub fn new() -> Self {
        Self {
            header: ObjHeader::new(ObjType::Map),
            entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: Value) -> Option<Value> {
        self.entries.get(&MapKey::new(key)).copied()
    }

    pub fn set(&mut self, key: Value, value: Value) {
        self.entries.insert(MapKey::new(key), value);
    }

    pub fn remove(&mut self, key: Value) -> Option<Value> {
        self.entries.remove(&MapKey::new(key))
    }

    pub fn contains(&self, key: Value) -> bool {
        self.entries.contains_key(&MapKey::new(key))
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl fmt::Debug for ObjMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjMap({} entries)", self.entries.len())
    }
}

// ---------------------------------------------------------------------------
// Range
// ---------------------------------------------------------------------------

/// A numeric range (from..to), inclusive or exclusive.
#[repr(C)]
pub struct ObjRange {
    pub header: ObjHeader,
    pub from: f64,
    pub to: f64,
    pub is_inclusive: bool,
}

impl ObjRange {
    pub fn new(from: f64, to: f64, is_inclusive: bool) -> Self {
        Self {
            header: ObjHeader::new(ObjType::Range),
            from,
            to,
            is_inclusive,
        }
    }

    /// Iterate integer steps (Wren ranges iterate as integers).
    pub fn iter_integers(&self) -> RangeIter {
        RangeIter {
            current: self.from as i64,
            end: self.to as i64,
            is_inclusive: self.is_inclusive,
            ascending: self.from <= self.to,
        }
    }
}

impl fmt::Debug for ObjRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_inclusive {
            write!(f, "ObjRange({}..{})", self.from, self.to)
        } else {
            write!(f, "ObjRange({}...{})", self.from, self.to)
        }
    }
}

/// Iterator for integer steps through a range.
pub struct RangeIter {
    current: i64,
    end: i64,
    is_inclusive: bool,
    ascending: bool,
}

impl Iterator for RangeIter {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        let done = if self.is_inclusive {
            if self.ascending {
                self.current > self.end
            } else {
                self.current < self.end
            }
        } else if self.ascending {
            self.current >= self.end
        } else {
            self.current <= self.end
        };

        if done {
            return None;
        }

        let val = self.current;
        if self.ascending {
            self.current += 1;
        } else {
            self.current -= 1;
        }
        Some(val)
    }
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

/// A compiled function (before closure capture).
#[repr(C)]
pub struct ObjFn {
    pub header: ObjHeader,
    /// Function name for debugging (SymbolId::MAX for anonymous).
    pub name: SymbolId,
    /// Number of parameters (excluding the receiver).
    pub arity: u8,
    /// Number of upvalues this function captures.
    pub upvalue_count: u16,
    /// Index into the code/module's function table.
    pub fn_id: u32,
}

impl ObjFn {
    pub fn new(name: SymbolId, arity: u8, upvalue_count: u16, fn_id: u32) -> Self {
        Self {
            header: ObjHeader::new(ObjType::Fn),
            name,
            arity,
            upvalue_count,
            fn_id,
        }
    }
}

impl fmt::Debug for ObjFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjFn({:?}, arity={})", self.name, self.arity)
    }
}

// ---------------------------------------------------------------------------
// Closure
// ---------------------------------------------------------------------------

/// A closure: a function + captured upvalues.
#[repr(C)]
pub struct ObjClosure {
    pub header: ObjHeader,
    /// The underlying function.
    pub function: *mut ObjFn,
    /// Captured upvalues (pointers to ObjUpvalue).
    pub upvalues: Vec<*mut ObjUpvalue>,
}

impl ObjClosure {
    pub fn new(function: *mut ObjFn, upvalue_count: usize) -> Self {
        Self {
            header: ObjHeader::new(ObjType::Closure),
            function,
            upvalues: vec![std::ptr::null_mut(); upvalue_count],
        }
    }
}

impl fmt::Debug for ObjClosure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjClosure(upvalues={})", self.upvalues.len())
    }
}

// ---------------------------------------------------------------------------
// Upvalue
// ---------------------------------------------------------------------------

/// A captured variable. While the variable is on the stack, `location` points
/// into the stack. When the variable goes out of scope, its value is "closed
/// over" — moved into `closed` and `location` updated to point to `closed`.
#[repr(C)]
pub struct ObjUpvalue {
    pub header: ObjHeader,
    /// Pointer to the variable's current storage (stack slot or `closed`).
    pub location: *mut Value,
    /// Storage for the closed-over value once the stack slot is gone.
    pub closed: Value,
    /// Intrusive list of open upvalues (sorted by stack slot, descending).
    pub next_upvalue: *mut ObjUpvalue,
}

impl ObjUpvalue {
    pub fn new(location: *mut Value) -> Self {
        Self {
            header: ObjHeader::new(ObjType::Upvalue),
            location,
            closed: Value::null(),
            next_upvalue: std::ptr::null_mut(),
        }
    }

    /// Close over the value: copy from stack to `closed` field,
    /// then redirect `location` to point to `closed`.
    pub fn close(&mut self) {
        unsafe {
            self.closed = *self.location;
        }
        self.location = &mut self.closed as *mut Value;
    }

    /// Read the current value of this upvalue.
    pub fn get(&self) -> Value {
        unsafe { *self.location }
    }

    /// Write a new value to this upvalue.
    pub fn set(&mut self, value: Value) {
        unsafe {
            *self.location = value;
        }
    }
}

impl fmt::Debug for ObjUpvalue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjUpvalue")
    }
}

// ---------------------------------------------------------------------------
// Fiber
// ---------------------------------------------------------------------------

/// Fiber execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberState {
    /// Created but not yet started.
    New,
    /// Currently executing.
    Running,
    /// Suspended (yielded or waiting).
    Suspended,
    /// Finished execution normally.
    Done,
    /// Finished with an error.
    Error,
}

/// A call frame on a fiber's call stack (for JIT/closure-based execution).
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// The closure being executed.
    pub closure: *mut ObjClosure,
    /// Instruction pointer (index into compiled code).
    pub ip: usize,
    /// Base index into the fiber's value stack for this frame's locals.
    pub stack_base: usize,
}

/// A call frame for MIR interpretation within a fiber.
///
/// Holds the per-frame state needed to walk MIR basic blocks:
/// which function, which block, which instruction, and the SSA value map.
#[derive(Clone)]
pub struct MirCallFrame {
    /// Which function in the engine we're executing.
    pub func_id: FuncId,
    /// Current basic block.
    pub current_block: BlockId,
    /// Instruction index within the current block (used for resumption after yield).
    pub ip: usize,
    /// SSA value map for this frame.
    pub values: HashMap<crate::mir::ValueId, crate::mir::interp::InterpValue>,
    /// Module variable storage for this frame's module.
    pub module_name: String,
    /// The ValueId in the *caller* frame that should receive our return value.
    pub return_dst: Option<crate::mir::ValueId>,
    /// The closure being executed (if any), for upvalue access.
    pub closure: Option<*mut ObjClosure>,
    /// The class that defines the current method (for super dispatch).
    pub defining_class: Option<*mut ObjClass>,
}

impl fmt::Debug for MirCallFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MirCallFrame(func={:?}, block={:?}, ip={})",
            self.func_id, self.current_block, self.ip
        )
    }
}

/// A snapshot of a call frame's location, used for spawn-site traces.
#[derive(Debug, Clone)]
pub struct StackFrame {
    pub func_name: String,
    pub module: String,
    pub line: Option<usize>,
}

impl std::fmt::Display for StackFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(line) => write!(f, "  at {} ({}:{})", self.func_name, self.module, line),
            None => write!(f, "  at {} ({})", self.func_name, self.module),
        }
    }
}

/// A fiber (lightweight coroutine / green thread).
#[repr(C)]
pub struct ObjFiber {
    pub header: ObjHeader,
    /// The value stack (used for passing args between frames).
    pub stack: Vec<Value>,
    /// Call frame stack for JIT/closure-based execution.
    pub frames: Vec<CallFrame>,
    /// Call frame stack for MIR interpretation.
    pub mir_frames: Vec<MirCallFrame>,
    /// Current state.
    pub state: FiberState,
    /// The fiber that will resume when this one finishes or yields.
    pub caller: *mut ObjFiber,
    /// Error value (if state == Error).
    pub error: Value,
    /// Where to store the resume value when this fiber is resumed after yielding.
    pub resume_value_dst: Option<crate::mir::ValueId>,
    /// Stack trace snapshot from where this fiber was spawned (opt-in).
    pub spawn_trace: Option<Vec<StackFrame>>,
    /// When true, runtime errors are caught and stored on the fiber instead
    /// of propagating (set by Fiber.try).
    pub is_try: bool,
}

impl Default for ObjFiber {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjFiber {
    pub fn new() -> Self {
        Self {
            header: ObjHeader::new(ObjType::Fiber),
            stack: Vec::with_capacity(256),
            frames: Vec::with_capacity(64),
            mir_frames: Vec::with_capacity(64),
            state: FiberState::New,
            caller: std::ptr::null_mut(),
            error: Value::null(),
            resume_value_dst: None,
            spawn_trace: None,
            is_try: false,
        }
    }

    pub fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    pub fn pop(&mut self) -> Value {
        self.stack.pop().unwrap_or(Value::null())
    }

    pub fn peek(&self, distance: usize) -> Value {
        let idx = self.stack.len().wrapping_sub(1 + distance);
        self.stack.get(idx).copied().unwrap_or(Value::null())
    }

    pub fn push_frame(&mut self, frame: CallFrame) {
        self.frames.push(frame);
    }

    pub fn pop_frame(&mut self) -> Option<CallFrame> {
        self.frames.pop()
    }
}

impl fmt::Debug for ObjFiber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObjFiber(state={:?}, stack={}, frames={})",
            self.state,
            self.stack.len(),
            self.frames.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Class
// ---------------------------------------------------------------------------

/// A Wren class.
///
/// Methods are indexed by `SymbolId` for O(1) dispatch. The method table
/// maps symbol IDs to closures (or native function pointers). Inheritance
/// is implemented by copying the superclass's method table at class creation.
#[repr(C)]
pub struct ObjClass {
    pub header: ObjHeader,
    /// Class name.
    pub name: SymbolId,
    /// Superclass (null for Object).
    pub superclass: *mut ObjClass,
    /// Method table: symbol → closure/native.
    pub methods: HashMap<SymbolId, Method>,
    /// Number of instance fields.
    pub num_fields: u16,
    /// Is this a foreign class?
    pub is_foreign: bool,
    /// Protocols this class conforms to (bitset, populated at class creation).
    pub protocols: crate::sema::protocol::ProtocolSet,
    /// Static fields (__name) — per-class storage shared across instances.
    pub static_fields: HashMap<SymbolId, Value>,
}

/// A method entry in the class method table.
#[derive(Clone)]
pub enum Method {
    /// A Wren closure.
    Closure(*mut ObjClosure),
    /// A constructor closure — allocates an instance then calls the body.
    Constructor(*mut ObjClosure),
    /// A native/foreign function.
    Native(NativeFn),
}

impl fmt::Debug for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Method::Closure(_) => write!(f, "Method::Closure(...)"),
            Method::Constructor(_) => write!(f, "Method::Constructor(...)"),
            Method::Native(_) => write!(f, "Method::Native(...)"),
        }
    }
}

/// Signature for native functions: (vm, args) → result.
pub type NativeFn = fn(&mut dyn NativeContext, &[Value]) -> Value;

/// Trait for native function context (implemented by the VM).
///
/// Core library primitives and foreign methods use this to interact with
/// the VM without creating circular dependencies.
pub trait NativeContext {
    // -- Slot API (for C FFI compatibility) --
    fn get_slot(&self, index: usize) -> Value;
    fn set_slot(&mut self, index: usize, value: Value);
    fn slot_count(&self) -> usize;

    // -- Allocation --
    fn alloc_string(&mut self, s: String) -> Value;
    fn alloc_list(&mut self, elements: Vec<Value>) -> Value;
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> Value;
    fn alloc_map(&mut self) -> Value;

    // -- Error handling --
    fn runtime_error(&mut self, msg: String);
    fn has_error(&self) -> bool;

    // -- Object introspection --
    fn get_class_of(&self, value: Value) -> *mut ObjClass;
    fn get_class_name_of(&self, value: Value) -> String;

    // -- Symbol interning --
    fn intern(&mut self, s: &str) -> SymbolId;
    fn resolve_symbol(&self, id: SymbolId) -> &str;

    // -- Output --
    fn write_output(&mut self, s: &str);

    // -- Fiber operations --
    fn alloc_fiber(&mut self) -> *mut ObjFiber;
    fn get_fiber_class(&self) -> *mut ObjClass;
    fn get_fn_class(&self) -> *mut ObjClass;
    fn set_fiber_action_call(&mut self, target: *mut ObjFiber, value: Value);
    fn set_fiber_action_yield(&mut self, value: Value);
    fn set_fiber_action_transfer(&mut self, target: *mut ObjFiber, value: Value);
    fn set_fiber_action_suspend(&mut self);
    fn get_current_fiber(&self) -> *mut ObjFiber;

    // -- Stack trace support (opt-in) --
    fn fiber_stack_traces_enabled(&self) -> bool;
    fn capture_spawn_trace(&self) -> Option<Vec<StackFrame>>;
    fn get_stack_trace_string(&self, fiber: *mut ObjFiber) -> String;

    // -- Method dispatch from native code --
    /// Call a method on a receiver from native code. Handles both Native and
    /// Closure methods (closures run on a temporary fiber).
    /// Returns None if the method doesn't exist or the call fails.
    fn call_method_on(&mut self, receiver: Value, method: &str, args: &[Value]) -> Option<Value>;

    // -- Foreign objects --
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign;
    fn new_list(&mut self) -> Value;

    // -- Module variable lookup (for Meta module) --
    fn get_module_variable_names(&self, module: &str) -> Option<Vec<String>>;

    // -- Runtime compilation (for Meta module) --
    /// Compile and execute Wren source in the given module scope.
    fn meta_eval(&mut self, module: &str, source: &str) -> bool;
    /// Compile Wren source into a closure value (does not execute).
    fn meta_compile(&mut self, source: &str) -> Option<Value>;
    /// Compile a Wren expression into a closure that returns the expression's value.
    fn meta_compile_expression(&mut self, expr: &str) -> Option<Value>;

    // -- Instance allocation --
    /// Allocate a new ObjInstance of the given class, returning its Value.
    fn alloc_instance(&mut self, class: *mut ObjClass) -> Value;

    // -- Class lookup --
    /// Look up a core class by name (e.g. "MapSequence").
    fn lookup_class(&self, name: &str) -> Option<*mut ObjClass>;

    // -- Garbage collection --
    fn trigger_gc(&mut self);
}

impl ObjClass {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(name: SymbolId, superclass: *mut ObjClass) -> Self {
        let mut methods = HashMap::new();
        let mut protocols = crate::sema::protocol::ProtocolSet::EMPTY;

        // Inherit methods and protocols from superclass.
        if !superclass.is_null() {
            unsafe {
                for (sym, method) in &(*superclass).methods {
                    methods.insert(*sym, method.clone());
                }
                protocols = (*superclass).protocols;
            }
        }

        Self {
            header: ObjHeader::new(ObjType::Class),
            name,
            superclass,
            methods,
            num_fields: 0,
            is_foreign: false,
            protocols,
            static_fields: HashMap::new(),
        }
    }

    /// Bind a closure method to this class.
    pub fn bind_method(&mut self, name: SymbolId, closure: *mut ObjClosure) {
        self.methods.insert(name, Method::Closure(closure));
    }

    /// Bind a native method to this class.
    pub fn bind_native(&mut self, name: SymbolId, func: NativeFn) {
        self.methods.insert(name, Method::Native(func));
    }

    /// Look up a method by symbol. Walks superclass chain.
    pub fn find_method(&self, name: SymbolId) -> Option<&Method> {
        if let Some(m) = self.methods.get(&name) {
            return Some(m);
        }
        let mut cls = self.superclass;
        while !cls.is_null() {
            let parent = unsafe { &*cls };
            if let Some(m) = parent.methods.get(&name) {
                return Some(m);
            }
            cls = parent.superclass;
        }
        None
    }
}

impl fmt::Debug for ObjClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObjClass({:?}, {} methods)",
            self.name,
            self.methods.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

/// A class instance with a fixed number of fields.
#[repr(C)]
pub struct ObjInstance {
    pub header: ObjHeader, // offset 0, 24 bytes
    pub num_fields: u32,   // offset 24
    // 4 bytes padding (implicit, for *mut alignment)
    pub fields: *mut Value, // offset 32, heap-allocated [Value; num_fields]
                            // total: 40 bytes
}

impl ObjInstance {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn new(class: *mut ObjClass) -> Self {
        let num_fields = if class.is_null() {
            0
        } else {
            unsafe { (*class).num_fields as usize }
        };

        let fields = if num_fields > 0 {
            let layout = std::alloc::Layout::array::<Value>(num_fields).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut Value };
            // Initialize all fields to null
            for i in 0..num_fields {
                unsafe {
                    ptr.add(i).write(Value::null());
                }
            }
            ptr
        } else {
            std::ptr::null_mut()
        };

        let mut instance = Self {
            header: ObjHeader::new(ObjType::Instance),
            num_fields: num_fields as u32,
            fields,
        };
        instance.header.class = class;
        instance
    }

    pub fn get_field(&self, index: usize) -> Option<Value> {
        if index < self.num_fields as usize {
            Some(unsafe { *self.fields.add(index) })
        } else {
            None
        }
    }

    pub fn set_field(&mut self, index: usize, value: Value) {
        if index < self.num_fields as usize {
            unsafe {
                self.fields.add(index).write(value);
            }
        }
    }
}

impl Drop for ObjInstance {
    fn drop(&mut self) {
        if !self.fields.is_null() && self.num_fields > 0 {
            let layout = std::alloc::Layout::array::<Value>(self.num_fields as usize).unwrap();
            unsafe {
                std::alloc::dealloc(self.fields as *mut u8, layout);
            }
        }
    }
}

impl fmt::Debug for ObjInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjInstance({} fields)", self.num_fields)
    }
}

// ---------------------------------------------------------------------------
// Foreign
// ---------------------------------------------------------------------------

/// A foreign object wrapping opaque host data.
#[repr(C)]
pub struct ObjForeign {
    pub header: ObjHeader,
    /// Opaque data managed by the host.
    pub data: Vec<u8>,
}

impl ObjForeign {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            header: ObjHeader::new(ObjType::Foreign),
            data,
        }
    }
}

impl fmt::Debug for ObjForeign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjForeign({} bytes)", self.data.len())
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// A Wren module (compilation unit).
#[repr(C)]
pub struct ObjModule {
    pub header: ObjHeader,
    /// Module name.
    pub name: SymbolId,
    /// Module-level variables.
    pub variables: Vec<Value>,
    /// Variable names (parallel to `variables` for debug/import resolution).
    pub variable_names: Vec<SymbolId>,
}

impl ObjModule {
    pub fn new(name: SymbolId) -> Self {
        Self {
            header: ObjHeader::new(ObjType::Module),
            name,
            variables: Vec::new(),
            variable_names: Vec::new(),
        }
    }

    pub fn define_variable(&mut self, name: SymbolId, value: Value) -> usize {
        let index = self.variables.len();
        self.variables.push(value);
        self.variable_names.push(name);
        index
    }

    pub fn find_variable(&self, name: SymbolId) -> Option<usize> {
        self.variable_names.iter().position(|&n| n == name)
    }
}

impl fmt::Debug for ObjModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObjModule({:?}, {} vars)",
            self.name,
            self.variables.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers: safe downcasting from ObjHeader
// ---------------------------------------------------------------------------

/// Cast an `ObjHeader` pointer to a specific object type.
///
/// # Safety
/// The pointer must be non-null, valid, and point to an object whose
/// `obj_type` matches the target type.
pub unsafe fn downcast_ref<T>(header: *const ObjHeader) -> &'static T {
    &*(header as *const T)
}

/// Cast an `ObjHeader` pointer to a mutable specific object type.
///
/// # Safety
/// Same as `downcast_ref`, plus the caller must have unique access.
pub unsafe fn downcast_mut<T>(header: *mut ObjHeader) -> &'static mut T {
    &mut *(header as *mut T)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;

    // Helper: create an interner with some symbols.
    fn test_interner() -> Interner {
        let mut i = Interner::new();
        i.intern("foo"); // 0
        i.intern("bar"); // 1
        i.intern("baz"); // 2
        i.intern("+"); // 3
        i.intern("init"); // 4
        i.intern("myClass"); // 5
        i
    }

    fn sym(interner: &mut Interner, s: &str) -> SymbolId {
        interner.intern(s)
    }

    // -- ObjString ----------------------------------------------------------

    #[test]
    fn test_string_creation() {
        let s = ObjString::new("hello".to_string());
        assert_eq!(s.header.obj_type, ObjType::String);
        assert_eq!(s.as_str(), "hello");
        assert_eq!(s.len(), 5);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_string_empty() {
        let s = ObjString::new(String::new());
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_string_hash_deterministic() {
        let a = ObjString::new("hello".to_string());
        let b = ObjString::new("hello".to_string());
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn test_string_hash_different() {
        let a = ObjString::new("hello".to_string());
        let b = ObjString::new("world".to_string());
        assert_ne!(a.hash, b.hash);
    }

    // -- ObjList ------------------------------------------------------------

    #[test]
    fn test_list_crud() {
        let mut list = ObjList::new();
        assert!(list.is_empty());

        list.add(Value::num(1.0));
        list.add(Value::num(2.0));
        list.add(Value::num(3.0));
        assert_eq!(list.len(), 3);

        assert_eq!(list.get(0), Some(Value::num(1.0)));
        assert_eq!(list.get(1), Some(Value::num(2.0)));
        assert_eq!(list.get(3), None);

        list.set(1, Value::num(42.0));
        assert_eq!(list.get(1), Some(Value::num(42.0)));

        list.remove(0);
        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0), Some(Value::num(42.0)));
    }

    #[test]
    fn test_list_insert() {
        let mut list = ObjList::new();
        list.add(Value::num(1.0));
        list.add(Value::num(3.0));
        list.insert(1, Value::num(2.0));
        assert_eq!(list.len(), 3);
        assert_eq!(list.get(1), Some(Value::num(2.0)));
    }

    #[test]
    fn test_list_clear() {
        let mut list = ObjList::new();
        list.add(Value::num(1.0));
        list.add(Value::num(2.0));
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_mixed_values() {
        let mut list = ObjList::new();
        list.add(Value::num(42.0));
        list.add(Value::bool(true));
        list.add(Value::null());
        assert_eq!(list.len(), 3);
        assert!(list.get(0).unwrap().is_num());
        assert!(list.get(1).unwrap().is_bool());
        assert!(list.get(2).unwrap().is_null());
    }

    // -- ObjMap -------------------------------------------------------------

    #[test]
    fn test_map_crud() {
        let mut map = ObjMap::new();
        assert!(map.is_empty());

        map.set(Value::num(1.0), Value::num(10.0));
        map.set(Value::num(2.0), Value::num(20.0));
        assert_eq!(map.len(), 2);

        assert_eq!(map.get(Value::num(1.0)), Some(Value::num(10.0)));
        assert_eq!(map.get(Value::num(3.0)), None);

        assert!(map.contains(Value::num(1.0)));
        assert!(!map.contains(Value::num(3.0)));

        map.remove(Value::num(1.0));
        assert_eq!(map.len(), 1);
        assert!(!map.contains(Value::num(1.0)));
    }

    #[test]
    fn test_map_overwrite() {
        let mut map = ObjMap::new();
        map.set(Value::num(1.0), Value::num(10.0));
        map.set(Value::num(1.0), Value::num(99.0));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(Value::num(1.0)), Some(Value::num(99.0)));
    }

    #[test]
    fn test_map_bool_null_keys() {
        let mut map = ObjMap::new();
        map.set(Value::bool(true), Value::num(1.0));
        map.set(Value::bool(false), Value::num(2.0));
        map.set(Value::null(), Value::num(3.0));

        assert_eq!(map.get(Value::bool(true)), Some(Value::num(1.0)));
        assert_eq!(map.get(Value::bool(false)), Some(Value::num(2.0)));
        assert_eq!(map.get(Value::null()), Some(Value::num(3.0)));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn test_map_clear() {
        let mut map = ObjMap::new();
        map.set(Value::num(1.0), Value::num(10.0));
        map.clear();
        assert!(map.is_empty());
    }

    // -- ObjRange -----------------------------------------------------------

    #[test]
    fn test_range_inclusive() {
        let r = ObjRange::new(1.0, 5.0, true);
        assert_eq!(r.header.obj_type, ObjType::Range);
        let vals: Vec<i64> = r.iter_integers().collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_range_exclusive() {
        let r = ObjRange::new(1.0, 5.0, false);
        let vals: Vec<i64> = r.iter_integers().collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_range_descending() {
        let r = ObjRange::new(5.0, 1.0, true);
        let vals: Vec<i64> = r.iter_integers().collect();
        assert_eq!(vals, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_range_single() {
        let r = ObjRange::new(3.0, 3.0, true);
        let vals: Vec<i64> = r.iter_integers().collect();
        assert_eq!(vals, vec![3]);
    }

    #[test]
    fn test_range_empty_exclusive() {
        let r = ObjRange::new(3.0, 3.0, false);
        let vals: Vec<i64> = r.iter_integers().collect();
        assert!(vals.is_empty());
    }

    // -- ObjClass + ObjInstance ---------------------------------------------

    #[test]
    fn test_class_creation() {
        let mut interner = test_interner();
        let name = sym(&mut interner, "MyClass");
        let class = ObjClass::new(name, std::ptr::null_mut());
        assert_eq!(class.header.obj_type, ObjType::Class);
        assert_eq!(class.name, name);
        assert!(class.superclass.is_null());
        assert!(class.methods.is_empty());
    }

    #[test]
    fn test_class_method_lookup() {
        let mut interner = test_interner();
        let class_name = sym(&mut interner, "Num");
        let method_name = sym(&mut interner, "+");

        let mut class = ObjClass::new(class_name, std::ptr::null_mut());

        // Bind a native method.
        fn native_add(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
            Value::num(0.0)
        }
        class.bind_native(method_name, native_add);

        assert!(class.find_method(method_name).is_some());
        let unknown = sym(&mut interner, "unknown");
        assert!(class.find_method(unknown).is_none());
    }

    #[test]
    fn test_class_inheritance() {
        let mut interner = test_interner();
        let parent_name = sym(&mut interner, "Parent");
        let child_name = sym(&mut interner, "Child");
        let method_name = sym(&mut interner, "greet");

        let mut parent = ObjClass::new(parent_name, std::ptr::null_mut());
        fn greet(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
            Value::null()
        }
        parent.bind_native(method_name, greet);

        // Child inherits from parent — methods should be copied.
        let parent_ptr = &mut parent as *mut ObjClass;
        let child = ObjClass::new(child_name, parent_ptr);
        assert!(child.find_method(method_name).is_some());
    }

    #[test]
    fn test_instance_fields() {
        let mut interner = test_interner();
        let name = sym(&mut interner, "Point");
        let mut class = ObjClass::new(name, std::ptr::null_mut());
        class.num_fields = 2;

        let class_ptr = &mut class as *mut ObjClass;
        let mut inst = ObjInstance::new(class_ptr);
        assert_eq!(inst.num_fields, 2);
        assert!(inst.get_field(0).unwrap().is_null());

        inst.set_field(0, Value::num(10.0));
        inst.set_field(1, Value::num(20.0));
        assert_eq!(inst.get_field(0), Some(Value::num(10.0)));
        assert_eq!(inst.get_field(1), Some(Value::num(20.0)));
        assert_eq!(inst.get_field(2), None);
    }

    // -- ObjUpvalue ---------------------------------------------------------

    #[test]
    fn test_upvalue_open() {
        let mut slot = Value::num(42.0);
        let mut uv = ObjUpvalue::new(&mut slot as *mut Value);
        assert_eq!(uv.get().as_num(), Some(42.0));

        uv.set(Value::num(99.0));
        assert_eq!(slot.as_num(), Some(99.0));
    }

    #[test]
    fn test_upvalue_close() {
        let mut slot = Value::num(42.0);
        let mut uv = ObjUpvalue::new(&mut slot as *mut Value);

        uv.close();
        // After closing, the upvalue holds its own copy.
        let _ = std::mem::replace(&mut slot, Value::num(0.0)); // original slot changes
        assert_eq!(uv.get().as_num(), Some(42.0)); // upvalue still has 42
    }

    // -- ObjFiber -----------------------------------------------------------

    #[test]
    fn test_fiber_stack() {
        let mut fiber = ObjFiber::new();
        assert_eq!(fiber.state, FiberState::New);

        fiber.push(Value::num(1.0));
        fiber.push(Value::num(2.0));
        fiber.push(Value::num(3.0));
        assert_eq!(fiber.peek(0).as_num(), Some(3.0));
        assert_eq!(fiber.peek(1).as_num(), Some(2.0));

        let v = fiber.pop();
        assert_eq!(v.as_num(), Some(3.0));
        assert_eq!(fiber.stack.len(), 2);
    }

    #[test]
    fn test_fiber_states() {
        let mut fiber = ObjFiber::new();
        assert_eq!(fiber.state, FiberState::New);
        fiber.state = FiberState::Running;
        assert_eq!(fiber.state, FiberState::Running);
        fiber.state = FiberState::Suspended;
        assert_eq!(fiber.state, FiberState::Suspended);
        fiber.state = FiberState::Done;
        assert_eq!(fiber.state, FiberState::Done);
    }

    // -- ObjModule ----------------------------------------------------------

    #[test]
    fn test_module_variables() {
        let mut interner = test_interner();
        let mod_name = sym(&mut interner, "main");
        let var_name = sym(&mut interner, "x");

        let mut module = ObjModule::new(mod_name);
        let idx = module.define_variable(var_name, Value::num(42.0));
        assert_eq!(idx, 0);
        assert_eq!(module.find_variable(var_name), Some(0));
        assert_eq!(module.variables[0].as_num(), Some(42.0));

        let unknown = sym(&mut interner, "unknown");
        assert_eq!(module.find_variable(unknown), None);
    }

    // -- ObjForeign ---------------------------------------------------------

    #[test]
    fn test_foreign_data() {
        let data = vec![1, 2, 3, 4];
        let foreign = ObjForeign::new(data.clone());
        assert_eq!(foreign.header.obj_type, ObjType::Foreign);
        assert_eq!(foreign.data, data);
    }

    // -- MapKey / Value as hash key -----------------------------------------

    #[test]
    fn test_value_as_map_key() {
        let mut map = ObjMap::new();
        // Numeric keys
        map.set(Value::num(1.0), Value::bool(true));
        map.set(Value::num(2.0), Value::bool(false));
        assert_eq!(map.get(Value::num(1.0)), Some(Value::bool(true)));

        // Bool and null keys
        map.set(Value::bool(true), Value::num(100.0));
        map.set(Value::null(), Value::num(200.0));
        assert_eq!(map.get(Value::bool(true)), Some(Value::num(100.0)));
        assert_eq!(map.get(Value::null()), Some(Value::num(200.0)));
    }

    // -- Header + type tag --------------------------------------------------

    #[test]
    fn test_obj_header_type_tags() {
        assert_eq!(ObjString::new("".into()).header.obj_type, ObjType::String);
        assert_eq!(ObjList::new().header.obj_type, ObjType::List);
        assert_eq!(ObjMap::new().header.obj_type, ObjType::Map);
        assert_eq!(
            ObjRange::new(0.0, 1.0, true).header.obj_type,
            ObjType::Range
        );
        assert_eq!(ObjFiber::new().header.obj_type, ObjType::Fiber);
        assert_eq!(ObjForeign::new(vec![]).header.obj_type, ObjType::Foreign);
    }

    #[test]
    fn test_gc_mark_default() {
        let s = ObjString::new("test".into());
        assert_eq!(s.header.gc_mark, 0); // white by default
    }
}
