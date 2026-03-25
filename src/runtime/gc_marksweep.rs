/// Simple mark-sweep garbage collector.
///
/// Non-generational, stop-the-world mark-sweep. All objects are Box-allocated
/// and tracked in a Vec. Collection walks all roots, marks reachable objects,
/// then sweeps (frees) unmarked objects.
///
/// Characteristics:
/// - Simple and predictable — no nursery, no promotion, no write barriers needed
/// - Collection time proportional to total heap size (not just live set)
/// - Good baseline for comparing against generational GC
/// - No moving/compaction — may fragment over long runs
///
/// Use case: baseline comparison, workloads where generational overhead
/// isn't justified (few short-lived objects).
use super::gc::GcStats;
use super::gc_trait::GcAllocator;
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;

use std::collections::HashMap;

// Mark constants
const WHITE: u8 = 0;
const GRAY: u8 = 1;
const BLACK: u8 = 2;

/// Threshold: collect when allocated bytes exceed this.
const INITIAL_THRESHOLD: usize = 4 * 1024 * 1024; // 4MB
/// After collection, set threshold to live_size * GROWTH_FACTOR.
const GROWTH_FACTOR: f64 = 2.0;

// ---------------------------------------------------------------------------
// Mark-Sweep GC
// ---------------------------------------------------------------------------

pub struct MarkSweepGc {
    /// All live objects.
    objects: Vec<*mut ObjHeader>,
    /// String intern table.
    intern_table: HashMap<u64, Vec<*mut ObjString>>,
    pub stats: GcStats,
    /// Bytes allocated since last collection.
    bytes_since_gc: usize,
    /// Collection threshold in bytes.
    gc_threshold: usize,
    /// Arena bump allocator for object memory.
    arena: super::gc::OldArena,
}

impl Default for MarkSweepGc {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkSweepGc {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            intern_table: HashMap::new(),
            stats: GcStats::default(),
            bytes_since_gc: 0,
            gc_threshold: INITIAL_THRESHOLD,
            arena: super::gc::OldArena::new(),
        }
    }

    fn alloc_boxed<T>(&mut self, obj: T) -> *mut T {
        let size = std::mem::size_of::<T>();
        let ptr = self.arena.alloc(obj);
        self.objects.push(ptr as *mut ObjHeader);
        self.stats.total_allocated += size;
        self.stats.objects_allocated += 1;
        self.bytes_since_gc += size;
        let total = self.objects.len();
        if total > self.stats.peak_objects {
            self.stats.peak_objects = total;
        }
        ptr
    }
}

// ---------------------------------------------------------------------------
// Mark phase — reuses the same trace_object from gc.rs patterns
// ---------------------------------------------------------------------------

fn mark_value(val: Value, gray_stack: &mut Vec<*mut ObjHeader>) {
    if val.is_object() {
        if let Some(ptr) = val.as_object() {
            let header = ptr as *mut ObjHeader;
            if !header.is_null() {
                mark_gray(header, gray_stack);
            }
        }
    }
}

fn mark_gray(header: *mut ObjHeader, gray_stack: &mut Vec<*mut ObjHeader>) {
    unsafe {
        if (*header).gc_mark != WHITE {
            return;
        }
        (*header).gc_mark = GRAY;
        gray_stack.push(header);
    }
}

unsafe fn trace_object(header: *mut ObjHeader, gray_stack: &mut Vec<*mut ObjHeader>) {
    if !(*header).class.is_null() {
        mark_gray((*header).class as *mut ObjHeader, gray_stack);
    }

    match (*header).obj_type {
        ObjType::String | ObjType::Fn | ObjType::Range | ObjType::Foreign => {}

        ObjType::List => {
            let list = &*(header as *mut ObjList);
            for &val in list.as_slice() {
                mark_value(val, gray_stack);
            }
        }

        ObjType::Map => {
            let map = &*(header as *mut ObjMap);
            for (key, &val) in &map.entries {
                mark_value(key.value(), gray_stack);
                mark_value(val, gray_stack);
            }
        }

        ObjType::Closure => {
            let closure = &*(header as *mut ObjClosure);
            if !closure.function.is_null() {
                mark_gray(closure.function as *mut ObjHeader, gray_stack);
            }
            for &uv in &closure.upvalues {
                if !uv.is_null() {
                    mark_gray(uv as *mut ObjHeader, gray_stack);
                }
            }
        }

        ObjType::Upvalue => {
            let uv = &*(header as *mut ObjUpvalue);
            mark_value(uv.closed, gray_stack);
        }

        ObjType::Fiber => {
            let fiber = &*(header as *mut ObjFiber);
            for &val in &fiber.stack {
                mark_value(val, gray_stack);
            }
            for frame in &fiber.frames {
                if !frame.closure.is_null() {
                    mark_gray(frame.closure as *mut ObjHeader, gray_stack);
                }
            }
            for frame in &fiber.mir_frames {
                for val in &frame.values {
                    mark_value(*val, gray_stack);
                }
                if let Some(closure) = frame.closure {
                    if !closure.is_null() {
                        mark_gray(closure as *mut ObjHeader, gray_stack);
                    }
                }
                if let Some(class) = frame.defining_class {
                    if !class.is_null() {
                        mark_gray(class as *mut ObjHeader, gray_stack);
                    }
                }
            }
            if !fiber.caller.is_null() {
                mark_gray(fiber.caller as *mut ObjHeader, gray_stack);
            }
            mark_value(fiber.error, gray_stack);
        }

        ObjType::Class => {
            let class = &*(header as *mut ObjClass);
            if !class.superclass.is_null() {
                mark_gray(class.superclass as *mut ObjHeader, gray_stack);
            }
            for method in class.methods.iter().flatten() {
                match method {
                    Method::Closure(ptr) | Method::Constructor(ptr) => {
                        if !ptr.is_null() {
                            mark_gray(*ptr as *mut ObjHeader, gray_stack);
                        }
                    }
                    Method::Native(_) => {}
                }
            }
            for &val in class.static_fields.values() {
                mark_value(val, gray_stack);
            }
        }

        ObjType::Instance => {
            let inst = &*(header as *mut ObjInstance);
            if !inst.fields.is_null() {
                for i in 0..inst.num_fields as usize {
                    mark_value(*inst.fields.add(i), gray_stack);
                }
            }
        }

        ObjType::Module => {
            let module = &*(header as *mut ObjModule);
            for &val in &module.variables {
                mark_value(val, gray_stack);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Drop / size helpers
// ---------------------------------------------------------------------------

unsafe fn object_size(header: *mut ObjHeader) -> usize {
    match (*header).obj_type {
        ObjType::String => std::mem::size_of::<ObjString>(),
        ObjType::List => std::mem::size_of::<ObjList>(),
        ObjType::Map => std::mem::size_of::<ObjMap>(),
        ObjType::Range => std::mem::size_of::<ObjRange>(),
        ObjType::Fn => std::mem::size_of::<ObjFn>(),
        ObjType::Closure => std::mem::size_of::<ObjClosure>(),
        ObjType::Upvalue => std::mem::size_of::<ObjUpvalue>(),
        ObjType::Fiber => std::mem::size_of::<ObjFiber>(),
        ObjType::Class => std::mem::size_of::<ObjClass>(),
        ObjType::Instance => std::mem::size_of::<ObjInstance>(),
        ObjType::Foreign => std::mem::size_of::<ObjForeign>(),
        ObjType::Module => std::mem::size_of::<ObjModule>(),
    }
}

unsafe fn drop_object(header: *mut ObjHeader) {
    // Use drop_in_place: objects are arena-allocated, not Box-allocated.
    // drop_in_place calls the destructor (freeing internal Vecs, Strings, etc.)
    // without trying to free the object's memory itself.
    match (*header).obj_type {
        ObjType::String => std::ptr::drop_in_place(header as *mut ObjString),
        ObjType::List => std::ptr::drop_in_place(header as *mut ObjList),
        ObjType::Map => std::ptr::drop_in_place(header as *mut ObjMap),
        ObjType::Range => std::ptr::drop_in_place(header as *mut ObjRange),
        ObjType::Fn => std::ptr::drop_in_place(header as *mut ObjFn),
        ObjType::Closure => std::ptr::drop_in_place(header as *mut ObjClosure),
        ObjType::Upvalue => std::ptr::drop_in_place(header as *mut ObjUpvalue),
        ObjType::Fiber => std::ptr::drop_in_place(header as *mut ObjFiber),
        ObjType::Class => std::ptr::drop_in_place(header as *mut ObjClass),
        ObjType::Instance => std::ptr::drop_in_place(header as *mut ObjInstance),
        ObjType::Foreign => std::ptr::drop_in_place(header as *mut ObjForeign),
        ObjType::Module => std::ptr::drop_in_place(header as *mut ObjModule),
    }
}

impl Drop for MarkSweepGc {
    fn drop(&mut self) {
        for &obj in &self.objects {
            unsafe {
                drop_object(obj);
            }
        }
        self.objects.clear();
    }
}

// ---------------------------------------------------------------------------
// GcAllocator implementation
// ---------------------------------------------------------------------------

impl GcAllocator for MarkSweepGc {
    fn alloc_string(&mut self, s: String) -> *mut ObjString {
        self.alloc_boxed(ObjString::new(s))
    }
    fn alloc_list(&mut self) -> *mut ObjList {
        self.alloc_boxed(ObjList::new())
    }
    fn alloc_map(&mut self) -> *mut ObjMap {
        self.alloc_boxed(ObjMap::new())
    }
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        self.alloc_boxed(ObjRange::new(from, to, inclusive))
    }
    fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        self.alloc_boxed(ObjFn::new(name, arity, upvalue_count, fn_id))
    }
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        let uv_count = if function.is_null() {
            0
        } else {
            unsafe { (*function).upvalue_count as usize }
        };
        self.alloc_boxed(ObjClosure::new(function, uv_count))
    }
    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        self.alloc_boxed(ObjUpvalue::new(location))
    }
    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        self.alloc_boxed(ObjFiber::new())
    }
    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        self.alloc_boxed(ObjClass::new(name, superclass))
    }
    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        let num_fields = if class.is_null() {
            0
        } else {
            unsafe { (*class).num_fields as usize }
        };

        // Allocate instance and fields from the arena (no malloc).
        let inst = ObjInstance::new_with_fields(class, num_fields as u32, std::ptr::null_mut());
        let ptr = self.alloc_boxed(inst);

        if num_fields > 0 {
            let fields_ptr = self.arena.alloc_bytes(
                num_fields * std::mem::size_of::<Value>(),
                std::mem::align_of::<Value>(),
            ) as *mut Value;
            unsafe {
                // Initialize fields to null
                for i in 0..num_fields {
                    fields_ptr.add(i).write(Value::null());
                }
                (*ptr).fields = fields_ptr;
                (*ptr).fields_owned = false; // arena manages the memory
            }
        }

        ptr
    }
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        self.alloc_boxed(ObjForeign::new(data))
    }
    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        self.alloc_boxed(ObjModule::new(name))
    }

    fn intern_string(&mut self, s: String) -> *mut ObjString {
        let hash = fnv1a_hash_bytes(s.as_bytes());
        if let Some(ptrs) = self.intern_table.get(&hash) {
            for &ptr in ptrs {
                if unsafe { (*ptr).value == s } {
                    return ptr;
                }
            }
        }
        let ptr = self.alloc_string(s);
        self.intern_table.entry(hash).or_default().push(ptr);
        ptr
    }

    fn write_barrier(&mut self, _source: *mut ObjHeader, _value: Value) {
        // No-op: non-generational GC doesn't need write barriers.
    }

    fn collect(&mut self, roots: &mut [Value]) {
        let start = std::time::Instant::now();

        // Mark phase.
        let mut gray_stack: Vec<*mut ObjHeader> = Vec::new();
        for &root in roots.iter() {
            mark_value(root, &mut gray_stack);
        }
        while let Some(obj) = gray_stack.pop() {
            unsafe {
                (*obj).gc_mark = BLACK;
                trace_object(obj, &mut gray_stack);
            }
        }

        // Sweep intern table: remove dead entries before sweeping objects.
        for (_, ptrs) in self.intern_table.iter_mut() {
            ptrs.retain(|&ptr| unsafe { (*(ptr as *mut ObjHeader)).gc_mark == BLACK });
        }
        self.intern_table.retain(|_, ptrs| !ptrs.is_empty());

        // Sweep phase: free unmarked, reset marks on survivors.
        let mut live: Vec<*mut ObjHeader> = Vec::with_capacity(self.objects.len());
        let mut freed_bytes: usize = 0;
        let mut freed_count: usize = 0;

        for &obj in &self.objects {
            unsafe {
                if (*obj).gc_mark == BLACK {
                    (*obj).gc_mark = WHITE;
                    live.push(obj);
                } else {
                    freed_bytes += object_size(obj);
                    freed_count += 1;
                    drop_object(obj);
                }
            }
        }

        self.stats.objects_freed += freed_count;
        self.stats.total_freed += freed_bytes;
        self.objects = live;
        self.bytes_since_gc = 0;

        // Adaptive threshold: grow based on live set size.
        let live_bytes: usize = self.objects.len() * 64; // rough estimate
        self.gc_threshold = ((live_bytes as f64 * GROWTH_FACTOR) as usize).max(INITIAL_THRESHOLD);
        self.stats.major_collections += 1;
        self.stats.gc_time_ns += start.elapsed().as_nanos() as u64;
    }

    fn should_collect(&self) -> bool {
        self.bytes_since_gc >= self.gc_threshold
    }

    fn stats(&self) -> &GcStats {
        &self.stats
    }
}

fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marksweep_alloc_and_drop() {
        let mut gc = MarkSweepGc::new();
        gc.alloc_string("hello".into());
        gc.alloc_string("world".into());
        gc.alloc_list();
        assert_eq!(gc.stats.objects_allocated, 3);
    }

    #[test]
    fn test_marksweep_intern() {
        let mut gc = MarkSweepGc::new();
        let a = gc.intern_string("hello".into());
        let b = gc.intern_string("hello".into());
        assert_eq!(a, b);
        assert_eq!(gc.stats.objects_allocated, 1);
    }

    #[test]
    fn test_marksweep_collect_frees_unreachable() {
        let mut gc = MarkSweepGc::new();
        gc.alloc_string("garbage".into());
        gc.alloc_string("garbage2".into());
        assert_eq!(gc.objects.len(), 2);
        gc.collect(&mut []);
        assert_eq!(gc.objects.len(), 0);
        assert_eq!(gc.stats.objects_freed, 2);
    }

    #[test]
    fn test_marksweep_collect_keeps_roots() {
        let mut gc = MarkSweepGc::new();
        let ptr = gc.alloc_string("keep".into());
        gc.alloc_string("discard".into());
        let mut roots = vec![Value::object(ptr as *mut u8)];
        gc.collect(&mut roots);
        assert_eq!(gc.objects.len(), 1);
        assert_eq!(gc.stats.objects_freed, 1);
    }
}
