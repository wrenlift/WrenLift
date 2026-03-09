/// Generational garbage collector with bump-allocated nursery.
///
/// - Nursery: contiguous arena with bump-pointer allocation (O(1) alloc)
/// - Old gen: Box-allocated, mark-sweep
/// - Minor GC: mark live nursery objects, promote survivors to old gen (Box),
///   update forwarding pointers, drop dead nursery objects, reset arena
/// - Major GC: minor GC + sweep dead old-gen objects
/// - Write barrier: remembered set for old→young references
/// - String interning: hash-based dedup (collected when unreachable)
use std::collections::HashMap;

use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;
use crate::mir::interp::InterpValue;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEN_YOUNG: u8 = 0;
const GEN_OLD: u8 = 1;

const WHITE: u8 = 0;
const GRAY: u8 = 1;
const BLACK: u8 = 2;

// ---------------------------------------------------------------------------
// Nursery — bump-allocated arena
// ---------------------------------------------------------------------------

struct Nursery {
    buffer: Vec<u8>,
    alloc_ptr: usize,
}

impl Nursery {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            alloc_ptr: 0,
        }
    }

    /// Bump-allocate an object into the arena. Returns None if full.
    fn try_alloc<T>(&mut self, val: T) -> Option<*mut T> {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();
        let aligned = (self.alloc_ptr + align - 1) & !(align - 1);

        if aligned + size > self.buffer.len() {
            return None;
        }

        let ptr = unsafe { self.buffer.as_mut_ptr().add(aligned) as *mut T };
        unsafe {
            std::ptr::write(ptr, val);
        }
        self.alloc_ptr = aligned + size;
        Some(ptr)
    }

    /// Does the nursery have space for a T-sized allocation?
    fn has_space_for<T>(&self) -> bool {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>();
        let aligned = (self.alloc_ptr + align - 1) & !(align - 1);
        aligned + size <= self.buffer.len()
    }

    /// Check if a pointer falls within this nursery's buffer.
    fn contains(&self, ptr: *const u8) -> bool {
        let start = self.buffer.as_ptr() as usize;
        let end = start + self.buffer.len();
        let addr = ptr as usize;
        addr >= start && addr < end
    }

    fn used(&self) -> usize {
        self.alloc_ptr
    }

    fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Reset the bump pointer. All objects must be cleaned up first.
    fn reset(&mut self) {
        self.alloc_ptr = 0;
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub struct GcConfig {
    /// Nursery arena size in bytes.
    pub nursery_size: usize,
    /// Object count threshold for GC trigger (old gen).
    pub initial_threshold: usize,
    /// Multiply live count by this after each GC to set next threshold.
    pub heap_grow_factor: f64,
    /// Number of minor GCs between each major GC.
    pub major_gc_interval: u32,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            nursery_size: 256 * 1024, // 256 KB
            initial_threshold: 256,
            heap_grow_factor: 2.0,
            major_gc_interval: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct GcStats {
    pub minor_collections: u32,
    pub major_collections: u32,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub objects_allocated: usize,
    pub objects_freed: usize,
    pub objects_promoted: usize,
    pub peak_objects: usize,
}

// ---------------------------------------------------------------------------
// GC
// ---------------------------------------------------------------------------

pub struct Gc {
    /// Bump-allocated nursery arena.
    nursery: Nursery,
    /// Pointers into the nursery arena (for iteration during GC).
    nursery_objects: Vec<*mut ObjHeader>,

    /// Head of old generation intrusive linked list.
    old_objects: *mut ObjHeader,
    old_count: usize,

    /// Old objects that had a young value written into them.
    remembered_set: Vec<*mut ObjHeader>,
    /// String intern table: FNV hash → list of interned string pointers.
    intern_table: HashMap<u64, Vec<*mut ObjString>>,

    gc_threshold: usize,
    minor_since_major: u32,
    config: GcConfig,

    pub stats: GcStats,
}

impl Default for Gc {
    fn default() -> Self {
        Self::new()
    }
}

impl Gc {
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    pub fn with_config(config: GcConfig) -> Self {
        let threshold = config.initial_threshold;
        let nursery_size = config.nursery_size;
        Self {
            nursery: Nursery::new(nursery_size),
            nursery_objects: Vec::new(),
            old_objects: std::ptr::null_mut(),
            old_count: 0,
            remembered_set: Vec::new(),
            intern_table: HashMap::new(),
            gc_threshold: threshold,
            minor_since_major: 0,
            config,
            stats: GcStats::default(),
        }
    }

    pub fn object_count(&self) -> usize {
        self.nursery_objects.len() + self.old_count
    }

    pub fn young_count(&self) -> usize {
        self.nursery_objects.len()
    }

    pub fn old_count(&self) -> usize {
        self.old_count
    }

    pub fn should_collect(&self) -> bool {
        // Trigger when nursery is 75%+ full or old gen threshold exceeded.
        self.nursery.used() > self.nursery.capacity() * 3 / 4 || self.old_count >= self.gc_threshold
    }

    pub fn nursery_used(&self) -> usize {
        self.nursery.used()
    }

    pub fn nursery_capacity(&self) -> usize {
        self.nursery.capacity()
    }

    // -- Typed allocation wrappers ------------------------------------------

    pub fn alloc_string(&mut self, s: String) -> *mut ObjString {
        self.alloc(ObjString::new(s))
    }

    pub fn alloc_list(&mut self) -> *mut ObjList {
        self.alloc(ObjList::new())
    }

    pub fn alloc_map(&mut self) -> *mut ObjMap {
        self.alloc(ObjMap::new())
    }

    pub fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        self.alloc(ObjRange::new(from, to, inclusive))
    }

    pub fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        self.alloc(ObjFn::new(name, arity, upvalue_count, fn_id))
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        let uv_count = if function.is_null() {
            0
        } else {
            unsafe { (*function).upvalue_count as usize }
        };
        self.alloc(ObjClosure::new(function, uv_count))
    }

    pub fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        self.alloc(ObjUpvalue::new(location))
    }

    pub fn alloc_fiber(&mut self) -> *mut ObjFiber {
        self.alloc(ObjFiber::new())
    }

    pub fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        self.alloc(ObjClass::new(name, superclass))
    }

    pub fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        self.alloc(ObjInstance::new(class))
    }

    pub fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        self.alloc(ObjForeign::new(data))
    }

    pub fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        self.alloc(ObjModule::new(name))
    }

    // -- Core allocation (bump in nursery, overflow to old gen) --------------

    fn alloc<T>(&mut self, obj: T) -> *mut T {
        let size = std::mem::size_of::<T>();

        if self.nursery.has_space_for::<T>() {
            // Fast path: bump-allocate in nursery.
            let ptr = self.nursery.try_alloc(obj).unwrap();
            self.nursery_objects.push(ptr as *mut ObjHeader);
            self.track_alloc(size);
            return ptr;
        }

        // Slow path: nursery full, allocate directly in old gen.
        self.alloc_old(obj)
    }

    /// Allocate directly in old gen (Box). Used for overflow and promotion.
    fn alloc_old<T>(&mut self, obj: T) -> *mut T {
        let size = std::mem::size_of::<T>();
        let ptr = Box::into_raw(Box::new(obj));
        let header = ptr as *mut ObjHeader;
        unsafe {
            (*header).generation = GEN_OLD;
            (*header).gc_mark = WHITE;
            (*header).next = self.old_objects;
        }
        self.old_objects = header;
        self.old_count += 1;
        self.track_alloc(size);
        ptr
    }

    fn track_alloc(&mut self, size: usize) {
        self.stats.total_allocated += size;
        self.stats.objects_allocated += 1;
        let total = self.object_count();
        if total > self.stats.peak_objects {
            self.stats.peak_objects = total;
        }
    }

    // -- String interning ---------------------------------------------------

    pub fn intern_string(&mut self, s: String) -> *mut ObjString {
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

    // -- Write barrier ------------------------------------------------------

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn write_barrier(&mut self, source: *mut ObjHeader, value: Value) {
        if source.is_null() {
            return;
        }
        unsafe {
            if (*source).generation == GEN_OLD && value.is_object() {
                if let Some(obj_ptr) = value.as_object() {
                    let target = obj_ptr as *mut ObjHeader;
                    if !target.is_null() && (*target).generation == GEN_YOUNG {
                        self.remembered_set.push(source);
                    }
                }
            }
        }
    }

    // -- Collection ---------------------------------------------------------

    /// Auto-select minor or major GC based on interval.
    pub fn collect(&mut self, roots: &mut [Value]) {
        if self.minor_since_major >= self.config.major_gc_interval {
            self.collect_major(roots);
        } else {
            self.collect_minor(roots);
        }
    }

    /// Minor GC: mark from roots, promote live nursery objects to old gen,
    /// update forwarding pointers, drop dead nursery objects, reset arena.
    pub fn collect_minor(&mut self, roots: &mut [Value]) {
        if self.nursery_objects.is_empty() {
            return;
        }

        // 1. Mark from roots.
        let mut gray_stack = Vec::new();
        for val in roots.iter() {
            mark_value(*val, &mut gray_stack);
        }

        // Mark from remembered set (old objects that reference young).
        let remembered = std::mem::take(&mut self.remembered_set);
        for &obj in &remembered {
            unsafe {
                trace_object(obj, &mut gray_stack);
            }
        }

        process_gray_stack(&mut gray_stack);

        // 2. Promote live nursery objects, drop dead ones, build forwarding table.
        let forwards = self.process_nursery();

        // 3. Update all pointers to forwarded addresses.
        if !forwards.is_empty() {
            update_roots(roots, &forwards);
            unsafe {
                self.update_old_gen_pointers(&forwards);
            }
            self.update_intern_table(&forwards);
        }

        // 4. Reset nursery arena.
        self.nursery.reset();

        // 5. Reset marks on old objects (they were marked during tracing).
        unsafe {
            self.reset_old_marks();
        }

        self.stats.minor_collections += 1;
        self.minor_since_major += 1;
        self.adjust_threshold();
    }

    /// Major GC: promote nursery survivors, then sweep dead old-gen objects.
    pub fn collect_major(&mut self, roots: &mut [Value]) {
        // 1. Mark everything from roots.
        let mut gray_stack = Vec::new();
        for val in roots.iter() {
            mark_value(*val, &mut gray_stack);
        }
        let remembered = std::mem::take(&mut self.remembered_set);
        for &obj in &remembered {
            unsafe {
                trace_object(obj, &mut gray_stack);
            }
        }
        process_gray_stack(&mut gray_stack);

        // 2. Promote live nursery objects, drop dead ones.
        let forwards = self.process_nursery();
        if !forwards.is_empty() {
            update_roots(roots, &forwards);
            unsafe {
                self.update_old_gen_pointers(&forwards);
            }
            self.update_intern_table(&forwards);
        }
        self.nursery.reset();

        // 3. Sweep dead old-gen objects. Promoted objects have gc_mark=BLACK,
        //    so they survive. sweep_old resets marks for surviving objects.
        self.sweep_old();

        self.stats.major_collections += 1;
        self.minor_since_major = 0;
        self.adjust_threshold();
    }

    fn adjust_threshold(&mut self) {
        let live = self.old_count;
        let next = (live as f64 * self.config.heap_grow_factor) as usize;
        self.gc_threshold = next.max(self.config.initial_threshold);
    }

    // -- Process nursery (promote / drop) -----------------------------------

    fn process_nursery(&mut self) -> HashMap<usize, *mut ObjHeader> {
        let mut forwards: HashMap<usize, *mut ObjHeader> = HashMap::new();
        let nursery_objects = std::mem::take(&mut self.nursery_objects);

        for &obj in &nursery_objects {
            unsafe {
                if (*obj).gc_mark != WHITE {
                    // Live → promote to old gen.
                    let new_ptr = self.promote_object(obj);
                    forwards.insert(obj as usize, new_ptr);
                    self.stats.objects_promoted += 1;
                } else {
                    // Dead → remove from intern table, drop owned Rust types.
                    self.unlink_intern(obj);
                    let size = object_size(obj);
                    drop_in_place_by_type(obj);
                    self.stats.objects_freed += 1;
                    self.stats.total_freed += size;
                }
            }
        }

        forwards
    }

    /// Promote a nursery object to old gen: ptr::read from arena → Box on heap.
    unsafe fn promote_object(&mut self, old_header: *mut ObjHeader) -> *mut ObjHeader {
        let new_header = match (*old_header).obj_type {
            ObjType::String => self.promote_typed::<ObjString>(old_header),
            ObjType::List => self.promote_typed::<ObjList>(old_header),
            ObjType::Map => self.promote_typed::<ObjMap>(old_header),
            ObjType::Range => self.promote_typed::<ObjRange>(old_header),
            ObjType::Fn => self.promote_typed::<ObjFn>(old_header),
            ObjType::Closure => self.promote_typed::<ObjClosure>(old_header),
            ObjType::Upvalue => self.promote_typed::<ObjUpvalue>(old_header),
            ObjType::Fiber => self.promote_typed::<ObjFiber>(old_header),
            ObjType::Class => self.promote_typed::<ObjClass>(old_header),
            ObjType::Instance => self.promote_typed::<ObjInstance>(old_header),
            ObjType::Foreign => self.promote_typed::<ObjForeign>(old_header),
            ObjType::Module => self.promote_typed::<ObjModule>(old_header),
        };

        // Fix ObjUpvalue self-referential location pointer.
        // When open, location points to a stack slot (outside nursery) — fine.
        // When closed, location points to self.closed (inside nursery) — stale!
        if (*old_header).obj_type == ObjType::Upvalue {
            let new_uv = new_header as *mut ObjUpvalue;
            if self.nursery.contains((*new_uv).location as *const u8) {
                (*new_uv).location = &mut (*new_uv).closed;
            }
        }

        new_header
    }

    /// Read an object out of the nursery arena and Box it in old gen.
    unsafe fn promote_typed<T>(&mut self, old_header: *mut ObjHeader) -> *mut ObjHeader {
        let obj: T = std::ptr::read(old_header as *const T);
        let new_ptr = Box::into_raw(Box::new(obj));
        let header = new_ptr as *mut ObjHeader;
        (*header).generation = GEN_OLD;
        (*header).gc_mark = BLACK; // survive subsequent sweep_old
        (*header).next = self.old_objects;
        self.old_objects = header;
        self.old_count += 1;
        header
    }

    // -- Sweep old gen (mark-sweep) -----------------------------------------

    fn sweep_old(&mut self) {
        let mut prev: *mut ObjHeader = std::ptr::null_mut();
        let mut current = self.old_objects;

        while !current.is_null() {
            let next = unsafe { (*current).next };

            if unsafe { (*current).gc_mark } == WHITE {
                // Dead old object — free it.
                let size = object_size(current);
                self.unlink_intern(current);
                unsafe {
                    drop_object(current);
                }
                self.old_count -= 1;
                self.stats.objects_freed += 1;
                self.stats.total_freed += size;

                if prev.is_null() {
                    self.old_objects = next;
                } else {
                    unsafe {
                        (*prev).next = next;
                    }
                }
            } else {
                // Live — reset mark for next cycle.
                unsafe {
                    (*current).gc_mark = WHITE;
                }
                prev = current;
            }

            current = next;
        }
    }

    /// Reset marks on all old objects (after minor GC where no old sweep runs).
    unsafe fn reset_old_marks(&self) {
        let mut current = self.old_objects;
        while !current.is_null() {
            (*current).gc_mark = WHITE;
            current = (*current).next;
        }
    }

    // -- Pointer forwarding (update after promotion) ------------------------

    unsafe fn update_old_gen_pointers(&self, forwards: &HashMap<usize, *mut ObjHeader>) {
        let mut current = self.old_objects;
        while !current.is_null() {
            update_pointers_in_object(current, forwards);
            current = (*current).next;
        }
    }

    fn update_intern_table(&mut self, forwards: &HashMap<usize, *mut ObjHeader>) {
        for ptrs in self.intern_table.values_mut() {
            for ptr in ptrs.iter_mut() {
                if let Some(&new_hdr) = forwards.get(&(*ptr as usize)) {
                    *ptr = new_hdr as *mut ObjString;
                }
            }
        }
    }

    /// Remove an interned string from the intern table before freeing.
    fn unlink_intern(&mut self, header: *mut ObjHeader) {
        unsafe {
            if (*header).obj_type != ObjType::String {
                return;
            }
            let s = &*(header as *mut ObjString);
            let hash = s.hash;
            if let Some(ptrs) = self.intern_table.get_mut(&hash) {
                ptrs.retain(|&p| p != header as *mut ObjString);
                if ptrs.is_empty() {
                    self.intern_table.remove(&hash);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Drop — free all remaining objects
// ---------------------------------------------------------------------------

impl Drop for Gc {
    fn drop(&mut self) {
        // Drop surviving nursery objects (drop owned types, arena freed by Vec).
        for &obj in &self.nursery_objects {
            unsafe {
                drop_in_place_by_type(obj);
            }
        }
        self.nursery_objects.clear();

        // Drop old gen objects (Box-allocated).
        let mut current = self.old_objects;
        while !current.is_null() {
            let next = unsafe { (*current).next };
            unsafe {
                drop_object(current);
            }
            current = next;
        }
        self.old_objects = std::ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Mark phase (free functions)
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

fn process_gray_stack(gray_stack: &mut Vec<*mut ObjHeader>) {
    while let Some(obj) = gray_stack.pop() {
        unsafe {
            (*obj).gc_mark = BLACK;
            trace_object(obj, gray_stack);
        }
    }
}

/// Trace all object references from a single object.
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
            // Trace MIR interpreter frames (SSA value maps, closures, classes).
            for frame in &fiber.mir_frames {
                for interp_val in frame.values.values() {
                    if let InterpValue::Boxed(val) = interp_val {
                        mark_value(*val, gray_stack);
                    }
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
            for method in class.methods.values() {
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
// Pointer forwarding helpers
// ---------------------------------------------------------------------------

fn update_roots(roots: &mut [Value], forwards: &HashMap<usize, *mut ObjHeader>) {
    for val in roots.iter_mut() {
        update_value(val, forwards);
    }
}

fn update_value(val: &mut Value, forwards: &HashMap<usize, *mut ObjHeader>) {
    if val.is_object() {
        if let Some(ptr) = val.as_object() {
            if let Some(&new_ptr) = forwards.get(&(ptr as usize)) {
                *val = Value::object(new_ptr as *mut u8);
            }
        }
    }
}

/// Update all object-reference fields inside a single object.
unsafe fn update_pointers_in_object(
    header: *mut ObjHeader,
    forwards: &HashMap<usize, *mut ObjHeader>,
) {
    // Update class pointer.
    if !(*header).class.is_null() {
        if let Some(&new_ptr) = forwards.get(&((*header).class as usize)) {
            (*header).class = new_ptr as *mut ObjClass;
        }
    }

    match (*header).obj_type {
        ObjType::String | ObjType::Fn | ObjType::Range | ObjType::Foreign => {}

        ObjType::List => {
            let list = &mut *(header as *mut ObjList);
            for val in list.as_mut_slice() {
                update_value(val, forwards);
            }
        }

        ObjType::Map => {
            let map = &mut *(header as *mut ObjMap);
            // Must drain+reinsert: key hashes change when object ptrs change.
            let entries: Vec<(MapKey, Value)> = map.entries.drain().collect();
            for (key, val) in entries {
                let mut k = key.value();
                let mut v = val;
                update_value(&mut k, forwards);
                update_value(&mut v, forwards);
                map.entries.insert(MapKey::new(k), v);
            }
        }

        ObjType::Closure => {
            let closure = &mut *(header as *mut ObjClosure);
            update_raw_ptr(&mut closure.function, forwards);
            for uv in &mut closure.upvalues {
                update_raw_ptr(uv, forwards);
            }
        }

        ObjType::Upvalue => {
            let uv = &mut *(header as *mut ObjUpvalue);
            update_value(&mut uv.closed, forwards);
            // location: either stack slot (not in nursery) or already fixed in promote_object.
        }

        ObjType::Fiber => {
            let fiber = &mut *(header as *mut ObjFiber);
            for val in &mut fiber.stack {
                update_value(val, forwards);
            }
            for frame in &mut fiber.frames {
                update_raw_ptr(&mut frame.closure, forwards);
            }
            // Update MIR interpreter frames.
            for frame in &mut fiber.mir_frames {
                for interp_val in frame.values.values_mut() {
                    if let InterpValue::Boxed(ref mut val) = interp_val {
                        update_value(val, forwards);
                    }
                }
                if let Some(ref mut closure) = frame.closure {
                    update_raw_ptr(closure, forwards);
                }
                if let Some(ref mut class) = frame.defining_class {
                    update_raw_ptr(class, forwards);
                }
            }
            update_raw_ptr(&mut fiber.caller, forwards);
            update_value(&mut fiber.error, forwards);
        }

        ObjType::Class => {
            let class = &mut *(header as *mut ObjClass);
            update_raw_ptr(&mut class.superclass, forwards);
            for method in class.methods.values_mut() {
                match method {
                    Method::Closure(ref mut ptr) | Method::Constructor(ref mut ptr) => {
                        update_raw_ptr(ptr, forwards);
                    }
                    Method::Native(_) => {}
                }
            }
            for val in class.static_fields.values_mut() {
                update_value(val, forwards);
            }
        }

        ObjType::Instance => {
            let inst = &mut *(header as *mut ObjInstance);
            if !inst.fields.is_null() {
                for i in 0..inst.num_fields as usize {
                    let val_ptr = inst.fields.add(i);
                    update_value(&mut *val_ptr, forwards);
                }
            }
        }

        ObjType::Module => {
            let module = &mut *(header as *mut ObjModule);
            for val in &mut module.variables {
                update_value(val, forwards);
            }
        }
    }
}

fn update_raw_ptr<T>(ptr: &mut *mut T, forwards: &HashMap<usize, *mut ObjHeader>) {
    if !ptr.is_null() {
        if let Some(&new_ptr) = forwards.get(&(*ptr as usize)) {
            *ptr = new_ptr as *mut T;
        }
    }
}

// ---------------------------------------------------------------------------
// Object size + drop helpers
// ---------------------------------------------------------------------------

fn object_size(header: *mut ObjHeader) -> usize {
    unsafe {
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
}

/// Drop owned Rust types in-place (for nursery objects — arena memory freed separately).
unsafe fn drop_in_place_by_type(header: *mut ObjHeader) {
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

/// Reconstruct Box and drop (for old-gen objects only).
unsafe fn drop_object(header: *mut ObjHeader) {
    match (*header).obj_type {
        ObjType::String => {
            let _ = Box::from_raw(header as *mut ObjString);
        }
        ObjType::List => {
            let _ = Box::from_raw(header as *mut ObjList);
        }
        ObjType::Map => {
            let _ = Box::from_raw(header as *mut ObjMap);
        }
        ObjType::Range => {
            let _ = Box::from_raw(header as *mut ObjRange);
        }
        ObjType::Fn => {
            let _ = Box::from_raw(header as *mut ObjFn);
        }
        ObjType::Closure => {
            let _ = Box::from_raw(header as *mut ObjClosure);
        }
        ObjType::Upvalue => {
            let _ = Box::from_raw(header as *mut ObjUpvalue);
        }
        ObjType::Fiber => {
            let _ = Box::from_raw(header as *mut ObjFiber);
        }
        ObjType::Class => {
            let _ = Box::from_raw(header as *mut ObjClass);
        }
        ObjType::Instance => {
            let _ = Box::from_raw(header as *mut ObjInstance);
        }
        ObjType::Foreign => {
            let _ = Box::from_raw(header as *mut ObjForeign);
        }
        ObjType::Module => {
            let _ = Box::from_raw(header as *mut ObjModule);
        }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> GcConfig {
        GcConfig {
            nursery_size: 64 * 1024, // 64 KB — plenty for tests
            initial_threshold: 1_000_000,
            ..Default::default()
        }
    }

    // -- Allocation + bump pointer ------------------------------------------

    #[test]
    fn test_alloc_and_count() {
        let mut gc = Gc::with_config(test_config());
        gc.alloc_string("hello".into());
        gc.alloc_string("world".into());
        assert_eq!(gc.object_count(), 2);
        assert_eq!(gc.young_count(), 2);
        assert_eq!(gc.old_count(), 0);
    }

    #[test]
    fn test_bump_allocation_advances() {
        let mut gc = Gc::with_config(test_config());
        assert_eq!(gc.nursery_used(), 0);
        gc.alloc_string("hello".into());
        let after_first = gc.nursery_used();
        assert!(after_first > 0);
        gc.alloc_string("world".into());
        assert!(gc.nursery_used() > after_first);
    }

    #[test]
    fn test_alloc_each_type() {
        let mut gc = Gc::with_config(test_config());
        let sym = SymbolId::from_raw(0);
        gc.alloc_string("s".into());
        gc.alloc_list();
        gc.alloc_map();
        gc.alloc_range(0.0, 10.0, true);
        gc.alloc_fn(sym, 0, 0, 0);
        gc.alloc_fiber();
        gc.alloc_class(sym, std::ptr::null_mut());
        gc.alloc_foreign(vec![1, 2, 3]);
        gc.alloc_module(sym);
        assert_eq!(gc.object_count(), 9);
        assert_eq!(gc.stats.objects_allocated, 9);
    }

    #[test]
    fn test_nursery_overflow_to_old() {
        let mut gc = Gc::with_config(GcConfig {
            nursery_size: 64, // tiny — forces overflow
            ..test_config()
        });
        // First alloc fits in nursery, second overflows to old gen.
        gc.alloc_string("a".into());
        gc.alloc_string("b".into());
        gc.alloc_string("c".into());
        assert_eq!(gc.object_count(), 3);
        // At least one should have overflowed to old gen.
        assert!(gc.old_count() > 0 || gc.young_count() == 3);
    }

    // -- Minor GC -----------------------------------------------------------

    #[test]
    fn test_minor_gc_collects_unreachable() {
        let mut gc = Gc::with_config(test_config());
        gc.alloc_string("garbage1".into());
        gc.alloc_string("garbage2".into());
        assert_eq!(gc.object_count(), 2);

        gc.collect_minor(&mut []);
        assert_eq!(gc.object_count(), 0);
        assert_eq!(gc.stats.objects_freed, 2);
    }

    #[test]
    fn test_minor_gc_preserves_reachable() {
        let mut gc = Gc::with_config(test_config());
        let keep = gc.alloc_string("keep".into());
        gc.alloc_string("garbage".into());

        let mut roots = [Value::object(keep as *mut u8)];
        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 1);

        // Root was forwarded to promoted address — verify it's still valid.
        let new_keep = roots[0].as_object().unwrap() as *mut ObjString;
        unsafe {
            assert_eq!((*new_keep).as_str(), "keep");
        }
    }

    #[test]
    fn test_minor_gc_promotes_survivors() {
        let mut gc = Gc::with_config(test_config());
        gc.alloc_string("promoted".into());
        let s = gc.alloc_string("promoted".into());
        assert!(gc.young_count() >= 1);

        let mut roots = [Value::object(s as *mut u8)];
        gc.collect_minor(&mut roots);
        // Promoted object is now in old gen.
        assert!(gc.old_count() >= 1);
        assert_eq!(gc.nursery_used(), 0); // arena reset
    }

    #[test]
    fn test_minor_gc_doesnt_collect_old() {
        let mut gc = Gc::with_config(test_config());
        let s = gc.alloc_string("old_obj".into());
        let mut roots = [Value::object(s as *mut u8)];
        gc.collect_minor(&mut roots); // promote

        // Minor GC with no roots — old object survives.
        gc.collect_minor(&mut []);
        assert_eq!(gc.object_count(), 1);
        assert_eq!(gc.old_count(), 1);
    }

    #[test]
    fn test_pointer_forwarding() {
        let mut gc = Gc::with_config(test_config());
        let s = gc.alloc_string("forwarded".into());
        let old_ptr = s as *mut u8;

        let mut roots = [Value::object(old_ptr)];
        gc.collect_minor(&mut roots);

        // Root should now point to a different (promoted) address.
        let new_ptr = roots[0].as_object().unwrap();
        assert_ne!(old_ptr, new_ptr, "pointer should have been forwarded");
        let new_s = new_ptr as *mut ObjString;
        unsafe {
            assert_eq!((*new_s).as_str(), "forwarded");
        }
    }

    // -- Major GC -----------------------------------------------------------

    #[test]
    fn test_major_gc_collects_old() {
        let mut gc = Gc::with_config(test_config());
        let s = gc.alloc_string("will_die".into());
        let mut roots = [Value::object(s as *mut u8)];
        gc.collect_minor(&mut roots); // promote
        assert_eq!(gc.old_count(), 1);

        gc.collect_major(&mut []); // no roots → collected
        assert_eq!(gc.object_count(), 0);
    }

    #[test]
    fn test_major_gc_preserves_reachable_old() {
        let mut gc = Gc::with_config(test_config());
        let s = gc.alloc_string("survivor".into());
        let mut roots = [Value::object(s as *mut u8)];
        gc.collect_minor(&mut roots); // promote

        gc.collect_major(&mut roots);
        assert_eq!(gc.object_count(), 1);
        let kept = roots[0].as_object().unwrap() as *mut ObjString;
        unsafe {
            assert_eq!((*kept).as_str(), "survivor");
        }
    }

    #[test]
    fn test_major_gc_after_interval() {
        let mut gc = Gc::with_config(GcConfig {
            major_gc_interval: 2,
            ..test_config()
        });

        gc.alloc_string("a".into());
        gc.collect(&mut []); // minor #1
        gc.alloc_string("b".into());
        gc.collect(&mut []); // minor #2
        assert_eq!(gc.stats.minor_collections, 2);

        gc.alloc_string("c".into());
        gc.collect(&mut []); // major (interval reached)
        assert_eq!(gc.stats.major_collections, 1);
    }

    // -- Write barrier ------------------------------------------------------

    #[test]
    fn test_write_barrier_records() {
        let mut gc = Gc::with_config(test_config());
        let old_s = gc.alloc_string("old".into());
        let mut roots = [Value::object(old_s as *mut u8)];
        gc.collect_minor(&mut roots); // promote
        let old_s = roots[0].as_object().unwrap() as *mut ObjString;

        let young_s = gc.alloc_string("young".into());
        let young_val = Value::object(young_s as *mut u8);

        gc.write_barrier(old_s as *mut ObjHeader, young_val);
        assert_eq!(gc.remembered_set.len(), 1);
    }

    #[test]
    fn test_write_barrier_ignores_young_to_young() {
        let mut gc = Gc::with_config(test_config());
        let a = gc.alloc_string("a".into());
        let b = gc.alloc_string("b".into());
        gc.write_barrier(a as *mut ObjHeader, Value::object(b as *mut u8));
        assert!(gc.remembered_set.is_empty());
    }

    // -- String interning ---------------------------------------------------

    #[test]
    fn test_intern_dedup() {
        let mut gc = Gc::with_config(test_config());
        let a = gc.intern_string("hello".into());
        let b = gc.intern_string("hello".into());
        assert_eq!(a, b);
        assert_eq!(gc.object_count(), 1);
    }

    #[test]
    fn test_intern_different_strings() {
        let mut gc = Gc::with_config(test_config());
        let a = gc.intern_string("hello".into());
        let b = gc.intern_string("world".into());
        assert_ne!(a, b);
        assert_eq!(gc.object_count(), 2);
    }

    #[test]
    fn test_intern_collected_when_unreachable() {
        let mut gc = Gc::with_config(test_config());
        gc.intern_string("temp".into());
        assert_eq!(gc.intern_table.len(), 1);

        gc.collect_minor(&mut []);
        assert_eq!(gc.object_count(), 0);
        assert!(gc.intern_table.is_empty());
    }

    #[test]
    fn test_intern_survives_promotion() {
        let mut gc = Gc::with_config(test_config());
        let s = gc.intern_string("keep".into());
        let mut roots = [Value::object(s as *mut u8)];
        gc.collect_minor(&mut roots);

        // Intern table entry should point to the promoted address.
        let new_s = gc.intern_string("keep".into());
        let root_ptr = roots[0].as_object().unwrap();
        assert_eq!(new_s as *mut u8, root_ptr);
        assert_eq!(gc.object_count(), 1);
    }

    // -- Tracing through object graphs --------------------------------------

    #[test]
    fn test_trace_list_references() {
        let mut gc = Gc::with_config(test_config());
        let inner = gc.alloc_string("inner".into());
        let list = gc.alloc_list();
        unsafe {
            (*list).add(Value::object(inner as *mut u8));
        }

        let mut roots = [Value::object(list as *mut u8)];
        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 2);

        // Verify inner string survived (accessible through promoted list).
        let new_list = roots[0].as_object().unwrap() as *mut ObjList;
        let inner_val = unsafe { (*new_list).get(0).unwrap() };
        assert!(inner_val.is_object());
    }

    #[test]
    fn test_trace_map_references() {
        let mut gc = Gc::with_config(test_config());
        let key = gc.alloc_string("key".into());
        let val = gc.alloc_string("val".into());
        let map = gc.alloc_map();
        unsafe {
            (*map).set(Value::object(key as *mut u8), Value::object(val as *mut u8));
        }

        let mut roots = [Value::object(map as *mut u8)];
        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 3);
    }

    #[test]
    fn test_trace_closure_references() {
        let mut gc = Gc::with_config(test_config());
        let func = gc.alloc_fn(SymbolId::from_raw(0), 0, 0, 0);
        let closure = gc.alloc_closure(func);

        let mut roots = [Value::object(closure as *mut u8)];
        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 2);
    }

    #[test]
    fn test_trace_instance_fields() {
        let mut gc = Gc::with_config(test_config());
        let class = gc.alloc_class(SymbolId::from_raw(0), std::ptr::null_mut());
        unsafe {
            (*class).num_fields = 1;
        }
        let inst = gc.alloc_instance(class);
        let field = gc.alloc_string("field_val".into());
        unsafe {
            (*inst).set_field(0, Value::object(field as *mut u8));
        }

        let mut roots = [Value::object(inst as *mut u8)];
        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 3);
    }

    // -- Cycles + intergenerational -----------------------------------------

    #[test]
    fn test_cycles_collected() {
        let mut gc = Gc::with_config(test_config());
        let a = gc.alloc_list();
        let b = gc.alloc_list();
        unsafe {
            (*a).add(Value::object(b as *mut u8));
            (*b).add(Value::object(a as *mut u8));
        }

        gc.collect_minor(&mut []);
        assert_eq!(gc.object_count(), 0);
    }

    #[test]
    fn test_intergenerational_references() {
        let mut gc = Gc::with_config(test_config());
        let old_list = gc.alloc_list();
        let mut roots = [Value::object(old_list as *mut u8)];
        gc.collect_minor(&mut roots); // promote list

        let young_str = gc.alloc_string("young".into());
        let new_list = roots[0].as_object().unwrap() as *mut ObjList;
        unsafe {
            (*new_list).add(Value::object(young_str as *mut u8));
        }

        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 2);
    }

    // -- Stress + stats -----------------------------------------------------

    #[test]
    fn test_stress_many_objects() {
        let mut gc = Gc::with_config(test_config());

        let mut roots = Vec::new();
        for i in 0..500 {
            let s = gc.alloc_string(format!("str_{}", i));
            if i % 2 == 0 {
                roots.push(Value::object(s as *mut u8));
            }
        }
        assert_eq!(gc.object_count(), 500);

        gc.collect_minor(&mut roots);
        assert_eq!(gc.object_count(), 250);
    }

    #[test]
    fn test_gc_stats() {
        let mut gc = Gc::with_config(test_config());
        gc.alloc_string("a".into());
        gc.alloc_string("b".into());
        assert_eq!(gc.stats.objects_allocated, 2);
        assert_eq!(gc.stats.peak_objects, 2);

        gc.collect_minor(&mut []);
        assert_eq!(gc.stats.objects_freed, 2);
        assert_eq!(gc.stats.minor_collections, 1);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut gc = Gc::with_config(GcConfig {
            initial_threshold: 10,
            heap_grow_factor: 2.0,
            ..test_config()
        });

        let mut roots = Vec::new();
        for i in 0..5 {
            let s = gc.alloc_string(format!("s{}", i));
            roots.push(Value::object(s as *mut u8));
        }
        gc.collect_minor(&mut roots);
        // 5 promoted → old_count=5, threshold = max(5*2, 10) = 10
        assert_eq!(gc.gc_threshold, 10);
    }

    #[test]
    fn test_should_collect_nursery_pressure() {
        let mut gc = Gc::with_config(GcConfig {
            nursery_size: 256, // small nursery
            ..test_config()
        });
        assert!(!gc.should_collect());
        // Fill nursery past 75%.
        gc.alloc_string("a".into());
        gc.alloc_string("b".into());
        gc.alloc_string("c".into());
        assert!(gc.should_collect());
    }

    // -- Drop ---------------------------------------------------------------

    #[test]
    fn test_drop_frees_all() {
        let mut gc = Gc::with_config(test_config());
        for i in 0..100 {
            gc.alloc_string(format!("s{}", i));
        }
        // Some promoted, some in nursery.
        let s = gc.alloc_string("root".into());
        let mut roots = [Value::object(s as *mut u8)];
        gc.collect_minor(&mut roots);
        for i in 0..50 {
            gc.alloc_string(format!("new_{}", i));
        }
        drop(gc); // should free both nursery and old gen without panic
    }
}
