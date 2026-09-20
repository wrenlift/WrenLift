//! The Immix strategy: precise marking through wren_lift's object
//! visitors over whatever memory the runtime table (`rt`) provides,
//! wren_lift's own block allocator (`gc_immix_heap`) unless a host
//! installed its own.
//!
//! The strategy owns the objects: every allocation is an `ObjHeader`-led
//! object it wrote, the trace from the VM's roots and the conservative
//! stack ranges runs through `gc::for_each_child`, the intern table is
//! here, and a dead object releases its Rust-owned containers through
//! `rt::object_drop`. Memory is the table's: allocation, resolving an
//! address to its allocation, the liveness claim per cycle, the scan of
//! a stack range, the collection trigger and the sweep. Nothing moves,
//! so there are no forwarding pointers, no write barrier, and roots are
//! read but never written.

use super::gc::{self, GcStats};
use super::gc_trait::GcAllocator;
use super::object::*;
use super::rt::{self, MAX_ALLOC};
use super::value::Value;
use crate::intern::SymbolId;
use crate::portable_time::Instant;

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct ImmixGc {
    /// The table's heap handle; opaque here.
    heap: *mut c_void,
    /// Locked only once the program has several threads.
    intern_table: Mutex<HashMap<u64, Vec<*mut ObjString>>>,
    /// Object counters and cycle timing; the byte counters come from
    /// the memory side at `stats()`.
    stats: GcStats,
    object_count: usize,
    /// The program has several threads: allocations are counted
    /// atomically and the tables are locked.
    threaded: bool,
    threaded_allocs: AtomicUsize,
    /// A sweep freed a class, closure, function or module, so
    /// address-keyed caches must be dropped.
    freed_code_objects: bool,
    /// Every fiber allocated and not yet swept, so a collection finds
    /// the fiber stacks without walking the heap.
    fibers: Mutex<Vec<*mut ObjFiber>>,
}

thread_local! {
    /// The collector closing a cycle on this thread: whose intern table
    /// `object_drop` unlinks from and whose counters it updates.
    static CLOSING: Cell<*mut ImmixGc> = const { Cell::new(std::ptr::null_mut()) };
    /// The heap the thread last allocated from or collected: where an
    /// object living in it grows its buffers.
    static ACTIVE_HEAP: Cell<*mut c_void> = const { Cell::new(std::ptr::null_mut()) };
}

/// Elements a list allocated without a size starts with room for.
const LIST_INLINE_CAPACITY: usize = 8;

/// The object at `owner` now owns malloc memory: if it is a plain
/// allocation of the active heap, the cycle that finds it dead drops it.
pub fn watch_malloc_owner(owner: *const u8) {
    let heap = ACTIVE_HEAP.get();
    if !heap.is_null() && rt::heap_is_builtin() {
        unsafe { rt::watch(heap, owner as *mut u8) };
    }
}

/// `bytes` of payload for the object at `owner`, as a `Buffer` object
/// in the heap that holds `owner`; null when there is no such heap,
/// the owner is not in it, or the payload is too large for it. The
/// owner must mark the buffer and leave it to the sweep.
pub fn alloc_buffer(owner: *const u8, bytes: usize) -> *mut u8 {
    let heap = ACTIVE_HEAP.get();
    if heap.is_null() || !rt::heap_is_builtin() {
        return std::ptr::null_mut();
    }
    let total = std::mem::size_of::<ObjHeader>() + bytes;
    if total > MAX_ALLOC || unsafe { !rt::is_heap_ptr(heap, owner as usize) } {
        return std::ptr::null_mut();
    }
    let p = unsafe { rt::alloc_plain(heap, total) };
    if p.is_null() {
        return p;
    }
    unsafe {
        (p as *mut ObjHeader).write(ObjHeader::new(ObjType::Buffer));
        p.add(std::mem::size_of::<ObjHeader>())
    }
}

impl Default for ImmixGc {
    fn default() -> Self {
        Self::new()
    }
}

impl ImmixGc {
    pub fn new() -> Self {
        Self {
            heap: rt::heap_new(),
            intern_table: Mutex::new(HashMap::new()),
            stats: GcStats::default(),
            object_count: 0,
            threaded: false,
            threaded_allocs: AtomicUsize::new(0),
            freed_code_objects: false,
            fibers: Mutex::new(Vec::new()),
        }
    }

    // -- Allocation ---------------------------------------------------------

    #[inline]
    fn alloc_raw(&mut self, size: usize) -> *mut u8 {
        self.count_allocation();
        ACTIVE_HEAP.set(self.heap);
        let p = unsafe { rt::alloc_raw(self.heap, size) };
        if p.is_null() {
            self.exhausted();
        }
        p
    }

    #[inline]
    fn alloc<T>(&mut self, obj: T) -> *mut T {
        let p = self.alloc_raw(std::mem::size_of::<T>()) as *mut T;
        unsafe { p.write(obj) };
        p
    }

    /// `alloc` for an object that owns nothing outside the heap.
    fn alloc_plain<T>(&mut self, obj: T) -> *mut T {
        self.count_allocation();
        ACTIVE_HEAP.set(self.heap);
        let p = unsafe { rt::alloc_plain(self.heap, std::mem::size_of::<T>()) } as *mut T;
        if p.is_null() {
            self.exhausted();
        }
        unsafe { p.write(obj) };
        p
    }

    fn count_allocation(&mut self) {
        if self.threaded {
            self.threaded_allocs.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.stats.objects_allocated += 1;
        self.object_count += 1;
        if self.object_count > self.stats.peak_objects {
            self.stats.peak_objects = self.object_count;
        }
    }

    #[cold]
    fn exhausted(&self) -> ! {
        let m = unsafe { rt::stats(self.heap) };
        panic!(
            "wren_lift: Immix heap exhausted at {} MiB (live {} MiB); raise WLIFT_GC_HEAP_MB",
            m.heap_bytes / (1024 * 1024),
            m.live_bytes / (1024 * 1024)
        );
    }

    // -- Enumeration --------------------------------------------------------

    /// Iterate every currently-allocated `ObjFiber`. See
    /// `GcImpl::for_each_fiber` for the contract.
    pub fn for_each_fiber<F: FnMut(*mut ObjFiber)>(&self, mut f: F) {
        for &fiber in lock(&self.fibers).iter() {
            f(fiber);
        }
    }

    fn for_each_object<F: FnMut(*mut ObjHeader)>(&self, mut f: F) {
        let mut visit: &mut dyn FnMut(*mut u8) = &mut |p| f(p as *mut ObjHeader);
        unsafe {
            rt::for_each_allocation(self.heap, visit_dyn, &mut visit as *mut _ as *mut c_void);
        }
    }

    /// The bump region compiled code may allocate small plain objects
    /// from, when the built-in heap is in use.
    pub fn bump_region_ptr(&self) -> usize {
        if rt::heap_is_builtin() {
            unsafe { rt::bump_region(self.heap) as usize }
        } else {
            0
        }
    }

    /// Close the compiled region: from here the program's threads
    /// each allocate through regions of their own. Only the built-in
    /// heap has one; a host's heap is its own to run.
    pub fn set_multithreaded(&mut self) {
        self.threaded = true;
        if rt::heap_is_builtin() {
            unsafe { (*(self.heap as *mut super::gc_immix_heap::ImmixHeap)).set_multithreaded() }
        }
    }

    /// Start of the allocation containing `addr`, if any.
    pub fn containing_allocation(&self, addr: usize) -> Option<*mut ObjHeader> {
        let p = unsafe { rt::containing_allocation(self.heap, addr) };
        (!p.is_null()).then_some(p as *mut ObjHeader)
    }

    pub fn track_external(&mut self, bytes: usize) {
        unsafe { rt::track_external(self.heap, bytes) };
    }

    /// True once since the last call if a sweep freed an object that
    /// inline caches or the method cache may point at.
    pub fn take_freed_code_objects(&mut self) -> bool {
        std::mem::replace(&mut self.freed_code_objects, false)
    }

    // -- Marking ------------------------------------------------------------

    /// Collect with precise `roots` plus conservative word scans of
    /// `ranges` (native stack windows, register spills).
    pub fn collect_with_ranges(&mut self, roots: &[Value], ranges: &[(usize, usize)]) {
        let start = Instant::now();
        ACTIVE_HEAP.set(self.heap);
        unsafe { rt::collect_begin(self.heap) };
        let marking = Instant::now();
        self.stats.stop_ns += (marking - start).as_nanos() as u64;
        let mut gray = Gray {
            heap: self.heap,
            builtin: rt::heap_is_builtin(),
            stack: Vec::with_capacity(1024),
        };
        for &root in roots {
            if let Some(header) = gc::object_of(root) {
                gray.claim(header);
            }
        }
        let heap = self.heap;
        let mut visit: &mut dyn FnMut(*mut u8) = &mut |start| gray.claim(start as *mut ObjHeader);
        for &(lo, hi) in ranges {
            if lo < hi {
                unsafe {
                    rt::scan_range(heap, lo, hi, visit_dyn, &mut visit as *mut _ as *mut c_void)
                };
            }
        }
        gray.drain();
        self.stats.mark_ns += marking.elapsed().as_nanos() as u64;
        self.finish_collection(start);
    }

    fn finish_collection(&mut self, start: Instant) {
        // `object_drop` reaches this collector through `CLOSING`; the
        // sweep runs on this thread and touches `self` only that way.
        let this: *mut ImmixGc = self;
        let heap = self.heap;
        let prev = CLOSING.replace(this);
        let sweeping = Instant::now();
        unsafe { rt::collect_end(heap) };
        self.stats.sweep_ns += sweeping.elapsed().as_nanos() as u64;
        CLOSING.set(prev);
        let m = unsafe { rt::stats(heap) };
        self.stats.objects_allocated += self.threaded_allocs.swap(0, Ordering::Relaxed);
        self.stats.objects_freed = m.freed_objects;
        self.object_count = self.stats.objects_allocated.saturating_sub(m.freed_objects);
        self.stats.peak_objects = self.stats.peak_objects.max(self.object_count);
        self.stats.collections += 1;
        self.stats.gc_time_ns += start.elapsed().as_nanos() as u64;
    }

    fn unlink_intern(&mut self, header: *mut ObjHeader) {
        unsafe {
            if (*header).obj_type != ObjType::String {
                return;
            }
            let s = &*(header as *mut ObjString);
            let mut table = lock(&self.intern_table);
            if let Some(ptrs) = table.get_mut(&s.hash) {
                ptrs.retain(|&p| p != header as *mut ObjString);
                if ptrs.is_empty() {
                    table.remove(&s.hash);
                }
            }
        }
    }

    #[cfg(test)]
    fn heap(&self) -> &super::gc_immix_heap::ImmixHeap {
        unsafe { &*(self.heap as *const super::gc_immix_heap::ImmixHeap) }
    }
}

/// `Visit` over a `&mut dyn FnMut(*mut u8)` passed as the context.
unsafe extern "C" fn visit_dyn(start: *mut u8, ctx: *mut c_void) {
    unsafe {
        let f = &mut *(ctx as *mut &mut dyn FnMut(*mut u8));
        f(start);
    }
}

/// The objects claimed for the open cycle and not yet traced.
struct Gray {
    heap: *mut c_void,
    /// The handle is an `ImmixHeap`, so a claim is a direct call rather
    /// than one through the memory slot.
    builtin: bool,
    stack: Vec<*mut ObjHeader>,
}

impl Gray {
    /// Claim `header` for this cycle and queue it for tracing if the
    /// claim is new.
    #[inline(always)]
    fn claim(&mut self, header: *mut ObjHeader) {
        let new = unsafe {
            if self.builtin {
                (*(self.heap as *mut super::gc_immix_heap::ImmixHeap)).mark(header as *mut u8)
            } else {
                rt::mark_allocation(self.heap, header as *mut u8)
            }
        };
        if new {
            self.stack.push(header);
        }
    }

    fn drain(&mut self) {
        while let Some(obj) = self.stack.pop() {
            unsafe { gc::for_each_child(obj, &mut |child| self.claim(child)) };
        }
    }
}

// ---------------------------------------------------------------------------
// The slots wren_lift fills
// ---------------------------------------------------------------------------

/// `mark` every object `obj` refers to.
///
/// # Safety
/// `obj` must be a live `ObjHeader`-led object.
pub(super) unsafe fn object_trace<F: FnMut(*mut u8)>(obj: *mut u8, mut mark: F) {
    unsafe {
        gc::for_each_child(obj as *mut ObjHeader, &mut |child| mark(child as *mut u8));
    }
}

fn lookup_interned(
    table: &HashMap<u64, Vec<*mut ObjString>>,
    hash: u64,
    s: &str,
) -> Option<*mut ObjString> {
    let ptrs = table.get(&hash)?;
    ptrs.iter()
        .copied()
        .find(|&ptr| unsafe { (*ptr).value == s })
}

/// Release what the dead object `obj` owns outside the heap. While the
/// collector that allocated it is closing a cycle on this thread, also
/// unlink it from the intern table and count it.
///
/// # Safety
/// `obj` must be a dead `ObjHeader`-led object that no live object
/// refers to; it is not touched again.
pub(super) unsafe fn object_drop(obj: *mut u8) {
    unsafe {
        let header = obj as *mut ObjHeader;
        let closing = CLOSING.get();
        if !closing.is_null() {
            let gc = &mut *closing;
            match (*header).obj_type {
                ObjType::Class | ObjType::Closure | ObjType::Fn | ObjType::Module => {
                    gc.freed_code_objects = true;
                }
                ObjType::Fiber => lock(&gc.fibers).retain(|&f| f as *mut ObjHeader != header),
                ObjType::String => gc.unlink_intern(header),
                _ => {}
            }
        }
        gc::drop_in_place_by_type(header);
    }
}

impl Drop for ImmixGc {
    fn drop(&mut self) {
        if ACTIVE_HEAP.get() == self.heap {
            ACTIVE_HEAP.set(std::ptr::null_mut());
        }
        let mut all = Vec::new();
        self.for_each_object(|h| all.push(h));
        // Nothing is closing: the intern table goes with `self` and the
        // counters stop here.
        let prev = CLOSING.replace(std::ptr::null_mut());
        for h in all {
            unsafe { rt::object_drop(h as *mut u8) };
        }
        CLOSING.set(prev);
        unsafe { rt::heap_drop(self.heap) };
    }
}

// ---------------------------------------------------------------------------
// GcAllocator implementation
// ---------------------------------------------------------------------------

impl GcAllocator for ImmixGc {
    fn alloc_string(&mut self, s: String) -> *mut ObjString {
        self.alloc(ObjString::new(s))
    }
    fn alloc_list(&mut self) -> *mut ObjList {
        self.alloc_list_sized(LIST_INLINE_CAPACITY)
    }
    /// The elements follow the header in the same allocation while
    /// they fit a line; growth moves them to a buffer object. The list
    /// is plain: it owns nothing outside the heap unless a buffer too
    /// large for the heap makes it watched.
    fn alloc_list_sized(&mut self, cap: usize) -> *mut ObjList {
        // An empty literal is usually about to be filled.
        let cap = if cap == 0 { LIST_INLINE_CAPACITY } else { cap };
        let total = std::mem::size_of::<ObjList>() + cap * std::mem::size_of::<Value>();
        if total > MAX_ALLOC {
            return self.alloc(ObjList::new());
        }
        self.count_allocation();
        ACTIVE_HEAP.set(self.heap);
        let p = unsafe { rt::alloc_plain(self.heap, total) } as *mut ObjList;
        if p.is_null() {
            self.exhausted();
        }
        unsafe {
            let mut list = ObjList::new();
            list.elements = (p as *mut u8).add(std::mem::size_of::<ObjList>()) as *mut Value;
            list.capacity = cap as u32;
            list.header.flags |= FLAG_HEAP_BUFFER;
            p.write(list);
        }
        p
    }
    fn alloc_map(&mut self) -> *mut ObjMap {
        self.alloc(ObjMap::new())
    }
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        self.alloc_plain(ObjRange::new(from, to, inclusive))
    }
    fn alloc_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> *mut ObjTypedArray {
        self.alloc(ObjTypedArray::new(count, kind))
    }
    fn alloc_simd(&mut self, kind: SimdKind, lanes: [u32; 4]) -> *mut ObjSimd {
        self.alloc_plain(ObjSimd::new(kind, lanes))
    }
    fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        self.alloc(ObjFn::new(name, arity, upvalue_count, fn_id))
    }
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        let uv_count = if function.is_null() {
            0
        } else {
            unsafe { (*function).upvalue_count as usize }
        };
        self.alloc(ObjClosure::new(function, uv_count))
    }
    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        self.alloc(ObjUpvalue::new(location))
    }
    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        let f = self.alloc(ObjFiber::new());
        lock(&self.fibers).push(f);
        f
    }
    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        self.alloc(ObjClass::new(name, superclass))
    }
    /// Instance header and field array are one allocation; the fields
    /// start right after the header and are not freed on drop.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        let num_fields = if class.is_null() {
            0
        } else {
            unsafe { (*class).num_fields as usize }
        };
        let header_size = std::mem::size_of::<ObjInstance>();
        let total = header_size + num_fields * std::mem::size_of::<Value>();
        // Compiled code addresses the fields after the header.
        debug_assert!(total <= MAX_ALLOC);
        self.count_allocation();
        ACTIVE_HEAP.set(self.heap);
        let p = unsafe { rt::alloc_plain(self.heap, total) } as *mut ObjInstance;
        if p.is_null() {
            self.exhausted();
        }
        let fields = if num_fields > 0 {
            let f = unsafe { (p as *mut u8).add(header_size) as *mut Value };
            // Written one slot at a time: a few fields are the common
            // case, and a pattern fill of that size costs a call.
            let mut i = 0;
            while i < num_fields {
                unsafe { f.add(i).write(std::hint::black_box(Value::null())) };
                i += 1;
            }
            f
        } else {
            std::ptr::null_mut()
        };
        unsafe {
            p.write(ObjInstance::new_with_fields(
                class,
                num_fields as u32,
                fields,
            ));
        }
        p
    }
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        self.alloc(ObjForeign::new(data))
    }
    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        self.alloc(ObjModule::new(name))
    }

    fn intern_string(&mut self, s: String) -> *mut ObjString {
        let hash = fnv1a_hash_bytes(s.as_bytes());
        // The lock is taken only when other threads can be in the
        // table; alone, the table is this thread's.
        let found = if self.threaded {
            lookup_interned(&lock(&self.intern_table), hash, &s)
        } else {
            lookup_interned(
                self.intern_table
                    .get_mut()
                    .unwrap_or_else(|e| e.into_inner()),
                hash,
                &s,
            )
        };
        if let Some(ptr) = found {
            return ptr;
        }
        let ptr = self.alloc_string(s);
        if self.threaded {
            lock(&self.intern_table).entry(hash).or_default().push(ptr);
        } else {
            self.intern_table
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .entry(hash)
                .or_default()
                .push(ptr);
        }
        ptr
    }

    fn collect(&mut self, roots: &[Value]) {
        self.collect_with_ranges(roots, &[]);
    }

    #[inline(always)]
    fn should_collect(&self) -> bool {
        unsafe { rt::should_collect(self.heap) }
    }

    fn stats(&self) -> GcStats {
        let m = unsafe { rt::stats(self.heap) };
        GcStats {
            total_allocated: m.allocated_bytes,
            total_freed: m.freed_bytes,
            ..self.stats
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

#[cfg(test)]
mod tests {
    use super::super::gc_immix_heap::LINE_SIZE;
    use super::*;

    fn roots_of(ptrs: &[*mut ObjHeader]) -> Vec<Value> {
        ptrs.iter().map(|&p| Value::object(p as *mut u8)).collect()
    }

    #[test]
    fn collect_frees_unreachable_and_keeps_roots() {
        let mut gc = ImmixGc::new();
        let keep = gc.alloc_list() as *mut ObjHeader;
        for _ in 0..50_000 {
            gc.alloc_string("garbage".to_string());
        }
        let before = gc.object_count;
        let roots = roots_of(&[keep]);
        gc.collect(&roots);
        assert_eq!(gc.object_count, 1);
        assert!(gc.stats.objects_freed >= before - 1);
        assert_eq!(unsafe { (*keep).obj_type }, ObjType::List);
        assert_eq!(gc.stats().collections, 1);
        assert!(gc.stats().total_freed > 0);
        // Reuse the freed lines.
        for _ in 0..50_000 {
            gc.alloc_string("again".to_string());
        }
        gc.collect(&roots);
        assert_eq!(gc.object_count, 1);
    }

    #[test]
    fn instance_fields_are_inline() {
        let mut gc = ImmixGc::new();
        let class = gc.alloc_class(SymbolId::from_raw(0), std::ptr::null_mut());
        unsafe { (*class).num_fields = 3 };
        let inst = gc.alloc_instance(class);
        unsafe {
            assert!(!(*inst).fields_owned);
            assert_eq!(
                (*inst).fields as usize,
                inst as usize + std::mem::size_of::<ObjInstance>()
            );
            (*inst).set_field(2, Value::num(7.0));
            assert_eq!((*inst).get_field(2).unwrap().as_num(), Some(7.0));
        }
        let roots = roots_of(&[class as *mut ObjHeader, inst as *mut ObjHeader]);
        gc.collect(&roots);
        assert_eq!(unsafe { (*inst).get_field(2).unwrap().as_num() }, Some(7.0));
    }

    #[test]
    fn interned_strings_survive_and_dead_ones_unlink() {
        let mut gc = ImmixGc::new();
        let a = gc.intern_string("hello".to_string());
        let b = gc.intern_string("hello".to_string());
        assert_eq!(a, b);
        let _dead = gc.intern_string("bye".to_string());
        let roots = roots_of(&[a as *mut ObjHeader]);
        gc.collect(&roots);
        assert_eq!(gc.intern_string("hello".to_string()), a);
        let bye_hash = fnv1a_hash_bytes(b"bye");
        assert!(
            !lock(&gc.intern_table).contains_key(&bye_hash),
            "dead interned string still in the table"
        );
    }

    #[test]
    fn interior_pointers_resolve_to_their_allocation() {
        let mut gc = ImmixGc::new();
        let small = gc.alloc_range(0.0, 1.0, false) as usize;
        let fiber = gc.alloc_fiber() as usize;
        let size = std::mem::size_of::<ObjRange>();
        for off in [0, 8, size - 1] {
            assert_eq!(
                gc.containing_allocation(small + off).map(|h| h as usize),
                Some(small)
            );
        }
        let fsize = std::mem::size_of::<ObjFiber>();
        for off in [0, 200, fsize - 1] {
            assert_eq!(
                gc.containing_allocation(fiber + off).map(|h| h as usize),
                Some(fiber)
            );
        }
        let span_end = fiber + fsize.div_ceil(LINE_SIZE) * LINE_SIZE;
        assert_eq!(
            gc.containing_allocation(span_end - 1).map(|h| h as usize),
            Some(fiber)
        );
        assert_eq!(gc.containing_allocation(0x1000), None);
    }

    #[test]
    fn conservative_scan_keeps_boxed_raw_and_interior_words() {
        let mut gc = ImmixGc::new();
        let boxed = gc.alloc_list() as *mut ObjHeader;
        let raw = gc.alloc_map() as *mut ObjHeader;
        let interior = gc.alloc_fiber() as *mut ObjHeader;
        let dead = gc.alloc_string("dead".to_string()) as *mut ObjHeader;
        let words: [usize; 4] = [
            Value::object(boxed as *mut u8).to_bits() as usize,
            raw as usize,
            interior as usize + 100,
            0xdead_beef,
        ];
        let lo = words.as_ptr() as usize;
        let hi = lo + std::mem::size_of_val(&words);
        gc.collect_with_ranges(&[], &[(lo, hi)]);
        let mut live = Vec::new();
        gc.for_each_object(|h| live.push(h));
        assert!(live.contains(&boxed));
        assert!(live.contains(&raw));
        assert!(live.contains(&interior));
        assert!(!live.contains(&dead));
    }

    #[test]
    fn children_keep_their_referents_and_code_objects_are_noticed() {
        let mut gc = ImmixGc::new();
        let list = gc.alloc_list();
        let held = gc.alloc_string("held".to_string());
        unsafe { (*list).add(Value::object(held as *mut u8)) };
        let _dead_class = gc.alloc_class(SymbolId::from_raw(0), std::ptr::null_mut());
        let roots = roots_of(&[list as *mut ObjHeader]);
        gc.collect(&roots);
        assert!(gc.take_freed_code_objects());
        assert!(!gc.take_freed_code_objects());
        let mut live = Vec::new();
        gc.for_each_object(|h| live.push(h));
        assert_eq!(live.len(), 2);
        assert!(live.contains(&(held as *mut ObjHeader)));
    }

    #[test]
    fn object_trace_marks_each_child_once_per_reference() {
        let mut gc = ImmixGc::new();
        let list = gc.alloc_list();
        let a = gc.alloc_string("a".to_string()) as *mut u8;
        let b = gc.alloc_string("b".to_string()) as *mut u8;
        unsafe {
            (*list).add(Value::object(a));
            (*list).add(Value::object(b));
            (*list).add(Value::object(a));
        }
        let mut seen = Vec::new();
        unsafe { object_trace(list as *mut u8, |child| seen.push(child)) };
        assert_eq!(seen, vec![a, b, a]);
    }

    #[test]
    fn for_each_fiber_finds_fibers() {
        let mut gc = ImmixGc::new();
        let f1 = gc.alloc_fiber();
        let _ = gc.alloc_list();
        let f2 = gc.alloc_fiber();
        let mut seen = Vec::new();
        gc.for_each_fiber(|f| seen.push(f));
        assert_eq!(seen, vec![f1, f2]);
    }

    #[test]
    fn the_default_heap_is_per_collector() {
        let mut a = ImmixGc::new();
        let mut b = ImmixGc::new();
        let in_a = a.alloc_list() as usize;
        let in_b = b.alloc_list() as usize;
        assert!(a.containing_allocation(in_a).is_some());
        assert!(a.containing_allocation(in_b).is_none());
        assert!(b.containing_allocation(in_b).is_some());
        assert!(!a.heap().is_heap_ptr(in_b));
    }
}
