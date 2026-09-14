//! The runtime seam: the memory under the Immix strategy, the fiber
//! stacks a host's collector scans, and the run a host guards, as one
//! atomic function-pointer slot per operation.
//!
//! Every slot starts as wren_lift's own block allocator (`gc_immix_heap`)
//! and is replaced entry by entry through [`wlift_rt_install`]; a `None`
//! entry keeps wren_lift's. `ImmixGc` reaches memory only through the
//! dispatchers here, so a host that fills the table owns the memory
//! without wren_lift naming the host anywhere. What stays wren_lift's
//! whatever hosts it: `ObjHeader` and every object layout, the precise
//! trace through the object visitors, the intern table, the VM's root
//! gathering. The two entry points a host's own collector needs from
//! wren_lift, `object_trace` and `object_drop`, are slots wren_lift
//! fills; a host reads them back with [`wlift_rt_object_trace`] and
//! [`wlift_rt_object_drop`].
//!
//! Dispatch is one relaxed load and an indirect call. That is sound only
//! because installation is refused once a heap exists: before that point
//! nothing but the installing thread runs, and every thread created later
//! observes the stores through its spawn.
//!
//! Every heap slot takes the handle `heap_new` returned. It is opaque to
//! wren_lift and minted once per Immix VM, so the default heap is per VM;
//! a host with one heap may return any token. Whatever the handle, the
//! slots that yield addresses (`containing_allocation`, `scan_range`,
//! `for_each_allocation`) must yield only allocations made through
//! `alloc_raw` on that handle: wren_lift reads an `ObjHeader` at every
//! address they return.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// Bumped whenever a slot is added, removed or changes signature.
pub const RT_VERSION: u32 = 2;

/// Largest size `alloc_raw` is ever asked for.
pub const MAX_ALLOC: usize = 32 * 1024;

/// Callback for the slots that yield allocations: the allocation's start
/// and the caller's context.
pub type Visit = unsafe extern "C" fn(*mut u8, *mut c_void);

/// `object_trace`'s shape, for a host storing it.
pub type ObjectTrace = unsafe extern "C" fn(*mut u8, Visit, *mut c_void);

/// `object_drop`'s shape, for a host storing it.
pub type ObjectDrop = unsafe extern "C" fn(*mut u8);

/// What the memory side reports; `GcStats` takes its byte counters from
/// here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtStats {
    /// Bytes the heap has reserved.
    pub heap_bytes: usize,
    /// Bytes live after the last cycle.
    pub live_bytes: usize,
    /// Bytes handed out since the heap was created.
    pub allocated_bytes: usize,
    /// Bytes reclaimed by sweeps since the heap was created.
    pub freed_bytes: usize,
    /// Allocations reclaimed by sweeps since the heap was created,
    /// dropped or not.
    pub freed_objects: usize,
}

macro_rules! runtime_table {
    ($( $(#[$doc:meta])* $name:ident ( $($arg:ident : $ty:ty),* ) $(-> $ret:ty)? = $default:expr ; )*) => {
        /// Entry points a host may replace. See the module doc.
        ///
        /// `version` and `size` say which table a host was built against;
        /// [`wlift_rt_install`] refuses either mismatching.
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct RuntimeVTable {
            pub version: u32,
            pub size: u32,
            $( $(#[$doc])* pub $name: Option<unsafe extern "C" fn($($ty),*) $(-> $ret)?>, )*
        }

        impl RuntimeVTable {
            /// Every entry `None`: installing it changes nothing.
            pub const fn new() -> Self {
                Self {
                    version: RT_VERSION,
                    size: std::mem::size_of::<Self>() as u32,
                    $( $name: None, )*
                }
            }
        }

        impl Default for RuntimeVTable {
            fn default() -> Self {
                Self::new()
            }
        }

        // Slots are named for the entry they hold.
        #[allow(non_upper_case_globals)]
        mod slot {
            use super::*;
            $( pub static $name: AtomicPtr<()> = AtomicPtr::new($default as *mut ()); )*
        }

        /// The dispatchers, C-typed. `ImmixGc` calls these and never
        /// `gc_immix_heap` directly.
        pub mod call {
            use super::*;
            $(
                /// The installed slot.
                ///
                /// # Safety
                /// The slot's contract on [`RuntimeVTable`].
                #[inline(always)]
                pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                    let f: unsafe extern "C" fn($($ty),*) $(-> $ret)? =
                        std::mem::transmute(slot::$name.load(Ordering::Relaxed));
                    f($($arg),*)
                }
            )*
        }

        unsafe fn install_entries(table: &RuntimeVTable) {
            $( if let Some(f) = table.$name {
                slot::$name.store(f as *mut (), Ordering::Release);
            } )*
        }
    };
}

runtime_table! {
    // ── Memory ──────────────────────────────────────────────────────────
    /// A heap. The handle is opaque to wren_lift and passed to every
    /// other memory slot.
    heap_new() -> *mut c_void = immix::heap_new;
    /// Release `heap`. wren_lift has dropped every object in it first.
    heap_drop(heap: *mut c_void) = immix::heap_drop;
    /// `size` bytes, 16-aligned, at most `MAX_ALLOC`; null when the heap
    /// is exhausted. Contents are unspecified: wren_lift writes an object
    /// before reading it.
    alloc_raw(heap: *mut c_void, size: usize) -> *mut u8 = immix::alloc_raw;
    /// `alloc_raw` for an object that owns nothing outside the heap:
    /// the sweep reclaims it without `object_drop`.
    alloc_plain(heap: *mut c_void, size: usize) -> *mut u8 = immix::alloc_plain;
    /// Start of the allocation containing `addr`, null if none.
    containing_allocation(heap: *mut c_void, addr: usize) -> *mut u8 = immix::containing_allocation;
    /// Whether `addr` lies inside an allocation.
    is_heap_ptr(heap: *mut c_void, addr: usize) -> bool = immix::is_heap_ptr;
    /// Claim the allocation at `ptr` live for the open cycle; false when
    /// it already was.
    mark_allocation(heap: *mut c_void, ptr: *mut u8) -> bool = immix::mark_allocation;
    /// Whether `ptr` has been claimed in the open cycle.
    is_marked(heap: *mut c_void, ptr: *mut u8) -> bool = immix::is_marked;
    /// `visit(start, ctx)` for every allocation a word in `lo..hi` may
    /// point at. A word is a candidate as a raw address and, on 64-bit
    /// targets, as a NaN-boxed object `Value`: its low 48 bits when its
    /// top 14 bits are all set.
    scan_range(heap: *mut c_void, lo: usize, hi: usize, visit: Visit, ctx: *mut c_void) = immix::scan_range;
    /// Charge `bytes` held outside the heap against the next collection.
    track_external(heap: *mut c_void, bytes: usize) = immix::track_external;
    /// Have `object_drop` run for the plain allocation at `ptr` once a
    /// cycle finds it dead, as if it had been allocated with `alloc_raw`;
    /// false when `ptr` is not a plain allocation of this heap.
    watch(heap: *mut c_void, ptr: *mut u8) -> bool = immix::watch;
    /// Whether the mutator should collect now.
    should_collect(heap: *mut c_void) -> bool = immix::should_collect;
    /// Open a cycle: no allocation is claimed.
    collect_begin(heap: *mut c_void) = immix::collect_begin;
    /// Close the cycle: `object_drop` every unclaimed allocation, recycle
    /// its memory, and return the live bytes.
    collect_end(heap: *mut c_void) -> usize = immix::collect_end;
    /// `visit(start, ctx)` for every allocation; `visit` must not
    /// allocate from `heap`.
    for_each_allocation(heap: *mut c_void, visit: Visit, ctx: *mut c_void) = immix::for_each_allocation;
    stats(heap: *mut c_void, out: *mut RtStats) = immix::stats;
    // ── Stacks ──────────────────────────────────────────────────────────
    // A fiber's stack is a root a host's collector scans itself, from
    // where the stack is suspended. wren_lift's own collectors find the
    // stacks through krio at collection time, so its defaults do nothing.
    /// A fiber's stack came to be: `[base, base + size)`, under the id
    /// krio gave the fiber.
    stack_new(id: u64, base: usize, size: usize) = stacks::stack_new;
    /// The stack `id` is suspended with its stack pointer at `sp`. `0`
    /// is the thread's own stack, suspended while a fiber runs.
    stack_suspended(id: u64, sp: usize) = stacks::stack_suspended;
    /// The stack `id` is about to be freed.
    stack_drop(id: u64) = stacks::stack_drop;
    // ── Runs ────────────────────────────────────────────────────────────
    /// Run `body(ctx)` as one run of `vm`'s active fiber. A host whose
    /// code the run calls into, and which leaves that code by a long
    /// jump, lands here instead of somewhere above: it puts back what it
    /// keeps per thread, raises the error on `vm` as a runtime error,
    /// and answers false. True when `body` returned.
    run_guarded(vm: *mut c_void, body: unsafe extern "C" fn(*mut c_void), ctx: *mut c_void) -> bool = stacks::run_guarded;
    // ── wren_lift's, for a host's collector ─────────────────────────────
    /// `mark(child, ctx)` for every object `obj` refers to.
    object_trace(obj: *mut u8, mark: Visit, ctx: *mut c_void) = wren::object_trace;
    /// Release what the dead object `obj` owns outside the heap. Its
    /// intern-table entry is unlinked only while the VM that allocated it
    /// is closing a cycle on the calling thread.
    object_drop(obj: *mut u8) = wren::object_drop;
}

pub use call::{
    alloc_plain, alloc_raw, collect_begin, collect_end, containing_allocation, for_each_allocation,
    heap_drop, is_heap_ptr, is_marked, mark_allocation, object_drop, object_trace, run_guarded,
    scan_range, should_collect, stack_drop, stack_new, stack_suspended, track_external, watch,
};

/// Whether the built-in heap serves the memory slots, so a handle is an
/// `ImmixHeap` whose bump region compiled code may advance itself.
pub fn heap_is_builtin() -> bool {
    slot::alloc_raw.load(Ordering::Relaxed) == immix::alloc_raw as *mut ()
        && slot::alloc_plain.load(Ordering::Relaxed) == immix::alloc_plain as *mut ()
        && slot::heap_new.load(Ordering::Relaxed) == immix::heap_new as *mut ()
        && slot::collect_end.load(Ordering::Relaxed) == immix::collect_end as *mut ()
}

/// The bump region of a built-in heap handle, for compiled code.
///
/// # Safety
/// `heap` must be a handle `heap_new` returned while `heap_is_builtin`.
pub unsafe fn bump_region(heap: *mut c_void) -> *const ImmixHeapBump {
    (*(heap as *const super::gc_immix_heap::ImmixHeap)).bump_region()
}

pub use super::gc_immix_heap::BumpRegion as ImmixHeapBump;

/// Mint a heap, sealing the table: from here on a slot may be in use.
#[inline(always)]
pub fn heap_new() -> *mut c_void {
    SEALED.store(true, Ordering::Release);
    unsafe { call::heap_new() }
}

/// The memory side's counters for `heap`.
///
/// # Safety
/// `heap` must be a handle `heap_new` returned and `heap_drop` has not
/// released.
#[inline(always)]
pub unsafe fn stats(heap: *mut c_void) -> RtStats {
    let mut out = RtStats::default();
    call::stats(heap, &mut out);
    out
}

// ── Installation ────────────────────────────────────────────────────────

static INSTALLED: AtomicBool = AtomicBool::new(false);
/// Set by the first `heap_new`: the point past which a slot may be in
/// use on any thread, and installation is refused.
static SEALED: AtomicBool = AtomicBool::new(false);

/// Copy the `Some` entries of `table` into the dispatch slots.
///
/// Returns false, changing nothing, when `table` is null, was built
/// against another table version or size, a table was already
/// installed, or a heap already exists -- an Immix VM has been created.
///
/// # Safety
/// `table` must be null or point at a readable `RuntimeVTable`, and every
/// `Some` entry must have the slot's signature and contract.
#[no_mangle]
pub unsafe extern "C" fn wlift_rt_install(table: *const RuntimeVTable) -> bool {
    if table.is_null() {
        return false;
    }
    let table = &*table;
    if table.version != RT_VERSION || table.size as usize != std::mem::size_of::<RuntimeVTable>() {
        return false;
    }
    if SEALED.load(Ordering::Acquire) || INSTALLED.load(Ordering::Acquire) {
        return false;
    }
    install_entries(table);
    INSTALLED.store(true, Ordering::Release);
    true
}

/// Whether a host table has been installed.
#[no_mangle]
pub extern "C" fn wlift_rt_installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// What the `object_trace` slot dispatches to: wren_lift's per-object
/// trace, for a host that traces wren_lift objects itself.
#[no_mangle]
pub extern "C" fn wlift_rt_object_trace() -> ObjectTrace {
    unsafe { std::mem::transmute(slot::object_trace.load(Ordering::Relaxed)) }
}

/// What the `object_drop` slot dispatches to: wren_lift's per-object
/// drop, for a host that reclaims wren_lift objects itself.
#[no_mangle]
pub extern "C" fn wlift_rt_object_drop() -> ObjectDrop {
    unsafe { std::mem::transmute(slot::object_drop.load(Ordering::Relaxed)) }
}

// ── wren_lift's own implementations, C-shaped ───────────────────────────

/// The stack slots' defaults: nothing, since wren_lift's collectors walk
/// krio's fibers when they scan.
mod stacks {
    use std::ffi::c_void;

    pub unsafe extern "C" fn stack_new(_id: u64, _base: usize, _size: usize) {}
    pub unsafe extern "C" fn stack_suspended(_id: u64, _sp: usize) {}
    pub unsafe extern "C" fn stack_drop(_id: u64) {}

    /// Nothing leaves wren_lift's own runs by a long jump.
    pub unsafe extern "C" fn run_guarded(
        _vm: *mut c_void,
        body: unsafe extern "C" fn(*mut c_void),
        ctx: *mut c_void,
    ) -> bool {
        body(ctx);
        true
    }
}

/// Adapters from the C slot signatures to `gc_immix_heap`.
mod immix {
    use super::*;
    use crate::runtime::gc_immix_heap::ImmixHeap;

    #[inline(always)]
    unsafe fn heap<'a>(heap: *mut c_void) -> &'a mut ImmixHeap {
        &mut *(heap as *mut ImmixHeap)
    }

    pub unsafe extern "C" fn heap_new() -> *mut c_void {
        Box::into_raw(Box::new(ImmixHeap::new())) as *mut c_void
    }

    pub unsafe extern "C" fn heap_drop(heap: *mut c_void) {
        drop(Box::from_raw(heap as *mut ImmixHeap));
    }

    pub unsafe extern "C" fn alloc_raw(heap: *mut c_void, size: usize) -> *mut u8 {
        self::heap(heap).alloc_raw(size)
    }

    pub unsafe extern "C" fn alloc_plain(heap: *mut c_void, size: usize) -> *mut u8 {
        self::heap(heap).alloc_plain(size)
    }

    pub unsafe extern "C" fn containing_allocation(heap: *mut c_void, addr: usize) -> *mut u8 {
        self::heap(heap).containing_allocation(addr)
    }

    pub unsafe extern "C" fn is_heap_ptr(heap: *mut c_void, addr: usize) -> bool {
        self::heap(heap).is_heap_ptr(addr)
    }

    pub unsafe extern "C" fn mark_allocation(heap: *mut c_void, ptr: *mut u8) -> bool {
        self::heap(heap).mark(ptr)
    }

    pub unsafe extern "C" fn is_marked(heap: *mut c_void, ptr: *mut u8) -> bool {
        self::heap(heap).is_marked(ptr)
    }

    pub unsafe extern "C" fn scan_range(
        heap: *mut c_void,
        lo: usize,
        hi: usize,
        visit: Visit,
        ctx: *mut c_void,
    ) {
        self::heap(heap).scan_range(lo, hi, |start| visit(start, ctx));
    }

    pub unsafe extern "C" fn track_external(heap: *mut c_void, bytes: usize) {
        self::heap(heap).track_external(bytes);
    }

    pub unsafe extern "C" fn watch(heap: *mut c_void, ptr: *mut u8) -> bool {
        self::heap(heap).watch(ptr)
    }

    pub unsafe extern "C" fn should_collect(heap: *mut c_void) -> bool {
        self::heap(heap).should_collect()
    }

    pub unsafe extern "C" fn collect_begin(heap: *mut c_void) {
        self::heap(heap).collect_begin()
    }

    pub unsafe extern "C" fn collect_end(heap: *mut c_void) -> usize {
        self::heap(heap).collect_end(|dead| call::object_drop(dead))
    }

    pub unsafe extern "C" fn for_each_allocation(
        heap: *mut c_void,
        visit: Visit,
        ctx: *mut c_void,
    ) {
        self::heap(heap).for_each_allocation(|start| visit(start, ctx));
    }

    pub unsafe extern "C" fn stats(heap: *mut c_void, out: *mut RtStats) {
        *out = self::heap(heap).stats();
    }
}

/// Adapters from the C slot signatures to `gc_immix`.
mod wren {
    use super::*;

    pub unsafe extern "C" fn object_trace(obj: *mut u8, mark: Visit, ctx: *mut c_void) {
        crate::runtime::gc_immix::object_trace(obj, |child| mark(child, ctx));
    }

    pub unsafe extern "C" fn object_drop(obj: *mut u8) {
        crate::runtime::gc_immix::object_drop(obj);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::engine::InterpretResult;
    use crate::runtime::gc_trait::GcStrategy;
    use crate::runtime::vm::{VMConfig, VM};
    use std::sync::atomic::AtomicUsize;

    fn immix_vm() -> VM {
        VM::new(VMConfig {
            gc_strategy: GcStrategy::Immix,
            ..VMConfig::default()
        })
    }

    static ALLOCS: AtomicUsize = AtomicUsize::new(0);

    /// Counts, then forwards to wren_lift's.
    unsafe extern "C" fn counting_alloc_raw(heap: *mut c_void, size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        immix::alloc_raw(heap, size)
    }

    const CHILD_ENV: &str = "WLIFT_RT_SEAM_CHILD";

    /// The table is process-global and any other test's Immix VM seals
    /// it, so the install has to happen in a process of its own: the test
    /// re-runs itself with only this test selected and checks the exit.
    #[test]
    fn an_entry_installed_before_the_first_vm_is_what_immix_allocates_through() {
        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "runtime::rt::tests::an_entry_installed_before_the_first_vm_is_what_immix_allocates_through",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, "1")
                .status()
                .expect("re-run the test binary");
            assert!(status.success(), "child test process failed: {status}");
            return;
        }

        assert!(!wlift_rt_installed());
        let mut table = RuntimeVTable::new();
        table.alloc_raw = Some(counting_alloc_raw);
        assert!(unsafe { wlift_rt_install(&table) });
        assert!(wlift_rt_installed());
        // Once: a second table is refused even before a VM exists.
        assert!(!unsafe { wlift_rt_install(&table) });

        let mut vm = immix_vm();
        let before = ALLOCS.load(Ordering::SeqCst);
        let result = vm.interpret(
            "main",
            r#"
                var xs = []
                for (i in 0...2000) xs.add("s%(i)")
                System.print(xs.count)
            "#,
        );
        assert_eq!(result, InterpretResult::Success);
        assert!(ALLOCS.load(Ordering::SeqCst) >= before + 2000);

        // Sealed by the VM above; the table is untouched from here on.
        assert!(!unsafe { wlift_rt_install(&table) });
    }

    #[test]
    fn install_after_a_vm_exists_is_refused() {
        let _vm = immix_vm();
        let mut table = RuntimeVTable::new();
        table.alloc_raw = Some(counting_alloc_raw);
        assert!(!unsafe { wlift_rt_install(&table) });
    }

    #[test]
    fn a_table_from_another_version_or_size_is_refused() {
        let mut table = RuntimeVTable::new();
        table.version = RT_VERSION + 1;
        assert!(!unsafe { wlift_rt_install(&table) });
        let mut table = RuntimeVTable::new();
        table.size += 8;
        assert!(!unsafe { wlift_rt_install(&table) });
        assert!(!unsafe { wlift_rt_install(std::ptr::null()) });
    }

    #[test]
    fn the_callbacks_read_back_are_wren_lifts() {
        assert_eq!(
            wlift_rt_object_trace() as usize,
            wren::object_trace as ObjectTrace as usize
        );
        assert_eq!(
            wlift_rt_object_drop() as usize,
            wren::object_drop as ObjectDrop as usize
        );
    }
}
