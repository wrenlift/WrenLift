//! The runtime seam: the memory under the Immix strategy, the fiber
//! stacks a host's collector scans, the threads that run Wren and when
//! each is safe, the run a host guards, and the world the scheduler's
//! tasks and waits go to, as one atomic function-pointer slot per
//! operation.
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
pub const RT_VERSION: u32 = 6;

/// Largest size `alloc_raw` is ever asked for.
pub const MAX_ALLOC: usize = 32 * 1024;

/// Callback for the slots that yield allocations: the allocation's start
/// and the caller's context.
pub type Visit = unsafe extern "C" fn(*mut u8, *mut c_void);

/// `object_trace`'s shape, for a host storing it.
pub type ObjectTrace = unsafe extern "C" fn(*mut u8, Visit, *mut c_void);

/// `object_drop`'s shape, for a host storing it.
pub type ObjectDrop = unsafe extern "C" fn(*mut u8);

/// `host_stop`'s shape, for a host storing it.
pub type HostStop = unsafe extern "C" fn(bool);

/// `task_step`'s shape, for a host storing it.
pub type TaskStep = unsafe extern "C" fn(*mut c_void) -> bool;

/// `task_suspend`'s shape, for a host storing it.
pub type TaskSuspend = unsafe extern "C" fn() -> bool;

/// A deadline as the world slots carry it: nanoseconds from now, or
/// this for none.
pub const NO_DEADLINE: u64 = u64::MAX;

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
                pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? { unsafe {
                    let f: unsafe extern "C" fn($($ty),*) $(-> $ret)? =
                        std::mem::transmute(slot::$name.load(Ordering::Relaxed));
                    f($($arg),*)
                }}
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
    /// The thread is switching from the stack `from` to the stack `to`,
    /// both by krio's id, 0 the thread's own: what a host keeps per
    /// stack, a chain of frames of its own, goes with it. Told before
    /// the switch and again, the other way, after the return.
    stack_switch(from: u64, to: u64) = stacks::stack_switch;
    // ── Threads ─────────────────────────────────────────────────────────
    // A host's collector stops every thread that touches its heap. These
    // tell it which threads run Wren and when each is safe: a safe thread
    // is in a wait or a native call, its stack published from `sp` up,
    // and the collector need not wait for it. wren_lift's own collectors
    // keep the same facts in `stw`, so the defaults do nothing.
    /// This thread will run Wren on the program's heap: a worker of the
    /// pool, from its start until `thread_stop`.
    thread_start() = threads::thread_start;
    /// This thread runs Wren no more.
    thread_stop() = threads::thread_stop;
    /// This thread is safe until `thread_running`, its stack standing at
    /// `sp`, and `[extra_lo, extra_hi)` a second range to scan: the
    /// registers saved when it was stopped at a compiled loop header,
    /// empty when both are 0.
    thread_safe(sp: usize, extra_lo: usize, extra_hi: usize) = threads::thread_safe;
    /// This thread runs Wren again. A host whose collection is under way
    /// holds the thread here until it is done.
    thread_running() = threads::thread_running;
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
    /// Have every thread running Wren reach a safepoint (`on`), or let
    /// them go: what a host's collector asks when it stops the world,
    /// and undoes once its collection is over. A thread reaching one
    /// passes through `thread_safe` and `thread_running`.
    host_stop(on: bool) = wren::host_stop;
    /// wren_lift's world asked every thread of a program to stop (`on`),
    /// or let them go. A host whose threads reach safepoints of its own
    /// while they run its code has them come to wren_lift's there: a
    /// thread that polls neither wren_lift's page nor its interpreter
    /// loop is otherwise waited for.
    host_poll(on: bool) = threads::host_poll;
    /// Run one turn of the task `task` (a context `world_spawn` was
    /// given): up to its next park or yield. False once the task is
    /// done, and the context is released with it. Called on the world
    /// the task was placed on, from no task.
    task_step(task: *mut c_void) -> bool = wren::task_step;
    /// Suspend the task being stepped from inside its step, through
    /// wren_lift's own switch, so its step returns to the host's world
    /// as at a park; the host resumes it with the next `task_step`. False
    /// when the calling stack cannot leave from here.
    task_suspend() -> bool = wren::task_suspend;
    // ── World ───────────────────────────────────────────────────────────
    // The scheduler's world: where its tasks run and its waits park. A
    // host with a world of its own fills these so a Wren fiber or thread
    // is a task of that world beside the host's; the defaults are
    // wren_lift's own world, one per view (`sched::local`). Every slot
    // takes the view `vm` it is asked on. A deadline is nanoseconds from
    // now, `NO_DEADLINE` for none.
    /// A fresh wait token for the calling context.
    world_waiter_new(vm: *mut c_void) -> u64 = world::waiter_new;
    /// Forget a token that will not be parked on.
    world_waiter_discard(vm: *mut c_void, token: u64) = world::waiter_discard;
    /// Notify `token`; false when it is unknown or already resolved.
    /// From any thread.
    world_wake(token: u64) -> bool = world::wake;
    /// Before a park: 1 when `token` was woken already, which consumes
    /// it; 0 when it waits; -1 when it is not one this world may park.
    world_waiter_ready(vm: *mut c_void, token: u64) -> i32 = world::waiter_ready;
    /// Record the stepped task's park on `token`; it takes effect when
    /// the task yields back to the world.
    world_park_request(vm: *mut c_void, token: u64, deadline_ns: u64) = world::park_request;
    /// Whether the stepped task has a park recorded.
    world_park_pending(vm: *mut c_void) -> bool = world::park_pending;
    /// How the stepped task's last park ended: woken, or timed out.
    world_resume_woken(vm: *mut c_void) -> bool = world::resume_woken;
    /// The park of a stack that is no task: drive the world until
    /// `token` is woken or the deadline passes. True when woken.
    world_park_drive(vm: *mut c_void, token: u64, deadline_ns: u64) -> bool = world::park_drive;
    /// Make `task` a task of this world, or of the least-loaded worker
    /// world when `on_pool`; the world steps it through `task_step`.
    world_spawn(vm: *mut c_void, task: *mut c_void, on_pool: bool) = world::spawn;
    /// Step every ready task once and keep going until none is or the
    /// deadline passes; true while the world holds live tasks.
    world_tick(vm: *mut c_void, deadline_ns: u64) -> bool = world::tick;
    /// Wait for a wake, a timer or the deadline, with nothing ready.
    world_idle(vm: *mut c_void, deadline_ns: u64) = world::idle;
    /// Tasks not yet finished on this world.
    world_live(vm: *mut c_void) -> usize = world::live;
    /// Worker worlds tasks may be placed on.
    world_workers(vm: *mut c_void) -> usize = world::workers;
}

pub use call::{
    alloc_plain, alloc_raw, collect_begin, collect_end, containing_allocation, for_each_allocation,
    heap_drop, host_poll, host_stop, is_heap_ptr, is_marked, mark_allocation, object_drop,
    object_trace, run_guarded, scan_range, should_collect, stack_drop, stack_new, stack_suspended,
    stack_switch, task_step, task_suspend, thread_running, thread_safe, thread_start, thread_stop,
    track_external, watch, world_idle, world_live, world_park_drive, world_park_pending,
    world_park_request, world_resume_woken, world_spawn, world_tick, world_waiter_discard,
    world_waiter_new, world_waiter_ready, world_wake, world_workers,
};

/// Whether wren_lift's own world serves the world slots.
pub fn world_is_builtin() -> bool {
    slot::world_spawn.load(Ordering::Relaxed) == world::spawn as *mut ()
}

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
    unsafe { (*(heap as *const super::gc_immix_heap::ImmixHeap)).bump_region() }
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
    unsafe {
        let mut out = RtStats::default();
        call::stats(heap, &mut out);
        out
    }
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_rt_install(table: *const RuntimeVTable) -> bool {
    unsafe {
        if table.is_null() {
            return false;
        }
        let table = &*table;
        if table.version != RT_VERSION
            || table.size as usize != std::mem::size_of::<RuntimeVTable>()
        {
            return false;
        }
        if SEALED.load(Ordering::Acquire) || INSTALLED.load(Ordering::Acquire) {
            return false;
        }
        install_entries(table);
        INSTALLED.store(true, Ordering::Release);
        true
    }
}

/// Whether a host table has been installed.
#[unsafe(no_mangle)]
pub extern "C" fn wlift_rt_installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// What the `object_trace` slot dispatches to: wren_lift's per-object
/// trace, for a host that traces wren_lift objects itself.
#[unsafe(no_mangle)]
pub extern "C" fn wlift_rt_object_trace() -> ObjectTrace {
    unsafe { std::mem::transmute(slot::object_trace.load(Ordering::Relaxed)) }
}

/// What the `object_drop` slot dispatches to: wren_lift's per-object
/// drop, for a host that reclaims wren_lift objects itself.
#[unsafe(no_mangle)]
pub extern "C" fn wlift_rt_object_drop() -> ObjectDrop {
    unsafe { std::mem::transmute(slot::object_drop.load(Ordering::Relaxed)) }
}

/// What the `host_stop` slot dispatches to: how a host's collector asks
/// every thread running Wren to reach a safepoint.
#[unsafe(no_mangle)]
pub extern "C" fn wlift_rt_host_stop() -> HostStop {
    unsafe { std::mem::transmute(slot::host_stop.load(Ordering::Relaxed)) }
}

/// What the `task_step` slot dispatches to: one turn of a task, for a
/// host's world that runs Wren's tasks.
#[unsafe(no_mangle)]
pub extern "C" fn wlift_rt_task_step() -> TaskStep {
    unsafe { std::mem::transmute(slot::task_step.load(Ordering::Relaxed)) }
}

/// What the `task_suspend` slot dispatches to: the switch a host's world
/// suspends a Wren task by from inside its step.
#[unsafe(no_mangle)]
pub extern "C" fn wlift_rt_task_suspend() -> TaskSuspend {
    unsafe { std::mem::transmute(slot::task_suspend.load(Ordering::Relaxed)) }
}

// ── wren_lift's own implementations, C-shaped ───────────────────────────

/// The thread slots' defaults: nothing, since wren_lift's own world
/// keeps the same facts.
mod threads {
    pub unsafe extern "C" fn thread_start() {}
    pub unsafe extern "C" fn thread_stop() {}
    pub unsafe extern "C" fn thread_safe(_sp: usize, _extra_lo: usize, _extra_hi: usize) {}
    pub unsafe extern "C" fn thread_running() {}
    pub unsafe extern "C" fn host_poll(_on: bool) {}
}

/// The world slots' defaults: wren_lift's own world on the view.
#[cfg(feature = "host")]
mod world {
    use std::ffi::c_void;

    use crate::runtime::sched::local;
    use crate::runtime::vm::VM;

    pub unsafe extern "C" fn waiter_new(vm: *mut c_void) -> u64 {
        unsafe { local::waiter_new(vm as *mut VM) }
    }
    pub unsafe extern "C" fn waiter_discard(vm: *mut c_void, token: u64) {
        unsafe { local::waiter_discard(vm as *mut VM, token) }
    }
    pub unsafe extern "C" fn wake(token: u64) -> bool {
        local::wake(token)
    }
    pub unsafe extern "C" fn waiter_ready(vm: *mut c_void, token: u64) -> i32 {
        unsafe { local::waiter_ready(vm as *mut VM, token) }
    }
    pub unsafe extern "C" fn park_request(vm: *mut c_void, token: u64, deadline_ns: u64) {
        unsafe { local::park_request(vm as *mut VM, token, deadline_ns) }
    }
    pub unsafe extern "C" fn park_pending(vm: *mut c_void) -> bool {
        unsafe { local::park_pending(vm as *mut VM) }
    }
    pub unsafe extern "C" fn resume_woken(vm: *mut c_void) -> bool {
        unsafe { local::resume_woken(vm as *mut VM) }
    }
    pub unsafe extern "C" fn park_drive(vm: *mut c_void, token: u64, deadline_ns: u64) -> bool {
        unsafe { local::park_drive(vm as *mut VM, token, deadline_ns) }
    }
    pub unsafe extern "C" fn spawn(vm: *mut c_void, task: *mut c_void, on_pool: bool) {
        unsafe {
            local::spawn(
                vm as *mut VM,
                task as *mut crate::runtime::sched::TaskCtx,
                on_pool,
            )
        }
    }
    pub unsafe extern "C" fn tick(vm: *mut c_void, deadline_ns: u64) -> bool {
        unsafe { local::tick(vm as *mut VM, deadline_ns) }
    }
    pub unsafe extern "C" fn idle(vm: *mut c_void, deadline_ns: u64) {
        unsafe { local::idle(vm as *mut VM, deadline_ns) }
    }
    pub unsafe extern "C" fn live(vm: *mut c_void) -> usize {
        unsafe { local::live(vm as *mut VM) }
    }
    pub unsafe extern "C" fn workers(vm: *mut c_void) -> usize {
        unsafe { local::workers(vm as *mut VM) }
    }
}

/// Without a host there is no scheduler: every world slot is a no-op.
#[cfg(not(feature = "host"))]
mod world {
    use std::ffi::c_void;

    pub unsafe extern "C" fn waiter_new(_vm: *mut c_void) -> u64 {
        0
    }
    pub unsafe extern "C" fn waiter_discard(_vm: *mut c_void, _token: u64) {}
    pub unsafe extern "C" fn wake(_token: u64) -> bool {
        false
    }
    pub unsafe extern "C" fn waiter_ready(_vm: *mut c_void, _token: u64) -> i32 {
        -1
    }
    pub unsafe extern "C" fn park_request(_vm: *mut c_void, _token: u64, _deadline_ns: u64) {}
    pub unsafe extern "C" fn park_pending(_vm: *mut c_void) -> bool {
        false
    }
    pub unsafe extern "C" fn resume_woken(_vm: *mut c_void) -> bool {
        false
    }
    pub unsafe extern "C" fn park_drive(_vm: *mut c_void, _token: u64, _deadline_ns: u64) -> bool {
        false
    }
    pub unsafe extern "C" fn spawn(_vm: *mut c_void, _task: *mut c_void, _on_pool: bool) {}
    pub unsafe extern "C" fn tick(_vm: *mut c_void, _deadline_ns: u64) -> bool {
        false
    }
    pub unsafe extern "C" fn idle(_vm: *mut c_void, _deadline_ns: u64) {}
    pub unsafe extern "C" fn live(_vm: *mut c_void) -> usize {
        0
    }
    pub unsafe extern "C" fn workers(_vm: *mut c_void) -> usize {
        0
    }
}

mod stacks {
    use std::ffi::c_void;

    pub unsafe extern "C" fn stack_new(_id: u64, _base: usize, _size: usize) {}
    pub unsafe extern "C" fn stack_suspended(_id: u64, _sp: usize) {}
    pub unsafe extern "C" fn stack_drop(_id: u64) {}
    pub unsafe extern "C" fn stack_switch(_from: u64, _to: u64) {}

    /// Nothing leaves wren_lift's own runs by a long jump.
    pub unsafe extern "C" fn run_guarded(
        _vm: *mut c_void,
        body: unsafe extern "C" fn(*mut c_void),
        ctx: *mut c_void,
    ) -> bool {
        unsafe {
            body(ctx);
            true
        }
    }
}

/// Adapters from the C slot signatures to `gc_immix_heap`.
mod immix {
    use super::*;
    use crate::runtime::gc_immix_heap::ImmixHeap;

    #[inline(always)]
    unsafe fn heap<'a>(heap: *mut c_void) -> &'a mut ImmixHeap {
        unsafe { &mut *(heap as *mut ImmixHeap) }
    }

    pub unsafe extern "C" fn heap_new() -> *mut c_void {
        Box::into_raw(Box::new(ImmixHeap::new())) as *mut c_void
    }

    pub unsafe extern "C" fn heap_drop(heap: *mut c_void) {
        unsafe {
            drop(Box::from_raw(heap as *mut ImmixHeap));
        }
    }

    pub unsafe extern "C" fn alloc_raw(heap: *mut c_void, size: usize) -> *mut u8 {
        unsafe { self::heap(heap).alloc_raw(size) }
    }

    pub unsafe extern "C" fn alloc_plain(heap: *mut c_void, size: usize) -> *mut u8 {
        unsafe { self::heap(heap).alloc_plain(size) }
    }

    pub unsafe extern "C" fn containing_allocation(heap: *mut c_void, addr: usize) -> *mut u8 {
        unsafe { self::heap(heap).containing_allocation(addr) }
    }

    pub unsafe extern "C" fn is_heap_ptr(heap: *mut c_void, addr: usize) -> bool {
        unsafe { self::heap(heap).is_heap_ptr(addr) }
    }

    pub unsafe extern "C" fn mark_allocation(heap: *mut c_void, ptr: *mut u8) -> bool {
        unsafe { self::heap(heap).mark(ptr) }
    }

    pub unsafe extern "C" fn is_marked(heap: *mut c_void, ptr: *mut u8) -> bool {
        unsafe { self::heap(heap).is_marked(ptr) }
    }

    pub unsafe extern "C" fn scan_range(
        heap: *mut c_void,
        lo: usize,
        hi: usize,
        visit: Visit,
        ctx: *mut c_void,
    ) {
        unsafe {
            self::heap(heap).scan_range(lo, hi, |start| visit(start, ctx));
        }
    }

    pub unsafe extern "C" fn track_external(heap: *mut c_void, bytes: usize) {
        unsafe {
            self::heap(heap).track_external(bytes);
        }
    }

    pub unsafe extern "C" fn watch(heap: *mut c_void, ptr: *mut u8) -> bool {
        unsafe { self::heap(heap).watch(ptr) }
    }

    pub unsafe extern "C" fn should_collect(heap: *mut c_void) -> bool {
        unsafe { self::heap(heap).should_collect() }
    }

    pub unsafe extern "C" fn collect_begin(heap: *mut c_void) {
        unsafe { self::heap(heap).collect_begin() }
    }

    pub unsafe extern "C" fn collect_end(heap: *mut c_void) -> usize {
        unsafe { self::heap(heap).collect_end(|dead| call::object_drop(dead)) }
    }

    pub unsafe extern "C" fn for_each_allocation(
        heap: *mut c_void,
        visit: Visit,
        ctx: *mut c_void,
    ) {
        unsafe {
            self::heap(heap).for_each_allocation(|start| visit(start, ctx));
        }
    }

    pub unsafe extern "C" fn stats(heap: *mut c_void, out: *mut RtStats) {
        unsafe {
            *out = self::heap(heap).stats();
        }
    }
}

/// Adapters from the C slot signatures to `gc_immix`.
mod wren {
    use super::*;

    pub unsafe extern "C" fn object_trace(obj: *mut u8, mark: Visit, ctx: *mut c_void) {
        unsafe {
            crate::runtime::gc_immix::object_trace(obj, |child| mark(child, ctx));
        }
    }

    pub unsafe extern "C" fn object_drop(obj: *mut u8) {
        unsafe {
            crate::runtime::gc_immix::object_drop(obj);
        }
    }

    #[cfg(feature = "host")]
    pub unsafe extern "C" fn host_stop(on: bool) {
        crate::runtime::stw::host_stop(on);
    }

    #[cfg(not(feature = "host"))]
    pub unsafe extern "C" fn host_stop(_on: bool) {}

    #[cfg(feature = "host")]
    pub unsafe extern "C" fn task_step(task: *mut c_void) -> bool {
        unsafe { crate::runtime::sched::task_step(task as *mut crate::runtime::sched::TaskCtx) }
    }

    #[cfg(not(feature = "host"))]
    pub unsafe extern "C" fn task_step(_task: *mut c_void) -> bool {
        false
    }

    #[cfg(feature = "host")]
    pub unsafe extern "C" fn task_suspend() -> bool {
        crate::runtime::sched::task_suspend()
    }

    #[cfg(not(feature = "host"))]
    pub unsafe extern "C" fn task_suspend() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::engine::InterpretResult;
    use crate::runtime::gc_trait::GcStrategy;
    use crate::runtime::vm::{VM, VMConfig};
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
        unsafe {
            ALLOCS.fetch_add(1, Ordering::SeqCst);
            immix::alloc_raw(heap, size)
        }
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
        assert_eq!(
            wlift_rt_host_stop() as usize,
            wren::host_stop as HostStop as usize
        );
    }

    static STARTS: AtomicUsize = AtomicUsize::new(0);
    static STOPS: AtomicUsize = AtomicUsize::new(0);
    static SAFES: AtomicUsize = AtomicUsize::new(0);
    static RUNNINGS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn count_start() {
        STARTS.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "C" fn count_stop() {
        STOPS.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "C" fn count_safe(sp: usize, _extra_lo: usize, _extra_hi: usize) {
        assert_ne!(sp, 0, "a safe thread publishes where its stack stands");
        SAFES.fetch_add(1, Ordering::SeqCst);
    }
    unsafe extern "C" fn count_running() {
        RUNNINGS.fetch_add(1, Ordering::SeqCst);
    }

    /// Every worker of the pool starts and stops through the seam, and
    /// every wait and every collection passes through safe and running.
    /// In a process of its own, as the install test is.
    #[test]
    fn the_threads_that_run_wren_and_their_waits_reach_the_seam() {
        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "runtime::rt::tests::the_threads_that_run_wren_and_their_waits_reach_the_seam",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, "1")
                .status()
                .expect("re-run the test binary");
            assert!(status.success(), "child test process failed: {status}");
            return;
        }
        let mut table = RuntimeVTable::new();
        table.thread_start = Some(count_start);
        table.thread_stop = Some(count_stop);
        table.thread_safe = Some(count_safe);
        table.thread_running = Some(count_running);
        assert!(unsafe { wlift_rt_install(&table) });

        let mut vm = immix_vm();
        let result = vm.interpret(
            "main",
            r#"
                import "thread" for Thread, Lock
                var done = Lock.new()
                for (i in 0...4) {
                  Thread.create {
                    var xs = []
                    for (k in 0...5000) xs.add([k])
                    done.release()
                  }
                }
                for (i in 0...4) done.wait()
                System.print("done")
            "#,
        );
        assert_eq!(result, InterpretResult::Success);
        // The main thread's waits and the workers' collections.
        assert!(SAFES.load(Ordering::SeqCst) >= 4);
        assert!(RUNNINGS.load(Ordering::SeqCst) >= 4);
        assert!(STARTS.load(Ordering::SeqCst) >= 1);
        drop(vm);
        // The pool stops with the main view; each worker told the seam.
        assert_eq!(STOPS.load(Ordering::SeqCst), STARTS.load(Ordering::SeqCst));
    }

    /// A host's stop holds every live page unreadable until it lets go;
    /// wren_lift's own stop on the same page is counted with it.
    #[test]
    fn a_host_stop_holds_the_pages() {
        let page = crate::runtime::stw::poll_page::PollPage::new();
        assert!(!page.is_protected());
        unsafe { host_stop(true) };
        assert!(page.is_protected());
        assert!(crate::runtime::stw::poll_page::static_page().is_protected());
        page.protect(true);
        unsafe { host_stop(false) };
        assert!(page.is_protected(), "wren_lift's own hold stays");
        page.protect(false);
        assert!(!page.is_protected());
        assert!(!crate::runtime::stw::poll_page::static_page().is_protected());
    }
}
