use crate::runtime::engine::FuncId;
use crate::runtime::object::{
    FiberState, MirCallFrame, NativeContext, ObjClosure, ObjFiber, ObjHeader, ObjMap, ObjType,
};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Helpers ---

/// Extract an ObjFiber pointer from a Value (args[0] for instance methods).
unsafe fn as_fiber(val: Value) -> Option<*mut ObjFiber> {
    let ptr = val.as_object()?;
    let header = ptr as *const ObjHeader;
    if (*header).obj_type == ObjType::Fiber {
        Some(ptr as *mut ObjFiber)
    } else {
        None
    }
}

/// Extract an ObjClosure pointer from a Value.
unsafe fn as_closure(val: Value) -> Option<*mut ObjClosure> {
    let ptr = val.as_object()?;
    let header = ptr as *const ObjHeader;
    if (*header).obj_type == ObjType::Closure {
        Some(ptr as *mut ObjClosure)
    } else {
        None
    }
}

/// Set up a fiber's initial MIR frame from a closure. `module_name`
/// inherits from the caller so module-scope identifiers (classes,
/// top-level vars) remain reachable inside the fiber's closure.
unsafe fn setup_fiber_from_closure(
    fiber: *mut ObjFiber,
    closure: *mut ObjClosure,
    module_name: std::rc::Rc<String>,
) {
    let fn_ptr = (*closure).function;
    let func_id = FuncId((*fn_ptr).fn_id);

    // The register file starts empty; it will be sized correctly when the
    // fiber's first frame is loaded in run_fiber.
    (*fiber).mir_frames.push(MirCallFrame {
        func_id,
        current_block: crate::mir::BlockId(0),
        ip: 0,
        pc: 0,
        values: Vec::new(),
        module_name,
        return_dst: None,
        closure: Some(closure),
        defining_class: None,
        bc_ptr: std::ptr::null(),
    });
    (*fiber).state = FiberState::New;
}

/// Pull the module name from the caller's topmost frame so a
/// newly-spawned fiber resolves `Greet` (etc.) against the module
/// where it was defined, not a hardcoded "main".
unsafe fn current_module_name(ctx: &mut dyn NativeContext) -> std::rc::Rc<String> {
    let caller = ctx.get_current_fiber();
    if caller.is_null() {
        return std::rc::Rc::new(String::from("main"));
    }
    (*caller)
        .mir_frames
        .last()
        .map(|f| f.module_name.clone())
        .unwrap_or_else(|| std::rc::Rc::new(String::from("main")))
}

// --- Static methods ---

fn fiber_new(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // args[0] = Fiber class (receiver), args[1] = closure
    fiber_new_inner(ctx, args[1], None)
}

/// `Fiber.new(closure, stackKb)` — explicit stack size hint for the
/// krio backing. Use when the body calls into foreign Rust code that
/// stages large values on the stack (`Http.get` → `flate2::GzDecoder`,
/// etc.). The default 256 KiB is fine for pure-Wren bodies; ask for
/// 1024 here for HTTP / decompression / similar.
fn fiber_new_with_stack(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let stack_kb = match args[2].as_num() {
        Some(n) if n > 0.0 && n.is_finite() => Some((n as usize) * 1024),
        _ => {
            ctx.runtime_error(
                "Fiber.new(closure, stackKb): stackKb must be a positive number.".to_string(),
            );
            return Value::null();
        }
    };
    fiber_new_inner(ctx, args[1], stack_kb)
}

fn fiber_new_inner(
    ctx: &mut dyn NativeContext,
    closure_val: Value,
    parent_stack_request: Option<usize>,
) -> Value {
    let closure_ptr = unsafe { as_closure(closure_val) };
    match closure_ptr {
        Some(closure) => {
            // Capture spawn-site stack trace if opt-in is enabled
            let spawn_trace = ctx.capture_spawn_trace();
            // Prefer the closure's defining module: a closure passed
            // across an import boundary still has to resolve
            // `GetModuleVar` against the slots it was compiled for.
            // Fall back to the caller's module for test closures
            // that aren't registered with a module.
            let fn_ptr = unsafe { (*closure).function };
            let func_id = unsafe { (*fn_ptr).fn_id };
            let module_name = ctx
                .func_module(func_id)
                .unwrap_or_else(|| unsafe { current_module_name(ctx) });

            // Snapshot parent's context bag + deadline before we alloc
            // (alloc can move objects on nursery GC, invalidating caller).
            // Cancellation does NOT inherit — each fiber decides its own
            // timeline, and a cancelled parent cascades only if the user
            // wires it up explicitly.
            let parent = ctx.get_current_fiber();
            let (parent_ctx_entries, parent_deadline) = unsafe {
                if parent.is_null() {
                    (Vec::new(), None)
                } else {
                    let entries = snapshot_context_entries(parent);
                    ((*parent).deadline_ms).map_or((entries.clone(), None), |d| (entries, Some(d)))
                }
            };

            let fiber = ctx.alloc_fiber();
            unsafe {
                (*fiber).header.class = ctx.get_fiber_class();
                setup_fiber_from_closure(fiber, closure, module_name);
                (*fiber).spawn_trace = spawn_trace;
                (*fiber).deadline_ms = parent_deadline;
            }
            // The just-pushed mir_frame holds `closure` as a raw
            // pointer — a young object in an old-gen fiber (fibers
            // are pinned in old gen per `alloc_fiber`). Without a
            // write barrier, the next minor GC won't trace the
            // closure through the fiber's mir_frames and the
            // frame's `closure` field will dangle. The validator
            // direction-1 check catches this; the barrier closes
            // it.
            ctx.write_barrier(
                Value::object(fiber as *mut u8),
                Value::object(closure as *mut u8),
            );

            // krio-fiber backing: when WLIFT_KRIO_FIBER is on, attach
            // a per-fiber mmap stack and a body closure that runs the
            // Wren-level fiber body on that stack. Fiber.call drives
            // krio.resume_with(input); Fiber.yield inside calls
            // krio_fiber::yield_value to switch back synchronously.
            // The fiber's mir_frames is already prepared above by
            // setup_fiber_from_closure — the body just needs to point
            // `vm.fiber` at us and run the existing interpreter loop.
            #[cfg(feature = "host")]
            if ctx.krio_fiber_active() {
                let vm_ptr = ctx.krio_vm_raw_ptr();
                if !vm_ptr.is_null() {
                    let target_ptr_usize = fiber as usize;
                    let vm_ptr_usize = vm_ptr as usize;
                    // 256 KiB default. Wren bodies that don't enter
                    // heavy foreign code touch well under 100 KiB, so a
                    // 1 MiB reservation per fiber wastes virtual address
                    // space and (more importantly on macOS arm64) widens
                    // the window where the allocator's zone manager can
                    // stumble on the fiber's mmap region. Heavy paths
                    // — `Http.get` pulls in `flate2::GzDecoder` which
                    // stages a ~256 KiB `InflateState` on the stack
                    // before boxing — should be created via
                    // `Fiber.new(closure, stackKb)` with an explicit
                    // larger size. `WLIFT_KRIO_STACK_KB=N` overrides
                    // the default for the whole process.
                    let stack_bytes = std::env::var("WLIFT_KRIO_STACK_KB")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|kb| kb * 1024)
                        .or(parent_stack_request)
                        .unwrap_or(256 * 1024);
                    let krio = krio_fiber::Fiber::with_stack_size(stack_bytes, move || {
                        krio_fiber_body(vm_ptr_usize, target_ptr_usize);
                    });
                    unsafe {
                        (*fiber).krio_fiber = Some(Box::new(krio));
                    }
                    // Charge the krio mmap stack against GC pressure.
                    // Without this the Wren heap accounting only sees
                    // the few-hundred-byte ObjFiber header and never
                    // triggers a collection, while each fiber's stack
                    // accumulates 1 MiB of resident memory per request.
                    ctx.track_external_alloc(stack_bytes);
                    // GC Pass 3 enumerates krio-backed fibers via
                    // GcImpl::for_each_fiber at scan time, so no
                    // per-fiber registration is needed here.
                }
            }

            // Per-fiber bump-allocator region: short-lived
            // allocations (Wren strings/lists/maps) made inside
            // this fiber route through the region instead of the
            // GC heap. `release_fiber_resources` (in
            // `try_krio_call`'s Done arm) drops the region, freeing
            // every page in one `munmap` per page — no GC walk,
            // no Cranelift stack-map consultation, no
            // libsystem_malloc free-list residue. Off unless
            // `WLIFT_FIBER_ARENA=1` was set at VM construction;
            // the routing is also a no-op when `region` is `None`
            // so the GC heap stays the supported default until
            // the escape barriers (cross-fiber field writes,
            // cache stores, etc.) are wired across every path.
            #[cfg(feature = "host")]
            if ctx.fiber_arena_active() {
                unsafe {
                    (*fiber).region = Some(Box::new(wlift_region::Region::new()));
                }
            }

            // Populate the child's context map (allocates) only if parent
            // had entries. Keeps the zero-context case allocation-free.
            if !parent_ctx_entries.is_empty() {
                let child_map = ctx.alloc_map();
                unsafe {
                    if let Some(map_ptr) = child_map.as_object() {
                        let map = map_ptr as *mut ObjMap;
                        for (k, v) in parent_ctx_entries {
                            (*map).set(k, v);
                        }
                    }
                    (*fiber).context_map = child_map;
                }
            }

            // Root the fresh fiber in JIT_ROOTS_STORE so a GC fired
            // before the AOT caller's stack-map safepoint reads it
            // back doesn't reap the fresh allocation. Without this,
            // sustained load on AOT-compiled servers (the hatch site
            // accept loop spawns a new ObjFiber per request) crashes
            // within a few requests as "Fiber has already been
            // called." — the slab was freed + reused. No-op in JIT
            // / interpreter mode (push gates on `aot_gc_enabled()`
            // via `finish_alloc`).
            let fiber_val = Value::object(fiber as *mut u8);
            #[cfg(feature = "host")]
            if crate::codegen::runtime_fns::aot_gc_enabled() {
                crate::codegen::runtime_fns::push_jit_root(fiber_val);
            }
            fiber_val
        }
        None => {
            ctx.runtime_error("Fiber.new expects a function.".to_string());
            Value::null()
        }
    }
}

/// Snapshot a fiber's context entries into an owned Vec so the caller
/// can pass them to a later `alloc_map` without worrying about GC
/// moving the source map mid-copy.
unsafe fn snapshot_context_entries(fiber: *mut ObjFiber) -> Vec<(Value, Value)> {
    if fiber.is_null() || (*fiber).context_map.is_null() {
        return Vec::new();
    }
    let ptr = match (*fiber).context_map.as_object() {
        Some(p) => p,
        None => return Vec::new(),
    };
    let header = ptr as *const ObjHeader;
    if (*header).obj_type != ObjType::Map {
        return Vec::new();
    }
    let map = ptr as *const ObjMap;
    (*map)
        .entries
        .iter()
        .map(|(k, &v)| (k.value(), v))
        .collect()
}

/// Return (or lazily create + cache) a fiber's context map.
unsafe fn get_or_init_context_map(ctx: &mut dyn NativeContext, fiber: *mut ObjFiber) -> Value {
    if !(*fiber).context_map.is_null() {
        return (*fiber).context_map;
    }
    let map = ctx.alloc_map();
    (*fiber).context_map = map;
    map
}

// --- Context / cancellation / deadline ---

fn fiber_context_get(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        ctx.runtime_error("Fiber.context called outside a fiber.".to_string());
        return Value::null();
    }
    unsafe { get_or_init_context_map(ctx, cur) }
}

fn fiber_is_cancelled_static(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        return Value::bool(false);
    }
    unsafe { Value::bool((*cur).cancelled) }
}

fn fiber_cancel_static(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if !cur.is_null() {
        unsafe {
            (*cur).cancelled = true;
        }
    }
    Value::null()
}

fn fiber_deadline_ms_static(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        return Value::null();
    }
    unsafe {
        match (*cur).deadline_ms {
            Some(ms) => Value::num(ms),
            None => Value::null(),
        }
    }
}

fn fiber_set_deadline_ms_static(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let cur = ctx.get_current_fiber();
    if cur.is_null() {
        ctx.runtime_error("Fiber.setDeadlineMs called outside a fiber.".to_string());
        return Value::null();
    }
    unsafe {
        (*cur).deadline_ms = if args[1].is_null() {
            None
        } else {
            args[1].as_num()
        };
    }
    Value::null()
}

fn fiber_context_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe { get_or_init_context_map(ctx, f) },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_cancel_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe {
            (*f).cancelled = true;
        },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_is_cancelled_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe { Value::bool((*f).cancelled) },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_deadline_ms_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe {
            match (*f).deadline_ms {
                Some(ms) => Value::num(ms),
                None => Value::null(),
            }
        },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_set_deadline_ms_instance(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe {
            (*f).deadline_ms = if args[1].is_null() {
                None
            } else {
                args[1].as_num()
            };
        },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_abort(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if args[1].is_object() {
        let class_name = ctx.get_class_name_of(args[1]);
        if class_name == "String" {
            let msg = super::as_string(args[1]);
            ctx.runtime_error(msg.to_string());
        } else {
            ctx.runtime_error("Runtime error.".to_string());
        }
    } else {
        ctx.runtime_error("Runtime error.".to_string());
    }
    Value::null()
}

fn fiber_current(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let fiber = ctx.get_current_fiber();
    if fiber.is_null() {
        Value::null()
    } else {
        Value::object(fiber as *mut u8)
    }
}

fn fiber_suspend(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    ctx.set_fiber_action_suspend();
    Value::null()
}

fn fiber_yield_0(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    if let Some(v) = try_krio_yield(Value::null()) {
        return v;
    }
    ctx.set_fiber_action_yield(Value::null());
    Value::null()
}

fn fiber_yield_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if let Some(v) = try_krio_yield(args[1]) {
        return v;
    }
    ctx.set_fiber_action_yield(args[1]);
    Value::null()
}

/// If the current thread is executing inside a `krio_fiber::Fiber`,
/// route the yield through krio's synchronous context switch and
/// return whatever value the host pumps back via `resume_with`. The
/// `pending_fiber_action` path is skipped entirely on this branch —
/// the krio context switch unwinds the fiber's physical call stack
/// instead of relying on the interpreter loop's stackless trampoline.
///
/// Returns `None` if no krio fiber is active, so the caller falls
/// back to the existing `set_fiber_action_yield` path. This makes the
/// change a no-op when `WLIFT_KRIO_FIBER` is off OR when the fiber
/// was allocated under the stackless path (krio fiber backing is
/// per-ObjFiber, not global).
#[cfg(feature = "host")]
fn try_krio_yield(value: Value) -> Option<Value> {
    krio_fiber::current_fiber_id()?;

    // Caller-aware split: krio's context switch is the right
    // answer for a top-level Fiber.yield (no Wren caller — the
    // implicit main fiber, wrapped by wlift_aot_invoke_module_body).
    // For nested fibers (anything driven by sched.tick / .try /
    // .call), yield must propagate through pending_fiber_action so
    // the interpreter switches vm.fiber back to the Wren caller —
    // otherwise the body just resumes immediately from the
    // module-body wrapper and the scheduler never gets control
    // back. The hatch site's catalog.refreshLoop's busy yield-poll
    // would burn 94% CPU without this split.
    let vm_ptr = crate::runtime::vm::current_vm_ptr();
    if !vm_ptr.is_null() {
        let current_fiber = unsafe { (*vm_ptr).fiber };
        if !current_fiber.is_null() && unsafe { !(*current_fiber).caller.is_null() } {
            // Nested fiber — defer to the stackless path. The BC
            // interpreter's run_fiber loop handles
            // pending_fiber_action::Yield by switching vm.fiber to
            // (*fiber).caller; that's the right semantic.
            return None;
        }
    }

    // Save the fiber's JIT roots and clear the thread-local store
    // before switching back to the host. The matching restore on
    // the next resume reinstalls them. Mirrors try_krio_call's
    // host-side save/restore — the two halves form the pair that
    // keeps each fiber's roots disjoint from the host's and from
    // sibling fibers' on the same thread.
    let fiber_roots = crate::codegen::runtime_fns::take_jit_roots();
    // We can't directly identify the active ObjFiber from here
    // (current_fiber_id is krio's, not ours), so stash the roots
    // in a thread-local handoff slot that the host's try_krio_call
    // drains after resume_with returns.
    KRIO_YIELD_ROOTS_HANDOFF.with(|cell| cell.replace(fiber_roots));

    // Pump the yield value to the host via krio's `u64` fast path,
    // then suspend. On resume the host calls `resume_with_u64(bits)`
    // with the value the next `Fiber.call(v)` passed in. The generic
    // `yield_value::<u64, u64>` path allocates one `Box<dyn Any>`
    // per yield-and-resume round-trip; under a tight
    // `while true { Fiber.yield() }` loop that's 2 Box::new + 2 drop
    // per tick, which dominates the allocator and OOMs the process
    // (see `project_krio_box_per_yield_leak` memory note). The u64
    // fast path stores the bits in a fixed Cell<Option<u64>> — zero
    // allocation per round-trip.
    let received: Option<u64> = krio_fiber::yield_u64(value.to_bits());

    // Back on the fiber stack — host has reinstalled this fiber's
    // saved roots into JIT_ROOTS_STORE before resuming us.
    Some(received.map(Value::from_bits).unwrap_or_else(Value::null))
}

// Thread-local handoff: the fiber-side `try_krio_yield` stashes
// its drained roots here; the host-side `try_krio_call` picks
// them up and parks them on the ObjFiber after `resume_with`
// returns. Necessary because the fiber doesn't know which
// ObjFiber it belongs to (krio's id is internal), and even if it
// did, mutating `*mut ObjFiber` from inside the fiber while the
// host holds `&mut self` would be UB.
#[cfg(feature = "host")]
thread_local! {
    static KRIO_YIELD_ROOTS_HANDOFF: std::cell::RefCell<Vec<Value>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(not(feature = "host"))]
fn try_krio_yield(_value: Value) -> Option<Value> {
    None
}

/// If the target fiber has a krio-fiber backing, drive it directly
/// via `resume_with(value)` and return the value the fiber yielded
/// (or `null` if it completed). Skips the `pending_fiber_action`
/// dance entirely.
///
/// Returns `None` if the target has no krio backing — caller falls
/// back to the existing stackless path.
///
/// # Safety
/// `target` is dereferenced; caller ensures it's a valid ObjFiber
/// pointer (the caller already validated via `as_fiber`).
#[cfg(feature = "host")]
/// Public wrapper around `try_krio_call` so the AOT/JIT fiber-action
/// helper in `runtime_fns::handle_jit_fiber_action` can route fresh-
/// fiber Calls through krio without duplicating its body.
#[cfg(feature = "host")]
pub fn try_krio_call_pub(target: *mut ObjFiber, input: Value) -> Option<Value> {
    try_krio_call(target, input)
}

/// True when this VM was constructed with krio backings active.
#[cfg(feature = "host")]
pub fn current_vm_krio_active(vm: &crate::runtime::vm::VM) -> bool {
    vm.krio_fiber_active
}

/// Set a fiber to its terminal state (Done / Error) and drop the
/// expensive per-fiber state the runtime no longer needs.
///
/// `krio_fiber` owns a 256 KiB-1 MiB mmap'd stack; under AOT mode
/// the `finish_alloc` safepoint is a no-op so GC may not fire for
/// minutes (or until catalog refresh's explicit `System.gc()`
/// 5-minute cadence), and the ObjFiber that owns the krio Box stays
/// pinned until then. Sustained request traffic on the hatch site
/// otherwise accumulated ~6 VM_ALLOCATE regions per nav with no
/// recovery — visible as monotonic footprint growth.
///
/// `mir_frames`, `stack`, and `frames` are the Wren-side register
/// files and call-frame metadata; their `Vec<Value>` buffers also
/// linger on libsystem_malloc's free list with the page kept
/// resident until the next pressure relief, which on macOS only
/// happens at major-GC boundaries. Clearing them eagerly returns
/// those Vecs' backing storage to the allocator now.
///
/// Safe to call when the krio body has returned (its trampoline
/// has unwound the per-fiber stack) and the fiber is transitioning
/// to a terminal state where no further code will read those
/// fields. The remaining `ObjFiber` shell is small (≈few hundred
/// bytes) and is reaped by a later GC pass.
#[cfg(feature = "host")]
unsafe fn release_fiber_resources(target: *mut ObjFiber, terminal: FiberState) {
    unsafe {
        (*target).state = terminal;
        (*target).krio_fiber = None;
        (*target).mir_frames = Vec::new();
        (*target).stack = Vec::new();
        (*target).frames = Vec::new();
        // Drop the bump-allocator region last. By this point the
        // fiber's `mir_frames` and `stack` have been cleared, so
        // any remaining region-allocated Value with a Wren-side
        // reference would have to come from a caller that crossed
        // the fiber boundary without going through the escape
        // barrier. That's a bug and we'd rather find it via a
        // post-drop dangling-pointer crash in tests than ship it
        // silently — leaving the Box drop here makes the unmap
        // happen now, surfacing any caller's stale pointer on its
        // next access.
        //
        // DEBUG TOGGLE: setting `WLIFT_FIBER_ARENA_HOLD=1` keeps
        // the region attached to the dead `ObjFiber` until the GC
        // reaps the fiber itself. Used during Phase 2 escape-
        // barrier development to isolate "this is an escape that
        // I missed" from "the drop point is wrong" — if holding
        // the region cures a crash, the bug is a missed barrier
        // somewhere; if it doesn't, the bug is elsewhere.
        if std::env::var_os("WLIFT_FIBER_ARENA_HOLD").is_some_and(|v| v == "1") {
            // Hold — let the ObjFiber's eventual GC sweep drop it.
        } else {
            (*target).region = None;
        }
    }
}

#[cfg(feature = "host")]
fn try_krio_call(target: *mut ObjFiber, input: Value) -> Option<Value> {
    // Establish that target has a krio backing before we start
    // doing any save/restore work. Early-return None lets the
    // caller fall back to the stackless path.
    if unsafe { (*target).krio_fiber.is_none() } {
        return None;
    }

    // Save the caller's JIT roots and install this fiber's saved
    // roots (or an empty Vec if first call). The thread-local
    // JIT_ROOTS_STORE is shared across fibers on the same thread;
    // without this swap, the fiber's AOT-compiled body would index
    // into the host's roots and read off the end of the array.
    let host_roots = crate::codegen::runtime_fns::take_jit_roots();
    let fiber_roots = std::mem::take(unsafe { &mut (*target).krio_jit_roots });
    crate::codegen::runtime_fns::set_jit_roots(fiber_roots);

    // Host-side vm.fiber swap, wrapping resume_with. The krio
    // body has its own swap inside (captured at body entry), but
    // that swap only restores when the body fully exits — between
    // yields it leaves vm.fiber pointing at `target`. Without the
    // host-side wrap, after the first yield the host's interpreter
    // loop runs with vm.fiber = target instead of its own fiber,
    // misdispatching subsequent opcodes. This bug surfaced on
    // proc.spec's "two fibers reading different processes
    // interleave" and the equivalent in events.spec.
    //
    // SAFETY: we own a raw *mut VM via the VM's own ObjFiber graph
    // — the fiber wouldn't exist without a live VM, and the VM is
    // single-threaded. The deref is bounded to the body of this
    // function.
    let vm_ptr: *mut crate::runtime::vm::VM = crate::runtime::vm::current_vm_ptr();
    let prev_vm_fiber = if !vm_ptr.is_null() {
        unsafe {
            let prev = (*vm_ptr).fiber;
            (*vm_ptr).fiber = target;
            prev
        }
    } else {
        std::ptr::null_mut()
    };

    unsafe {
        (*target).state = FiberState::Suspended; // running, but not New
    }
    // SAFETY of the raw split: `krio` is reborrowed mutably from
    // `target`'s field. The fiber's body inside `resume_with` may
    // freely interact with the rest of `*target` and the VM via
    // raw pointers — the host's borrow ends at the resume_with
    // call boundary because the body's first statement is a
    // context switch onto the fiber's stack.
    let krio_ptr: *mut krio_fiber::Fiber = unsafe { (*target).krio_fiber.as_deref_mut().unwrap() };
    // `resume_with_u64` is the alloc-free counterpart of the
    // generic `resume_with::<u64>` — see the matching comment on
    // `krio_fiber::yield_u64` in `try_krio_yield`.
    let step = unsafe { (*krio_ptr).resume_with_u64(input.to_bits()) };

    // Restore host's vm.fiber. Even if the body's own restore line
    // already ran (natural-completion path), re-asserting it here
    // is fine: prev_vm_fiber is the value vm.fiber had on entry to
    // this function, which is the canonical "host caller" identity.
    if !vm_ptr.is_null() {
        unsafe {
            (*vm_ptr).fiber = prev_vm_fiber;
        }
    }

    // Drain the fiber's roots: if it yielded, the fiber-side
    // `try_krio_yield` stashed them into the handoff thread-local
    // just before suspending; if it returned naturally, the
    // body emptied JIT_ROOTS_STORE on exit (or it's still got
    // whatever leaked into it, which we collect for the next
    // re-entry).
    let saved_fiber_roots = if matches!(step, krio_fiber::FiberStep::Yielded) {
        KRIO_YIELD_ROOTS_HANDOFF.with(|cell| cell.replace(Vec::new()))
    } else {
        crate::codegen::runtime_fns::take_jit_roots()
    };
    unsafe {
        (*target).krio_jit_roots = saved_fiber_roots;
    }
    // Reinstall the host's roots.
    crate::codegen::runtime_fns::set_jit_roots(host_roots);

    let result = match step {
        krio_fiber::FiberStep::Yielded => {
            let krio = unsafe { &mut *krio_ptr };
            // `take_yield_u64` reads the alloc-free fast-path slot
            // populated by `krio_fiber::yield_u64`. `take_yield_any`
            // would force the fiber side to box the u64 into a
            // `Box<dyn Any>` per yield — the leak this whole path
            // is unwinding.
            let v = krio
                .take_yield_u64()
                .map(Value::from_bits)
                .unwrap_or_else(Value::null);
            unsafe {
                (*target).state = FiberState::Suspended;
            }
            // Escape barrier: if the child yielded a value that
            // points into its own region (Suspended fibers keep
            // their region alive across yields, so the strict
            // "region will drop" condition doesn't apply here yet
            // — but the parent might suspend US too before resuming
            // the child, by which point the child may have been
            // reaped). Copying is also forward-compatible with the
            // Phase 2.5 escape paths for List / Map.
            //
            // No-op when the arena feature is off or the child
            // didn't use it for this Value (early-return on the
            // FLAG_ARENA_ALLOCATED check inside the helper).
            if !vm_ptr.is_null() {
                unsafe { (*vm_ptr).escape_to_gc_if_arena(v) }
            } else {
                v
            }
        }
        krio_fiber::FiberStep::Done => {
            // The body stashed the closure's last-expression value
            // into krio_return_value just before exiting; surface
            // it as the return of this Fiber.call(_).
            //
            // Try-fiber abort path: when the body called
            // Fiber.abort or hit a runtime error, vm_interp.rs's
            // try-handling (line ~2418) parked the message on
            // `(*fiber).error` and returned `Ok(err_val)` from
            // run_fiber. The krio body forwards that into
            // `krio_return_value`, so the normal `stored` read
            // picks it up. The `(*target).error` fallback covers
            // the case where the body aborted but the err value
            // didn't propagate through run_fiber's Result (e.g.
            // an error path that returned `Ok(Value::null())` or
            // `Err` with the message parked elsewhere) — Wren
            // callers expect `f.try()` to return the same string
            // they see in `f.error`.
            let stored = unsafe { (*target).krio_return_value };
            let v = if stored.is_null() {
                let err = unsafe { (*target).error };
                if !err.is_null() {
                    err
                } else {
                    stored
                }
            } else {
                stored
            };
            // Escape barrier (the critical case): we're about to
            // drop the child's region in `release_fiber_resources`.
            // If `v` points into that region, the parent would be
            // left with a dangling pointer. Deep-copy to the GC
            // heap first.
            let v = if !vm_ptr.is_null() {
                unsafe { (*vm_ptr).escape_to_gc_if_arena(v) }
            } else {
                v
            };
            unsafe {
                release_fiber_resources(target, FiberState::Done);
                (*target).krio_return_value = Value::null();
            }
            v
        }
        krio_fiber::FiberStep::Errored => {
            // Escape barrier: the `(*target).error` Value may live
            // in the child's region — same dangling-pointer hazard
            // as the Done arm. Copy out before the region drops.
            // `release_fiber_resources` clears the error slot
            // separately, so we have to read + copy here, then
            // re-store the GC-heap copy onto the target before the
            // region releases.
            if !vm_ptr.is_null() {
                let err = unsafe { (*target).error };
                let err = unsafe { (*vm_ptr).escape_to_gc_if_arena(err) };
                unsafe {
                    (*target).error = err;
                    // target fiber is old-gen pinned; err may be a
                    // young object. Without the barrier the next
                    // minor GC won't trace err through fiber.error
                    // and the slot dangles.
                    (*vm_ptr).gc.write_barrier(target as *mut ObjHeader, err);
                }
            }
            unsafe {
                release_fiber_resources(target, FiberState::Error);
            }
            Value::null()
        }
    };
    Some(result)
}

#[cfg(not(feature = "host"))]
fn try_krio_call(_target: *mut ObjFiber, _input: Value) -> Option<Value> {
    None
}

/// Body of a krio-backed fiber. Runs on the fiber's per-fiber mmap
/// stack the first time the host calls `resume_with` (the initial
/// `Fiber.call(v)`); subsequent resumes re-enter via the saved
/// context-switch point — they don't invoke this function again.
///
/// Captures the VM and target ObjFiber as `usize`s (rather than
/// borrowed references) because:
///   1. The closure is `'static + FnOnce()` per krio's API.
///   2. The host's `&mut VM` aliasing rules permit re-entrance
///      because the fiber's `resume_with` is synchronous — the
///      host yields its borrow while the fiber runs.
///
/// # Safety
/// `vm_ptr` must be a valid `*mut VM` for the lifetime of the
/// fiber (the VM outlives every ObjFiber it allocates). `target_ptr`
/// must be a valid `*mut ObjFiber` whose mir_frames were prepared
/// by `setup_fiber_from_closure` before this body runs.
#[cfg(feature = "host")]
fn krio_fiber_body(vm_ptr_usize: usize, target_ptr_usize: usize) {
    use crate::runtime::object::FiberState;
    use crate::runtime::vm::VM;

    let vm_ptr = vm_ptr_usize as *mut VM;
    let target_ptr = target_ptr_usize as *mut ObjFiber;

    // Pull the initial input the host passed via `resume_with`.
    // For `Fiber.call(v)`, v is the closure's first argument; for
    // `Fiber.call()` (no arg) it's null. We bind it into the
    // closure's first arg slot before running the interpreter.
    let initial_bits: u64 = krio_fiber::take_input::<u64>().unwrap_or(0);
    let initial_val = Value::from_bits(initial_bits);

    unsafe {
        // Inject the initial input into the closure's first arg
        // slot. The MIR frame was set up by setup_fiber_from_closure
        // but its register file is empty; the interpreter will size
        // it from the function's metadata on first dispatch. We
        // stash the initial value on the stack instead, which the
        // closure's entry sees as arg 0.
        (*target_ptr).stack.clear();
        (*target_ptr).stack.push(initial_val);
        (*target_ptr).state = FiberState::Running;

        // Swap the VM's active fiber to `target_ptr` so the
        // interpreter loop runs the right body. The previous
        // fiber's identity is preserved on the host stack (krio
        // returns to the resume_with call site, which is in a
        // foreign-method handler running on behalf of `prev`),
        // so we restore it before returning.
        let prev_fiber = (*vm_ptr).fiber;
        (*vm_ptr).fiber = target_ptr;

        // Run the fiber to completion on this physical stack. If
        // the Wren body calls `Fiber.yield(_)`, try_krio_yield
        // routes through krio_fiber::yield_value, which performs a
        // synchronous context switch — control returns to the
        // host's resume_with call site, and run_fiber pauses here.
        // On the next resume_with, control re-enters the switch
        // and run_fiber continues from inside the foreign-method
        // dispatch frame.
        //
        // When the Wren body returns naturally (no more yields),
        // run_fiber returns. The Ok value is the closure's last-
        // expression value, which Wren surfaces as the return of
        // the final `Fiber.call(_)`. Stash it on the ObjFiber so
        // try_krio_call can return it after the body exits.
        let result = crate::runtime::vm_interp::run_fiber(&mut *vm_ptr);
        (*target_ptr).krio_return_value = result.unwrap_or(Value::null());

        (*vm_ptr).fiber = prev_fiber;
        if (*target_ptr).state == FiberState::Running {
            (*target_ptr).state = FiberState::Done;
        }
    }
}

// --- Instance methods ---

fn fiber_call_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    if let Some(v) = try_krio_call(f, Value::null()) {
                        return v;
                    }
                    ctx.set_fiber_action_call(f, Value::null());
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot call a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber has already been called.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_call_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    if let Some(v) = try_krio_call(f, args[1]) {
                        return v;
                    }
                    ctx.set_fiber_action_call(f, args[1]);
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot call a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber has already been called.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_error(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => unsafe { (*f).error },
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_is_done(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let done = unsafe { (*f).state == FiberState::Done };
            Value::bool(done)
        }
        None => Value::bool(true),
    }
}

fn fiber_transfer_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    ctx.set_fiber_action_transfer(f, Value::null());
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot transfer to a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber is already running.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_transfer_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    ctx.set_fiber_action_transfer(f, args[1]);
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot transfer to a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber is already running.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_transfer_error(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    unsafe {
                        (*f).error = args[1];
                    }
                    // Fiber is old-gen pinned; args[1] may be young.
                    ctx.write_barrier(Value::object(f as *mut u8), args[1]);
                    ctx.set_fiber_action_transfer(f, args[1]);
                }
                _ => {
                    ctx.runtime_error("Cannot transfer to fiber.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_stack_trace(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    if !ctx.fiber_stack_traces_enabled() {
        return ctx.alloc_string(
            "<stack traces not enabled — set VMConfig.fiber_stack_traces = true>".to_string(),
        );
    }
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let trace = ctx.get_stack_trace_string(f);
            ctx.alloc_string(trace)
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
            Value::null()
        }
    }
}

fn fiber_try_0(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    unsafe { (*f).is_try = true };
                    // Full krio routing for both fresh and suspended
                    // fibers. Required on AOT: handle_jit_fiber_action's
                    // Yield arm is a no-op (the Mechanism B leak), so
                    // any fiber whose body calls Fiber.yield needs the
                    // krio context-switch to actually suspend. Falling
                    // through to set_fiber_action_call on a New fiber
                    // runs the body on the main stack and yields are
                    // silently dropped.
                    if let Some(v) = try_krio_call(f, Value::null()) {
                        return v;
                    }
                    ctx.set_fiber_action_call(f, Value::null());
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot call a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber has already been called.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

fn fiber_try_1(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let fiber = unsafe { as_fiber(args[0]) };
    match fiber {
        Some(f) => {
            let state = unsafe { (*f).state };
            match state {
                FiberState::New | FiberState::Suspended => {
                    unsafe { (*f).is_try = true };
                    if let Some(v) = try_krio_call(f, args[1]) {
                        return v;
                    }
                    ctx.set_fiber_action_call(f, args[1]);
                }
                FiberState::Done => {
                    ctx.runtime_error("Cannot call a finished fiber.".to_string());
                }
                _ => {
                    ctx.runtime_error("Fiber has already been called.".to_string());
                }
            }
        }
        None => {
            ctx.runtime_error("Expected a fiber.".to_string());
        }
    }
    Value::null()
}

pub fn bind(vm: &mut VM) {
    let class = vm.fiber_class;

    // Static methods
    vm.primitive_static(class, "new(_)", fiber_new);
    vm.primitive_static(class, "new(_,_)", fiber_new_with_stack);
    vm.primitive_static(class, "abort(_)", fiber_abort);
    vm.primitive_static(class, "current", fiber_current);
    vm.primitive_static(class, "suspend()", fiber_suspend);
    vm.primitive_static(class, "yield()", fiber_yield_0);
    vm.primitive_static(class, "yield(_)", fiber_yield_1);

    // Context / cancellation / deadline — current-fiber shortcuts.
    vm.primitive_static(class, "context", fiber_context_get);
    vm.primitive_static(class, "isCancelled", fiber_is_cancelled_static);
    vm.primitive_static(class, "cancel()", fiber_cancel_static);
    vm.primitive_static(class, "deadlineMs", fiber_deadline_ms_static);
    vm.primitive_static(class, "setDeadlineMs(_)", fiber_set_deadline_ms_static);

    // Instance methods
    vm.primitive(class, "call()", fiber_call_0);
    vm.primitive(class, "call(_)", fiber_call_1);
    vm.primitive(class, "error", fiber_error);
    vm.primitive(class, "isDone", fiber_is_done);
    vm.primitive(class, "transfer()", fiber_transfer_0);
    vm.primitive(class, "transfer(_)", fiber_transfer_1);
    vm.primitive(class, "transferError(_)", fiber_transfer_error);
    vm.primitive(class, "try()", fiber_try_0);
    vm.primitive(class, "try(_)", fiber_try_1);
    vm.primitive(class, "stackTrace", fiber_stack_trace);

    // Per-instance context / cancellation / deadline (inspect / control
    // a fiber from outside it).
    vm.primitive(class, "context", fiber_context_instance);
    vm.primitive(class, "cancel()", fiber_cancel_instance);
    vm.primitive(class, "isCancelled", fiber_is_cancelled_instance);
    vm.primitive(class, "deadlineMs", fiber_deadline_ms_instance);
    vm.primitive(class, "setDeadlineMs(_)", fiber_set_deadline_ms_instance);
}
