//! Cooperative scheduler: one world per VM, driven on the thread that
//! owns the VM. A task is a fiber with a stack of its own; the world
//! resumes it with `Fiber.call` machinery and takes it back at every
//! yield or park. Waits are tokens in a process-wide registry, so a
//! wake can come from any thread; the world learns of it through its
//! endpoint, and a timer park is a min-heap entry, never a thread
//! sleep. Every notification is claimed before it is delivered, so a
//! stale timer or a second wake for the same token is harmless.
//!
//! The world is behind the runtime seam (`rt`'s World slots). The
//! functions at the top of this module are what the natives call; each
//! goes to the slot, whose default is the world here (`local`), one
//! per view. A host with a world of its own fills the slots, and then
//! a task is one of the host's: the host holds a [`TaskCtx`] and steps
//! it through [`task_step`], on the world it placed it on. The view's
//! [`Sched`] stays the registry of the tasks stepped on it, whichever
//! world steps them: their fibers and handles are its roots.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::runtime::object::{FiberState, NativeContext, ObjFiber};
use crate::runtime::rt::{self, NO_DEADLINE};
use crate::runtime::value::Value;
use crate::runtime::vm::{SharedCell, VM};

pub type TaskId = u64;
pub type Token = u64;
pub type WorldId = u64;

/// Where a park was asked for, relative to the world it belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Context {
    /// Under the task the world is stepping, on its own stack or on a
    /// fiber it called: the park is a yield back to the world, which a
    /// fiber in between passes on (see `try_krio_call`).
    Task,
    /// No task is being stepped: whoever asked drives the world
    /// itself until the park resolves.
    Driver,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunState {
    Runnable,
    Running,
    Waiting(Token),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaitStatus {
    Waiting,
    Notified,
}

struct Registration {
    world: WorldId,
    status: WaitStatus,
}

/// The world's mailbox: wakes and closures to run from other threads
/// land here, and the idle driver sleeps on the condvar.
pub struct Endpoint {
    wakes: Mutex<VecDeque<Token>>,
    /// Closures another thread asked this world to run as tasks,
    /// each with its `Thread` handle or null; roots of the program
    /// until the world takes them.
    spawns: Mutex<VecDeque<(Value, Value)>>,
    /// Tasks not yet finished plus closures not yet taken, for
    /// placing new ones.
    live: AtomicUsize,
    changed: Condvar,
}

impl Endpoint {
    /// Ask the world to run `closure` as a task, finishing `handle`
    /// when it ends.
    pub fn push_spawn(&self, closure: Value, handle: Value) {
        self.spawns
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back((closure, handle));
        self.live.fetch_add(1, Ordering::Relaxed);
        // The idle wait checks the mailbox under the wakes lock, so
        // the notice is given under it too, or could fall between
        // that check and the wait.
        let _wakes = self.wakes.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }

    /// Closures waiting to become tasks and their handles: roots for
    /// the collector.
    pub fn pending_spawns(&self) -> Vec<Value> {
        self.spawns
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .flat_map(|&(c, h)| [c, h])
            .collect()
    }

    pub fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }
}

struct Task {
    fiber: *mut ObjFiber,
    /// The `Thread` handle to finish when the task ends, or null.
    handle: Value,
    state: RunState,
    /// How the last park ended: a wake, or the timer.
    woken: bool,
}

/// A task on its way to a world and, once there, what the world steps:
/// the program, the closure until the fiber is made on the thread that
/// will run it, the fiber after, and the `Thread` handle to finish. The
/// program's `transit` list roots it until its first step; the view's
/// registry does from then on.
pub struct TaskCtx {
    shared: Weak<SharedCell>,
    closure: Value,
    handle: Value,
    fiber: *mut ObjFiber,
}

impl TaskCtx {
    /// The values a context in transit holds.
    pub fn roots(&self) -> [Value; 3] {
        [
            self.closure,
            self.handle,
            if self.fiber.is_null() {
                Value::null()
            } else {
                Value::object(self.fiber as *mut u8)
            },
        ]
    }
}

// ── The front: what the natives call, each one slot ─────────────────────

fn to_ns(deadline: Option<Instant>) -> u64 {
    deadline.map_or(NO_DEADLINE, |d| {
        d.saturating_duration_since(Instant::now()).as_nanos() as u64
    })
}

fn from_ns(ns: u64) -> Option<Instant> {
    (ns != NO_DEADLINE).then(|| Instant::now() + Duration::from_nanos(ns))
}

fn vm_ptr(vm: *mut VM) -> *mut std::ffi::c_void {
    vm as *mut std::ffi::c_void
}

/// A fresh token, waiting on this world.
pub fn new_waiter(vm: *mut VM) -> Token {
    unsafe { rt::world_waiter_new(vm_ptr(vm)) }
}

/// Forget a token that will not be parked on.
pub fn discard_waiter(vm: *mut VM, token: Token) {
    unsafe { rt::world_waiter_discard(vm_ptr(vm), token) }
}

/// Mark `token` notified and tell its world. `false` when the token
/// is unknown or already resolved, so a duplicate wake does nothing.
pub fn wake(token: Token) -> bool {
    unsafe { rt::world_wake(token) }
}

/// Check a token belongs to this world; `Ok(true)` when it was
/// woken before the park, which consumes it.
pub fn waiter_ready(vm: *mut VM, token: Token) -> Result<bool, String> {
    match unsafe { rt::world_waiter_ready(vm_ptr(vm), token) } {
        1 => Ok(true),
        0 => Ok(false),
        _ => Err(format!("Fiber.park: unknown waiter {token}.")),
    }
}

/// Record the active task's park; it takes effect when the task
/// yields back to the world.
pub fn request_park(vm: *mut VM, token: Token, deadline: Option<Instant>) {
    unsafe { rt::world_park_request(vm_ptr(vm), token, to_ns(deadline)) }
}

/// How the active task's last park ended.
pub fn resume_woken(vm: *mut VM) -> bool {
    unsafe { rt::world_resume_woken(vm_ptr(vm)) }
}

/// Drive the world until `token` is woken or `deadline` passes: the
/// park of a stack that is not a task.
pub fn park_drive(vm: *mut VM, token: Token, deadline: Option<Instant>) -> bool {
    unsafe { rt::world_park_drive(vm_ptr(vm), token, to_ns(deadline)) }
}

/// Make a task of `fiber`, made on this thread, with `handle` finished
/// when it ends.
pub fn spawn_fiber(vm: *mut VM, fiber: *mut ObjFiber, handle: Value) {
    spawn_ctx(vm, Value::null(), handle, fiber, false)
}

/// Make a task of `closure` on a worker world; its fiber is made
/// there, and `handle` is finished when it ends.
pub fn spawn_on_pool(vm: *mut VM, closure: Value, handle: Value) {
    spawn_ctx(vm, closure, handle, std::ptr::null_mut(), true)
}

fn spawn_ctx(vm: *mut VM, closure: Value, handle: Value, fiber: *mut ObjFiber, on_pool: bool) {
    let vm_ref = unsafe { &mut *vm };
    if on_pool {
        // Another thread will run Wren from here on.
        vm_ref.mark_threaded();
    }
    let ctx = Box::into_raw(Box::new(TaskCtx {
        shared: vm_ref.shared_weak(),
        closure,
        handle,
        fiber,
    }));
    vm_ref
        .transit
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(ctx);
    unsafe { rt::world_spawn(vm_ptr(vm), ctx as *mut std::ffi::c_void, on_pool) }
}

/// Where the calling stack stands relative to a step in progress on
/// this view.
pub fn context(vm: *mut VM) -> Context {
    sched_of(vm).context()
}

/// A step is in progress and the calling stack is neither the stepped
/// task's nor the one the step was started from: a park that reaches
/// it belongs to the task above it.
pub fn park_travels_through_here(vm: *mut VM) -> bool {
    let sched = sched_of(vm);
    match sched.active {
        Some((_, host)) => {
            let pending = unsafe { rt::world_park_pending(vm_ptr(vm)) };
            pending && current_stack_id() != host
        }
        None => false,
    }
}

/// The handle of the task being stepped, or null.
pub fn active_handle(vm: *mut VM) -> Value {
    sched_of(vm).active_handle()
}

/// Step until nothing is ready or `deadline` passes; whether live
/// tasks remain.
pub fn tick(vm: *mut VM, deadline: Option<Instant>) -> bool {
    unsafe { rt::world_tick(vm_ptr(vm), to_ns(deadline)) }
}

/// Wait until a wake arrives, a timer is due, or `deadline` passes.
pub fn idle(vm: *mut VM, deadline: Option<Instant>) {
    unsafe { rt::world_idle(vm_ptr(vm), to_ns(deadline)) }
}

/// Tasks not yet finished.
pub fn live(vm: *mut VM) -> usize {
    unsafe { rt::world_live(vm_ptr(vm)) }
}

/// Worker worlds tasks may be placed on.
pub fn workers(vm: *mut VM) -> usize {
    unsafe { rt::world_workers(vm_ptr(vm)) }
}

/// Wait for `ms` milliseconds from now, or forever when `None`.
pub fn deadline_from_ms(ms: Option<f64>) -> Option<Instant> {
    ms.map(|ms| Instant::now() + Duration::from_secs_f64(ms.max(0.0) / 1000.0))
}

fn sched_of<'a>(vm: *mut VM) -> &'a mut Sched {
    unsafe { (*vm).sched.get_or_insert_with(Default::default) }
}

/// Views a host's world made on this thread to step tasks of programs
/// the thread had no view of, by program, each with the tasks of that
/// program on this thread. A view goes with its last task, so a program
/// whose main view is gone is held by nothing once its tasks are done;
/// the seam hears the thread stop once for each.
#[derive(Default)]
struct HostViews(Vec<HostView>);

struct HostView {
    program: *const SharedCell,
    vm: Box<VM>,
    tasks: usize,
}

impl HostViews {
    fn retire(&mut self, program: *const SharedCell) {
        if let Some(i) = self.0.iter().position(|v| v.program == program) {
            let view = self.0.remove(i);
            drop(view.vm);
            unsafe { rt::thread_stop() };
        }
    }
}

impl Drop for HostViews {
    fn drop(&mut self) {
        while let Some(view) = self.0.pop() {
            drop(view.vm);
            unsafe { rt::thread_stop() };
        }
    }
}

thread_local! {
    static HOST_VIEWS: RefCell<HostViews> = RefCell::new(HostViews::default());
}

/// One step of `ctx` on the calling thread, for a host's world: the
/// view of the program on this thread (made if the thread has none),
/// the fiber on its first step, and one run to its next park or yield.
/// The view is safe for wren_lift's collector outside the step when it
/// was found safe, as a worker's is between steps. False once the task
/// is done, and `ctx` is released.
///
/// # Safety
/// `ctx` is one `world_spawn` was given, not yet released, on the world
/// it was placed on, from no task.
pub unsafe fn task_step(ctx: *mut TaskCtx) -> bool {
    let shared = match unsafe { (*ctx).shared.upgrade() } {
        Some(shared) => shared,
        None => {
            drop(unsafe { Box::from_raw(ctx) });
            return false;
        }
    };
    let program = Arc::as_ptr(&shared);
    let vm = view_for(&shared);
    let prev = crate::runtime::vm::__set_thread_local_current_vm(vm);
    let was_safe = unsafe { (*vm).thread.is_safe() };
    if was_safe {
        unsafe { (*vm).leave_safe_here() };
    }
    let id = ctx as u64;
    let sched = sched_of(vm);
    let first = matches!(
        sched.tasks.entry(id),
        std::collections::hash_map::Entry::Vacant(_)
    );
    if first {
        host_view_tasks(program, 1);
    }
    if let std::collections::hash_map::Entry::Vacant(entry) = sched.tasks.entry(id) {
        let (closure, handle) = unsafe { ((*ctx).closure, (*ctx).handle) };
        let fiber = if unsafe { (*ctx).fiber.is_null() } {
            let vm_ctx: &mut dyn NativeContext = unsafe { &mut *vm };
            let made = crate::runtime::core::fiber::fiber_new_inner(vm_ctx, closure, None);
            let fiber = made
                .as_object()
                .map_or(std::ptr::null_mut(), |p| p as *mut ObjFiber);
            if !fiber.is_null() {
                crate::runtime::core::thread::attach(vm_ctx, handle, fiber);
            }
            fiber
        } else {
            unsafe { (*ctx).fiber }
        };
        unsafe {
            (*ctx).fiber = fiber;
            (*ctx).closure = Value::null();
        }
        entry.insert(Task {
            fiber,
            handle,
            state: RunState::Runnable,
            woken: false,
        });
        unsafe { &*vm }
            .transit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&c| c != ctx);
    }
    let done = sched.tasks[&id].fiber.is_null() || sched.step_task(id);
    if done {
        sched.tasks.remove(&id);
        drop(unsafe { Box::from_raw(ctx) });
    }
    if was_safe {
        let mut spill = crate::runtime::vm::Spill::new();
        unsafe { (*vm).enter_safe_here(&mut spill) };
        std::hint::black_box(&spill);
    }
    crate::runtime::vm::__set_thread_local_current_vm(prev);
    if done && host_view_tasks(program, -1) == 0 {
        // The program's last task on this thread: a view the host's
        // world made here goes with it.
        drop(shared);
        HOST_VIEWS.with(|views| views.borrow_mut().retire(program));
    }
    !done
}

/// Count a task of `program` on this thread's host-made view in or out;
/// the tasks left, or `usize::MAX` when the view is the thread's own.
fn host_view_tasks(program: *const SharedCell, by: isize) -> usize {
    HOST_VIEWS.with(|views| {
        let mut views = views.borrow_mut();
        match views.0.iter_mut().find(|v| v.program == program) {
            Some(view) => {
                view.tasks = view.tasks.saturating_add_signed(by);
                view.tasks
            }
            None => usize::MAX,
        }
    })
}

/// Suspend the task being stepped from inside, through wren_lift's own
/// switch: its `task_step` returns as at a park. False when the calling
/// stack is no fiber that can leave from here.
pub fn task_suspend() -> bool {
    crate::runtime::core::fiber::try_krio_yield_pub(Value::null()).is_some()
}

/// This thread's view of `shared`: the one it runs, else one a host's
/// world had made here, made now if neither.
fn view_for(shared: &Arc<SharedCell>) -> *mut VM {
    let current = crate::runtime::vm::current_vm_ptr();
    if !current.is_null() && unsafe { (*current).is_view_of(shared) } {
        return current;
    }
    HOST_VIEWS.with(|views| {
        let program = Arc::as_ptr(shared);
        let mut views = views.borrow_mut();
        if let Some(view) = views.0.iter_mut().find(|v| v.program == program) {
            return &mut *view.vm as *mut VM;
        }
        unsafe { rt::thread_start() };
        let mut vm = Box::new(VM::view_of(shared));
        let ptr: *mut VM = &mut *vm;
        views.0.push(HostView {
            program,
            vm,
            tasks: 0,
        });
        ptr
    })
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);
static NEXT_WORLD: AtomicU64 = AtomicU64::new(1);
static WAIT_REGISTRY: Mutex<Option<HashMap<Token, Registration>>> = Mutex::new(None);
static WORLDS: Mutex<Option<HashMap<WorldId, Weak<Endpoint>>>> = Mutex::new(None);

fn with_registry<R>(f: impl FnOnce(&mut HashMap<Token, Registration>) -> R) -> R {
    let mut guard = WAIT_REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(HashMap::new))
}

fn with_worlds<R>(f: impl FnOnce(&mut HashMap<WorldId, Weak<Endpoint>>) -> R) -> R {
    let mut guard = WORLDS.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(HashMap::new))
}

/// Mark `token` notified and tell its world. `false` when the token
/// is unknown or already resolved, so a duplicate wake does nothing.
fn wake_local(token: Token) -> bool {
    let world = with_registry(|reg| {
        let entry = reg.get_mut(&token)?;
        if entry.status != WaitStatus::Waiting {
            return None;
        }
        entry.status = WaitStatus::Notified;
        Some(entry.world)
    });
    let Some(world) = world else {
        return false;
    };
    if let Some(endpoint) = with_worlds(|w| w.get(&world).and_then(Weak::upgrade)) {
        endpoint
            .wakes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(token);
        endpoint.changed.notify_all();
    }
    true
}

/// Take the registration if the token was notified. `Some(true)`
/// when it was, `Some(false)` when it was still waiting and `timed_out`
/// asked to claim the timeout, `None` when it stays parked.
fn claim(token: Token, timed_out: bool) -> Option<bool> {
    with_registry(|reg| {
        let status = reg.get(&token)?.status;
        match status {
            WaitStatus::Notified => {
                reg.remove(&token);
                Some(true)
            }
            WaitStatus::Waiting if timed_out => {
                reg.remove(&token);
                Some(false)
            }
            WaitStatus::Waiting => None,
        }
    })
}

fn current_stack_id() -> u64 {
    krio_fiber::current_fiber_id().unwrap_or(0)
}

pub struct Sched {
    world: WorldId,
    endpoint: Arc<Endpoint>,
    /// The tasks stepped on this view, by whichever world steps them.
    tasks: HashMap<TaskId, Task>,
    ready: VecDeque<TaskId>,
    /// Tokens parked by tasks of this world.
    waiting: HashMap<Token, TaskId>,
    timers: BinaryHeap<Reverse<(Instant, Token, TaskId)>>,
    next_task: TaskId,
    /// The task being stepped and the stack the step was started from.
    active: Option<(TaskId, u64)>,
    /// Park asked for by the active task, taken when it yields.
    pending_park: Option<(Token, Option<Instant>)>,
}

impl Default for Sched {
    fn default() -> Self {
        Self::new()
    }
}

impl Sched {
    pub fn new() -> Self {
        let world = NEXT_WORLD.fetch_add(1, Ordering::Relaxed);
        let endpoint = Arc::new(Endpoint {
            wakes: Mutex::new(VecDeque::new()),
            spawns: Mutex::new(VecDeque::new()),
            live: AtomicUsize::new(0),
            changed: Condvar::new(),
        });
        with_worlds(|w| {
            w.insert(world, Arc::downgrade(&endpoint));
        });
        Sched {
            world,
            endpoint,
            tasks: HashMap::new(),
            ready: VecDeque::new(),
            waiting: HashMap::new(),
            timers: BinaryHeap::new(),
            next_task: 1,
            active: None,
            pending_park: None,
        }
    }

    /// Tasks not yet finished.
    pub fn live(&self) -> usize {
        self.tasks.len()
    }

    /// The world's mailbox, for another thread to reach it.
    pub fn endpoint(&self) -> Arc<Endpoint> {
        Arc::clone(&self.endpoint)
    }

    /// Closures other threads asked this world to run, with their
    /// handles.
    pub fn take_spawns(&self) -> Vec<(Value, Value)> {
        let taken: Vec<(Value, Value)> = self
            .endpoint
            .spawns
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
            .collect();
        // Each becomes a task at once, so the count carries over.
        self.endpoint.live.store(
            self.tasks.len() + taken.iter().filter(|(c, _)| !c.is_null()).count(),
            Ordering::Relaxed,
        );
        taken
    }

    /// Whether anything waits in the mailbox.
    pub fn has_mail(&self) -> bool {
        let ep = &self.endpoint;
        !ep.wakes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
            || !ep
                .spawns
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
    }

    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// What the world holds for the collector: its tasks' fibers and
    /// handles.
    pub fn roots(&self) -> impl Iterator<Item = Value> + '_ {
        self.tasks
            .values()
            .flat_map(|t| [Value::object(t.fiber as *mut u8), t.handle])
    }

    /// The handle of the task being stepped, or null.
    pub fn active_handle(&self) -> Value {
        self.active
            .and_then(|(id, _)| self.tasks.get(&id))
            .map_or_else(Value::null, |t| t.handle)
    }

    pub fn spawn(&mut self, fiber: *mut ObjFiber) -> TaskId {
        self.spawn_with(fiber, Value::null())
    }

    /// Make a task of `fiber`; `handle` is finished when it ends.
    pub fn spawn_with(&mut self, fiber: *mut ObjFiber, handle: Value) -> TaskId {
        let id = self.next_task;
        self.next_task += 1;
        self.tasks.insert(
            id,
            Task {
                fiber,
                handle,
                state: RunState::Runnable,
                woken: false,
            },
        );
        self.ready.push_back(id);
        self.endpoint
            .live
            .store(self.tasks.len(), Ordering::Relaxed);
        self.endpoint.changed.notify_all();
        id
    }

    /// A fresh token, waiting on this world.
    pub fn new_waiter(&self) -> Token {
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
        with_registry(|reg| {
            reg.insert(
                token,
                Registration {
                    world: self.world,
                    status: WaitStatus::Waiting,
                },
            );
        });
        token
    }

    /// Forget a token that will not be parked on.
    pub fn discard_waiter(&self, token: Token) {
        with_registry(|reg| {
            reg.remove(&token);
        });
    }

    /// Where the calling stack stands relative to the step in progress.
    pub fn context(&self) -> Context {
        if self.active.is_some() {
            Context::Task
        } else {
            Context::Driver
        }
    }

    /// A step is in progress and the calling stack is neither the
    /// stepped task's nor the one the step was started from: a park
    /// that reaches it belongs to the task above it.
    pub fn park_travels_through_here(&self) -> bool {
        match self.active {
            Some((_, host)) if self.pending_park.is_some() => current_stack_id() != host,
            _ => false,
        }
    }

    /// Check a token belongs to this world; `Ok(true)` when it was
    /// woken before the park, which consumes it.
    pub fn check_token(&self, token: Token) -> Result<bool, String> {
        with_registry(|reg| match reg.get(&token) {
            None => Err(format!("Fiber.park: unknown waiter {token}.")),
            Some(r) if r.world != self.world => {
                Err("Fiber.park: a waiter parks on the world that made it.".to_string())
            }
            Some(r) if r.status == WaitStatus::Notified => {
                reg.remove(&token);
                Ok(true)
            }
            Some(_) => Ok(false),
        })
    }

    /// Record the active task's park; it takes effect when the task
    /// yields back to the world.
    pub fn request_park(&mut self, token: Token, deadline: Option<Instant>) {
        self.pending_park = Some((token, deadline));
    }

    /// How the active task's last park ended.
    pub fn resume_woken(&self) -> bool {
        self.active
            .and_then(|(t, _)| self.tasks.get(&t))
            .is_some_and(|t| t.woken)
    }

    /// Deliver wakes and due timers, then step every task that was
    /// ready at the start once. Returns whether any task ran.
    pub fn step(&mut self) -> bool {
        self.drain_wakes();
        self.fire_timers();
        let count = self.ready.len();
        for _ in 0..count {
            let Some(id) = self.ready.pop_front() else {
                break;
            };
            self.resume(id);
        }
        count > 0
    }

    fn drain_wakes(&mut self) {
        let tokens: Vec<Token> = {
            let mut wakes = self
                .endpoint
                .wakes
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            wakes.drain(..).collect()
        };
        for token in tokens {
            // A driver-side park watches the registry itself; only a
            // task's token is in `waiting`.
            if let Some(id) = self.waiting.remove(&token) {
                with_registry(|reg| {
                    reg.remove(&token);
                });
                self.make_runnable(id, true);
            }
        }
    }

    fn fire_timers(&mut self) {
        let now = Instant::now();
        while let Some(Reverse((when, token, id))) = self.timers.peek().copied() {
            if when > now {
                break;
            }
            self.timers.pop();
            // A wake that beat the timer is on its way through the
            // endpoint; it resolves the park instead.
            if claim(token, true) == Some(false) && self.waiting.remove(&token).is_some() {
                self.make_runnable(id, false);
            }
        }
    }

    fn make_runnable(&mut self, id: TaskId, woken: bool) {
        if let Some(task) = self.tasks.get_mut(&id) {
            if matches!(task.state, RunState::Waiting(_)) {
                task.state = RunState::Runnable;
                task.woken = woken;
                self.ready.push_back(id);
            }
        }
    }

    /// One run of task `id`'s fiber, to its next park or yield. True
    /// when the task is done, its handle finished; the registry entry
    /// stays for the caller to remove.
    fn step_task(&mut self, id: TaskId) -> bool {
        let task = self.tasks.get_mut(&id).expect("task");
        task.state = RunState::Running;
        let fiber = task.fiber;
        self.active = Some((id, current_stack_id()));
        // An abort ends the task and stays on its fiber, as under
        // `try`; the driver is not the one to unwind.
        unsafe { (*fiber).is_try = true };
        let stepped = crate::runtime::core::fiber::try_krio_call_pub(fiber, Value::null());
        self.active = None;
        let done = stepped.is_none()
            || matches!(
                unsafe { (*fiber).state },
                FiberState::Done | FiberState::Error
            );
        if done {
            let handle = self.tasks[&id].handle;
            if !handle.is_null() {
                crate::runtime::core::thread::finish(handle, fiber);
            }
        }
        done
    }

    fn resume(&mut self, id: TaskId) {
        if !self.tasks.contains_key(&id) {
            return;
        }
        self.pending_park = None;
        let done = self.step_task(id);
        let park = self.pending_park.take();
        if done {
            self.tasks.remove(&id);
            self.endpoint
                .live
                .store(self.tasks.len(), Ordering::Relaxed);
            return;
        }
        let task = self.tasks.get_mut(&id).expect("task");
        match park {
            Some((token, deadline)) => {
                task.state = RunState::Waiting(token);
                self.waiting.insert(token, id);
                if let Some(when) = deadline {
                    self.timers.push(Reverse((when, token, id)));
                }
            }
            None => {
                task.state = RunState::Runnable;
                self.ready.push_back(id);
            }
        }
    }

    /// Wait until a wake arrives, a timer is due, or `deadline`
    /// passes. Returns at once while something is ready. The thread
    /// is safe for a collector while it waits.
    ///
    /// # Safety
    /// `vm` is the view this world belongs to, on the calling thread.
    pub unsafe fn idle(&self, deadline: Option<Instant>, vm: *mut crate::runtime::vm::VM) {
        if !self.ready.is_empty() {
            return;
        }
        let mut until = deadline;
        if let Some(Reverse((when, _, _))) = self.timers.peek() {
            until = Some(until.map_or(*when, |d| d.min(*when)));
        }
        let mut spill = crate::runtime::vm::Spill::new();
        unsafe { (*vm).enter_safe(&mut spill) };
        self.wait_for_wake(until);
        unsafe { (*vm).leave_safe() };
    }

    fn wait_for_wake(&self, until: Option<Instant>) {
        let mut wakes = self
            .endpoint
            .wakes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let spawns_waiting = || {
            !self
                .endpoint
                .spawns
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        };
        while wakes.is_empty() && !spawns_waiting() {
            match until {
                Some(until) => {
                    let now = Instant::now();
                    if until <= now {
                        return;
                    }
                    let (guard, _) = self
                        .endpoint
                        .changed
                        .wait_timeout(wakes, until - now)
                        .unwrap_or_else(|e| e.into_inner());
                    wakes = guard;
                }
                None => {
                    wakes = self
                        .endpoint
                        .changed
                        .wait(wakes)
                        .unwrap_or_else(|e| e.into_inner());
                }
            }
        }
    }

    /// Step until nothing is ready or `deadline` passes; at least one
    /// step runs.
    pub fn tick(&mut self, deadline: Option<Instant>) {
        loop {
            self.step();
            if self.ready.is_empty() || deadline.is_some_and(|d| Instant::now() >= d) {
                return;
            }
        }
    }

    /// Drive the world until `token` is woken or `deadline` passes:
    /// the park of a stack that is not a task.
    ///
    /// # Safety
    /// As [`Sched::idle`].
    pub unsafe fn drive_until(
        &mut self,
        token: Token,
        deadline: Option<Instant>,
        vm: *mut crate::runtime::vm::VM,
    ) -> bool {
        loop {
            self.step();
            if let Some(woken) = claim(token, deadline.is_some_and(|d| Instant::now() >= d)) {
                return woken;
            }
            unsafe { self.idle(deadline, vm) };
        }
    }
}

/// wren_lift's own world, the World slots' defaults: the `Sched` of
/// the view, and the view's pool for placement.
pub(crate) mod local {
    use super::*;

    pub(crate) unsafe fn waiter_new(vm: *mut VM) -> Token {
        sched_of(vm).new_waiter()
    }

    pub(crate) unsafe fn waiter_discard(vm: *mut VM, token: Token) {
        sched_of(vm).discard_waiter(token)
    }

    pub(crate) fn wake(token: Token) -> bool {
        super::wake_local(token)
    }

    pub(crate) unsafe fn waiter_ready(vm: *mut VM, token: Token) -> i32 {
        match sched_of(vm).check_token(token) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(_) => -1,
        }
    }

    pub(crate) unsafe fn park_request(vm: *mut VM, token: Token, deadline_ns: u64) {
        sched_of(vm).request_park(token, from_ns(deadline_ns))
    }

    pub(crate) unsafe fn park_pending(vm: *mut VM) -> bool {
        sched_of(vm).pending_park.is_some()
    }

    pub(crate) unsafe fn resume_woken(vm: *mut VM) -> bool {
        sched_of(vm).resume_woken()
    }

    pub(crate) unsafe fn park_drive(vm: *mut VM, token: Token, deadline_ns: u64) -> bool {
        unsafe { sched_of(vm).drive_until(token, from_ns(deadline_ns), vm) }
    }

    /// A pooled task's closure goes to a worker, which makes the
    /// fiber; a local one's fiber is made already. Either way the
    /// context, the one `world_spawn` was just given, has served.
    pub(crate) unsafe fn spawn(vm: *mut VM, ctx: *mut TaskCtx, on_pool: bool) {
        let ctx = unsafe { Box::from_raw(ctx) };
        let ctx_ptr: *const TaskCtx = &*ctx;
        unsafe { &*vm }
            .transit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&c| !std::ptr::eq(c, ctx_ptr));
        if on_pool {
            unsafe { (*vm).create_thread(ctx.closure, ctx.handle) };
        } else {
            sched_of(vm).spawn_with(ctx.fiber, ctx.handle);
        }
    }

    pub(crate) unsafe fn tick(vm: *mut VM, deadline_ns: u64) -> bool {
        let sched = sched_of(vm);
        sched.tick(from_ns(deadline_ns));
        sched.live() > 0
    }

    pub(crate) unsafe fn idle(vm: *mut VM, deadline_ns: u64) {
        unsafe { sched_of(vm).idle(from_ns(deadline_ns), vm) }
    }

    pub(crate) unsafe fn live(vm: *mut VM) -> usize {
        sched_of(vm).live()
    }

    pub(crate) unsafe fn workers(vm: *mut VM) -> usize {
        unsafe { &*vm }
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .endpoints()
            .len()
    }
}

impl Drop for Sched {
    fn drop(&mut self) {
        with_worlds(|w| {
            w.remove(&self.world);
        });
        let tokens: Vec<Token> = self.waiting.keys().copied().collect();
        with_registry(|reg| {
            for t in tokens {
                reg.remove(&t);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wake_is_claimed_once() {
        let sched = Sched::new();
        let token = sched.new_waiter();
        assert!(wake_local(token));
        assert!(!wake_local(token));
        assert_eq!(sched.check_token(token), Ok(true));
        assert!(sched.check_token(token).is_err());
    }

    #[test]
    fn a_driver_park_times_out_or_wakes() {
        let mut sched = Sched::new();
        let token = sched.new_waiter();
        let deadline = Some(Instant::now() + Duration::from_millis(5));
        let mut vm = crate::runtime::vm::VM::new_default();
        let vm_ptr: *mut crate::runtime::vm::VM = &mut vm;
        assert!(!unsafe { sched.drive_until(token, deadline, vm_ptr) });

        let token = sched.new_waiter();
        let woken = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            wake_local(token)
        });
        assert!(unsafe {
            sched.drive_until(token, Some(Instant::now() + Duration::from_secs(5)), vm_ptr)
        });
        assert!(woken.join().unwrap());
    }
}
