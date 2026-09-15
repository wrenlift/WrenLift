//! Cooperative scheduler: one world per VM, driven on the thread that
//! owns the VM. A task is a fiber with a stack of its own; the world
//! resumes it with `Fiber.call` machinery and takes it back at every
//! yield or park. Waits are tokens in a process-wide registry, so a
//! wake can come from any thread; the world learns of it through its
//! endpoint, and a timer park is a min-heap entry, never a thread
//! sleep. Every notification is claimed before it is delivered, so a
//! stale timer or a second wake for the same token is harmless.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::runtime::object::{FiberState, ObjFiber};
use crate::runtime::value::Value;

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
pub fn wake(token: Token) -> bool {
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

    fn resume(&mut self, id: TaskId) {
        let Some(task) = self.tasks.get_mut(&id) else {
            return;
        };
        task.state = RunState::Running;
        let fiber = task.fiber;
        self.active = Some((id, current_stack_id()));
        self.pending_park = None;
        // An abort ends the task and stays on its fiber, as under
        // `try`; the driver is not the one to unwind.
        unsafe { (*fiber).is_try = true };
        let stepped = crate::runtime::core::fiber::try_krio_call_pub(fiber, Value::null());
        self.active = None;
        let park = self.pending_park.take();
        let done = stepped.is_none()
            || matches!(
                unsafe { (*fiber).state },
                FiberState::Done | FiberState::Error
            );
        if done {
            let task = self.tasks.remove(&id).expect("task");
            self.endpoint
                .live
                .store(self.tasks.len(), Ordering::Relaxed);
            if !task.handle.is_null() {
                crate::runtime::core::thread::finish(task.handle, fiber);
            }
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

    /// Wait for `ms` milliseconds from now, or forever when `None`.
    pub fn deadline_from_ms(ms: Option<f64>) -> Option<Instant> {
        ms.map(|ms| Instant::now() + Duration::from_secs_f64(ms.max(0.0) / 1000.0))
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
        assert!(wake(token));
        assert!(!wake(token));
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
            wake(token)
        });
        assert!(unsafe {
            sched.drive_until(token, Some(Instant::now() + Duration::from_secs(5)), vm_ptr)
        });
        assert!(woken.join().unwrap());
    }
}
