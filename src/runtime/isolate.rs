//! Isolates: one VM per thread, each with its own heap and scheduler
//! world. A value crosses between them by copy, in a thread-neutral
//! form (`Xfer`), never by reference; a channel is a shared queue of
//! such copies whose receive parks the caller on its world, so a send
//! from any isolate wakes it through the scheduler's wait registry.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjType};
use crate::runtime::sched::{self, Token};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

/// Builds a VM ready to import the program's modules, on whatever
/// thread it is called from. Set by the embedder; an isolate inherits
/// its parent's.
pub type Factory = Arc<dyn Fn() -> VM + Send + Sync>;

/// A value in transit between isolates.
#[derive(Clone, Debug, PartialEq)]
pub enum Xfer {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Xfer>),
    Map(Vec<(Xfer, Xfer)>),
    /// A `Channel` or `Isolate` instance, by its handle.
    Channel(u64),
    Isolate(u64),
}

/// Copy `v` out of the heap. Only data crosses: null, booleans,
/// numbers, strings, lists and maps of those, and channel and
/// isolate handles.
pub fn export(ctx: &dyn NativeContext, v: Value) -> Result<Xfer, String> {
    if v.is_null() {
        return Ok(Xfer::Null);
    }
    if let Some(b) = v.as_bool() {
        return Ok(Xfer::Bool(b));
    }
    if let Some(n) = v.as_num() {
        return Ok(Xfer::Num(n));
    }
    let Some(ptr) = v.as_object() else {
        return Err("cannot send this value across isolates".to_string());
    };
    let obj_type = unsafe { (*(ptr as *const ObjHeader)).obj_type };
    match obj_type {
        ObjType::String => Ok(Xfer::Str(crate::runtime::core::as_string(v).to_owned())),
        ObjType::List => {
            let list = unsafe { &*(ptr as *const ObjList) };
            list.as_slice()
                .iter()
                .map(|&e| export(ctx, e))
                .collect::<Result<Vec<_>, _>>()
                .map(Xfer::List)
        }
        ObjType::Map => {
            let map = unsafe { &*(ptr as *const ObjMap) };
            map.entries
                .iter()
                .map(|(k, &val)| Ok((export(ctx, k.0)?, export(ctx, val)?)))
                .collect::<Result<Vec<_>, String>>()
                .map(Xfer::Map)
        }
        ObjType::Instance => {
            let class = ctx.get_class_of(v);
            let id = crate::runtime::core::isolate::handle_of(v);
            if !class.is_null() && Some(class) == ctx.lookup_class("Channel") {
                Ok(Xfer::Channel(id))
            } else if !class.is_null() && Some(class) == ctx.lookup_class("Isolate") {
                Ok(Xfer::Isolate(id))
            } else {
                Err(format!(
                    "cannot send a {} across isolates",
                    ctx.get_class_name_of(v)
                ))
            }
        }
        _ => Err(format!(
            "cannot send a {} across isolates",
            ctx.get_class_name_of(v)
        )),
    }
}

/// Build `x` on this VM's heap. Allocation from a native is not a
/// safepoint, so the parts of a nested value stay put while the rest
/// is built.
pub fn import(ctx: &mut dyn NativeContext, x: &Xfer) -> Value {
    match x {
        Xfer::Null => Value::null(),
        Xfer::Bool(b) => Value::bool(*b),
        Xfer::Num(n) => Value::num(*n),
        Xfer::Str(s) => ctx.alloc_string(s.clone()),
        Xfer::List(items) => {
            let elements = items.iter().map(|e| import(ctx, e)).collect();
            ctx.alloc_list(elements)
        }
        Xfer::Channel(id) => crate::runtime::core::isolate::wrap_handle(ctx, "Channel", *id),
        Xfer::Isolate(id) => crate::runtime::core::isolate::wrap_handle(ctx, "Isolate", *id),
        Xfer::Map(entries) => {
            let map = ctx.alloc_map();
            for (k, v) in entries {
                let key = import(ctx, k);
                let val = import(ctx, v);
                unsafe {
                    (*(map.as_object().unwrap() as *mut ObjMap)).set(key, val);
                }
            }
            map
        }
    }
}

// --- Channels ---------------------------------------------------

pub struct Channel {
    queue: Mutex<VecDeque<Xfer>>,
    /// Tokens of receivers parked on an empty queue.
    receivers: Mutex<VecDeque<Token>>,
    closed: AtomicBool,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static CHANNELS: Mutex<Option<HashMap<u64, Arc<Channel>>>> = Mutex::new(None);
static ISOLATES: Mutex<Option<HashMap<u64, Arc<Isolate>>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn new_channel() -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let ch = Arc::new(Channel {
        queue: Mutex::new(VecDeque::new()),
        receivers: Mutex::new(VecDeque::new()),
        closed: AtomicBool::new(false),
    });
    lock(&CHANNELS)
        .get_or_insert_with(HashMap::new)
        .insert(id, ch);
    id
}

pub fn channel(id: u64) -> Option<Arc<Channel>> {
    lock(&CHANNELS).as_ref()?.get(&id).cloned()
}

/// Forget a channel; receivers parked on it are woken to find it
/// closed.
pub fn drop_channel(id: u64) {
    let ch = lock(&CHANNELS).as_mut().and_then(|m| m.remove(&id));
    if let Some(ch) = ch {
        ch.close();
    }
}

/// What `Channel::wait_with` found.
pub enum Receive {
    Value(Xfer),
    Closed,
    Parked,
}

impl Channel {
    /// Queue `x`; `false` once the channel is closed.
    pub fn send(&self, x: Xfer) -> bool {
        if self.closed.load(Ordering::Acquire) {
            return false;
        }
        lock(&self.queue).push_back(x);
        self.wake_one();
        true
    }

    fn wake_one(&self) {
        // A token that no longer parks anyone (its receiver timed out)
        // wakes nobody, so keep going until one lands.
        loop {
            let Some(token) = lock(&self.receivers).pop_front() else {
                return;
            };
            if sched::wake(token) {
                return;
            }
        }
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let tokens: Vec<Token> = lock(&self.receivers).drain(..).collect();
        for t in tokens {
            sched::wake(t);
        }
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub fn pending(&self) -> usize {
        lock(&self.queue).len()
    }

    pub fn try_receive(&self) -> Option<Xfer> {
        lock(&self.queue).pop_front()
    }

    /// Register `token` as a receiver, unless a value is already
    /// queued or the channel is closed.
    pub fn wait_with(&self, token: Token) -> Receive {
        let mut queue = lock(&self.queue);
        if let Some(x) = queue.pop_front() {
            return Receive::Value(x);
        }
        if self.is_closed() {
            return Receive::Closed;
        }
        // Registered under the queue lock so a send between the pop
        // and the park cannot miss the receiver.
        lock(&self.receivers).push_back(token);
        Receive::Parked
    }

    pub fn forget_receiver(&self, token: Token) {
        lock(&self.receivers).retain(|t| *t != token);
    }
}

// --- Isolates ---------------------------------------------------

pub struct Isolate {
    state: Mutex<IsolateState>,
}

struct IsolateState {
    /// `Some` once the isolate has run: its error, if it ended in one.
    outcome: Option<Option<String>>,
    joiners: Vec<Token>,
}

impl Isolate {
    pub fn is_done(&self) -> bool {
        lock(&self.state).outcome.is_some()
    }

    pub fn error(&self) -> Option<String> {
        lock(&self.state).outcome.clone().flatten()
    }

    /// Register a joiner unless already done (`true`).
    pub fn join_with(&self, token: Token) -> bool {
        let mut st = lock(&self.state);
        if st.outcome.is_some() {
            return true;
        }
        st.joiners.push(token);
        false
    }

    pub fn forget_joiner(&self, token: Token) {
        lock(&self.state).joiners.retain(|t| *t != token);
    }

    fn finish(&self, error: Option<String>) {
        let joiners = {
            let mut st = lock(&self.state);
            st.outcome = Some(error);
            std::mem::take(&mut st.joiners)
        };
        for t in joiners {
            sched::wake(t);
        }
    }
}

pub fn isolate(id: u64) -> Option<Arc<Isolate>> {
    lock(&ISOLATES).as_ref()?.get(&id).cloned()
}

/// The factory an isolate is made with: the parent's, or a VM with
/// the parent's execution settings and no module loader.
fn factory_of(parent: &VM) -> Factory {
    if let Some(f) = &parent.isolate_factory {
        return f.clone();
    }
    let config = parent.isolate_config();
    let (mode, jit, opt, traces, steps, depth, gc) = (
        config.execution_mode,
        config.jit_threshold,
        config.opt_threshold,
        config.fiber_stack_traces,
        config.step_limit,
        config.max_call_depth,
        config.gc_strategy,
    );
    Arc::new(move || {
        VM::new(crate::runtime::vm::VMConfig {
            execution_mode: mode,
            jit_threshold: jit,
            opt_threshold: opt,
            fiber_stack_traces: traces,
            step_limit: steps,
            max_call_depth: depth,
            gc_strategy: gc,
            ..Default::default()
        })
    })
}

/// Start a thread running `import "module"` on a fresh VM with `arg`
/// as its `IsolateCore.arg`. Returns the isolate's id.
pub fn spawn(parent: &VM, module: String, arg: Xfer) -> Result<u64, String> {
    let factory = factory_of(parent);
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let handle = Arc::new(Isolate {
        state: Mutex::new(IsolateState {
            outcome: None,
            joiners: Vec::new(),
        }),
    });
    lock(&ISOLATES)
        .get_or_insert_with(HashMap::new)
        .insert(id, handle.clone());
    let entry = format!(
        "import \"{}\"",
        module.replace('\\', "\\\\").replace('"', "\\\"")
    );
    let spawned = std::thread::Builder::new()
        .name(format!("isolate-{id}"))
        .spawn(move || {
            let mut vm = factory();
            vm.isolate_factory = Some(factory);
            vm.isolate_arg = Some(arg);
            let error = match vm.interpret(&format!("isolate-{id}"), &entry) {
                crate::runtime::engine::InterpretResult::Success => None,
                crate::runtime::engine::InterpretResult::CompileError => {
                    Some(format!("compile error in \"{module}\""))
                }
                crate::runtime::engine::InterpretResult::RuntimeError => Some(
                    vm.last_error
                        .clone()
                        .unwrap_or_else(|| "runtime error".to_string()),
                ),
            };
            drop(vm);
            handle.finish(error);
        });
    match spawned {
        Ok(_) => Ok(id),
        Err(e) => {
            lock(&ISOLATES).as_mut().map(|m| m.remove(&id));
            Err(format!("cannot start an isolate: {e}"))
        }
    }
}
