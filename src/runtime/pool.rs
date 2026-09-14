//! The worker threads a program's tasks run on. Each worker is an OS
//! thread with a view of the program and a scheduler world of its
//! own; `Thread.create` places a closure on the world with the
//! fewest tasks, and the worker makes a fiber of it there. The pool
//! starts on first use, one worker per hardware thread, and stops
//! when the main view goes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::runtime::sched::Endpoint;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

struct Worker {
    endpoint: Arc<Endpoint>,
    join: Option<JoinHandle<()>>,
}

#[derive(Default)]
pub struct Pool {
    workers: Vec<Worker>,
    shutdown: Arc<AtomicBool>,
}

impl Pool {
    pub fn is_started(&self) -> bool {
        !self.workers.is_empty()
    }

    pub fn endpoints(&self) -> Vec<Arc<Endpoint>> {
        self.workers
            .iter()
            .map(|w| Arc::clone(&w.endpoint))
            .collect()
    }

    /// The world with the fewest tasks.
    pub fn least_loaded(&self) -> Option<Arc<Endpoint>> {
        self.workers
            .iter()
            .min_by_key(|w| w.endpoint.live())
            .map(|w| Arc::clone(&w.endpoint))
    }

    /// Tell the workers to finish and hand back their threads to join.
    pub fn shutdown(&mut self) -> Vec<JoinHandle<()>> {
        self.shutdown.store(true, Ordering::Release);
        for w in &self.workers {
            w.endpoint.push_spawn(Value::null());
        }
        self.workers
            .drain(..)
            .filter_map(|mut w| w.join.take())
            .collect()
    }
}

impl VM {
    /// Start the workers if none run yet; `n` threads, or one per
    /// hardware thread.
    pub fn start_pool(&mut self, n: Option<usize>) {
        if self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_started()
        {
            return;
        }
        let n = n
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |c| c.get()))
            .max(1);
        let shutdown = Arc::clone(&self.pool.lock().unwrap_or_else(|e| e.into_inner()).shutdown);
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx) = std::sync::mpsc::channel::<Arc<Endpoint>>();
            let stop = Arc::clone(&shutdown);
            let join = self.spawn_thread(move |vm| {
                let sched = vm.sched.get_or_insert_with(Default::default);
                let _ = tx.send(sched.endpoint());
                worker_main(vm, stop);
            });
            let endpoint = rx.recv().expect("worker endpoint");
            workers.push(Worker {
                endpoint,
                join: Some(join),
            });
        }
        self.pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .workers
            .extend(workers);
    }

    /// Run `closure` as a task on a worker.
    pub fn create_thread(&mut self, closure: Value) {
        self.start_pool(None);
        let target = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .least_loaded()
            .expect("pool started");
        target.push_spawn(closure);
    }
}

/// A worker's life: make tasks of the closures sent to it, run its
/// world, and rest when nothing is ready.
fn worker_main(vm: &mut VM, shutdown: Arc<AtomicBool>) {
    let vm_ptr: *mut VM = vm;
    crate::runtime::vm::__set_thread_local_current_vm(vm_ptr);
    vm.leave_safe();
    loop {
        let spawns = vm
            .sched
            .as_ref()
            .map(|s| s.take_spawns())
            .unwrap_or_default();
        for closure in spawns {
            if closure.is_null() {
                continue;
            }
            let fiber = crate::runtime::core::fiber::fiber_new_inner(vm, closure, None);
            if let Some(ptr) = fiber.as_object() {
                let fiber = ptr as *mut crate::runtime::object::ObjFiber;
                vm.sched.get_or_insert_with(Default::default).spawn(fiber);
            }
        }
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let sched: *mut crate::runtime::sched::Sched =
            &mut **vm.sched.get_or_insert_with(Default::default);
        // SAFETY: the world is this view's, on this thread; a task it
        // steps reaches the view through the thread-local pointer.
        unsafe {
            (*sched).step();
            if !(*sched).has_ready() && !(*sched).has_mail() {
                (*sched).idle(None, vm_ptr);
            }
        }
    }
    let mut spill = crate::runtime::vm::Spill::new();
    vm.enter_safe(&mut spill);
    std::hint::black_box(&spill);
}
