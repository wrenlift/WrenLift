//! Stopping the world: the threads of a program and how a collection
//! brings them to rest. Each thread's view of the program has a
//! record here; the record says whether the thread is running Wren
//! or is safe, and while safe, where its stack stands and what it
//! holds precisely, so the collector on another thread can scan it.
//!
//! A collector sets the request and waits for every other thread to
//! be safe. A running thread reaches a safepoint (an allocation, the
//! interpreter's poll), publishes itself and parks until the request
//! clears; a thread already safe (blocked in a native, idle in its
//! scheduler, or back in the embedder) is scanned where it stands
//! and waits at its next transition to running.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::runtime::value::Value;

pub const RUNNING: u8 = 0;
pub const SAFE: u8 = 1;

/// One thread's standing with the collector.
pub struct ThreadState {
    pub state: AtomicU8,
    /// Where the thread's stack stands while safe: the low end of the
    /// range to scan. 0 until the thread has run.
    pub sp: AtomicUsize,
    /// The high end of the thread's stack, recorded when it first
    /// becomes safe on its own thread.
    pub stack_top: AtomicUsize,
    /// The krio fiber the thread was inside when it became safe, or 0.
    pub fiber_id: AtomicU64,
    /// What the thread holds outside any stack: its fiber, api stack,
    /// pools, compiled-code roots. Published when it becomes safe.
    pub roots: Mutex<Vec<Value>>,
}

impl ThreadState {
    fn new() -> Arc<ThreadState> {
        Arc::new(ThreadState {
            state: AtomicU8::new(SAFE),
            sp: AtomicUsize::new(0),
            stack_top: AtomicUsize::new(0),
            fiber_id: AtomicU64::new(0),
            roots: Mutex::new(Vec::new()),
        })
    }

    pub fn is_safe(&self) -> bool {
        self.state.load(Ordering::SeqCst) == SAFE
    }
}

/// The program's threads and the collector's request to them.
pub struct World {
    threads: Mutex<Vec<Arc<ThreadState>>>,
    requested: AtomicBool,
    /// Held by the collecting thread for the length of a collection.
    collector: Mutex<()>,
    /// Signalled when a thread becomes safe and when the request
    /// clears.
    changed: Condvar,
    gate: Mutex<()>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> World {
        World {
            threads: Mutex::new(Vec::new()),
            requested: AtomicBool::new(false),
            collector: Mutex::new(()),
            changed: Condvar::new(),
            gate: Mutex::new(()),
        }
    }

    /// Register a thread; it starts safe with nothing to scan.
    pub fn join(&self) -> Arc<ThreadState> {
        let t = ThreadState::new();
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(t.clone());
        t
    }

    pub fn leave(&self, t: &Arc<ThreadState>) {
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|x| !Arc::ptr_eq(x, t));
    }

    /// The threads of the program other than `me`.
    pub fn others(&self, me: &Arc<ThreadState>) -> Vec<Arc<ThreadState>> {
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|x| !Arc::ptr_eq(x, me))
            .cloned()
            .collect()
    }

    pub fn thread_count(&self) -> usize {
        self.threads.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether a collection is waiting for the threads to stop.
    #[inline(always)]
    pub fn requested(&self) -> bool {
        self.requested.load(Ordering::Relaxed)
    }

    /// Mark `t` safe with its stack and roots published, and tell a
    /// waiting collector.
    pub fn become_safe(&self, t: &ThreadState, sp: usize, fiber_id: u64, roots: Vec<Value>) {
        *t.roots.lock().unwrap_or_else(|e| e.into_inner()) = roots;
        t.sp.store(sp, Ordering::Relaxed);
        t.fiber_id.store(fiber_id, Ordering::Relaxed);
        t.state.store(SAFE, Ordering::SeqCst);
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }

    /// Wait out any collection in progress, then mark `t` running.
    /// The store and the check are both sequentially consistent
    /// against the collector's request and its read of the state, so
    /// one side always sees the other.
    pub fn become_running(&self, t: &ThreadState) {
        loop {
            t.state.store(RUNNING, Ordering::SeqCst);
            if !self.requested.load(Ordering::SeqCst) {
                return;
            }
            t.state.store(SAFE, Ordering::SeqCst);
            let mut g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            // The collector may have seen the thread running just now.
            self.changed.notify_all();
            while self.requested() {
                g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    /// Become the collector: every other thread is safe when this
    /// returns, and stays so until `resume`. `t` must be safe while
    /// it waits here, so a collector already at work can scan it.
    pub fn stop(&self, t: &ThreadState) -> std::sync::MutexGuard<'_, ()> {
        let guard = self.collector.lock().unwrap_or_else(|e| e.into_inner());
        self.requested.store(true, Ordering::SeqCst);
        t.state.store(RUNNING, Ordering::SeqCst);
        let mut g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let all_safe = self
                .threads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .all(|x| std::ptr::eq(x.as_ref(), t) || x.is_safe());
            if all_safe {
                break;
            }
            g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        guard
    }

    /// Let the world go after a collection.
    pub fn resume(&self, guard: std::sync::MutexGuard<'_, ()>) {
        self.requested.store(false, Ordering::SeqCst);
        {
            let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            self.changed.notify_all();
        }
        drop(guard);
    }
}

/// A lock a thread may take again while it holds it: the tier
/// bookkeeping's entry points call each other.
pub struct ReentrantLock {
    mutex: Mutex<()>,
    owner: AtomicU64,
    depth: AtomicUsize,
    changed: Condvar,
}

impl Default for ReentrantLock {
    fn default() -> Self {
        Self::new()
    }
}

/// A number naming the calling OS thread, stable for its lifetime.
pub fn thread_key() -> u64 {
    thread_local! {
        static KEY: u8 = const { 0 };
    }
    KEY.with(|k| k as *const u8 as u64)
}

impl ReentrantLock {
    pub fn new() -> ReentrantLock {
        ReentrantLock {
            mutex: Mutex::new(()),
            owner: AtomicU64::new(0),
            depth: AtomicUsize::new(0),
            changed: Condvar::new(),
        }
    }

    pub fn lock(&self) -> ReentrantGuard<'_> {
        let me = thread_key();
        if self.owner.load(Ordering::Acquire) == me {
            self.depth.fetch_add(1, Ordering::Relaxed);
            return ReentrantGuard(self);
        }
        let mut g = self.mutex.lock().unwrap_or_else(|e| e.into_inner());
        while self.owner.load(Ordering::Acquire) != 0 {
            g = self.changed.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        self.owner.store(me, Ordering::Release);
        self.depth.store(1, Ordering::Relaxed);
        ReentrantGuard(self)
    }
}

pub struct ReentrantGuard<'a>(&'a ReentrantLock);

impl Drop for ReentrantGuard<'_> {
    fn drop(&mut self) {
        let lock = self.0;
        if lock.depth.fetch_sub(1, Ordering::Relaxed) == 1 {
            let _g = lock.mutex.lock().unwrap_or_else(|e| e.into_inner());
            lock.owner.store(0, Ordering::Release);
            lock.changed.notify_one();
        }
    }
}
