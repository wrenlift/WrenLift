//! Fibers with a stack of their own, which a compiled body suspends on.
//! Natively they are krio's. In the wasm AOT runtime they are ash's guest
//! fibers, which suspend through the host: a program linked with the
//! unwind/rewind transform unwinds its frames into the fiber's side stack
//! and rewinds them on resume. The wasm half keeps krio's interface, a
//! value handed in on resume and one handed out on yield, so the fiber
//! code above it is the same on both.

#[cfg(feature = "host")]
pub use krio_fiber::{Fiber, FiberStep, current_fiber_id, take_input, yield_u64};

#[cfg(not(feature = "host"))]
pub use wasm::*;

#[cfg(not(feature = "host"))]
mod wasm {
    use std::cell::{Cell, RefCell};
    use std::ffi::c_void;

    use ash_wasm_runtime::guest::{self, FiberState};

    #[link(wasm_import_module = "env")]
    unsafe extern "C" {
        /// The transform's side stack and state, and the shadow stack
        /// pointer: see `ash_wasm_runtime::guest`.
        fn ash_host_fiber_arm(data: *mut c_void, rewind: i32, sp: i32) -> i32;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FiberStep {
        Yielded,
        Done,
        Errored,
    }

    /// A fiber the runtime resumed and has not yet returned from.
    struct Running {
        id: u64,
        /// The fiber's own region, which the transform is pointed at.
        base: *mut c_void,
        /// The shadow stack pointer of the frame that resumed it.
        caller_sp: usize,
    }

    thread_local! {
        static NEXT_ID: Cell<u64> = const { Cell::new(1) };
        /// Innermost last.
        static RUNNING: RefCell<Vec<Running>> = const { RefCell::new(Vec::new()) };
        static INPUT: Cell<Option<u64>> = const { Cell::new(None) };
        static YIELDED: Cell<Option<u64>> = const { Cell::new(None) };
    }

    pub struct Fiber {
        inner: guest::Fiber,
        id: u64,
        yielded: Option<u64>,
    }

    impl Fiber {
        /// `body` is entered again to rewind it, and the transform walks
        /// that entry back to where it suspended, so it must be callable
        /// more than once.
        pub fn with_stack_size(bytes: usize, body: impl FnMut() + 'static) -> Self {
            let id = NEXT_ID.with(|n| {
                let id = n.get();
                n.set(id + 1);
                id
            });
            Fiber {
                inner: guest::Fiber::with_stack_size(bytes, body),
                id,
                yielded: None,
            }
        }

        pub fn id(&self) -> u64 {
            self.id
        }

        pub fn is_done(&self) -> bool {
            matches!(self.inner.state(), FiberState::Done | FiberState::Errored)
        }

        /// The fiber's region: its side stack, then its shadow stack.
        pub fn stack_range(&self) -> (*const u8, usize) {
            self.inner.stack_range()
        }

        pub fn saved_sp(&self) -> *const u8 {
            self.inner.saved_sp()
        }

        /// Run the fiber, or rewind it to where it yielded, handing it
        /// `input`; how it stopped.
        pub fn resume_with_u64(&mut self, input: u64) -> FiberStep {
            let (start, _) = self.inner.stack_range();
            // The region starts past the two-word header the transform's
            // data global points at.
            let base = start.wrapping_sub(8) as *mut c_void;
            let caller_sp = crate::codegen::runtime_fns::wasm_stack_here();
            INPUT.set(Some(input));
            RUNNING.with(|r| {
                r.borrow_mut().push(Running {
                    id: self.id,
                    base,
                    caller_sp,
                })
            });
            self.inner.resume();
            RUNNING.with(|r| {
                let mut r = r.borrow_mut();
                r.pop();
                // `resume` leaves the transform pointed at nothing; the
                // fiber that resumed this one is still running and may
                // yield in turn.
                if let Some(outer) = r.last() {
                    unsafe { ash_host_fiber_arm(outer.base, 0, 0) };
                }
            });
            match self.inner.state() {
                FiberState::Suspended => {
                    self.yielded = YIELDED.take();
                    FiberStep::Yielded
                }
                FiberState::Errored => FiberStep::Errored,
                _ => FiberStep::Done,
            }
        }

        /// What the fiber handed out when it last yielded.
        pub fn take_yield_u64(&mut self) -> Option<u64> {
            self.yielded.take()
        }
    }

    /// The fiber running on this thread, if any.
    pub fn current_fiber_id() -> Option<u64> {
        RUNNING.with(|r| r.borrow().last().map(|f| f.id))
    }

    /// The value the fiber was resumed with, once.
    pub fn take_input<T>() -> Option<u64> {
        INPUT.take()
    }

    /// Suspend the running fiber, handing out `value`; the value it is
    /// resumed with.
    pub fn yield_u64(value: u64) -> Option<u64> {
        YIELDED.set(Some(value));
        guest::yield_now();
        INPUT.take()
    }

    /// Where the frames under the outermost running fiber begin: the
    /// shadow stack from here up to the program's top holds the frames
    /// that resumed it. `None` when no fiber is running.
    pub fn outermost_caller_sp() -> Option<usize> {
        RUNNING.with(|r| r.borrow().first().map(|f| f.caller_sp))
    }
}
