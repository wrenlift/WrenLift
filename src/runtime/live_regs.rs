//! Live register files of running interpreter activations.
//!
//! An interpreter activation works on its frame's register file as a
//! Rust-local `Vec<Value>` and only stows it back into the fiber at
//! safepoints, native calls and JIT dispatch. A collection triggered
//! from an allocation helper the interpreter calls directly would not
//! see those registers through the fiber, so each activation registers
//! the address of its `Vec` for as long as it runs and the collector
//! reads the current buffer through that address. An activation on a
//! fiber's own stack stays registered while the fiber is suspended,
//! and is forgotten with the stack when a suspended fiber is dropped
//! without unwinding.

use super::value::Value;
use std::cell::RefCell;

thread_local! {
    static LIVE: RefCell<Vec<(u64, *const Vec<Value>)>> = const { RefCell::new(Vec::new()) };
}

/// The fiber stack this activation runs on; 0 for the thread's own.
fn stack_id() -> u64 {
    #[cfg(feature = "host")]
    {
        krio_fiber::current_fiber_id().unwrap_or(0)
    }
    #[cfg(not(feature = "host"))]
    {
        0
    }
}

/// Keeps one register file visible to the collector until dropped.
pub struct LiveRegsGuard(*const Vec<Value>);

impl LiveRegsGuard {
    /// `regs` must outlive the guard and stay at the same address; the
    /// buffer it owns may be replaced freely.
    pub fn register(regs: &Vec<Value>) -> Self {
        let p = regs as *const Vec<Value>;
        LIVE.with(|l| l.borrow_mut().push((stack_id(), p)));
        LiveRegsGuard(p)
    }
}

impl Drop for LiveRegsGuard {
    fn drop(&mut self) {
        LIVE.with(|l| {
            let mut l = l.borrow_mut();
            if let Some(i) = l.iter().rposition(|&(_, p)| p == self.0) {
                l.remove(i);
            }
        });
    }
}

/// Drop every registration made on fiber stack `id`: the stack is
/// going away with its activations unfinished.
pub fn forget_stack(id: u64) {
    LIVE.with(|l| l.borrow_mut().retain(|&(s, _)| s != id));
}

/// Append every live register file's values to `out`.
pub fn collect_live_values(out: &mut Vec<Value>) {
    LIVE.with(|l| {
        for &(_, p) in l.borrow().iter() {
            // SAFETY: the guard contract keeps the Vec alive and in
            // place while it is registered.
            let regs = unsafe { &*p };
            out.extend_from_slice(regs);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_buffers_are_visible_and_pop_on_drop() {
        let mut regs = vec![Value::num(1.0)];
        let mut seen = Vec::new();
        {
            let _g = LiveRegsGuard::register(&regs);
            regs = vec![Value::num(2.0), Value::num(3.0)];
            collect_live_values(&mut seen);
        }
        assert_eq!(seen.len(), 2);
        seen.clear();
        collect_live_values(&mut seen);
        assert!(seen.is_empty());
        let _ = regs;
    }

    #[test]
    fn a_guard_removes_its_own_entry_and_a_stack_is_forgotten_whole() {
        let a = vec![Value::num(1.0)];
        let b = vec![Value::num(2.0), Value::num(3.0)];
        let ga = LiveRegsGuard::register(&a);
        let gb = LiveRegsGuard::register(&b);
        // Dropped out of order, as activations on different stacks are.
        drop(ga);
        let mut seen = Vec::new();
        collect_live_values(&mut seen);
        assert_eq!(seen.len(), 2);
        drop(gb);
        LIVE.with(|l| l.borrow_mut().push((7, &a as *const Vec<Value>)));
        forget_stack(7);
        seen.clear();
        collect_live_values(&mut seen);
        assert!(seen.is_empty());
    }
}
