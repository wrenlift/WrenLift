//! Live register files of running interpreter activations.
//!
//! An interpreter activation works on its frame's register file as a
//! Rust-local `Vec<Value>` and only stows it back into the fiber at
//! safepoints, native calls and JIT dispatch. A collection triggered
//! from an allocation helper the interpreter calls directly would not
//! see those registers through the fiber, so each activation registers
//! the address of its `Vec` for as long as it runs and the collector
//! reads the current buffer through that address. Registrations nest
//! strictly, so a guard pops on drop.

use super::value::Value;
use std::cell::RefCell;

thread_local! {
    static LIVE: RefCell<Vec<*const Vec<Value>>> = const { RefCell::new(Vec::new()) };
}

/// Keeps one register file visible to the collector until dropped.
pub struct LiveRegsGuard(());

impl LiveRegsGuard {
    /// `regs` must outlive the guard and stay at the same address; the
    /// buffer it owns may be replaced freely.
    pub fn register(regs: &Vec<Value>) -> Self {
        LIVE.with(|l| l.borrow_mut().push(regs as *const Vec<Value>));
        LiveRegsGuard(())
    }
}

impl Drop for LiveRegsGuard {
    fn drop(&mut self) {
        LIVE.with(|l| {
            l.borrow_mut().pop();
        });
    }
}

/// Append every live register file's values to `out`.
pub fn collect_live_values(out: &mut Vec<Value>) {
    LIVE.with(|l| {
        for &p in l.borrow().iter() {
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
}
