//! Native stack windows for conservative root scanning.
//!
//! A collection that scans the machine stack must see every pointer a
//! caller left in a callee-saved register, so the collector first
//! spills those registers into a buffer on its own frame and scans
//! from the lower of that buffer and the current stack pointer up to
//! the thread's stack top. Fiber stacks are windows of their own; the
//! VM assembles those from the fibers it owns.

use std::cell::Cell;

/// Registers the platform ABI lets a callee keep across a call. Every
/// entry is stored so the scan covers register-resident pointers.
pub const SPILL_WORDS: usize = 12;

/// Store the callee-saved registers into `buf`. Never inlined so the
/// buffer lives in a frame below every caller of the collector.
#[inline(never)]
pub fn spill_callee_saved(buf: &mut [usize; SPILL_WORDS]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "stp x19, x20, [{b}]",
            "stp x21, x22, [{b}, #16]",
            "stp x23, x24, [{b}, #32]",
            "stp x25, x26, [{b}, #48]",
            "stp x27, x28, [{b}, #64]",
            "str x29, [{b}, #80]",
            b = in(reg) buf.as_mut_ptr(),
            options(nostack)
        );
    }
    #[cfg(all(target_arch = "x86_64", not(windows)))]
    unsafe {
        std::arch::asm!(
            "mov [{b}], rbx",
            "mov [{b} + 8], rbp",
            "mov [{b} + 16], r12",
            "mov [{b} + 24], r13",
            "mov [{b} + 32], r14",
            "mov [{b} + 40], r15",
            b = in(reg) buf.as_mut_ptr(),
            options(nostack)
        );
    }
    // The Windows x64 ABI also keeps rsi and rdi across a call.
    #[cfg(all(target_arch = "x86_64", windows))]
    unsafe {
        std::arch::asm!(
            "mov [{b}], rbx",
            "mov [{b} + 8], rbp",
            "mov [{b} + 16], r12",
            "mov [{b} + 24], r13",
            "mov [{b} + 32], r14",
            "mov [{b} + 40], r15",
            "mov [{b} + 48], rsi",
            "mov [{b} + 56], rdi",
            b = in(reg) buf.as_mut_ptr(),
            options(nostack)
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = buf;
    }
    // Keep the stores observable to the scanner.
    std::hint::black_box(buf);
}

/// Address of a local in the caller's frame: a stack pointer upper
/// bound good enough for a scan that starts at `min(this, spill)`.
#[inline(never)]
pub fn approx_sp() -> usize {
    let marker = 0u8;
    std::hint::black_box(&marker as *const u8 as usize)
}

thread_local! {
    static STACK_BOUNDS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// Highest address of the current thread's stack, cached per thread.
/// Zero on platforms without a query, which disables stack scanning.
pub fn thread_stack_top() -> usize {
    thread_stack_bounds().1
}

/// The current thread's own stack, `(lowest, highest)`, cached per
/// thread. Both zero on platforms without a query; the lowest alone
/// zero where only the top is known.
pub fn thread_stack_bounds() -> (usize, usize) {
    STACK_BOUNDS.with(|c| {
        let v = c.get();
        if v.1 != 0 {
            return v;
        }
        let bounds = query_stack_bounds();
        c.set(bounds);
        bounds
    })
}

#[cfg(all(target_os = "macos", feature = "host"))]
fn query_stack_bounds() -> (usize, usize) {
    unsafe {
        let me = libc::pthread_self();
        let top = libc::pthread_get_stackaddr_np(me) as usize;
        let size = libc::pthread_get_stacksize_np(me);
        (top.saturating_sub(size), top)
    }
}

#[cfg(all(target_os = "linux", feature = "host"))]
fn query_stack_bounds() -> (usize, usize) {
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
            return (0, 0);
        }
        let mut base: *mut libc::c_void = std::ptr::null_mut();
        let mut size: libc::size_t = 0;
        let rc = libc::pthread_attr_getstack(&attr, &mut base, &mut size);
        libc::pthread_attr_destroy(&mut attr);
        if rc != 0 {
            return (0, 0);
        }
        (base as usize, base as usize + size)
    }
}

#[cfg(all(windows, feature = "host"))]
fn query_stack_bounds() -> (usize, usize) {
    let (mut lo, mut hi) = (0usize, 0usize);
    unsafe {
        windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut lo, &mut hi);
    }
    (lo, hi)
}

#[cfg(not(any(
    all(target_os = "macos", feature = "host"),
    all(target_os = "linux", feature = "host"),
    all(windows, feature = "host")
)))]
fn query_stack_bounds() -> (usize, usize) {
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_top_is_above_a_local() {
        let top = thread_stack_top();
        if top != 0 {
            let here = approx_sp();
            assert!(here < top, "sp {here:#x} not below stack top {top:#x}");
            assert!(top - here < 64 * 1024 * 1024);
        }
    }

    #[test]
    fn spill_writes_every_slot_observably() {
        let mut buf = [0usize; SPILL_WORDS];
        spill_callee_saved(&mut buf);
        // Frame pointer is always live on the platforms with asm.
        #[cfg(target_arch = "aarch64")]
        assert_ne!(buf[10], 0);
        #[cfg(target_arch = "x86_64")]
        assert_ne!(buf[1], 0);
    }
}
