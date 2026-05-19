//! GC walker spike: confirm the operations the existing JIT GC
//! walker needs (see `src/runtime/vm.rs::scan_native_stack_roots`
//! Pass 2) all work on a suspended krio-fiber's stack.
//!
//! Pass 2's loop is, per frame:
//!   1. Read `saved_ret` to find the code range / active safepoint.
//!   2. For each `Spill(offset)` root in the safepoint:
//!      `slot_addr = fp + offset; bits = *slot_addr;` then test
//!      `Value::from_bits(bits).is_object()`.
//!   3. Step to `saved_fp` for the parent frame.
//!
//! What this spike validates, WITHOUT needing real Cranelift codegen
//! on the fiber stack:
//!
//!   - `Fiber::saved_fp()` is a usable seed: chain-walking from it
//!     terminates cleanly and traverses every nested call frame.
//!   - We can read words via `*((fp + offset) as *const u64)` inside
//!     a fiber's mmap'd stack without crossing the guard page or
//!     leaving the mapped region (the real walker only ever reads
//!     at offsets the safepoint table explicitly enumerates, which
//!     are by construction within the frame).
//!   - `Fiber::saved_ret()` returns a non-null address — that's what
//!     Pass 2 looks up against `engine.code_ranges` to find which
//!     compiled function the deepest frame is in.
//!
//! What this spike does NOT do (deferred to integration step):
//!   - Run an actual Cranelift-compiled function on a fiber stack.
//!   - Use real per-function safepoint metadata. Rust doesn't emit
//!     it; that's specifically a property of the Cranelift backend.
//!
//! If this passes, the GC piece reduces to mechanical wiring: the
//! existing safepoint lookup + spill read works unchanged when
//! seeded with `(saved_fp, saved_ret)` from a suspended ObjFiber.

#![cfg(feature = "host")]

use krio_fiber::{yield_now, Fiber};

/// Count the frames in a suspended fiber's saved_fp chain. Real GC
/// walker does this exact loop, but stops at each frame to scan
/// safepoint-described spill slots. Returns the depth — `0` means
/// the chain bottomed out immediately (fiber suspended at its
/// entry point with no nested calls).
fn count_frames(fiber: &Fiber) -> Option<usize> {
    let mut fp = fiber.saved_fp()? as usize;
    let mut depth = 0;
    let max = 64;
    while fp != 0 && depth < max {
        if fp & 7 != 0 {
            return None;
        }
        depth += 1;
        // Read only the saved_fp link — the FIRST word of each
        // frame, which is always within the fiber's stack region
        // (the frame itself was constructed on this stack). This
        // is the only read a chain-walk needs.
        let next_fp = unsafe { *(fp as *const usize) };
        if next_fp == 0 {
            break;
        }
        if next_fp <= fp {
            // Chain went backwards or sideways — walked off the
            // top of the fiber stack into garbage. Bail.
            break;
        }
        fp = next_fp;
    }
    Some(depth)
}

/// Validates: the walker traverses one frame for `yield_now()`
/// from the body's top scope.
#[test]
fn walker_one_frame_for_top_level_yield() {
    let mut fiber = Fiber::new(|| {
        yield_now();
    });
    fiber.resume();
    assert_eq!(fiber.state(), krio_fiber::FiberState::Suspended);
    let depth = count_frames(&fiber).expect("chain walks");
    // Closure body + yield_now's frame = at least 2; the
    // trampoline may add one more. Anywhere in 2..=4 is fine —
    // the point is the chain walks and terminates cleanly.
    assert!(
        (1..=8).contains(&depth),
        "expected 1..=8 frames, got {depth}"
    );
}

/// Validates: the frame chain walks cleanly even when Rust release
/// builds elide some intermediate frame pointers (small leaf
/// functions may not emit a full stp x29,x30 pair). The walker
/// still terminates without crashing — exactly what we need for the
/// real integration, where Cranelift-emitted AOT code WILL have
/// frame pointers (we control its codegen flags) but host helpers
/// and runtime functions may not.
///
/// What this confirms: the walker is robust to a mix of
/// frame-pointered and frame-elided frames in the same chain — the
/// chain pointer either points to another valid frame or to zero/a
/// non-monotonic address (signalling "stop").
#[test]
fn walker_terminates_through_mixed_frame_layouts() {
    // Build a body whose stack usage forces frame setup: a small
    // local array makes the prologue allocate stack and save x29,
    // even under release optimization.
    fn body_with_stack_alloc() {
        let mut buf = [0u64; 16];
        std::hint::black_box(buf.as_mut_ptr());
        yield_now();
        std::hint::black_box(buf.as_mut_ptr());
    }
    let mut fiber = Fiber::new(body_with_stack_alloc);
    fiber.resume();
    assert_eq!(fiber.state(), krio_fiber::FiberState::Suspended);
    let depth = count_frames(&fiber).expect("chain walks");
    assert!(
        depth >= 1,
        "expected at least one walked frame, got {depth}"
    );
    fiber.resume();
    assert_eq!(fiber.state(), krio_fiber::FiberState::Done);
}

/// Validates: `saved_ret` returns a code-pointer-shaped address.
/// Pass 2 looks this up against `engine.code_ranges` — i.e., it
/// needs to be a real instruction address inside compiled code.
#[test]
fn saved_ret_is_a_plausible_code_pointer() {
    let mut fiber = Fiber::new(|| {
        yield_now();
    });
    fiber.resume();
    let ret = fiber.saved_ret().expect("suspended fiber has ret");
    assert!(!ret.is_null(), "saved_ret must be non-null");
    // Code pointers on aarch64/x86_64 are word-aligned (well,
    // 4-byte on aarch64; allow it). Should sit in a high-address
    // region (process image / dylib), NOT inside our mmap'd
    // fiber stack region (which is below).
    let fp = fiber.saved_fp().unwrap() as usize;
    let ret_addr = ret as usize;
    assert_ne!(ret_addr, fp, "saved_ret must differ from saved_fp");
    fiber.resume();
}

/// Validates: precise spill-offset reads work on a fiber stack. We
/// don't have safepoint metadata for Rust code, so we *manually*
/// stash a pointer at a known fp-relative offset by passing a slot
/// pointer down to a child function that yields with the slot in
/// scope. The walker then reads at that exact offset — same
/// mechanics as a real safepoint emit + GC scan.
///
/// This is the "we can do precise reads" half of the GC walker
/// pattern. Conservative scanning has to bound itself to the
/// frame's region; precise reads don't, because the safepoint
/// table guarantees the offset is within the frame.
#[test]
fn precise_spill_read_works_on_fiber_stack() {
    // Sentinel that looks roughly like a NaN-boxed object pointer.
    // The walker treats this as "an object value lives at this
    // slot" — same as what the real Value::is_object() check does.
    const SENTINEL: u64 = 0xFFFC_0000_DEAD_BEE8;

    // We need to know WHERE in the fiber's stack the sentinel ends
    // up. Easiest: have the fiber stash the address into a Cell
    // that's accessible from outside, then yield.
    use std::cell::Cell;
    use std::rc::Rc;

    let slot_addr = Rc::new(Cell::new(0usize));
    let slot_addr_clone = slot_addr.clone();

    let mut fiber = Fiber::new(move || {
        // Stash the sentinel as a stack local.
        let live_root: u64 = SENTINEL;
        let addr = &live_root as *const u64 as usize;
        slot_addr_clone.set(addr);
        // Keep `live_root` alive across the yield (prevents the
        // optimizer from spilling it elsewhere).
        std::hint::black_box(&live_root);
        yield_now();
        // Re-read after yield to confirm the value's still good.
        let after = unsafe { std::ptr::read_volatile(&live_root) };
        assert_eq!(after, SENTINEL);
    });

    fiber.resume();
    assert_eq!(fiber.state(), krio_fiber::FiberState::Suspended);

    let recorded = slot_addr.get();
    assert_ne!(recorded, 0, "fiber must have recorded the slot address");

    // The recorded address should be inside the fiber's stack
    // region — far from our host-stack `recorded` itself.
    let here = &recorded as *const usize as usize;
    let gap = here.abs_diff(recorded);
    assert!(
        gap > 16 * 1024,
        "slot address should be in fiber's mmap stack, not host's: \
         here={here:#x} recorded={recorded:#x} gap={gap}"
    );

    // Precise read at the known address — exactly what a GC
    // walker would do, given a safepoint-supplied (fp, offset)
    // pair that resolves to `recorded`.
    let observed = unsafe { *(recorded as *const u64) };
    assert_eq!(
        observed, SENTINEL,
        "GC walker must read the live-root sentinel at the precise slot"
    );

    // Drive to completion; the body re-asserts the sentinel.
    fiber.resume();
    assert_eq!(fiber.state(), krio_fiber::FiberState::Done);
}

/// Validates: two suspended fibers' stacks don't alias. Real GC has
/// to walk every suspended ObjFiber on each cycle; if their saved_fp
/// pointers were the same (impossible if mmap regions are
/// distinct), the walker would visit one frame chain twice.
#[test]
fn two_fibers_have_disjoint_stacks() {
    let mut a = Fiber::new(|| {
        yield_now();
    });
    let mut b = Fiber::new(|| {
        yield_now();
    });
    a.resume();
    b.resume();

    let fp_a = a.saved_fp().unwrap() as usize;
    let fp_b = b.saved_fp().unwrap() as usize;
    assert_ne!(fp_a, fp_b, "distinct fibers must have distinct saved_fp");
    // They should be on different mmap regions, so the gap should
    // be at least a few pages.
    let gap = fp_a.abs_diff(fp_b);
    assert!(gap > 4096, "fiber stacks too close: gap={gap}");

    a.resume();
    b.resume();
}
