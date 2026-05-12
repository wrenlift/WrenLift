//! Smoke tests for `wlift_alloc`. These exercise the allocator
//! through its public Rust API rather than via `#[global_allocator]`
//! — installing the global allocator is per-binary and would
//! conflict with `cargo test`'s default test harness.

use std::alloc::{GlobalAlloc, Layout};
use wlift_alloc::{Wlift, pressure_release, try_alloc, wlift_free, wlift_malloc, wlift_realloc};

#[test]
fn round_trip_small() {
    let a = Wlift;
    let layouts = [
        Layout::from_size_align(8, 8),
        Layout::from_size_align(16, 8),
        Layout::from_size_align(48, 8),
        Layout::from_size_align(120, 8),
        Layout::from_size_align(255, 8),
        Layout::from_size_align(1000, 8),
    ];
    for l in layouts {
        let l = l.unwrap();
        unsafe {
            let p = a.alloc(l);
            assert!(!p.is_null());
            // Write a pattern and read it back.
            std::ptr::write_bytes(p, 0xAB, l.size());
            for i in 0..l.size() {
                assert_eq!(*p.add(i), 0xAB);
            }
            a.dealloc(p, l);
        }
    }
}

#[test]
fn round_trip_large() {
    let a = Wlift;
    let l = Layout::from_size_align(64 * 1024, 16).unwrap();
    unsafe {
        let p = a.alloc(l);
        assert!(!p.is_null());
        std::ptr::write_bytes(p, 0xCD, l.size());
        assert_eq!(*p.add(l.size() / 2), 0xCD);
        a.dealloc(p, l);
    }
}

#[test]
fn realloc_in_place() {
    let a = Wlift;
    let l = Layout::from_size_align(40, 8).unwrap();
    unsafe {
        let p = a.alloc(l);
        std::ptr::write_bytes(p, 0xEE, 40);
        let p2 = a.realloc(p, l, 60);
        // 40 and 60 are both in the 64-byte class — same chunk.
        assert_eq!(p, p2);
        // Original bytes preserved.
        for i in 0..40 {
            assert_eq!(*p2.add(i), 0xEE);
        }
        a.dealloc(p2, Layout::from_size_align(60, 8).unwrap());
    }
}

#[test]
fn realloc_across_classes() {
    let a = Wlift;
    let l = Layout::from_size_align(40, 8).unwrap();
    unsafe {
        let p = a.alloc(l);
        std::ptr::write_bytes(p, 0x77, 40);
        let p2 = a.realloc(p, l, 800);
        // 40 → 64 class, 800 → 1024 class. Different chunk.
        assert!(!p2.is_null());
        // Original 40 bytes copied.
        for i in 0..40 {
            assert_eq!(*p2.add(i), 0x77);
        }
        a.dealloc(p2, Layout::from_size_align(800, 8).unwrap());
    }
}

#[test]
fn c_abi_symmetry() {
    let p = wlift_malloc(64, 8);
    assert!(!p.is_null());
    unsafe {
        std::ptr::write_bytes(p, 0x42, 64);
        let p2 = wlift_realloc(p, 64, 8, 128);
        assert!(!p2.is_null());
        for i in 0..64 {
            assert_eq!(*p2.add(i), 0x42);
        }
        wlift_free(p2, 128, 8);
    }
}

#[test]
fn many_small_round_trips() {
    let a = Wlift;
    let l = Layout::from_size_align(32, 8).unwrap();
    let mut ptrs = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        unsafe {
            let p = a.alloc(l);
            assert!(!p.is_null());
            ptrs.push(p);
        }
    }
    // Free in reverse order to stress free-list LIFO behaviour.
    while let Some(p) = ptrs.pop() {
        unsafe {
            a.dealloc(p, l);
        }
    }
    // After freeing, pressure_release should hint many slabs as
    // reclaimable.
    let released = pressure_release();
    assert!(released > 0, "expected some bytes hinted reclaimable");
}

#[test]
fn try_alloc_returns_nonnull() {
    let p = try_alloc(128, 8).expect("alloc failed");
    unsafe {
        wlift_free(p.as_ptr(), 128, 8);
    }
}

#[test]
fn zero_size_returns_aligned_dangling() {
    let p = wlift_malloc(0, 16);
    assert_eq!(p as usize, 16);
    // No free needed for zero-sized allocations.
}
