//! wlift_alloc — in-tree allocator for the wren_lift runtime.
//!
//! Why this crate exists: the wlift binary plus the cdylib plugins
//! (`wlift_sqlite`, `wlift_gpu`, `…`) need to share one allocator
//! instance, otherwise cross-FFI buffer ownership goes wrong —
//! plugin allocates a `Vec`, host's GC tries to free it, the two
//! ends route through different mimallocs / different system
//! arenas, and `free()` corrupts the heap. macOS's
//! `libsystem_malloc` adds the further wrinkle that freed pages
//! aren't returned to the OS aggressively, so `Activity Monitor`
//! shows the process holding gigabytes of memory long after the
//! GC reclaimed it logically.
//!
//! Owning the allocator solves both: one code path, one set of
//! free-list head pointers, return-to-OS policy under our control.
//!
//! Design (kept boring on purpose):
//!
//! - **Size classes** (16, 32, 64, 128, 256, 512, 1024 B). Each
//!   class has a process-wide free list of fixed-size chunks. New
//!   chunks come from 64 KiB *slabs* — `mmap`'d up front, divided
//!   into N fixed-size chunks, freed back to the OS when the whole
//!   slab is empty.
//! - **Large pool** for `size > 1024`: direct `mmap` per allocation,
//!   `munmap` on `dealloc`. The kernel handles return-to-OS for us.
//! - **Madvise on idle.** [`pressure_release`] walks free slabs and
//!   `madvise(MADV_FREE)`s their pages so the kernel can reclaim
//!   under pressure without us giving up the virtual mapping.
//!
//! Thread-safety: a global `Mutex` over each size class's free
//! list. Single-threaded fast path on the host (Wren runs on one
//! thread); the JIT compile broker is the only other allocator
//! caller. Contention would warrant per-thread caches later.
//!
//! Stable C ABI: [`wlift_malloc`], [`wlift_free`], [`wlift_realloc`].
//! Plugins call these directly to ensure single-instance state
//! regardless of how many cdylibs link `wlift_alloc` statically —
//! all of them resolve to the binary's symbols at dynamic-link
//! time.

use std::alloc::{GlobalAlloc, Layout};
use std::ptr::NonNull;

// We can't use `std::sync::Mutex` here — its first lock
// acquisition lazily allocates a pthread mutex via `OnceBox`,
// which re-enters our global allocator and stack-overflows
// during early startup. `spin::Mutex` is a pure-atomic
// spin-wait with no OS handles, no lazy init.
use spin::Mutex;

mod large;
mod slab;
#[cfg(feature = "stats")]
mod stats;

pub use slab::Slab;

// Size classes in bytes. Allocations round up to the next class.
// 7 classes is a balance between internal fragmentation (no class
// would imply ~50% slop on power-of-two misses) and metadata
// overhead (each class costs a Mutex + free-list head).
const SIZE_CLASSES: &[usize] = &[16, 32, 64, 128, 256, 512, 1024];

/// Largest size routed through the slab pool. Allocations strictly
/// above this go through [`large`] which `mmap`s directly.
const MAX_SLAB_BYTES: usize = 1024;

/// Per-slab chunk count is `SLAB_BYTES / class_size`. 64 KiB hits
/// the macOS small-arena boundary so the kernel can reclaim slabs
/// cleanly without splitting pages.
const SLAB_BYTES: usize = 64 * 1024;

/// Round `n` up to the next size class. Returns `None` for
/// allocations that should go to the large pool.
#[inline]
fn classify(n: usize) -> Option<usize> {
    for &c in SIZE_CLASSES {
        if n <= c {
            return Some(c);
        }
    }
    None
}

/// Process-wide allocator state. One instance, lives in a
/// `static` so there's no `OnceLock` semaphore — the `Mutex`
/// in each class is `const`-constructed and never deadlocks
/// against early-startup allocations from std::rt.
///
/// **Bootstrap invariant.** Construction of this static must
/// not allocate. We rely on `Mutex::new` being `const fn`
/// (stable since Rust 1.63) and `ClassPool::new` being a
/// `const fn` we provide.
struct Alloc {
    classes: [Mutex<slab::ClassPool>; 7],
}

static STATE: Alloc = Alloc {
    classes: [
        Mutex::new(slab::ClassPool::new(16, SLAB_BYTES)),
        Mutex::new(slab::ClassPool::new(32, SLAB_BYTES)),
        Mutex::new(slab::ClassPool::new(64, SLAB_BYTES)),
        Mutex::new(slab::ClassPool::new(128, SLAB_BYTES)),
        Mutex::new(slab::ClassPool::new(256, SLAB_BYTES)),
        Mutex::new(slab::ClassPool::new(512, SLAB_BYTES)),
        Mutex::new(slab::ClassPool::new(1024, SLAB_BYTES)),
    ],
};

fn state() -> &'static Alloc {
    &STATE
}

/// Public global-allocator type. Install via
/// `#[global_allocator] static GLOBAL: wlift_alloc::Wlift = wlift_alloc::Wlift;`
pub struct Wlift;

unsafe impl GlobalAlloc for Wlift {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        wlift_alloc_impl(layout.size(), layout.align())
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { wlift_dealloc_impl(ptr, layout.size(), layout.align()) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = wlift_alloc_impl(layout.size(), layout.align());
        if !ptr.is_null() {
            unsafe {
                std::ptr::write_bytes(ptr, 0, layout.size());
            }
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { wlift_realloc_impl(ptr, layout.size(), layout.align(), new_size) }
    }
}

// ---------------------------------------------------------------------------
// Implementation entry points (shared with the C ABI exports below).
// ---------------------------------------------------------------------------

fn wlift_alloc_impl(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        // Match `std::alloc::System`: zero-sized allocations get a
        // dangling-but-aligned pointer. The caller must never
        // dereference it; `Vec::new()` etc. rely on this contract.
        return align as *mut u8;
    }
    let needed = size.max(align);
    if let Some(class) = classify(needed) {
        let idx = SIZE_CLASSES.iter().position(|&c| c == class).unwrap();
        let mut pool = state().classes[idx].lock();
        pool.alloc()
    } else {
        large::alloc(size, align)
    }
}

unsafe fn wlift_dealloc_impl(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    let needed = size.max(align);
    if let Some(class) = classify(needed) {
        let idx = SIZE_CLASSES.iter().position(|&c| c == class).unwrap();
        let mut pool = state().classes[idx].lock();
        unsafe {
            pool.free(ptr);
        }
    } else {
        unsafe {
            large::dealloc(ptr, size);
        }
    }
}

unsafe fn wlift_realloc_impl(
    ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    if ptr.is_null() {
        return wlift_alloc_impl(new_size, align);
    }
    if new_size == 0 {
        unsafe {
            wlift_dealloc_impl(ptr, old_size, align);
        }
        return align as *mut u8;
    }
    let old_class = classify(old_size.max(align));
    let new_class = classify(new_size.max(align));
    if old_class == new_class && old_class.is_some() {
        // Same size class — in place. Slab chunks are fixed-size so
        // the existing allocation already has room for `new_size`
        // as long as `new_size <= class`.
        return ptr;
    }
    // Cross-class realloc: copy through a fresh allocation.
    let new_ptr = wlift_alloc_impl(new_size, align);
    if new_ptr.is_null() {
        return new_ptr;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, new_ptr, old_size.min(new_size));
        wlift_dealloc_impl(ptr, old_size, align);
    }
    new_ptr
}

/// Best-effort hint: walk free slabs and ask the kernel to reclaim
/// their pages via `madvise(MADV_FREE)`. Pages stay mapped (so
/// subsequent allocations don't re-page-fault hot), but the kernel
/// can reclaim them under memory pressure without touching us.
/// Call after a major GC if you want the process's RSS to drop.
///
/// Returns the number of bytes hinted as reclaimable.
pub fn pressure_release() -> usize {
    let mut total = 0usize;
    for class in &state().classes {
        let mut pool = class.lock();
        total += pool.pressure_release();
    }
    total += large::pressure_release();
    total
}

// ---------------------------------------------------------------------------
// Stable C ABI for cdylib plugins.
//
// Plugins link `wlift_alloc` statically *and* call these symbols
// from their foreign-method code instead of libc malloc/free —
// then regardless of which cdylib's copy resolves at runtime, the
// allocation routes through one global `OnceLock` state and there
// is no dual-instance hazard.
// ---------------------------------------------------------------------------

/// Allocate `size` bytes with `align` alignment. Returns null on
/// failure. `size == 0` returns a dangling-aligned pointer (do not
/// dereference; match `std::alloc::System` semantics).
#[unsafe(no_mangle)]
pub extern "C" fn wlift_malloc(size: usize, align: usize) -> *mut u8 {
    wlift_alloc_impl(size, align)
}

/// Free `ptr` previously returned by [`wlift_malloc`] /
/// [`wlift_realloc`]. The caller must pass the *original* `size` /
/// `align` the buffer was allocated with — we don't store
/// per-block metadata (cuts small-alloc overhead by 16 B / chunk).
///
/// # Safety
/// `ptr` must come from one of this allocator's entry points and
/// the size / align must match the original request. Passing
/// arbitrary libc-malloc'd pointers here is undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_free(ptr: *mut u8, size: usize, align: usize) {
    unsafe { wlift_dealloc_impl(ptr, size, align) }
}

/// Resize an allocation. Falls back to copy-and-free across size
/// classes; in-place within the same class. Null `ptr` is treated
/// as a fresh alloc, `new_size == 0` as a free.
///
/// # Safety
/// Same constraints as [`wlift_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_realloc(
    ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    unsafe { wlift_realloc_impl(ptr, old_size, align, new_size) }
}

/// Hint to the OS to reclaim free pages. See [`pressure_release`].
#[unsafe(no_mangle)]
pub extern "C" fn wlift_pressure_release() -> usize {
    pressure_release()
}

// ---------------------------------------------------------------------------
// `NonNull` helpers re-exported so callers can hand back the
// originally-typed pointer if they prefer.
// ---------------------------------------------------------------------------

/// Same as [`wlift_malloc`] but returns `Option<NonNull<u8>>`.
pub fn try_alloc(size: usize, align: usize) -> Option<NonNull<u8>> {
    NonNull::new(wlift_alloc_impl(size, align))
}
