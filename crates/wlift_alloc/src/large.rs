//! Large-allocation path. Anything above the biggest size class
//! goes through `mmap` directly — the kernel handles fragmentation
//! and page returns. We embed the original allocation size in a
//! small header `before` the returned pointer so `dealloc` knows
//! how many bytes to `munmap`.
//!
//! **Bootstrap constraint.** No heap allocations in this path —
//! everything is mmap-backed. A previous draft used a
//! `HashMap<ptr, size>` registry, but the HashMap itself allocates
//! via our global allocator and that recurses during early startup.

use std::ptr;

/// Bytes reserved before the returned user pointer for our size
/// header. Aligned to 16 to satisfy any reasonable `align`
/// request up to that boundary; for stricter alignment we'd
/// over-allocate (not done yet — none of our hot paths request it).
const HEADER_BYTES: usize = 16;

#[repr(C)]
struct LargeHdr {
    /// Total mmap'd size including this header.
    total: usize,
    /// Padding to 16 bytes so the user pointer is `header + 16`,
    /// always 16-byte-aligned regardless of mmap's page alignment.
    _pad: usize,
}

pub fn alloc(size: usize, align: usize) -> *mut u8 {
    // We always over-allocate by `HEADER_BYTES`. For requests
    // whose alignment is > 16, the simple `header + 16` offset
    // breaks; assert defensively so we catch the misuse during
    // development rather than silently returning a misaligned ptr.
    debug_assert!(
        align <= HEADER_BYTES,
        "wlift_alloc::large: alignment > 16 not yet supported (got {align})"
    );
    let total = size + HEADER_BYTES;
    let base = match unsafe { mmap_anon(total) } {
        Some(p) => p,
        None => return ptr::null_mut(),
    };
    let hdr = base as *mut LargeHdr;
    unsafe {
        ptr::write(hdr, LargeHdr { total, _pad: 0 });
        base.add(HEADER_BYTES)
    }
}

/// # Safety
/// `ptr` must come from `alloc()` and not have been freed yet.
pub unsafe fn dealloc(ptr: *mut u8, _size_hint: usize) {
    if ptr.is_null() {
        return;
    }
    let base = unsafe { ptr.sub(HEADER_BYTES) };
    let total = unsafe { (*(base as *const LargeHdr)).total };
    unsafe {
        munmap(base, total);
    }
}

/// Large allocations already return memory to the OS on `munmap`,
/// so nothing additional to do here. Returns 0.
pub fn pressure_release() -> usize {
    0
}

// ---------------------------------------------------------------------------
// mmap primitives (shared shape with slab.rs but kept local so
// each file is self-contained).
// ---------------------------------------------------------------------------

#[cfg(unix)]
unsafe fn mmap_anon(bytes: usize) -> Option<*mut u8> {
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        None
    } else {
        Some(p as *mut u8)
    }
}

#[cfg(unix)]
unsafe fn munmap(ptr: *mut u8, bytes: usize) {
    unsafe {
        libc::munmap(ptr as *mut libc::c_void, bytes);
    }
}

#[cfg(not(unix))]
unsafe fn mmap_anon(_bytes: usize) -> Option<*mut u8> {
    None
}

#[cfg(not(unix))]
unsafe fn munmap(_ptr: *mut u8, _bytes: usize) {}
