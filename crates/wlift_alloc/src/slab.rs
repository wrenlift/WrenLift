//! Size-class slab allocator. Each class owns an intrusive
//! free-list threaded through unused chunks of mmap'd 64 KiB
//! slabs. New slabs are pushed when the free list is empty; empty
//! slabs are dropped back to the OS on the next
//! `pressure_release()` call.
//!
//! **Bootstrap constraint.** Our allocator's `Alloc::new()` runs
//! before `#[global_allocator]` is fully wired up — it's invoked
//! from `OnceLock::get_or_init` on the very first call, which can
//! be from inside `std::rt::lang_start`'s own setup. Any heap
//! allocation here would recurse into our global allocator and
//! deadlock on the `OnceLock` semaphore. So slab bookkeeping
//! lives **inside the mmap'd region itself** — no `Vec<Slab>`,
//! no `Box<Slab>`, no internal allocations during alloc/free.
//!
//! Layout per slab:
//!
//! ```text
//! [ Slab header | chunk 1 | chunk 2 | ... | chunk N-1 ]
//!   (chunk_size      one chunk's worth lost to bookkeeping)
//! ```
//!
//! The header (`next: *mut SlabHdr`) makes slabs an intrusive
//! singly-linked list rooted in `ClassPool::head`. Pressure
//! release walks the list and `madvise(MADV_FREE)`s any slab
//! whose chunks are all on the free list.

use core::sync::atomic::{AtomicPtr, Ordering};
use std::ptr::{self, NonNull};

/// In-slab metadata header. Sits at offset 0 of every mmap'd
/// slab; the first chunk starts at offset `chunk_size` so the
/// header doesn't collide with caller data.
///
/// We deliberately keep this to a single `*mut` (8 B on
/// 64-bit) so it fits inside the smallest size class (16 B).
/// `chunk_size` and `slab_bytes` are class-wide constants
/// stored on `ClassPool`, not per-slab — there's no need to
/// duplicate them here.
#[repr(C)]
struct SlabHdr {
    next: *mut SlabHdr,
}

/// Public alias so the rest of the crate can talk about slabs
/// without exposing the in-arena header layout.
pub type Slab = SlabHdr;

/// Per-size-class state. Holds the free-list head + the intrusive
/// list of slabs backing it. Lives behind a `Mutex` in the global
/// `Alloc`.
pub struct ClassPool {
    chunk_size: usize,
    slab_bytes: usize,
    /// Intrusive free-list head. Each free chunk's first 8 bytes
    /// point to the next free chunk (or null).
    free: *mut u8,
    /// Head of the intrusive slab list. Each slab carries its
    /// `next` pointer in its `SlabHdr`.
    slabs: AtomicPtr<SlabHdr>,
}

// SAFETY: every access to `free` is through the `Mutex` wrapping
// the `ClassPool` in `Alloc::classes`. `slabs` is `AtomicPtr` to
// keep `ClassPool` `Sync` (so it can be stored in a `Mutex` and
// referenced from a `&'static`); the pointer itself is only
// written while the mutex is held, but reads in `pressure_release`
// can happen via `&self`.
unsafe impl Send for ClassPool {}
unsafe impl Sync for ClassPool {}

impl ClassPool {
    /// `const fn` so it can be invoked from a `static` initializer
    /// or other compile-time constant context — guarantees no
    /// allocation during construction.
    pub const fn new(chunk_size: usize, slab_bytes: usize) -> Self {
        Self {
            chunk_size,
            slab_bytes,
            free: ptr::null_mut(),
            slabs: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Pop a chunk off the free list, refilling with a fresh slab
    /// if the list is empty. Returns uninitialised memory (the
    /// caller may zero it for `alloc_zeroed`). O(1).
    pub fn alloc(&mut self) -> *mut u8 {
        if self.free.is_null() {
            self.add_slab();
        }
        let chunk = self.free;
        unsafe {
            // SAFETY: every free chunk's first word is its next-
            // free-list link, written at slab construction or by a
            // previous `free` call.
            self.free = ptr::read(chunk as *const *mut u8);
        }
        chunk
    }

    /// Push `ptr` onto the free list. O(1).
    ///
    /// # Safety
    /// `ptr` must be a previously-returned allocation from
    /// `alloc()` on *this* pool, not currently on the free list.
    /// Its first `mem::size_of::<*mut u8>()` bytes are overwritten
    /// with the next-free-list link.
    pub unsafe fn free(&mut self, ptr: *mut u8) {
        unsafe {
            ptr::write(ptr as *mut *mut u8, self.free);
        }
        self.free = ptr;
    }

    /// `mmap` a fresh slab, lay out the header + chunks, chain the
    /// chunks onto the free list, and push the slab onto the
    /// intrusive slab list.
    fn add_slab(&mut self) {
        assert!(self.chunk_size >= std::mem::size_of::<*mut u8>());
        assert!(self.chunk_size >= std::mem::size_of::<SlabHdr>());
        assert!(self.slab_bytes % self.chunk_size == 0);
        let base = unsafe { mmap_anon(self.slab_bytes) }.unwrap_or_else(|| {
            panic!("wlift_alloc: mmap({}B) failed", self.slab_bytes)
        });
        // Write the header at offset 0.
        let hdr = base.as_ptr() as *mut SlabHdr;
        unsafe {
            ptr::write(
                hdr,
                SlabHdr {
                    next: self.slabs.load(Ordering::Relaxed),
                },
            );
        }
        self.slabs.store(hdr, Ordering::Relaxed);

        // First usable chunk starts at offset chunk_size (so the
        // header sits in chunk 0's slot and we lose one chunk to
        // bookkeeping).
        let n = self.slab_bytes / self.chunk_size;
        let mut tail_link: *mut *mut u8 = ptr::null_mut();
        unsafe {
            for i in 1..n {
                let chunk = base.as_ptr().add(i * self.chunk_size);
                let next = if i + 1 < n {
                    base.as_ptr().add((i + 1) * self.chunk_size)
                } else {
                    ptr::null_mut()
                };
                ptr::write(chunk as *mut *mut u8, next);
                if i + 1 == n {
                    tail_link = chunk as *mut *mut u8;
                }
            }
        }
        // Splice this slab's chain onto the front of the pool's
        // free list. The tail of our new chain (last chunk, pointed
        // by `tail_link`) currently has null `next`; rewrite it to
        // the existing free head, then point head at our first
        // chunk.
        let first_chunk = unsafe { base.as_ptr().add(self.chunk_size) };
        if !tail_link.is_null() {
            unsafe {
                ptr::write(tail_link, self.free);
            }
        }
        self.free = first_chunk;
    }

    /// `madvise(MADV_FREE_REUSABLE)` slabs whose chunks are all on
    /// the free list. Returns total bytes hinted. O(n_chunks_total)
    /// — cold path (only called from the major GC tail).
    pub fn pressure_release(&self) -> usize {
        // First, count free chunks per slab by walking the free
        // list. We use the slab list as the index: for each free
        // chunk, find which slab it falls in.
        let mut total = 0usize;
        let chunks_per_slab = (self.slab_bytes / self.chunk_size) - 1; // minus header
        let mut slab = self.slabs.load(Ordering::Relaxed);
        while !slab.is_null() {
            let slab_base = slab as usize;
            let slab_end = slab_base + self.slab_bytes;
            // Count free chunks within this slab's range.
            let mut count = 0;
            let mut cur = self.free;
            while !cur.is_null() {
                let p = cur as usize;
                if p >= slab_base && p < slab_end {
                    count += 1;
                }
                unsafe {
                    cur = ptr::read(cur as *const *mut u8);
                }
            }
            if count == chunks_per_slab {
                unsafe {
                    // Don't madvise over the header itself —
                    // we still read its `next` pointer in this
                    // very loop. Madvise from the first chunk
                    // onward.
                    let chunks_start = (slab_base + self.chunk_size) as *mut u8;
                    let chunks_bytes = self.slab_bytes - self.chunk_size;
                    madvise_free(chunks_start, chunks_bytes);
                }
                total += self.slab_bytes - self.chunk_size;
            }
            slab = unsafe { (*slab).next };
        }
        total
    }
}

impl Drop for ClassPool {
    fn drop(&mut self) {
        // Walk the intrusive slab list and munmap each. `slab_bytes`
        // is a class-wide constant, not stored in the per-slab
        // header (which is one pointer wide so it fits in the
        // smallest size class).
        let mut cur = self.slabs.load(Ordering::Relaxed);
        while !cur.is_null() {
            let next = unsafe { (*cur).next };
            unsafe {
                munmap(cur as *mut u8, self.slab_bytes);
            }
            cur = next;
        }
    }
}

// ---------------------------------------------------------------------------
// Page-level primitives (Unix-only for now; Windows would use
// `VirtualAlloc` / `VirtualFree`).
// ---------------------------------------------------------------------------

#[cfg(unix)]
unsafe fn mmap_anon(bytes: usize) -> Option<NonNull<u8>> {
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
        NonNull::new(p as *mut u8)
    }
}

#[cfg(unix)]
unsafe fn munmap(ptr: *mut u8, bytes: usize) {
    unsafe {
        libc::munmap(ptr as *mut libc::c_void, bytes);
    }
}

#[cfg(all(unix, target_os = "macos"))]
unsafe fn madvise_free(ptr: *mut u8, bytes: usize) {
    // macOS prefers `MADV_FREE_REUSABLE` over `MADV_FREE` because
    // it pairs with `MADV_FREE_REUSE` to mark pages reusable
    // *and* deduct them from the process's footprint — which is
    // the metric Activity Monitor reports. Plain `MADV_FREE`
    // leaves footprint untouched.
    unsafe {
        libc::madvise(ptr as *mut libc::c_void, bytes, libc::MADV_FREE_REUSABLE);
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
unsafe fn madvise_free(ptr: *mut u8, bytes: usize) {
    unsafe {
        libc::madvise(ptr as *mut libc::c_void, bytes, libc::MADV_FREE);
    }
}

#[cfg(not(unix))]
unsafe fn mmap_anon(_bytes: usize) -> Option<NonNull<u8>> {
    None
}

#[cfg(not(unix))]
unsafe fn munmap(_ptr: *mut u8, _bytes: usize) {}

#[cfg(not(unix))]
unsafe fn madvise_free(_ptr: *mut u8, _bytes: usize) {}
