//! Large-allocation path. Anything above the biggest size class
//! goes through `mmap` directly — the kernel handles fragmentation
//! and page returns. We embed the original allocation size in a
//! small header `before` the returned pointer so `dealloc` knows
//! how many bytes to `munmap`.
//!
//! **Recycle free-list.** A previous draft `munmap`'d on every
//! large `dealloc`, which is correct but creates one fresh
//! `VM_ALLOCATE` region per allocation. macOS's `vmmap` accounting
//! counts each region toward the process's "Physical footprint"
//! (even when compressed), so a web server doing thousands of
//! small-medium allocations sees footprint climb into the GBs even
//! while RSS stays low. The free-list caches recently-freed
//! regions in power-of-2 buckets so the *same* virtual range gets
//! reused on the next allocation of similar size, dramatically
//! cutting region count.
//!
//! **Bootstrap constraint.** No heap allocations in this path —
//! everything is mmap-backed. The free-list buckets are stored
//! in a `static` array of `spin::Mutex<LargeBucket>` with fixed-
//! size arrays inside (no `Vec`).

use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;
use std::ptr;

/// Bytes reserved before the returned user pointer for our size
/// header. Aligned to 16 to satisfy any reasonable `align`
/// request up to that boundary.
const HEADER_BYTES: usize = 16;

/// Per-bucket cache cap. 4 cached regions per power-of-2 size
/// bucket absorbs typical request-burst churn while keeping the
/// total cache ceiling bounded.
const BUCKET_CAP: usize = 4;

/// Number of power-of-2 buckets. Bucket 0 = 2 KiB (2^11),
/// bucket 7 = 256 KiB (2^18). Anything > 256 KiB bypasses the
/// cache and goes straight through `mmap` / `munmap` — caching
/// huge regions wastes memory for low reuse rates.
const NUM_BUCKETS: usize = 8;

/// First bucket index. `total_bytes >= 2 KiB` lands in bucket 0
/// (corresponds to log2=11 → 2 KiB capacity). Smaller large allocs
/// (1024 < n <= 2048 bytes) also land here.
const MIN_LOG2: u32 = 11;

#[repr(C)]
struct LargeHdr {
    /// Total mmap'd size including this header. Rounded up to
    /// page granularity by the kernel, but we store the rounded
    /// value here too so the recycle path knows the actual size.
    total: usize,
    /// Padding to 16 bytes so the user pointer is `header + 16`,
    /// always 16-byte-aligned regardless of mmap's page alignment.
    _pad: usize,
}

struct LargeBucket {
    /// Fixed-size stack of cached free regions. Each entry is the
    /// **base** of the mmap'd region (header at offset 0), not the
    /// user pointer.
    cached: [*mut u8; BUCKET_CAP],
    /// Number of valid entries in `cached`.
    len: usize,
    /// Bucket's nominal capacity (`2^(MIN_LOG2 + bucket_index)`).
    /// All cached regions in this bucket are mmap'd at this size.
    bucket_bytes: usize,
}

// SAFETY: every access is through the bucket's `Mutex`. The raw
// pointers in `cached` are only dereferenced after the mutex is
// released (the caller takes ownership).
unsafe impl Send for LargeBucket {}
unsafe impl Sync for LargeBucket {}

impl LargeBucket {
    const fn new(bucket_bytes: usize) -> Self {
        Self {
            cached: [ptr::null_mut(); BUCKET_CAP],
            len: 0,
            bucket_bytes,
        }
    }
}

/// Power-of-2 bucket index for a given total allocation size,
/// or `None` if it's outside the cached range (too big — go
/// direct to mmap, skip the cache).
fn bucket_for(total: usize) -> Option<usize> {
    // Round up to the next power of 2.
    if total == 0 {
        return None;
    }
    let log2 = (usize::BITS - (total - 1).leading_zeros()).max(MIN_LOG2);
    let idx = (log2 - MIN_LOG2) as usize;
    if idx < NUM_BUCKETS { Some(idx) } else { None }
}

/// Rounded-up allocation size for a given bucket. Every region
/// cached in `BUCKETS[i]` has exactly this many bytes mmap'd.
fn bucket_bytes(idx: usize) -> usize {
    1usize << (MIN_LOG2 + idx as u32)
}

static BUCKETS: [Mutex<LargeBucket>; NUM_BUCKETS] = [
    Mutex::new(LargeBucket::new(1 << MIN_LOG2)), //   2 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 1))), //   4 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 2))), //   8 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 3))), //  16 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 4))), //  32 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 5))), //  64 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 6))), // 128 KiB
    Mutex::new(LargeBucket::new(1 << (MIN_LOG2 + 7))), // 256 KiB
];

/// Total bytes currently cached across all buckets. Useful for
/// `pressure_release` to know how much memory we can drain.
static CACHED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn alloc(size: usize, align: usize) -> *mut u8 {
    debug_assert!(
        align <= HEADER_BYTES,
        "wlift_alloc::large: alignment > 16 not yet supported (got {align})"
    );
    let needed_total = size + HEADER_BYTES;
    // Try to pull a cached region from the matching bucket. If the
    // bucket holds a region of `bucket_bytes(idx) >= needed_total`,
    // it's a valid fit — we waste at most 2× the requested bytes
    // of internal fragmentation, same as malloc's standard
    // size-class trade-off.
    if let Some(idx) = bucket_for(needed_total) {
        let mut b = BUCKETS[idx].lock();
        if b.len > 0 {
            let slot = b.len - 1;
            let base = b.cached[slot];
            b.cached[slot] = ptr::null_mut();
            b.len = slot;
            let cached_bytes = b.bucket_bytes;
            CACHED_BYTES.fetch_sub(cached_bytes, Ordering::Relaxed);
            unsafe {
                ptr::write(
                    base as *mut LargeHdr,
                    LargeHdr {
                        total: cached_bytes,
                        _pad: 0,
                    },
                );
                return base.add(HEADER_BYTES);
            }
        }
    }
    // Cache miss. Round the request up to the next bucket size
    // *only* if that's within 2× the asked-for total — we'd
    // rather make the eventual `dealloc` cache-friendly than
    // skip the cache, but doubling memory is the worst we
    // accept. For misses outside that range, mmap exact size
    // (callers asking for, say, 6 KiB get a 6 KiB region; on
    // `dealloc` it won't go back into the cache because no
    // bucket matches the exact size).
    let total = match bucket_for(needed_total) {
        Some(idx) if bucket_bytes(idx) <= needed_total * 2 => bucket_bytes(idx),
        _ => needed_total,
    };
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
    // If the region fits a bucket and there's room to cache,
    // park it. Otherwise munmap.
    if let Some(idx) = bucket_for(total)
        && bucket_bytes(idx) == total
    {
        let mut b = BUCKETS[idx].lock();
        if b.len < BUCKET_CAP {
            let slot = b.len;
            b.cached[slot] = base;
            b.len = slot + 1;
            CACHED_BYTES.fetch_add(total, Ordering::Relaxed);
            return;
        }
    }
    unsafe {
        munmap(base, total);
    }
}

/// Drain the free-list — `munmap` every cached region. Called
/// from the major GC tail (via `pressure_release` on the top-
/// level allocator) so footprint actually drops after the Wren
/// heap reclaims a batch of large objects.
pub fn pressure_release() -> usize {
    let mut total = 0usize;
    for bucket in &BUCKETS {
        let mut b = bucket.lock();
        for i in 0..b.len {
            let base = b.cached[i];
            b.cached[i] = ptr::null_mut();
            unsafe {
                munmap(base, b.bucket_bytes);
            }
            total += b.bucket_bytes;
        }
        b.len = 0;
    }
    CACHED_BYTES.fetch_sub(total, Ordering::Relaxed);
    total
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
