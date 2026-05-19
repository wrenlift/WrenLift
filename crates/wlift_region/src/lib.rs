//! `wlift_region` — bump-allocator regions for short-lived
//! per-fiber allocations in the WrenLift runtime.
//!
//! The motivating problem: AOT-compiled bodies almost never query
//! `vm.gc.should_collect()`, so per-request allocations on the Wren
//! heap accumulate until the next 5-minute `System.gc()` cycle.
//! Sustained traffic on the hatch site grew the process's footprint
//! ~3 MB/sec. Running `finish_alloc`-driven GC under AOT is unsafe
//! (Cranelift stack-map false-negatives reap live `Value`s and
//! SIGSEGV the next access).
//!
//! Regions sidestep both: each fiber gets a private mmap'd arena.
//! Allocations bump-allocate inside the arena. When the fiber
//! completes (`release_fiber_resources` already exists for this),
//! the region drops — every page `munmap`s. No GC walk, no stack
//! maps consulted, no libsystem_malloc free-list residue (pages
//! return to the OS immediately).
//!
//! ## Layout
//!
//! - **Page size:** 64 KiB on the initial page, geometric growth
//!   (256 KiB, 1 MiB, 4 MiB) until either the fiber completes or
//!   the per-fiber cap (default 32 MiB) is hit. Fallback past the
//!   cap routes back to the GC heap as today.
//! - **Pages are mmap'd anonymous.** They're zero-initialized,
//!   private to the process, and freed via `munmap` on `Drop`. No
//!   guard pages — the bump pointer is bounds-checked and an
//!   overflow returns `None`, no fault required.
//! - **Each allocation is 8-byte aligned.** Wren `Value`s are
//!   `u64`, every object header starts with a pointer-sized field;
//!   8-byte alignment covers every type the runtime allocates.
//!
//! ## Safety contract
//!
//! Pointers returned by `try_alloc` are valid only as long as the
//! `Region` is alive. Callers MUST NOT let those pointers escape
//! the fiber that owns the region — when the region drops, every
//! pointer becomes a dangling reference into munmap'd memory.
//!
//! The host runtime enforces this via an escape barrier at fiber-
//! boundary handoff (`Fiber.try` return, foreign-method return,
//! cross-fiber field write). The barrier copies arena-allocated
//! values onto the GC heap before the source region can drop.
//! Anything that escapes without copying is a bug the runtime
//! catches in debug builds via a walk of `mir_frames` immediately
//! before `Drop` runs — see the panic-on-leaked-arena-pointer
//! sanity check in `vm.rs`.

use std::ptr::NonNull;

/// First page's usable bytes. Sized to cover a typical small-
/// request workload's transient allocations (template render
/// intermediates, response string + map) without spilling to a
/// second page.
const INITIAL_PAGE_BYTES: usize = 64 * 1024;
/// Geometric growth cap. Past this, additional pages stay this
/// size — we'd rather pay more `mmap` syscalls than risk a runaway
/// fiber pinning a giant arena.
const MAX_PAGE_BYTES: usize = 4 * 1024 * 1024;
/// Per-region byte cap. Allocations that would push the region
/// past this fall back to the GC heap (the runtime's existing
/// allocation path). Bounds steady-state worst-case to 32 MiB per
/// in-flight fiber.
const DEFAULT_REGION_CAP: usize = 32 * 1024 * 1024;
/// Every allocation is aligned to this boundary. Picked to cover
/// `*mut ObjHeader` (pointer-sized) and `Value` (8 bytes) without
/// per-allocation alignment math.
const ALLOC_ALIGN: usize = 8;

/// A single mmap'd page in a region's chain.
struct Page {
    /// Bottom of the usable bytes.
    base: NonNull<u8>,
    /// End of the usable bytes (one past the last byte).
    end: NonNull<u8>,
}

impl Page {
    fn bytes(&self) -> usize {
        unsafe { self.end.as_ptr().offset_from(self.base.as_ptr()) as usize }
    }
}

/// One registered destructor that fires when the region drops.
/// `drop_fn` invokes `std::ptr::drop_in_place::<T>` for the
/// specific `T` the caller passed to `alloc`. Stored as a fn
/// pointer rather than a `Box<dyn FnOnce>` to keep the per-alloc
/// overhead at one pointer instead of a vtable + heap box.
struct DropEntry {
    ptr: *mut u8,
    drop_fn: unsafe fn(*mut u8),
}

/// Bump-allocator region. Owns a chain of mmap'd pages. Pages
/// are munmap'd in `Drop` so the OS reclaims them immediately —
/// no `madvise` round-trip, no libsystem_malloc free-list.
pub struct Region {
    pages: Vec<Page>,
    /// Bump pointer into the last page.
    cur: *mut u8,
    /// One-past-end of the last page.
    cur_end: *mut u8,
    /// Total bytes the region has allocated across all pages.
    /// Compared against `cap` before each new page.
    bytes_used: usize,
    /// Hard ceiling on total region size. Past this, `try_alloc`
    /// returns `None` so the caller falls back to the GC heap.
    cap: usize,
    /// Size of the *next* page to allocate. Doubles each grow
    /// until `MAX_PAGE_BYTES`.
    next_page_size: usize,
    /// Drop list: every typed allocation registers an entry so the
    /// region can call `drop_in_place` on it before unmapping
    /// pages. Without this, types like `ObjString` (which owns a
    /// `String` whose buffer lives in libc malloc) would leak the
    /// nested allocation when the region's pages were unmapped
    /// without running their `Drop`. The list itself lives on
    /// libc malloc — small per-entry cost (16 bytes), one Vec
    /// resize per ~thousand allocations.
    drops: Vec<DropEntry>,
}

impl Region {
    /// Allocate a fresh region with one initial page. The actual
    /// usable bytes from the first allocation are `INITIAL_PAGE_BYTES`;
    /// subsequent overflow grows the chain.
    pub fn new() -> Self {
        Self::with_cap(DEFAULT_REGION_CAP)
    }

    pub fn with_cap(cap: usize) -> Self {
        let mut r = Region {
            pages: Vec::with_capacity(4),
            cur: std::ptr::null_mut(),
            cur_end: std::ptr::null_mut(),
            bytes_used: 0,
            cap,
            next_page_size: INITIAL_PAGE_BYTES,
            drops: Vec::new(),
        };
        r.push_page(INITIAL_PAGE_BYTES);
        r
    }

    /// Allocate `val` into the region and register its destructor.
    /// The returned pointer is valid until the region's `Drop`
    /// (or `reset`) runs. On region drop, `std::ptr::drop_in_place`
    /// is called on the value before the pages are unmapped — so
    /// nested heap allocations (the `String` inside an `ObjString`,
    /// the `Vec` inside an `ObjList`) are freed correctly.
    ///
    /// Returns `None` when the region has hit its byte cap; the
    /// caller should fall back to the GC heap.
    ///
    /// # Safety
    /// Same lifetime contract as [`try_alloc_bytes`]: the returned
    /// pointer must not outlive the region.
    pub fn try_alloc<T>(&mut self, val: T) -> Option<*mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        // ALLOC_ALIGN covers the runtime's types (every `ObjHeader`
        // / `Value` is 8-byte aligned). Bail out if a caller asks
        // for a stricter alignment so we don't silently
        // misalign — there's no realistic Wren-runtime type with
        // align > 8 today, but check defensively.
        if align > ALLOC_ALIGN {
            return None;
        }
        let ptr = self.try_alloc_bytes(size)?.as_ptr() as *mut T;
        unsafe {
            std::ptr::write(ptr, val);
        }
        // Register destructor. The fn pointer captures `T` at the
        // call site; trampoline through a generic `drop_glue::<T>`
        // so the closure has no captured state.
        unsafe fn drop_glue<T>(p: *mut u8) {
            unsafe {
                std::ptr::drop_in_place(p as *mut T);
            }
        }
        if std::mem::needs_drop::<T>() {
            self.drops.push(DropEntry {
                ptr: ptr as *mut u8,
                drop_fn: drop_glue::<T>,
            });
        }
        Some(ptr)
    }

    /// Allocate `size` bytes aligned to `ALLOC_ALIGN`. Returns a
    /// pointer into the region or `None` if the region's cap is
    /// reached. The returned pointer is valid for the lifetime of
    /// the region; once the region drops, the pointer is invalid.
    ///
    /// Untyped — no destructor runs when the region drops. Use
    /// [`try_alloc`] for any type whose `Drop` matters (anything
    /// owning a `String`, `Vec`, `Box`, etc.).
    ///
    /// # Safety
    /// The caller is responsible for not retaining the returned
    /// pointer past the region's lifetime. See the crate-level
    /// safety contract.
    pub fn try_alloc_bytes(&mut self, size: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            // Match `Vec::with_capacity(0)` semantics: return a
            // unique non-null pointer the caller can't dereference.
            return NonNull::new(ALLOC_ALIGN as *mut u8);
        }
        let aligned = (size + ALLOC_ALIGN - 1) & !(ALLOC_ALIGN - 1);
        loop {
            // Fast path: fits in the current page.
            let remaining = (self.cur_end as usize).saturating_sub(self.cur as usize);
            if aligned <= remaining {
                let p = self.cur;
                self.cur = unsafe { self.cur.add(aligned) };
                self.bytes_used += aligned;
                return NonNull::new(p);
            }
            // Slow path: need a new page. Bail out if we'd exceed
            // the region's cap.
            let next_size = self.next_page_size.max(aligned);
            if self.bytes_used.saturating_add(next_size) > self.cap {
                return None;
            }
            self.push_page(next_size);
        }
    }

    /// Total bytes currently held by the region's chain (live and
    /// dead). For diagnostics.
    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    /// Number of mmap'd pages backing the region. For diagnostics.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Reset the bump pointer to the *start* of the first page,
    /// dropping every subsequent page. Lets a long-lived fiber
    /// reuse its region across loop iterations without re-mapping
    /// the initial page each time. Pointers handed out before
    /// `reset()` become invalid after — callers must respect the
    /// same escape-barrier discipline as `Drop`.
    pub fn reset(&mut self) {
        // Run every registered destructor first. Same order as
        // allocation (FIFO) — we don't track dependencies between
        // arena objects, and for the runtime's types (each
        // self-contained: an `ObjString` owns its bytes, an
        // `ObjList` owns its `Vec`) drop order doesn't matter.
        let drops = std::mem::take(&mut self.drops);
        for d in drops {
            unsafe {
                (d.drop_fn)(d.ptr);
            }
        }
        // Drop every page past the first.
        if self.pages.len() > 1 {
            let extras: Vec<Page> = self.pages.drain(1..).collect();
            for p in extras {
                let bytes = p.bytes();
                unsafe {
                    munmap(p.base.as_ptr(), bytes);
                }
            }
        }
        // Rewind the bump pointer to the start of the first page.
        if let Some(p) = self.pages.first() {
            self.cur = p.base.as_ptr();
            self.cur_end = p.end.as_ptr();
        } else {
            self.cur = std::ptr::null_mut();
            self.cur_end = std::ptr::null_mut();
        }
        self.bytes_used = 0;
        self.next_page_size = INITIAL_PAGE_BYTES;
    }

    fn push_page(&mut self, size: usize) {
        let aligned_size = page_align_up(size);
        let base = unsafe { mmap_anon(aligned_size) }
            .expect("wlift_region: mmap failed for fiber arena page");
        let end = unsafe { NonNull::new_unchecked(base.as_ptr().add(aligned_size)) };
        self.cur = base.as_ptr();
        self.cur_end = end.as_ptr();
        self.pages.push(Page { base, end });
        // Geometric growth, capped at MAX_PAGE_BYTES.
        self.next_page_size = (self.next_page_size * 4).min(MAX_PAGE_BYTES);
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // Run every registered destructor before unmapping. Order
        // matches `reset`; see the comment there for why FIFO is
        // safe for the runtime's types.
        let drops = std::mem::take(&mut self.drops);
        for d in drops {
            unsafe {
                (d.drop_fn)(d.ptr);
            }
        }
        for p in self.pages.drain(..) {
            let bytes = p.bytes();
            unsafe {
                munmap(p.base.as_ptr(), bytes);
            }
        }
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: `Region` owns its pages exclusively. No interior pointers
// alias between fibers — the runtime never shares a region across
// threads. `Send` so an `ObjFiber` (allocated on one thread, reaped
// by GC on possibly another) can carry it.
unsafe impl Send for Region {}

// ---------------------------------------------------------------------------
// Page-size + mmap helpers. Kept private — only `Region` needs them.
// ---------------------------------------------------------------------------

fn page_align_up(n: usize) -> usize {
    let p = page_size();
    (n + p - 1) & !(p - 1)
}

#[cfg(unix)]
fn page_size() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CACHED: AtomicUsize = AtomicUsize::new(0);
    let v = CACHED.load(Ordering::Relaxed);
    if v != 0 {
        return v;
    }
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    CACHED.store(p, Ordering::Relaxed);
    p
}

#[cfg(not(unix))]
fn page_size() -> usize {
    4096
}

#[cfg(unix)]
unsafe fn mmap_anon(bytes: usize) -> Option<NonNull<u8>> {
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
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

#[cfg(not(unix))]
unsafe fn mmap_anon(bytes: usize) -> Option<NonNull<u8>> {
    // Non-unix fallback: heap allocation. Wasm-only target; the
    // runtime there doesn't use this crate.
    let layout = std::alloc::Layout::from_size_align(bytes, 4096).ok()?;
    let p = unsafe { std::alloc::alloc_zeroed(layout) };
    NonNull::new(p)
}

#[cfg(unix)]
unsafe fn munmap(ptr: *mut u8, bytes: usize) {
    if !ptr.is_null() {
        let _ = unsafe { libc::munmap(ptr as *mut libc::c_void, bytes) };
    }
}

#[cfg(not(unix))]
unsafe fn munmap(ptr: *mut u8, bytes: usize) {
    if !ptr.is_null() {
        if let Ok(layout) = std::alloc::Layout::from_size_align(bytes, 4096) {
            unsafe {
                std::alloc::dealloc(ptr, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_region_serves_small_alloc() {
        let mut r = Region::new();
        let p = r
            .try_alloc_bytes(16)
            .expect("first alloc fits the initial page");
        assert_eq!(p.as_ptr() as usize % ALLOC_ALIGN, 0);
        assert_eq!(r.bytes_used(), 16);
    }

    #[test]
    fn many_allocs_grow_into_new_pages() {
        let mut r = Region::with_cap(8 * 1024 * 1024);
        // Allocate 128 KiB total — more than the 64 KiB initial page.
        for _ in 0..128 {
            r.try_alloc_bytes(1024).expect("under cap");
        }
        assert!(r.page_count() >= 2);
        assert!(r.bytes_used() >= 128 * 1024);
    }

    #[test]
    fn alloc_past_cap_returns_none() {
        let mut r = Region::with_cap(128 * 1024); // 128 KiB cap
        let mut ok = 0;
        for _ in 0..200 {
            if r.try_alloc_bytes(1024).is_some() {
                ok += 1;
            }
        }
        assert!(ok > 0);
        assert!(ok <= 130, "should bail out near the cap, got {ok}");
        // Subsequent small allocs also fail.
        assert!(r.try_alloc_bytes(1024).is_none());
    }

    #[test]
    fn reset_keeps_first_page_drops_extras() {
        let mut r = Region::new();
        for _ in 0..128 {
            r.try_alloc_bytes(1024).unwrap();
        }
        assert!(r.page_count() >= 2);
        r.reset();
        assert_eq!(r.page_count(), 1);
        assert_eq!(r.bytes_used(), 0);
        // Can still allocate after reset.
        r.try_alloc_bytes(64).unwrap();
    }

    #[test]
    fn zero_size_alloc_returns_non_null() {
        let mut r = Region::new();
        let p = r.try_alloc_bytes(0).unwrap();
        assert!(p.as_ptr() as usize >= ALLOC_ALIGN);
    }

    #[test]
    fn drop_unmaps_pages() {
        // Smoke test: no leak, no crash. Real leak detection would
        // need a heavier harness (Instruments / valgrind), but this
        // exercises the Drop path.
        for _ in 0..16 {
            let mut r = Region::new();
            for _ in 0..32 {
                r.try_alloc_bytes(1024).unwrap();
            }
        }
    }

    /// Tracks Drop calls for the `runs_drop_*` tests.
    struct DropCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[test]
    fn runs_drop_on_region_drop() {
        use std::sync::atomic::Ordering;
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let mut r = Region::new();
            for _ in 0..50 {
                r.try_alloc(DropCounter(count.clone())).unwrap();
            }
            assert_eq!(count.load(Ordering::Relaxed), 0);
            // Region drops at end of block.
        }
        assert_eq!(count.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn runs_drop_on_region_reset() {
        use std::sync::atomic::Ordering;
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut r = Region::new();
        for _ in 0..30 {
            r.try_alloc(DropCounter(count.clone())).unwrap();
        }
        r.reset();
        assert_eq!(count.load(Ordering::Relaxed), 30);
        // After reset, region is reusable.
        r.try_alloc(DropCounter(count.clone())).unwrap();
        drop(r);
        assert_eq!(count.load(Ordering::Relaxed), 31);
    }

    #[test]
    fn frees_nested_heap_allocations() {
        // The critical test: types that own libc-heap allocations
        // (like String) must have their Drop called so the nested
        // allocations are freed — otherwise the region's `munmap`
        // would leak them.
        let mut r = Region::new();
        for i in 0..100 {
            let s = format!("test string with content {i}");
            r.try_alloc(s).unwrap();
        }
        // Just checking no leak / no crash. Miri would catch the
        // libc leak if we skipped destructors; here we rely on the
        // counter test above for proof + valgrind for verification
        // in dedicated runs.
        drop(r);
    }

    #[test]
    fn pod_types_skip_drop_list() {
        // `needs_drop::<u64>()` is false, so the drop list stays
        // empty for POD values. Verified by checking `drops` after.
        let mut r = Region::new();
        for i in 0..1000_u64 {
            r.try_alloc(i).unwrap();
        }
        assert_eq!(r.drops.len(), 0);
    }
}
