//! wren_lift's own memory under the Immix strategy: the default behind
//! every heap slot of `rt`.
//!
//! The heap is a set of chunk-aligned chunks carved into 32 KiB blocks
//! of 128-byte lines. Objects up to one line bump-allocate through a
//! region and never straddle a line; larger objects take whole lines
//! and never share their last line. Allocation starts and span sizes
//! live in side tables keyed by block, so any address inside an
//! allocation resolves to its start. Every allocation begins with an
//! `ObjHeader`, so the header's mark byte is the memory-side liveness
//! claim; sweep derives line occupancy from the surviving starts, runs
//! the strategy's drop on every dead object, and hands runs of free
//! lines back as bump regions. Nothing moves.
//!
//! Several threads allocate from one heap: each bumps through regions
//! of its own, the side tables are reserved once for the largest heap
//! so they never move, and the block pool sits behind a lock that only
//! a region refill takes. Compiled code bumps the heap's own region
//! while the program has one thread; with more, that region is closed
//! and compiled code allocates through the runtime like anything else.
//! Marking and sweeping run with every other thread stopped.

use super::object::ObjHeader;
use super::rt::{RtStats, MAX_ALLOC};
use super::value::Value;
use crate::portable_time::Instant;
use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering::Relaxed,
};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub const BLOCK_SIZE: usize = MAX_ALLOC;
pub const LINE_SIZE: usize = 128;
const QUANTUM: usize = 16;
const LINES_PER_BLOCK: usize = BLOCK_SIZE / LINE_SIZE;
const QUANTA_PER_BLOCK: usize = BLOCK_SIZE / QUANTUM;
const QUANTA_PER_LINE: usize = LINE_SIZE / QUANTUM;
const LINE_WORDS: usize = LINES_PER_BLOCK / 64;

/// Start-byte code for an allocation that owns whole lines; its length
/// is `alloc_sizes[line] * LINE_SIZE`. Small allocations store their
/// size in quanta (1..=8) instead.
const SPAN_OBJECT: u8 = (QUANTA_PER_LINE + 1) as u8;
/// The size code of a start byte.
const CODE_MASK: u8 = 0x0F;
/// Start-byte bit of an allocation that owns nothing outside the heap:
/// the sweep reclaims it without calling the drop.
const PLAIN: u8 = 0x10;

/// Largest allocation that goes through the bump region.
const SMALL_MAX: usize = LINE_SIZE;

/// Heap grows one chunk at a time up to the configured maximum. Chunks
/// are aligned to their size so an address maps to its chunk by shift.
const CHUNK_SHIFT: u32 = 25;
const CHUNK_BYTES: usize = 1 << CHUNK_SHIFT;
const BLOCKS_PER_CHUNK: usize = CHUNK_BYTES / BLOCK_SIZE;
/// Address bits the chunk map covers.
#[cfg(target_pointer_width = "32")]
const ADDRESS_BITS: u32 = 32;
#[cfg(not(target_pointer_width = "32"))]
const ADDRESS_BITS: u32 = 48;

#[cfg(not(target_pointer_width = "32"))]
const DEFAULT_HEAP_MAX: usize = 4 * 1024 * 1024 * 1024;
#[cfg(target_pointer_width = "32")]
const DEFAULT_HEAP_MAX: usize = 256 * 1024 * 1024;

const TRIGGER_FLOOR: usize = 8 * 1024 * 1024;
const TRIGGER_CEILING: usize = 512 * 1024 * 1024;
const DEFAULT_GROWTH: usize = 4;
const HEARTBEAT: Duration = Duration::from_secs(30);

/// The two mark colours a cycle alternates between: an object marked
/// in the previous cycle is unmarked in this one without a reset pass.
/// Neither is 0 (fresh) or 3 (a moving collector's forwarding mark).
const COLOR_A: u8 = 2;
const COLOR_B: u8 = 4;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// `WLIFT_GC_HEAP_MB`: cap on heap bytes. Safe to raise; lowering it
/// below a program's live set aborts with a heap-exhausted message.
fn heap_max_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        env_usize("WLIFT_GC_HEAP_MB")
            .map(|mb| (mb.max(32) * 1024 * 1024).next_multiple_of(CHUNK_BYTES))
            .unwrap_or(DEFAULT_HEAP_MAX)
    })
}

/// `WLIFT_GC_TRIGGER_MB`: allocation volume between collections when
/// the live set is small. Safe to tune.
fn trigger_floor_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        env_usize("WLIFT_GC_TRIGGER_MB")
            .map(|mb| mb.max(1) * 1024 * 1024)
            .unwrap_or(TRIGGER_FLOOR)
    })
}

/// `WLIFT_GC_GROWTH`: next trigger as a multiple of the live set. Safe
/// to tune.
fn growth_factor() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        env_usize("WLIFT_GC_GROWTH")
            .unwrap_or(DEFAULT_GROWTH)
            .max(1)
    })
}

/// `WLIFT_GC_STRESS`: collect at every poll. Diagnostic only; the bump
/// path stays on so reuse is exercised.
fn stress_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("WLIFT_GC_STRESS").is_some())
}

// ---------------------------------------------------------------------------
// Reservations
// ---------------------------------------------------------------------------

/// Zeroed memory reserved once; pages commit on first touch.
struct Reservation {
    ptr: *mut u8,
    #[cfg(all(unix, feature = "host"))]
    len: usize,
    #[cfg(not(all(unix, feature = "host")))]
    layout: std::alloc::Layout,
}

impl Reservation {
    fn new(bytes: usize, align: usize) -> Option<Reservation> {
        #[cfg(all(unix, feature = "host"))]
        {
            let len = bytes + align;
            #[allow(unused_mut)]
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANON;
            #[cfg(target_os = "linux")]
            {
                flags |= libc::MAP_NORESERVE;
            }
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return None;
            }
            Some(Reservation {
                ptr: p as *mut u8,
                len,
            })
        }
        #[cfg(not(all(unix, feature = "host")))]
        {
            let layout = std::alloc::Layout::from_size_align(bytes, align).ok()?;
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            if p.is_null() {
                return None;
            }
            Some(Reservation { ptr: p, layout })
        }
    }

    /// First `align`-aligned address inside the reservation.
    fn aligned(&self, align: usize) -> usize {
        (self.ptr as usize).next_multiple_of(align)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        #[cfg(all(unix, feature = "host"))]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
        #[cfg(not(all(unix, feature = "host")))]
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

/// One chunk-aligned chunk of blocks.
struct Chunk {
    base: usize,
    _raw: Reservation,
}

impl Chunk {
    fn reserve() -> Option<Chunk> {
        let raw = Reservation::new(CHUNK_BYTES, CHUNK_BYTES)?;
        Some(Chunk {
            base: raw.aligned(CHUNK_BYTES),
            _raw: raw,
        })
    }
}

/// Words another thread may be writing: every access is a relaxed
/// atomic, which is a plain load or store on the targets we run on.
trait Word: Copy {
    fn load(p: *const Self) -> Self;
    fn store(p: *mut Self, v: Self);
}

macro_rules! word {
    ($t:ty, $a:ty) => {
        impl Word for $t {
            #[inline(always)]
            fn load(p: *const Self) -> Self {
                unsafe { <$a>::from_ptr(p as *mut Self).load(Relaxed) }
            }
            #[inline(always)]
            fn store(p: *mut Self, v: Self) {
                unsafe { <$a>::from_ptr(p).store(v, Relaxed) }
            }
        }
    };
}
word!(u8, AtomicU8);
word!(u32, AtomicU32);
word!(u64, AtomicU64);
word!(usize, AtomicUsize);

/// A side table sized for the largest heap, so it never moves.
struct Table<T: Word> {
    ptr: *mut T,
    len: usize,
    _raw: Reservation,
}

impl<T: Word> Table<T> {
    fn new(len: usize) -> Table<T> {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        let raw = Reservation::new(bytes, 4096).expect("Immix: cannot reserve a side table");
        Table {
            ptr: raw.aligned(4096) as *mut T,
            len,
            _raw: raw,
        }
    }

    #[inline(always)]
    fn get(&self, i: usize) -> T {
        debug_assert!(i < self.len);
        T::load(unsafe { self.ptr.add(i) })
    }

    #[inline(always)]
    fn set(&self, i: usize, v: T) {
        debug_assert!(i < self.len);
        T::store(unsafe { self.ptr.add(i) }, v)
    }

    /// A slice for the collector's bulk passes, when no thread writes.
    #[inline(always)]
    fn slice(&self, lo: usize, hi: usize) -> &[T] {
        debug_assert!(lo <= hi && hi <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(lo), hi - lo) }
    }
}

impl Table<u8> {
    #[inline(always)]
    fn fetch_add(&self, i: usize, n: u8) {
        debug_assert!(i < self.len);
        unsafe { AtomicU8::from_ptr(self.ptr.add(i)).fetch_add(n, Relaxed) };
    }

    #[inline(always)]
    fn fetch_sub(&self, i: usize, n: u8) {
        debug_assert!(i < self.len);
        unsafe { AtomicU8::from_ptr(self.ptr.add(i)).fetch_sub(n, Relaxed) };
    }

    fn fill(&self, lo: usize, hi: usize, v: u8) {
        debug_assert!(lo <= hi && hi <= self.len);
        unsafe { std::ptr::write_bytes(self.ptr.add(lo), v, hi - lo) };
    }
}

impl Table<u32> {
    fn fill(&self, lo: usize, hi: usize, v: u32) {
        for i in lo..hi {
            self.set(i, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Bump regions
// ---------------------------------------------------------------------------

/// A run of lines the allocator is bumping through. `cur == limit`
/// means empty.
#[derive(Clone, Copy)]
struct Region {
    cur: usize,
    limit: usize,
    block: u32,
}

impl Region {
    const EMPTY: Region = Region {
        cur: 0,
        limit: 0,
        block: 0,
    };

    fn is_empty(&self) -> bool {
        self.cur >= self.limit
    }
}

/// The small bump region as compiled code sees it: it allocates an
/// object up to a line by advancing `cur`, never past `limit` or across
/// a line, and records the start in `objects[q0 + (start - base) / 16]`.
/// First in the heap so the heap handle addresses it.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BumpRegion {
    pub cur: usize,
    pub limit: usize,
    /// The region's block base address.
    pub base: usize,
    /// The block's first start-byte index.
    pub q0: usize,
    /// The start-byte table.
    pub objects: *mut u8,
    /// `objects + q0 - base / 16`, so the start byte of the object at
    /// `p` is at `codes + p / 16`; compiled code reads only this.
    pub codes: usize,
}

impl BumpRegion {
    fn refresh_codes(&mut self) {
        self.codes = (self.objects as usize)
            .wrapping_add(self.q0)
            .wrapping_sub(self.base / QUANTUM);
    }
}

pub const BUMP_CUR: i32 = 0;
pub const BUMP_LIMIT: i32 = 8;
pub const BUMP_CODES: i32 = 40;
/// The start byte compiled code writes for an instance of `q` quanta.
pub const BUMP_PLAIN_FLAG: u8 = PLAIN;

/// The regions a thread bumps through, one set per heap it allocates
/// from; a set from another heap is dropped on the first allocation.
struct ThreadRegions {
    heap: u64,
    small: Region,
    medium: Region,
}

thread_local! {
    static REGIONS: UnsafeCell<ThreadRegions> = const {
        UnsafeCell::new(ThreadRegions {
            heap: 0,
            small: Region::EMPTY,
            medium: Region::EMPTY,
        })
    };
}

static NEXT_HEAP: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Heap
// ---------------------------------------------------------------------------

/// What a region refill takes the lock for.
struct Pool {
    chunks: Vec<Chunk>,
    /// Blocks in existence: the first `blocks` entries of every table.
    blocks: usize,
    /// Free block numbers; popped from the end.
    free_blocks: Vec<u32>,
    /// Runs of free lines found by the last sweep: (block, first line,
    /// line count). Consumed from the end.
    recycle_spans: Vec<(u32, u16, u16)>,
    /// Plain allocations that came to own memory outside the heap:
    /// dropped by the cycle that finds them dead.
    watched: Vec<usize>,
    last_collect: Instant,
}

#[repr(C)]
pub struct ImmixHeap {
    /// The region compiled code bumps, and the runtime's small region
    /// while the program has one thread. Closed for good once it has
    /// more.
    bump: BumpRegion,
    id: u64,
    multithreaded: AtomicBool,
    pool: Mutex<Pool>,
    /// Absolute base address of every block, indexed by block number.
    block_bases: Table<usize>,
    /// Blocks holding allocations (or handed out as a region).
    in_use: Table<u8>,
    /// Blocks that hold at least one line-owning allocation; gates the
    /// interior-pointer walk-back.
    has_span: Table<u8>,
    /// Free blocks whose pages were handed back to the OS and must be
    /// reclaimed before reuse (macOS keeps a reusable/reuse ledger).
    handed_back: Table<u8>,
    /// Regions currently bumping through the block: the sweep leaves
    /// its free lines to them.
    regions_in: Table<u8>,
    /// Chunk address (by `CHUNK_SHIFT`) to its first block number + 1.
    chunk_map: Table<u32>,
    /// One byte per quantum: 0 not a start, 1..=8 small object size in
    /// quanta, `SPAN_OBJECT` for a line-owning allocation.
    objects: Table<u8>,
    /// Lines owned by the span starting at that line, else 0.
    alloc_sizes: Table<u32>,
    /// One bit per line: a live object was marked in it this cycle.
    line_marks: Table<u64>,
    /// One byte per line: an allocation the sweep must drop starts in
    /// it, so the sweep walks it instead of clearing it whole. A byte
    /// so that a thread's note is a plain store.
    line_drop: Table<u8>,
    /// The colour that means "marked" in the open cycle.
    live_color: u8,
    /// The runtime's medium region while the program has one thread.
    medium: Region,

    total_allocated: AtomicUsize,
    total_freed: usize,
    freed_objects: usize,
    bytes_since_gc: AtomicUsize,
    external_since_gc: AtomicUsize,
    live_bytes: usize,
    trigger_threshold: AtomicUsize,
    polls: AtomicU32,
}

impl Default for ImmixHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl ImmixHeap {
    pub fn new() -> Self {
        let max_blocks = heap_max_bytes() / BLOCK_SIZE;
        let objects = Table::new(max_blocks * QUANTA_PER_BLOCK);
        let mut bump = BumpRegion {
            objects: objects.ptr,
            ..BumpRegion::default()
        };
        bump.refresh_codes();
        Self {
            bump,
            id: NEXT_HEAP.fetch_add(1, Relaxed),
            multithreaded: AtomicBool::new(false),
            pool: Mutex::new(Pool {
                chunks: Vec::new(),
                blocks: 0,
                free_blocks: Vec::new(),
                recycle_spans: Vec::new(),
                watched: Vec::new(),
                last_collect: Instant::now(),
            }),
            block_bases: Table::new(max_blocks),
            in_use: Table::new(max_blocks),
            has_span: Table::new(max_blocks),
            handed_back: Table::new(max_blocks),
            regions_in: Table::new(max_blocks),
            chunk_map: Table::new(1usize << (ADDRESS_BITS - CHUNK_SHIFT)),
            objects,
            alloc_sizes: Table::new(max_blocks * LINES_PER_BLOCK),
            line_marks: Table::new(max_blocks * LINE_WORDS),
            line_drop: Table::new(max_blocks * LINES_PER_BLOCK),
            live_color: COLOR_A,
            medium: Region::EMPTY,
            total_allocated: AtomicUsize::new(0),
            total_freed: 0,
            freed_objects: 0,
            bytes_since_gc: AtomicUsize::new(0),
            external_since_gc: AtomicUsize::new(0),
            live_bytes: 0,
            trigger_threshold: AtomicUsize::new(trigger_floor_bytes()),
            polls: AtomicU32::new(0),
        }
    }

    /// Close the compiled region: from here every thread allocates
    /// through the runtime, each bumping regions of its own. Called by
    /// the one running thread before it starts another.
    pub fn set_multithreaded(&mut self) {
        if self.multithreaded.swap(true, Relaxed) {
            return;
        }
        let small = self.bump_as_region();
        self.release(&small);
        self.set_bump(Region::EMPTY);
        let medium = self.medium;
        self.release(&medium);
        self.medium = Region::EMPTY;
    }

    pub fn is_multithreaded(&self) -> bool {
        self.multithreaded.load(Relaxed)
    }

    fn pool(&self) -> std::sync::MutexGuard<'_, Pool> {
        self.pool.lock().unwrap_or_else(|e| e.into_inner())
    }

    // -- Geometry helpers ---------------------------------------------------

    #[inline(always)]
    fn quantum_index(&self, block: u32, addr: usize) -> usize {
        block as usize * QUANTA_PER_BLOCK + (addr - self.block_bases.get(block as usize)) / QUANTUM
    }

    #[inline(always)]
    fn line_index(&self, block: u32, addr: usize) -> usize {
        block as usize * LINES_PER_BLOCK + (addr - self.block_bases.get(block as usize)) / LINE_SIZE
    }

    fn heap_bytes(&self) -> usize {
        self.pool().chunks.len() * CHUNK_BYTES
    }

    // -- Growth -------------------------------------------------------------

    fn add_chunk(&self, pool: &mut Pool) -> bool {
        if pool.chunks.len() * CHUNK_BYTES + CHUNK_BYTES > heap_max_bytes() {
            return false;
        }
        let Some(chunk) = Chunk::reserve() else {
            return false;
        };
        if chunk.base >> CHUNK_SHIFT >= self.chunk_map.len {
            return false;
        }
        let first = pool.blocks as u32;
        for i in 0..BLOCKS_PER_CHUNK {
            let b = pool.blocks + i;
            self.block_bases.set(b, chunk.base + i * BLOCK_SIZE);
            self.in_use.set(b, 0);
            self.has_span.set(b, 0);
            self.handed_back.set(b, 0);
            self.regions_in.set(b, 0);
        }
        // Published after the block tables it indexes.
        self.chunk_map.set(chunk.base >> CHUNK_SHIFT, first + 1);
        pool.blocks += BLOCKS_PER_CHUNK;
        // Push high blocks first so the lowest address pops next.
        for i in (0..BLOCKS_PER_CHUNK).rev() {
            pool.free_blocks.push(first + i as u32);
        }
        pool.chunks.push(chunk);
        true
    }

    fn acquire_free_block(&self, pool: &mut Pool) -> Option<u32> {
        if pool.free_blocks.is_empty() && !self.add_chunk(pool) {
            return None;
        }
        let b = pool.free_blocks.pop()?;
        self.in_use.set(b as usize, 1);
        if self.handed_back.get(b as usize) != 0 {
            self.handed_back.set(b as usize, 0);
            reclaim_pages(self.block_bases.get(b as usize), BLOCK_SIZE);
        }
        self.clear_metadata(b, 0, LINES_PER_BLOCK);
        Some(b)
    }

    fn clear_metadata(&self, block: u32, first_line: usize, lines: usize) {
        let b = block as usize;
        let q0 = b * QUANTA_PER_BLOCK + first_line * QUANTA_PER_LINE;
        let q1 = q0 + lines * QUANTA_PER_LINE;
        self.objects.fill(q0, q1, 0);
        let l0 = b * LINES_PER_BLOCK + first_line;
        self.alloc_sizes.fill(l0, l0 + lines, 0);
        self.line_drop.fill(l0, l0 + lines, 0);
    }

    /// Note that an allocation the sweep must drop starts at `addr`.
    #[inline(always)]
    fn note_droppable(&self, block: u32, addr: usize) {
        self.line_drop.set(self.line_index(block, addr), 1);
    }

    // -- Regions ------------------------------------------------------------

    /// Take a run of lines for a region: a recycled run of at least
    /// `lines`, else a fresh block. Empty once the heap is exhausted.
    fn take_region(&self, lines: usize) -> Region {
        let mut pool = self.pool();
        let taken = match pool
            .recycle_spans
            .iter()
            .rposition(|&(_, _, n)| n as usize >= lines)
        {
            Some(idx) => {
                let (b, first, n) = pool.recycle_spans.swap_remove(idx);
                let base = self.block_bases.get(b as usize) + first as usize * LINE_SIZE;
                self.clear_metadata(b, first as usize, n as usize);
                Region {
                    cur: base,
                    limit: base + n as usize * LINE_SIZE,
                    block: b,
                }
            }
            None => match self.acquire_free_block(&mut pool) {
                Some(b) => {
                    let base = self.block_bases.get(b as usize);
                    Region {
                        cur: base,
                        limit: base + BLOCK_SIZE,
                        block: b,
                    }
                }
                None => return Region::EMPTY,
            },
        };
        drop(pool);
        self.regions_in.fetch_add(taken.block as usize, 1);
        let len = taken.limit - taken.cur;
        self.bytes_since_gc.fetch_add(len, Relaxed);
        self.total_allocated.fetch_add(len, Relaxed);
        taken
    }

    /// Give a region up; its block's lines are the sweep's again.
    fn release(&self, region: &Region) {
        if region.limit != 0 {
            self.regions_in.fetch_sub(region.block as usize, 1);
        }
    }

    fn bump_as_region(&self) -> Region {
        Region {
            cur: self.bump.cur,
            limit: self.bump.limit,
            block: if self.bump.limit == 0 {
                0
            } else {
                ((self.bump.q0) / QUANTA_PER_BLOCK) as u32
            },
        }
    }

    fn set_bump(&mut self, region: Region) {
        self.bump.cur = region.cur;
        self.bump.limit = region.limit;
        if region.is_empty() {
            self.bump.base = 0;
            self.bump.q0 = 0;
        } else {
            self.bump.base = self.block_bases.get(region.block as usize);
            self.bump.q0 = region.block as usize * QUANTA_PER_BLOCK;
        }
        self.bump.refresh_codes();
    }

    /// The bump region compiled code allocates from.
    pub fn bump_region(&self) -> *const BumpRegion {
        &self.bump
    }

    /// This thread's regions of this heap. The pointer is to this
    /// thread's own cell; it is used only for the allocation that
    /// asked for it.
    #[inline(always)]
    fn thread_regions(&self) -> *mut ThreadRegions {
        REGIONS.with(|cell| {
            let r = cell.get();
            unsafe {
                if (*r).heap != self.id {
                    (*r).heap = self.id;
                    (*r).small = Region::EMPTY;
                    (*r).medium = Region::EMPTY;
                }
            }
            r
        })
    }

    // -- Allocation ---------------------------------------------------------

    /// Bump `size` bytes out of `region`, never straddling a line.
    #[inline(always)]
    fn bump_small(&self, region: &mut Region, size: usize, flags: u8) -> *mut u8 {
        let mut p = region.cur;
        if (p & (LINE_SIZE - 1)) + size > LINE_SIZE {
            p = p.next_multiple_of(LINE_SIZE);
        }
        let np = p + size;
        if np > region.limit {
            return std::ptr::null_mut();
        }
        region.cur = np;
        let b = region.block;
        let q = self.quantum_index(b, p);
        self.objects.set(q, (size / QUANTUM) as u8 | flags);
        if flags & PLAIN == 0 {
            self.note_droppable(b, p);
        }
        p as *mut u8
    }

    /// Hand out `size` bytes (16-aligned, at most one line) from the
    /// small region, never straddling a line. Null once the heap is
    /// exhausted.
    #[inline]
    fn alloc_small(&mut self, size: usize, flags: u8) -> *mut u8 {
        if !self.is_multithreaded() {
            loop {
                // Compiled code bumps the same region; `bump` is the
                // truth for `cur`.
                let mut region = self.bump_as_region();
                let p = self.bump_small(&mut region, size, flags);
                if !p.is_null() {
                    self.bump.cur = region.cur;
                    return p;
                }
                self.release(&region);
                let fresh = self.take_region(1);
                self.set_bump(fresh);
                if fresh.is_empty() {
                    return std::ptr::null_mut();
                }
            }
        }
        let regions = unsafe { &mut *self.thread_regions() };
        loop {
            let p = self.bump_small(&mut regions.small, size, flags);
            if !p.is_null() {
                return p;
            }
            self.release(&regions.small);
            regions.small = self.take_region(1);
            if regions.small.is_empty() {
                return std::ptr::null_mut();
            }
        }
    }

    /// Bump whole lines for an allocation larger than a line.
    #[inline]
    fn bump_medium(&self, region: &mut Region, lines: usize, flags: u8) -> *mut u8 {
        let p = region.cur.next_multiple_of(LINE_SIZE);
        let np = p + lines * LINE_SIZE;
        if np > region.limit || region.is_empty() {
            return std::ptr::null_mut();
        }
        region.cur = np;
        let b = region.block;
        let q = self.quantum_index(b, p);
        self.objects.set(q, SPAN_OBJECT | flags);
        if flags & PLAIN == 0 {
            self.note_droppable(b, p);
        }
        let l = self.line_index(b, p);
        self.alloc_sizes.set(l, lines as u32);
        self.has_span.set(b as usize, 1);
        p as *mut u8
    }

    /// Hand out whole lines for an allocation larger than a line. Null
    /// once the heap is exhausted.
    fn alloc_medium(&mut self, size: usize, flags: u8) -> *mut u8 {
        let lines = size.div_ceil(LINE_SIZE);
        debug_assert!(lines <= LINES_PER_BLOCK);
        if !self.is_multithreaded() {
            loop {
                let mut region = self.medium;
                let p = self.bump_medium(&mut region, lines, flags);
                if !p.is_null() {
                    self.medium = region;
                    return p;
                }
                self.release(&region);
                self.medium = self.take_region(lines);
                if self.medium.is_empty() {
                    return std::ptr::null_mut();
                }
            }
        }
        let regions = unsafe { &mut *self.thread_regions() };
        loop {
            let p = self.bump_medium(&mut regions.medium, lines, flags);
            if !p.is_null() {
                return p;
            }
            self.release(&regions.medium);
            regions.medium = self.take_region(lines);
            if regions.medium.is_empty() {
                return std::ptr::null_mut();
            }
        }
    }

    /// `size` bytes, 16-aligned, contents unspecified; null once the
    /// heap is exhausted.
    #[inline]
    pub fn alloc_raw(&mut self, size: usize) -> *mut u8 {
        self.alloc_with(size, 0)
    }

    /// `alloc_raw` for contents that own nothing outside the heap; the
    /// sweep reclaims the allocation without the drop.
    #[inline]
    pub fn alloc_plain(&mut self, size: usize) -> *mut u8 {
        self.alloc_with(size, PLAIN)
    }

    #[inline]
    fn alloc_with(&mut self, size: usize, flags: u8) -> *mut u8 {
        let size = size.max(QUANTUM).next_multiple_of(QUANTUM);
        assert!(
            size <= BLOCK_SIZE,
            "Immix: allocation of {size} bytes exceeds a block"
        );
        if size <= SMALL_MAX {
            self.alloc_small(size, flags)
        } else {
            self.alloc_medium(size, flags)
        }
    }

    // -- Enumeration --------------------------------------------------------

    fn block_count(&self) -> usize {
        self.pool().blocks
    }

    /// Every allocation start, in address order.
    pub fn for_each_allocation<F: FnMut(*mut u8)>(&self, mut f: F) {
        for b in 0..self.block_count() {
            if self.in_use.get(b) == 0 {
                continue;
            }
            let base = self.block_bases.get(b);
            let q0 = b * QUANTA_PER_BLOCK;
            let mut q = 0;
            while q < QUANTA_PER_BLOCK {
                let code = self.objects.get(q0 + q);
                if code == 0 {
                    q += 1;
                    continue;
                }
                let quanta = self.alloc_quanta(b, q, code);
                f((base + q * QUANTUM) as *mut u8);
                q += quanta;
            }
        }
    }

    #[inline(always)]
    fn alloc_quanta(&self, block: usize, q: usize, code: u8) -> usize {
        let code = code & CODE_MASK;
        if code == SPAN_OBJECT {
            self.alloc_sizes
                .get(block * LINES_PER_BLOCK + q / QUANTA_PER_LINE) as usize
                * QUANTA_PER_LINE
        } else {
            code as usize
        }
    }

    // -- Address resolution -------------------------------------------------

    /// Block number holding `addr`, if it is inside the heap.
    #[inline]
    fn block_containing(&self, addr: usize) -> Option<u32> {
        if ADDRESS_BITS < usize::BITS && addr >> ADDRESS_BITS != 0 {
            return None;
        }
        let first = self.chunk_map.get(addr >> CHUNK_SHIFT);
        if first == 0 {
            return None;
        }
        Some(first - 1 + ((addr & (CHUNK_BYTES - 1)) / BLOCK_SIZE) as u32)
    }

    /// Whether `addr` lies inside an allocation.
    pub fn is_heap_ptr(&self, addr: usize) -> bool {
        !self.containing_allocation(addr).is_null()
    }

    /// Start of the allocation containing `addr`, or null. Walks back
    /// within the line for a small object, then across lines for a
    /// span when the block holds one.
    pub fn containing_allocation(&self, addr: usize) -> *mut u8 {
        let Some(b) = self.block_containing(addr) else {
            return std::ptr::null_mut();
        };
        if self.in_use.get(b as usize) == 0 {
            return std::ptr::null_mut();
        }
        let base = self.block_bases.get(b as usize);
        let off = addr - base;
        let q = off / QUANTUM;
        let line_first_q = q - q % QUANTA_PER_LINE;
        let q0 = b as usize * QUANTA_PER_BLOCK;
        let mut i = q;
        loop {
            let code = self.objects.get(q0 + i);
            if code != 0 {
                let quanta = self.alloc_quanta(b as usize, i, code);
                return if q < i + quanta {
                    (base + i * QUANTUM) as *mut u8
                } else {
                    std::ptr::null_mut()
                };
            }
            if i == line_first_q {
                break;
            }
            i -= 1;
        }
        if self.has_span.get(b as usize) == 0 {
            return std::ptr::null_mut();
        }
        let line = off / LINE_SIZE;
        let l0 = b as usize * LINES_PER_BLOCK;
        let mut l = line;
        loop {
            let n = self.alloc_sizes.get(l0 + l) as usize;
            if n != 0 {
                let start_q = l * QUANTA_PER_LINE;
                return if self.objects.get(q0 + start_q) & CODE_MASK == SPAN_OBJECT && line < l + n
                {
                    (base + start_q * QUANTUM) as *mut u8
                } else {
                    std::ptr::null_mut()
                };
            }
            if l == 0 {
                return std::ptr::null_mut();
            }
            l -= 1;
        }
    }

    /// Call `visit` with every allocation a word in `lo..hi` might point
    /// at: raw addresses and NaN-boxed object payloads that land inside
    /// an allocation.
    pub fn scan_range<F: FnMut(*mut u8)>(&self, lo: usize, hi: usize, mut visit: F) {
        let word = std::mem::size_of::<usize>();
        let mut p = lo.next_multiple_of(word);
        while p + word <= hi {
            let w = unsafe { std::ptr::read_volatile(p as *const usize) };
            let mut candidate = self.containing_allocation(w);
            if candidate.is_null() && word == 8 {
                let v = Value::from_bits(w as u64);
                if let Some(obj) = v.as_object() {
                    candidate = self.containing_allocation(obj as usize);
                }
            }
            if !candidate.is_null() {
                visit(candidate);
            }
            p += word;
        }
    }

    // -- Marking ------------------------------------------------------------

    /// Claim `ptr` live for the open cycle; false if it already was.
    ///
    /// # Safety
    /// `ptr` must be the start of an `ObjHeader`-led object.
    #[inline(always)]
    pub unsafe fn mark(&mut self, ptr: *mut u8) -> bool {
        let header = ptr as *mut ObjHeader;
        if (*header).gc_mark == self.live_color {
            return false;
        }
        (*header).gc_mark = self.live_color;
        // The lines the object covers are live; a sweep frees the rest
        // without reading them.
        let addr = ptr as usize;
        if let Some(b) = self.block_containing(addr) {
            let b = b as usize;
            let q = (addr - self.block_bases.get(b)) / QUANTUM;
            let code = self.objects.get(b * QUANTA_PER_BLOCK + q);
            let first = q / QUANTA_PER_LINE;
            // Marking is the collector's alone, so the words are updated
            // with plain loads and stores.
            if code & CODE_MASK != SPAN_OBJECT {
                // A small object never straddles a line.
                let w = b * LINE_WORDS + first / 64;
                self.line_marks
                    .set(w, self.line_marks.get(w) | 1u64 << (first % 64));
            } else {
                let quanta = self.alloc_quanta(b, q, code).max(1);
                let last = (q + quanta - 1) / QUANTA_PER_LINE;
                for l in first..=last {
                    let w = b * LINE_WORDS + l / 64;
                    self.line_marks
                        .set(w, self.line_marks.get(w) | 1u64 << (l % 64));
                }
            }
        }
        true
    }

    /// # Safety
    /// As [`ImmixHeap::mark`].
    #[inline(always)]
    pub unsafe fn is_marked(&self, ptr: *mut u8) -> bool {
        (*(ptr as *mut ObjHeader)).gc_mark == self.live_color
    }

    /// Open a cycle: the other colour now means marked, so last cycle's
    /// survivors count as unmarked without a reset pass.
    pub fn collect_begin(&mut self) {
        self.live_color = if self.live_color == COLOR_A {
            COLOR_B
        } else {
            COLOR_A
        };
        let n = self.block_count() * LINE_WORDS;
        for w in 0..n {
            self.line_marks.set(w, 0);
        }
    }

    // -- Trigger ------------------------------------------------------------

    pub fn track_external(&mut self, bytes: usize) {
        self.external_since_gc.fetch_add(bytes, Relaxed);
    }

    /// Run the drop on the plain allocation at `ptr` once it dies.
    pub fn watch(&mut self, ptr: *mut u8) -> bool {
        let addr = ptr as usize;
        let Some(b) = self.block_containing(addr) else {
            return false;
        };
        let b = b as usize;
        if self.in_use.get(b) == 0 {
            return false;
        }
        let q = (addr - self.block_bases.get(b)) / QUANTUM;
        let code = self.objects.get(b * QUANTA_PER_BLOCK + q);
        if code & CODE_MASK == 0 || code & PLAIN == 0 {
            return false;
        }
        let mut pool = self.pool();
        if !pool.watched.contains(&addr) {
            pool.watched.push(addr);
        }
        true
    }

    pub fn should_collect(&self) -> bool {
        let since = self.bytes_since_gc.load(Relaxed);
        if stress_enabled() {
            return since > 0;
        }
        if since + self.external_since_gc.load(Relaxed) >= self.trigger_threshold.load(Relaxed) {
            return true;
        }
        // The clock is read rarely: a long-idle heap with some garbage
        // gets collected on a heartbeat rather than never.
        let n = self.polls.fetch_add(1, Relaxed).wrapping_add(1);
        n & 1023 == 0 && since > 0 && self.pool().last_collect.elapsed() >= HEARTBEAT
    }

    pub fn stats(&self) -> RtStats {
        RtStats {
            heap_bytes: self.heap_bytes(),
            live_bytes: self.live_bytes,
            allocated_bytes: self.total_allocated.load(Relaxed),
            freed_bytes: self.total_freed,
            freed_objects: self.freed_objects,
        }
    }

    // -- Sweep --------------------------------------------------------------

    /// Close a cycle: `drop` every unmarked allocation, recycle its
    /// lines, reset the trigger, and return the live bytes.
    pub fn collect_end<F: FnMut(*mut u8)>(&mut self, drop: F) -> usize {
        // This thread's regions are swept like any other lines and
        // fresh ones taken on the next allocation; other threads keep
        // theirs, and the sweep leaves those blocks' free lines alone.
        if !self.is_multithreaded() {
            let small = self.bump_as_region();
            self.release(&small);
            self.set_bump(Region::EMPTY);
            let medium = self.medium;
            self.release(&medium);
            self.medium = Region::EMPTY;
        } else {
            let regions = unsafe { &mut *self.thread_regions() };
            self.release(&regions.small);
            self.release(&regions.medium);
            regions.small = Region::EMPTY;
            regions.medium = Region::EMPTY;
        }

        let quiet = self.pool().last_collect.elapsed() >= HEARTBEAT;
        let mut drop = drop;
        let live = self.sweep(&mut drop);
        // A dead watched object's memory is intact until a region is
        // handed out again, so its mark byte still says it died.
        let color = self.live_color;
        let mut watched = std::mem::take(&mut self.pool().watched);
        watched.retain(|&p| {
            let alive = unsafe { (*(p as *const ObjHeader)).gc_mark == color };
            if !alive {
                drop(p as *mut u8);
            }
            alive
        });
        self.pool().watched = watched;
        if quiet {
            self.hand_back_free_blocks();
        }
        self.live_bytes = live;
        let floor = trigger_floor_bytes();
        let ceiling = TRIGGER_CEILING.max(live).max(floor);
        self.trigger_threshold.store(
            (live.saturating_mul(growth_factor())).clamp(floor, ceiling),
            Relaxed,
        );
        self.bytes_since_gc.store(0, Relaxed);
        self.external_since_gc.store(0, Relaxed);
        self.pool().last_collect = Instant::now();
        live
    }

    /// Free dead objects, rebuild the free-line runs, and return the
    /// live bytes.
    fn sweep<F: FnMut(*mut u8)>(&mut self, mut drop: F) -> usize {
        let mut pool = self.pool();
        pool.recycle_spans.clear();
        let mut live_bytes = 0usize;
        let mut freed_bytes = 0usize;
        let mut freed_objects = 0usize;
        let nblocks = pool.blocks;
        let color = self.live_color;
        for b in 0..nblocks {
            if self.in_use.get(b) == 0 {
                continue;
            }
            let base = self.block_bases.get(b);
            let q0 = b * QUANTA_PER_BLOCK;
            let mut line_live = [0u64; LINE_WORDS];
            line_live.copy_from_slice(self.line_marks.slice(b * LINE_WORDS, (b + 1) * LINE_WORDS));
            let any_live = line_live.iter().any(|w| *w != 0);
            let any_droppable = self
                .line_drop
                .slice(b * LINES_PER_BLOCK, (b + 1) * LINES_PER_BLOCK)
                .as_chunks::<8>()
                .0
                .iter()
                .any(|w| *w != [0u8; 8]);
            // A block some thread is bumping through keeps its free
            // lines for that thread.
            let held = self.regions_in.get(b) != 0;
            if !any_live && !any_droppable && !held {
                // Only dead plain objects: count them a word of start
                // bytes at a time and free the block whole; the next
                // owner clears its tables.
                let (n, bytes) = count_dead_plain(self.objects.slice(q0, q0 + QUANTA_PER_BLOCK));
                freed_objects += n;
                freed_bytes += bytes;
                let l0 = b * LINES_PER_BLOCK;
                for l in 0..LINES_PER_BLOCK {
                    let s = self.alloc_sizes.get(l0 + l) as usize;
                    if s != 0 {
                        freed_bytes += s * LINE_SIZE;
                    }
                }
                self.in_use.set(b, 0);
                self.has_span.set(b, 0);
                pool.free_blocks.push(b as u32);
                continue;
            }
            // A run of lines nothing was marked in and nothing to drop
            // starts in holds only dead plain objects: its start bytes
            // are cleared without reading the objects. A dead span
            // starting there is cleared with its size table entry.
            // Every other line is walked object by object.
            let mut l = 0usize;
            while l < LINES_PER_BLOCK {
                let live = line_live[l / 64] & (1u64 << (l % 64)) != 0;
                let droppable = self.line_drop.get(b * LINES_PER_BLOCK + l) != 0;
                let lq = l * QUANTA_PER_LINE;
                if !live && !droppable {
                    for i in q0 + lq..q0 + lq + QUANTA_PER_LINE {
                        let c = self.objects.get(i);
                        if c != 0 {
                            freed_objects += 1;
                            freed_bytes += if c & CODE_MASK == SPAN_OBJECT {
                                0
                            } else {
                                (c & CODE_MASK) as usize * QUANTUM
                            };
                            self.objects.set(i, 0);
                        }
                    }
                    let li = b * LINES_PER_BLOCK + l;
                    let s = self.alloc_sizes.get(li);
                    if s != 0 {
                        freed_bytes += s as usize * LINE_SIZE;
                        self.alloc_sizes.set(li, 0);
                    }
                    l += 1;
                    continue;
                }
                let mut q = lq;
                let end = lq + QUANTA_PER_LINE;
                // The line keeps its droppable bit only for a live
                // object the next sweep may have to drop.
                let mut keep_droppable = false;
                while q < end {
                    let code = self.objects.get(q0 + q);
                    if code == 0 {
                        q += 1;
                        continue;
                    }
                    let quanta = self.alloc_quanta(b, q, code);
                    let header = (base + q * QUANTUM) as *mut ObjHeader;
                    let marked = unsafe { (*header).gc_mark == color };
                    if marked {
                        live_bytes += quanta * QUANTUM;
                        keep_droppable |= code & PLAIN == 0;
                    } else {
                        if code & PLAIN == 0 {
                            drop(header as *mut u8);
                        }
                        self.objects.set(q0 + q, 0);
                        if code & CODE_MASK == SPAN_OBJECT {
                            self.alloc_sizes
                                .set(b * LINES_PER_BLOCK + q / QUANTA_PER_LINE, 0);
                        }
                        freed_bytes += quanta * QUANTUM;
                        freed_objects += 1;
                    }
                    q += quanta;
                }
                self.line_drop
                    .set(b * LINES_PER_BLOCK + l, keep_droppable as u8);
                // A span walked from its first line covers the lines
                // it owns; continue past them.
                l = (q.max(end) - 1) / QUANTA_PER_LINE + 1;
            }
            if held {
                continue;
            }
            if !any_live {
                self.in_use.set(b, 0);
                self.has_span.set(b, 0);
                pool.free_blocks.push(b as u32);
                continue;
            }
            // Runs of free lines become bump regions.
            let mut l = 0usize;
            while l < LINES_PER_BLOCK {
                if line_live[l / 64] & (1u64 << (l % 64)) != 0 {
                    l += 1;
                    continue;
                }
                let start = l;
                while l < LINES_PER_BLOCK && line_live[l / 64] & (1u64 << (l % 64)) == 0 {
                    l += 1;
                }
                pool.recycle_spans
                    .push((b as u32, start as u16, (l - start) as u16));
            }
        }
        std::mem::drop(pool);
        self.total_freed += freed_bytes;
        self.freed_objects += freed_objects;
        live_bytes
    }
}

/// `(objects, bytes)` of the allocation starts in `codes`, none of them
/// spans: eight start bytes are summed as one word. A span's bytes are
/// counted from the size table by the caller.
fn count_dead_plain(codes: &[u8]) -> (usize, usize) {
    const LOW: u64 = 0x0101_0101_0101_0101;
    let mut objects = 0usize;
    let mut quanta = 0usize;
    for chunk in codes.as_chunks::<8>().0 {
        let w = u64::from_ne_bytes(*chunk);
        if w == 0 {
            continue;
        }
        // A start byte is at most PLAIN | SPAN_OBJECT, so folding its
        // five low bits into bit 0 marks each nonzero byte once.
        let nz = (w | (w >> 1) | (w >> 2) | (w >> 3) | (w >> 4)) & LOW;
        objects += nz.count_ones() as usize;
        // Low nibbles hold sizes in quanta (a span's reads as
        // SPAN_OBJECT and is replaced by its table entry below).
        let nibbles = w & (LOW * 0x0F);
        quanta += (nibbles.wrapping_mul(LOW) >> 56) as usize;
        let spans = (nibbles & (LOW * SPAN_OBJECT as u64)) ^ (LOW * SPAN_OBJECT as u64);
        // Bytes equal to SPAN_OBJECT have zero in every bit of `spans`.
        let is_span = !((spans | (spans >> 1) | (spans >> 2) | (spans >> 3)) & LOW) & LOW;
        quanta -= is_span.count_ones() as usize * SPAN_OBJECT as usize;
    }
    (objects, quanta * QUANTUM)
}

/// Free blocks kept resident so a burst after an idle period does not
/// pay page faults immediately.
const RESIDENT_FLOAT: usize = 16;

impl ImmixHeap {
    /// Return the pages of free blocks beyond the resident float to the
    /// OS. Only called after a quiet collection: a churning workload
    /// would otherwise pay a madvise pair per block per cycle.
    fn hand_back_free_blocks(&mut self) {
        let mut pool = self.pool();
        if pool.free_blocks.len() <= RESIDENT_FLOAT {
            return;
        }
        pool.free_blocks.sort_unstable_by(|a, b| b.cmp(a));
        // The float stays at the end of the list (lowest addresses),
        // which is what pops next.
        let n = pool.free_blocks.len() - RESIDENT_FLOAT;
        let mut run_start: Option<(usize, usize)> = None;
        for &b in pool.free_blocks[..n].iter().rev() {
            let b = b as usize;
            if self.handed_back.get(b) != 0 {
                continue;
            }
            self.handed_back.set(b, 1);
            let base = self.block_bases.get(b);
            match run_start {
                Some((lo, len)) if lo + len == base => run_start = Some((lo, len + BLOCK_SIZE)),
                Some((lo, len)) => {
                    hand_back_pages(lo, len);
                    run_start = Some((base, BLOCK_SIZE));
                }
                None => run_start = Some((base, BLOCK_SIZE)),
            }
        }
        if let Some((lo, len)) = run_start {
            hand_back_pages(lo, len);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "host"))]
fn hand_back_pages(addr: usize, len: usize) {
    unsafe {
        libc::madvise(addr as *mut libc::c_void, len, libc::MADV_FREE_REUSABLE);
    }
}
#[cfg(all(target_os = "macos", feature = "host"))]
fn reclaim_pages(addr: usize, len: usize) {
    unsafe {
        libc::madvise(addr as *mut libc::c_void, len, libc::MADV_FREE_REUSE);
    }
}
#[cfg(all(target_os = "linux", feature = "host"))]
fn hand_back_pages(addr: usize, len: usize) {
    unsafe {
        libc::madvise(addr as *mut libc::c_void, len, libc::MADV_DONTNEED);
    }
}
#[cfg(all(target_os = "linux", feature = "host"))]
fn reclaim_pages(_addr: usize, _len: usize) {}
#[cfg(not(any(
    all(target_os = "macos", feature = "host"),
    all(target_os = "linux", feature = "host")
)))]
fn hand_back_pages(_addr: usize, _len: usize) {}
#[cfg(not(any(
    all(target_os = "macos", feature = "host"),
    all(target_os = "linux", feature = "host")
)))]
fn reclaim_pages(_addr: usize, _len: usize) {}

// Chunks unmap on their own drop; the objects in them are the
// strategy's to drop first.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::object::{ObjHeader, ObjType};

    #[test]
    fn bump_region_layout_matches_compiled_code() {
        assert_eq!(std::mem::offset_of!(BumpRegion, cur), BUMP_CUR as usize);
        assert_eq!(std::mem::offset_of!(BumpRegion, limit), BUMP_LIMIT as usize);
        assert_eq!(std::mem::offset_of!(BumpRegion, codes), BUMP_CODES as usize);
        assert_eq!(std::mem::offset_of!(ImmixHeap, bump), 0);
    }

    #[test]
    fn bump_codes_addresses_the_start_byte() {
        let mut heap = ImmixHeap::new();
        let p = alloc(&mut heap, 48);
        let bump = unsafe { *heap.bump_region() };
        let code = unsafe { *((bump.codes.wrapping_add(p as usize / QUANTUM)) as *const u8) };
        assert_eq!(code & CODE_MASK, 3);
    }

    /// A header-led allocation of `size` bytes, white.
    fn alloc(heap: &mut ImmixHeap, size: usize) -> *mut u8 {
        let p = heap.alloc_raw(size);
        assert!(!p.is_null());
        unsafe { (p as *mut ObjHeader).write(ObjHeader::new(ObjType::Range)) };
        p
    }

    fn sweep_unmarked(heap: &mut ImmixHeap) -> (usize, usize) {
        let mut dropped = 0;
        let live = heap.collect_end(|_| dropped += 1);
        (live, dropped)
    }

    #[test]
    fn small_objects_never_straddle_a_line() {
        let mut heap = ImmixHeap::new();
        let mut last: usize = 0;
        for _ in 0..10_000 {
            let p = alloc(&mut heap, 40) as usize;
            let size = 40usize.next_multiple_of(QUANTUM);
            assert_eq!(
                p / LINE_SIZE,
                (p + size - 1) / LINE_SIZE,
                "straddles a line"
            );
            assert!(p != last);
            last = p;
        }
    }

    #[test]
    fn spans_own_whole_lines() {
        let mut heap = ImmixHeap::new();
        let f = alloc(&mut heap, 700) as usize;
        assert_eq!(f % LINE_SIZE, 0);
        let s = alloc(&mut heap, 24) as usize;
        let lines = 700usize.div_ceil(LINE_SIZE);
        assert!(
            s >= f + lines * LINE_SIZE,
            "small object packed into a span's tail line"
        );
    }

    #[test]
    fn interior_pointers_resolve_to_their_allocation() {
        let mut heap = ImmixHeap::new();
        let small = alloc(&mut heap, 40) as usize;
        let span = alloc(&mut heap, 700) as usize;
        for off in [0, 8, 39] {
            assert_eq!(heap.containing_allocation(small + off) as usize, small);
        }
        for off in [0, 200, 699] {
            assert_eq!(heap.containing_allocation(span + off) as usize, span);
        }
        // A span owns its reserved lines end to end; addresses outside
        // the heap resolve to nothing.
        let span_end = span + 700usize.div_ceil(LINE_SIZE) * LINE_SIZE;
        assert_eq!(heap.containing_allocation(span_end - 1) as usize, span);
        assert!(heap.containing_allocation(0x1000).is_null());
        assert!(heap.containing_allocation(small.wrapping_sub(1)).is_null());
        assert!(heap.is_heap_ptr(small + 8));
        assert!(!heap.is_heap_ptr(0x1000));
        assert!(heap.containing_allocation(usize::MAX).is_null());
    }

    #[test]
    fn sweep_frees_unmarked_and_reuses_their_lines() {
        let mut heap = ImmixHeap::new();
        let keep = alloc(&mut heap, 40);
        for _ in 0..50_000 {
            alloc(&mut heap, 40);
        }
        heap.collect_begin();
        assert!(unsafe { heap.mark(keep) });
        assert!(!unsafe { heap.mark(keep) });
        assert!(unsafe { heap.is_marked(keep) });
        let (live, dropped) = sweep_unmarked(&mut heap);
        assert_eq!(dropped, 50_000);
        assert_eq!(live, 48);
        // The next cycle's colour is the other one, so last cycle's
        // mark no longer counts.
        heap.collect_begin();
        assert!(!unsafe { heap.is_marked(keep) });
        let allocated = heap.total_allocated.load(Relaxed);
        for _ in 0..50_000 {
            alloc(&mut heap, 40);
        }
        assert!(unsafe { heap.mark(keep) });
        let (_, dropped) = sweep_unmarked(&mut heap);
        assert_eq!(dropped, 50_000);
        assert!(
            heap.total_allocated.load(Relaxed) < allocated * 2,
            "freed lines were not reused"
        );
    }

    #[test]
    fn threads_allocate_into_regions_of_their_own() {
        let mut heap = ImmixHeap::new();
        let before = alloc(&mut heap, 40);
        heap.set_multithreaded();
        assert_eq!(unsafe { (*heap.bump_region()).limit }, 0);
        let here = alloc(&mut heap, 40) as usize;
        let heap_ptr = &mut heap as *mut ImmixHeap as usize;
        let there = std::thread::spawn(move || {
            let heap = unsafe { &mut *(heap_ptr as *mut ImmixHeap) };
            alloc(heap, 40) as usize
        })
        .join()
        .unwrap();
        // Different threads bump different blocks.
        let there_block = heap.block_containing(there).unwrap() as usize;
        assert_ne!(here / BLOCK_SIZE, there / BLOCK_SIZE);
        assert!(heap.is_heap_ptr(there));
        // A sweep leaves the block a thread's region is in alone and
        // frees the rest.
        heap.collect_begin();
        assert!(unsafe { heap.mark(before) });
        let (live, dropped) = sweep_unmarked(&mut heap);
        assert_eq!(live, 48);
        assert_eq!(dropped, 2);
        assert!(heap.regions_in.get(there_block) != 0);
        assert!(heap.in_use.get(there_block) != 0);
    }

    #[test]
    fn conservative_scan_finds_boxed_raw_and_interior_words() {
        let mut heap = ImmixHeap::new();
        let boxed = alloc(&mut heap, 40);
        let raw = alloc(&mut heap, 40);
        let interior = alloc(&mut heap, 700);
        let _other = alloc(&mut heap, 40);
        let words: [usize; 4] = [
            Value::object(boxed).to_bits() as usize,
            raw as usize,
            interior as usize + 100,
            0xdead_beef,
        ];
        let lo = words.as_ptr() as usize;
        let hi = lo + std::mem::size_of_val(&words);
        let mut seen = Vec::new();
        heap.scan_range(lo, hi, |p| seen.push(p));
        assert_eq!(seen, vec![boxed, raw, interior]);
    }

    #[test]
    fn handed_back_blocks_are_reclaimed_before_reuse() {
        let mut heap = ImmixHeap::new();
        for _ in 0..200_000 {
            alloc(&mut heap, 40);
        }
        sweep_unmarked(&mut heap);
        assert!(heap.pool().free_blocks.len() > RESIDENT_FLOAT);
        heap.hand_back_free_blocks();
        let blocks = heap.block_count();
        let handed = (0..blocks)
            .filter(|&b| heap.handed_back.get(b) != 0)
            .count();
        assert_eq!(handed, heap.pool().free_blocks.len() - RESIDENT_FLOAT);
        // Allocate through every handed-back block and verify the
        // memory is usable and the ledger clears.
        for _ in 0..200_000 {
            let p = alloc(&mut heap, 40);
            unsafe { p.add(32).write(0x5a) };
            assert_eq!(unsafe { p.add(32).read() }, 0x5a);
        }
        let still = (0..blocks)
            .filter(|&b| heap.handed_back.get(b) != 0 && heap.in_use.get(b) != 0)
            .count();
        assert_eq!(still, 0, "an in-use block is still marked handed back");
    }

    #[test]
    fn allocations_enumerate_in_address_order() {
        let mut heap = ImmixHeap::new();
        let a = alloc(&mut heap, 40);
        let b = alloc(&mut heap, 700);
        let c = alloc(&mut heap, 40);
        let mut seen = Vec::new();
        heap.for_each_allocation(|p| seen.push(p));
        let mut expected = vec![a, b, c];
        expected.sort_unstable();
        assert_eq!(seen, expected);
    }
}
