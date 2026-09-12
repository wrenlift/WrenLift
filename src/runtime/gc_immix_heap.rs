//! wren_lift's own memory under the Immix strategy: the default behind
//! every heap slot of `rt`.
//!
//! The heap is a set of block-aligned chunks carved into 32 KiB blocks
//! of 128-byte lines. Objects up to one line bump-allocate through a
//! region and never straddle a line; larger objects take whole lines
//! and never share their last line. Allocation starts and span sizes
//! live in side tables keyed by block, so any address inside an
//! allocation resolves to its start. Every allocation begins with an
//! `ObjHeader`, so the header's mark byte is the memory-side liveness
//! claim; sweep derives line occupancy from the surviving starts, runs
//! the strategy's drop on every dead object, and hands runs of free
//! lines back as bump regions. Nothing moves.

use super::object::ObjHeader;
use super::rt::{RtStats, MAX_ALLOC};
use super::value::Value;
use crate::portable_time::Instant;

use std::cell::Cell;
use std::sync::OnceLock;
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

/// Largest allocation that goes through the bump region.
const SMALL_MAX: usize = LINE_SIZE;

/// Heap grows one chunk at a time up to the configured maximum.
const CHUNK_BYTES: usize = 32 * 1024 * 1024;
const BLOCKS_PER_CHUNK: usize = CHUNK_BYTES / BLOCK_SIZE;

#[cfg(not(target_pointer_width = "32"))]
const DEFAULT_HEAP_MAX: usize = 4 * 1024 * 1024 * 1024;
#[cfg(target_pointer_width = "32")]
const DEFAULT_HEAP_MAX: usize = 256 * 1024 * 1024;

const TRIGGER_FLOOR: usize = 8 * 1024 * 1024;
const TRIGGER_CEILING: usize = 512 * 1024 * 1024;
const DEFAULT_GROWTH: usize = 4;
const HEARTBEAT: Duration = Duration::from_secs(30);

const WHITE: u8 = 0;
const BLACK: u8 = 2;

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
// Chunks
// ---------------------------------------------------------------------------

/// One block-aligned mapping. Pages are committed on first touch.
struct Chunk {
    base: usize,
    #[cfg(all(unix, feature = "host"))]
    raw: (*mut u8, usize),
    #[cfg(not(all(unix, feature = "host")))]
    layout: std::alloc::Layout,
}

impl Chunk {
    fn reserve() -> Option<Chunk> {
        #[cfg(all(unix, feature = "host"))]
        {
            let len = CHUNK_BYTES + BLOCK_SIZE;
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
            let base = (p as usize).next_multiple_of(BLOCK_SIZE);
            Some(Chunk {
                base,
                raw: (p as *mut u8, len),
            })
        }
        #[cfg(not(all(unix, feature = "host")))]
        {
            let layout = std::alloc::Layout::from_size_align(CHUNK_BYTES, BLOCK_SIZE).ok()?;
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            if p.is_null() {
                return None;
            }
            Some(Chunk {
                base: p as usize,
                layout,
            })
        }
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        #[cfg(all(unix, feature = "host"))]
        unsafe {
            libc::munmap(self.raw.0 as *mut libc::c_void, self.raw.1);
        }
        #[cfg(not(all(unix, feature = "host")))]
        unsafe {
            std::alloc::dealloc(self.base as *mut u8, self.layout);
        }
    }
}

// ---------------------------------------------------------------------------
// Bump regions
// ---------------------------------------------------------------------------

/// A run of lines the allocator is bumping through. `cur == limit`
/// means empty.
#[derive(Clone, Copy, Default)]
struct Region {
    cur: usize,
    limit: usize,
    block: u32,
}

impl Region {
    fn is_empty(&self) -> bool {
        self.cur >= self.limit
    }
}

// ---------------------------------------------------------------------------
// Heap
// ---------------------------------------------------------------------------

pub struct ImmixHeap {
    chunks: Vec<Chunk>,
    /// Absolute base address of every block, indexed by block number.
    block_bases: Vec<usize>,
    /// Blocks holding allocations (or handed out as a region).
    in_use: Vec<bool>,
    /// Blocks that hold at least one line-owning allocation; gates the
    /// interior-pointer walk-back.
    has_span: Vec<bool>,
    /// Free blocks whose pages were handed back to the OS and must be
    /// reclaimed before reuse (macOS keeps a reusable/reuse ledger).
    handed_back: Vec<bool>,
    /// (chunk base, first block number) sorted by base, for address
    /// lookups.
    chunk_index: Vec<(usize, u32)>,
    /// Free block numbers; popped from the end.
    free_blocks: Vec<u32>,
    /// One byte per quantum: 0 not a start, 1..=8 small object size in
    /// quanta, `SPAN_OBJECT` for a line-owning allocation.
    objects: Vec<u8>,
    /// Lines owned by the span starting at that line, else 0.
    alloc_sizes: Vec<u32>,
    /// Runs of free lines found by the last sweep: (block, first line,
    /// line count). Consumed from the end.
    recycle_spans: Vec<(u32, u16, u16)>,
    /// Bump region for allocations up to a line.
    small: Region,
    /// Bump region for line-owning allocations; always line-aligned.
    medium: Region,

    total_allocated: usize,
    total_freed: usize,
    bytes_since_gc: usize,
    external_since_gc: usize,
    live_bytes: usize,
    trigger_threshold: usize,
    last_collect: Instant,
    polls: Cell<u32>,
}

impl Default for ImmixHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl ImmixHeap {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            block_bases: Vec::new(),
            in_use: Vec::new(),
            has_span: Vec::new(),
            handed_back: Vec::new(),
            chunk_index: Vec::new(),
            free_blocks: Vec::new(),
            objects: Vec::new(),
            alloc_sizes: Vec::new(),
            recycle_spans: Vec::new(),
            small: Region::default(),
            medium: Region::default(),
            total_allocated: 0,
            total_freed: 0,
            bytes_since_gc: 0,
            external_since_gc: 0,
            live_bytes: 0,
            trigger_threshold: trigger_floor_bytes(),
            last_collect: Instant::now(),
            polls: Cell::new(0),
        }
    }

    // -- Geometry helpers ---------------------------------------------------

    #[inline(always)]
    fn quantum_index(&self, block: u32, addr: usize) -> usize {
        block as usize * QUANTA_PER_BLOCK + (addr - self.block_bases[block as usize]) / QUANTUM
    }

    #[inline(always)]
    fn line_index(&self, block: u32, addr: usize) -> usize {
        block as usize * LINES_PER_BLOCK + (addr - self.block_bases[block as usize]) / LINE_SIZE
    }

    fn heap_bytes(&self) -> usize {
        self.chunks.len() * CHUNK_BYTES
    }

    // -- Growth -------------------------------------------------------------

    fn add_chunk(&mut self) -> bool {
        if self.heap_bytes() + CHUNK_BYTES > heap_max_bytes() {
            return false;
        }
        let Some(chunk) = Chunk::reserve() else {
            return false;
        };
        let first = self.block_bases.len() as u32;
        for i in 0..BLOCKS_PER_CHUNK {
            self.block_bases.push(chunk.base + i * BLOCK_SIZE);
            self.in_use.push(false);
            self.has_span.push(false);
            self.handed_back.push(false);
        }
        self.chunk_index.push((chunk.base, first));
        self.chunk_index.sort_unstable();
        self.objects
            .resize(self.block_bases.len() * QUANTA_PER_BLOCK, 0);
        self.alloc_sizes
            .resize(self.block_bases.len() * LINES_PER_BLOCK, 0);
        // Push high blocks first so the lowest address pops next.
        for i in (0..BLOCKS_PER_CHUNK).rev() {
            self.free_blocks.push(first + i as u32);
        }
        self.chunks.push(chunk);
        true
    }

    fn acquire_free_block(&mut self) -> Option<u32> {
        if self.free_blocks.is_empty() && !self.add_chunk() {
            return None;
        }
        let b = self.free_blocks.pop()?;
        self.in_use[b as usize] = true;
        if self.handed_back[b as usize] {
            self.handed_back[b as usize] = false;
            reclaim_pages(self.block_bases[b as usize], BLOCK_SIZE);
        }
        self.clear_metadata(b, 0, LINES_PER_BLOCK);
        Some(b)
    }

    fn clear_metadata(&mut self, block: u32, first_line: usize, lines: usize) {
        let b = block as usize;
        let q0 = b * QUANTA_PER_BLOCK + first_line * QUANTA_PER_LINE;
        let q1 = q0 + lines * QUANTA_PER_LINE;
        self.objects[q0..q1].fill(0);
        let l0 = b * LINES_PER_BLOCK + first_line;
        self.alloc_sizes[l0..l0 + lines].fill(0);
    }

    // -- Allocation ---------------------------------------------------------

    /// Hand out `size` bytes (16-aligned, at most one line) from the
    /// small region, never straddling a line. Null once the heap is
    /// exhausted.
    #[inline]
    fn alloc_small(&mut self, size: usize) -> *mut u8 {
        loop {
            let mut p = self.small.cur;
            if (p & (LINE_SIZE - 1)) + size > LINE_SIZE {
                p = p.next_multiple_of(LINE_SIZE);
            }
            let np = p + size;
            if np <= self.small.limit {
                self.small.cur = np;
                let b = self.small.block;
                let q = self.quantum_index(b, p);
                self.objects[q] = (size / QUANTUM) as u8;
                return p as *mut u8;
            }
            if !self.refill_small() {
                return std::ptr::null_mut();
            }
        }
    }

    fn refill_small(&mut self) -> bool {
        // Recycled line runs first; a run shorter than a line is
        // impossible by construction, so any span serves.
        if let Some((b, first, n)) = self.recycle_spans.pop() {
            let base = self.block_bases[b as usize] + first as usize * LINE_SIZE;
            let len = n as usize * LINE_SIZE;
            self.clear_metadata(b, first as usize, n as usize);
            self.small = Region {
                cur: base,
                limit: base + len,
                block: b,
            };
            self.bytes_since_gc += len;
            self.total_allocated += len;
            return true;
        }
        match self.acquire_free_block() {
            Some(b) => {
                let base = self.block_bases[b as usize];
                self.small = Region {
                    cur: base,
                    limit: base + BLOCK_SIZE,
                    block: b,
                };
                self.bytes_since_gc += BLOCK_SIZE;
                self.total_allocated += BLOCK_SIZE;
                true
            }
            None => false,
        }
    }

    /// Hand out whole lines for an allocation larger than a line. Null
    /// once the heap is exhausted.
    fn alloc_medium(&mut self, size: usize) -> *mut u8 {
        let lines = size.div_ceil(LINE_SIZE);
        debug_assert!(lines <= LINES_PER_BLOCK);
        loop {
            let p = self.medium.cur.next_multiple_of(LINE_SIZE);
            let np = p + lines * LINE_SIZE;
            if np <= self.medium.limit && !self.medium.is_empty() {
                self.medium.cur = np;
                let b = self.medium.block;
                let q = self.quantum_index(b, p);
                self.objects[q] = SPAN_OBJECT;
                let l = self.line_index(b, p);
                self.alloc_sizes[l] = lines as u32;
                self.has_span[b as usize] = true;
                return p as *mut u8;
            }
            // Prefer a recycled run that fits; otherwise a fresh block.
            if let Some(idx) = self
                .recycle_spans
                .iter()
                .rposition(|&(_, _, n)| n as usize >= lines)
            {
                let (b, first, n) = self.recycle_spans.swap_remove(idx);
                let base = self.block_bases[b as usize] + first as usize * LINE_SIZE;
                let len = n as usize * LINE_SIZE;
                self.clear_metadata(b, first as usize, n as usize);
                self.medium = Region {
                    cur: base,
                    limit: base + len,
                    block: b,
                };
                self.bytes_since_gc += len;
                self.total_allocated += len;
                continue;
            }
            match self.acquire_free_block() {
                Some(b) => {
                    let base = self.block_bases[b as usize];
                    self.medium = Region {
                        cur: base,
                        limit: base + BLOCK_SIZE,
                        block: b,
                    };
                    self.bytes_since_gc += BLOCK_SIZE;
                    self.total_allocated += BLOCK_SIZE;
                }
                None => return std::ptr::null_mut(),
            }
        }
    }

    /// `size` bytes, 16-aligned, contents unspecified; null once the
    /// heap is exhausted.
    #[inline]
    pub fn alloc_raw(&mut self, size: usize) -> *mut u8 {
        let size = size.max(QUANTUM).next_multiple_of(QUANTUM);
        assert!(
            size <= BLOCK_SIZE,
            "Immix: allocation of {size} bytes exceeds a block"
        );
        if size <= SMALL_MAX {
            self.alloc_small(size)
        } else {
            self.alloc_medium(size)
        }
    }

    // -- Enumeration --------------------------------------------------------

    /// Every allocation start, in address order.
    pub fn for_each_allocation<F: FnMut(*mut u8)>(&self, mut f: F) {
        for b in 0..self.block_bases.len() {
            if !self.in_use[b] {
                continue;
            }
            let base = self.block_bases[b];
            let q0 = b * QUANTA_PER_BLOCK;
            let mut q = 0;
            while q < QUANTA_PER_BLOCK {
                let code = self.objects[q0 + q];
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
        if code == SPAN_OBJECT {
            self.alloc_sizes[block * LINES_PER_BLOCK + q / QUANTA_PER_LINE] as usize
                * QUANTA_PER_LINE
        } else {
            code as usize
        }
    }

    // -- Address resolution -------------------------------------------------

    /// Block number holding `addr`, if it is inside the heap.
    #[inline]
    fn block_containing(&self, addr: usize) -> Option<u32> {
        let i = self.chunk_index.partition_point(|&(base, _)| base <= addr);
        if i == 0 {
            return None;
        }
        let (base, first) = self.chunk_index[i - 1];
        if addr >= base + CHUNK_BYTES {
            return None;
        }
        Some(first + ((addr - base) / BLOCK_SIZE) as u32)
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
        if !self.in_use[b as usize] {
            return std::ptr::null_mut();
        }
        let base = self.block_bases[b as usize];
        let off = addr - base;
        let q = off / QUANTUM;
        let line_first_q = q - q % QUANTA_PER_LINE;
        let q0 = b as usize * QUANTA_PER_BLOCK;
        let mut i = q;
        loop {
            let code = self.objects[q0 + i];
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
        if !self.has_span[b as usize] {
            return std::ptr::null_mut();
        }
        let line = off / LINE_SIZE;
        let l0 = b as usize * LINES_PER_BLOCK;
        let mut l = line;
        loop {
            let n = self.alloc_sizes[l0 + l] as usize;
            if n != 0 {
                let start_q = l * QUANTA_PER_LINE;
                return if self.objects[q0 + start_q] == SPAN_OBJECT && line < l + n {
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
    pub unsafe fn mark(&self, ptr: *mut u8) -> bool {
        let header = ptr as *mut ObjHeader;
        if (*header).gc_mark == BLACK {
            return false;
        }
        (*header).gc_mark = BLACK;
        true
    }

    /// # Safety
    /// As [`ImmixHeap::mark`].
    #[inline(always)]
    pub unsafe fn is_marked(&self, ptr: *mut u8) -> bool {
        (*(ptr as *mut ObjHeader)).gc_mark == BLACK
    }

    // -- Trigger ------------------------------------------------------------

    pub fn track_external(&mut self, bytes: usize) {
        self.external_since_gc += bytes;
    }

    pub fn should_collect(&self) -> bool {
        if stress_enabled() {
            return self.bytes_since_gc > 0;
        }
        if self.bytes_since_gc + self.external_since_gc >= self.trigger_threshold {
            return true;
        }
        // The clock is read rarely: a long-idle heap with some garbage
        // gets collected on a heartbeat rather than never.
        let n = self.polls.get().wrapping_add(1);
        self.polls.set(n);
        n & 1023 == 0 && self.bytes_since_gc > 0 && self.last_collect.elapsed() >= HEARTBEAT
    }

    pub fn stats(&self) -> RtStats {
        RtStats {
            heap_bytes: self.heap_bytes(),
            live_bytes: self.live_bytes,
            allocated_bytes: self.total_allocated,
            freed_bytes: self.total_freed,
        }
    }

    // -- Sweep --------------------------------------------------------------

    /// Close a cycle: `drop` every unmarked allocation, recycle its
    /// lines, reset the trigger, and return the live bytes.
    pub fn collect_end<F: FnMut(*mut u8)>(&mut self, drop: F) -> usize {
        // The current regions are swept like any other lines; a fresh
        // region is taken on the next allocation.
        self.small = Region::default();
        self.medium = Region::default();

        let quiet = self.last_collect.elapsed() >= HEARTBEAT;
        let live = self.sweep(drop);
        if quiet {
            self.hand_back_free_blocks();
        }
        self.live_bytes = live;
        let floor = trigger_floor_bytes();
        let ceiling = TRIGGER_CEILING.max(live).max(floor);
        self.trigger_threshold = (live.saturating_mul(growth_factor())).clamp(floor, ceiling);
        self.bytes_since_gc = 0;
        self.external_since_gc = 0;
        self.last_collect = Instant::now();
        live
    }

    /// Free dead objects, rebuild the free-line runs, and return the
    /// live bytes.
    fn sweep<F: FnMut(*mut u8)>(&mut self, mut drop: F) -> usize {
        self.recycle_spans.clear();
        let mut live_bytes = 0usize;
        let mut freed_bytes = 0usize;
        let nblocks = self.block_bases.len();
        for b in 0..nblocks {
            if !self.in_use[b] {
                continue;
            }
            let base = self.block_bases[b];
            let q0 = b * QUANTA_PER_BLOCK;
            let mut line_live = [0u64; LINE_WORDS];
            let mut any_live = false;
            let mut q = 0;
            while q < QUANTA_PER_BLOCK {
                let code = self.objects[q0 + q];
                if code == 0 {
                    q += 1;
                    continue;
                }
                let quanta = self.alloc_quanta(b, q, code);
                let header = (base + q * QUANTUM) as *mut ObjHeader;
                let marked = unsafe { (*header).gc_mark == BLACK };
                if marked {
                    unsafe { (*header).gc_mark = WHITE };
                    any_live = true;
                    live_bytes += quanta * QUANTUM;
                    let first_line = q / QUANTA_PER_LINE;
                    let last_line = (q + quanta - 1) / QUANTA_PER_LINE;
                    for l in first_line..=last_line {
                        line_live[l / 64] |= 1u64 << (l % 64);
                    }
                } else {
                    drop(header as *mut u8);
                    self.objects[q0 + q] = 0;
                    if code == SPAN_OBJECT {
                        self.alloc_sizes[b * LINES_PER_BLOCK + q / QUANTA_PER_LINE] = 0;
                    }
                    freed_bytes += quanta * QUANTUM;
                }
                q += quanta;
            }
            if !any_live {
                self.in_use[b] = false;
                self.has_span[b] = false;
                self.free_blocks.push(b as u32);
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
                self.recycle_spans
                    .push((b as u32, start as u16, (l - start) as u16));
            }
        }
        self.total_freed += freed_bytes;
        live_bytes
    }
}

/// Free blocks kept resident so a burst after an idle period does not
/// pay page faults immediately.
const RESIDENT_FLOAT: usize = 16;

impl ImmixHeap {
    /// Return the pages of free blocks beyond the resident float to the
    /// OS. Only called after a quiet collection: a churning workload
    /// would otherwise pay a madvise pair per block per cycle.
    fn hand_back_free_blocks(&mut self) {
        if self.free_blocks.len() <= RESIDENT_FLOAT {
            return;
        }
        self.free_blocks.sort_unstable_by(|a, b| b.cmp(a));
        // The float stays at the end of the list (lowest addresses),
        // which is what pops next.
        let n = self.free_blocks.len() - RESIDENT_FLOAT;
        let mut run_start: Option<(usize, usize)> = None;
        for &b in self.free_blocks[..n].iter().rev() {
            let b = b as usize;
            if self.handed_back[b] {
                continue;
            }
            self.handed_back[b] = true;
            let base = self.block_bases[b];
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
    }

    #[test]
    fn sweep_frees_unmarked_and_reuses_their_lines() {
        let mut heap = ImmixHeap::new();
        let keep = alloc(&mut heap, 40);
        for _ in 0..50_000 {
            alloc(&mut heap, 40);
        }
        assert!(unsafe { heap.mark(keep) });
        assert!(!unsafe { heap.mark(keep) });
        assert!(unsafe { heap.is_marked(keep) });
        let (live, dropped) = sweep_unmarked(&mut heap);
        assert_eq!(dropped, 50_000);
        assert_eq!(live, 48);
        assert!(!unsafe { heap.is_marked(keep) });
        let allocated = heap.total_allocated;
        for _ in 0..50_000 {
            alloc(&mut heap, 40);
        }
        assert!(unsafe { heap.mark(keep) });
        let (_, dropped) = sweep_unmarked(&mut heap);
        assert_eq!(dropped, 50_000);
        assert!(
            heap.total_allocated < allocated * 2,
            "freed lines were not reused"
        );
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
        assert!(heap.free_blocks.len() > RESIDENT_FLOAT);
        heap.hand_back_free_blocks();
        let handed: usize = heap.handed_back.iter().filter(|&&h| h).count();
        assert_eq!(handed, heap.free_blocks.len() - RESIDENT_FLOAT);
        // Allocate through every handed-back block and verify the
        // memory is usable and the ledger clears.
        for _ in 0..200_000 {
            let p = alloc(&mut heap, 40);
            unsafe { p.add(32).write(0x5a) };
            assert_eq!(unsafe { p.add(32).read() }, 0x5a);
        }
        let still: usize = heap
            .handed_back
            .iter()
            .zip(heap.in_use.iter())
            .filter(|(&h, &u)| h && u)
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
