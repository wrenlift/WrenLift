//! Immix-geometry, non-moving mark-sweep collector.
//!
//! The heap is a set of block-aligned chunks carved into 32 KiB blocks
//! of 128-byte lines. Objects up to one line bump-allocate through a
//! per-VM region and never straddle a line; larger objects take whole
//! lines and never share their last line. Object starts and span sizes
//! live in side tables keyed by block, so any address inside an
//! allocation resolves to its start. Marking is object-granular through
//! the header's mark byte and traces the heap precisely; sweep derives
//! line occupancy from the surviving starts, runs the destructor of
//! every dead object, and hands runs of free lines back as bump regions.
//! Nothing moves, so there are no forwarding pointers, no write barrier,
//! and roots are read but never written.

use super::gc::{drop_in_place_by_type, mark_value, process_gray_stack, GcStats};
use super::gc_trait::GcAllocator;
use super::object::*;
use super::value::Value;
use crate::intern::SymbolId;
use crate::portable_time::Instant;

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;

pub const BLOCK_SIZE: usize = 32 * 1024;
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
    *CACHED.get_or_init(|| env_usize("WLIFT_GC_GROWTH").unwrap_or(DEFAULT_GROWTH).max(1))
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
// Collector
// ---------------------------------------------------------------------------

pub struct ImmixGc {
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

    intern_table: HashMap<u64, Vec<*mut ObjString>>,
    pub stats: GcStats,
    bytes_since_gc: usize,
    external_since_gc: usize,
    live_bytes: usize,
    trigger_threshold: usize,
    last_collect: Instant,
    polls: Cell<u32>,
    object_count: usize,
    /// A sweep freed a class, closure, function or module, so
    /// address-keyed caches must be dropped.
    freed_code_objects: bool,
}

impl Default for ImmixGc {
    fn default() -> Self {
        Self::new()
    }
}

impl ImmixGc {
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
            intern_table: HashMap::new(),
            stats: GcStats::default(),
            bytes_since_gc: 0,
            external_since_gc: 0,
            live_bytes: 0,
            trigger_threshold: trigger_floor_bytes(),
            last_collect: Instant::now(),
            polls: Cell::new(0),
            object_count: 0,
            freed_code_objects: false,
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

    fn exhausted(&self) -> ! {
        panic!(
            "wren_lift: Immix heap exhausted at {} MiB (live {} MiB); raise WLIFT_GC_HEAP_MB",
            self.heap_bytes() / (1024 * 1024),
            self.live_bytes / (1024 * 1024)
        );
    }

    // -- Allocation ---------------------------------------------------------

    /// Hand out `size` bytes (16-aligned, at most one line) from the
    /// small region, never straddling a line.
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
            self.refill_small();
        }
    }

    fn refill_small(&mut self) {
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
            self.stats.total_allocated += len;
            return;
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
                self.stats.total_allocated += BLOCK_SIZE;
            }
            None => self.exhausted(),
        }
    }

    /// Hand out whole lines for an allocation larger than a line.
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
                self.stats.total_allocated += len;
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
                    self.stats.total_allocated += BLOCK_SIZE;
                }
                None => self.exhausted(),
            }
        }
    }

    #[inline]
    fn alloc_raw(&mut self, size: usize) -> *mut u8 {
        let size = size.max(QUANTUM).next_multiple_of(QUANTUM);
        assert!(
            size <= BLOCK_SIZE,
            "Immix: allocation of {size} bytes exceeds a block"
        );
        self.stats.objects_allocated += 1;
        self.object_count += 1;
        if self.object_count > self.stats.peak_objects {
            self.stats.peak_objects = self.object_count;
        }
        if size <= SMALL_MAX {
            self.alloc_small(size)
        } else {
            self.alloc_medium(size)
        }
    }

    #[inline]
    fn alloc<T>(&mut self, obj: T) -> *mut T {
        let p = self.alloc_raw(std::mem::size_of::<T>()) as *mut T;
        unsafe { p.write(obj) };
        p
    }

    /// Iterate every currently-allocated `ObjFiber`. See
    /// `GcImpl::for_each_fiber` for the contract.
    pub fn for_each_fiber<F: FnMut(*mut ObjFiber)>(&self, mut f: F) {
        self.for_each_object(|h| unsafe {
            if (*h).obj_type == ObjType::Fiber {
                f(h as *mut ObjFiber);
            }
        });
    }

    fn for_each_object<F: FnMut(*mut ObjHeader)>(&self, mut f: F) {
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
                f((base + q * QUANTUM) as *mut ObjHeader);
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

    pub fn track_external(&mut self, bytes: usize) {
        self.external_since_gc += bytes;
    }

    /// True once since the last call if a sweep freed an object that
    /// inline caches or the method cache may point at.
    pub fn take_freed_code_objects(&mut self) -> bool {
        std::mem::replace(&mut self.freed_code_objects, false)
    }

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

    /// Start of the allocation containing `addr`, if any. Walks back
    /// within the line for a small object, then across lines for a
    /// span when the block holds one.
    pub fn containing_allocation(&self, addr: usize) -> Option<*mut ObjHeader> {
        let b = self.block_containing(addr)?;
        if !self.in_use[b as usize] {
            return None;
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
                    Some((base + i * QUANTUM) as *mut ObjHeader)
                } else {
                    None
                };
            }
            if i == line_first_q {
                break;
            }
            i -= 1;
        }
        if !self.has_span[b as usize] {
            return None;
        }
        let line = off / LINE_SIZE;
        let l0 = b as usize * LINES_PER_BLOCK;
        let mut l = line;
        loop {
            let n = self.alloc_sizes[l0 + l] as usize;
            if n != 0 {
                let start_q = l * QUANTA_PER_LINE;
                return if self.objects[q0 + start_q] == SPAN_OBJECT && line < l + n {
                    Some((base + start_q * QUANTUM) as *mut ObjHeader)
                } else {
                    None
                };
            }
            if l == 0 {
                return None;
            }
            l -= 1;
        }
    }

    /// Mark everything a word range might point at: raw addresses and
    /// NaN-boxed object payloads that land inside an allocation.
    fn scan_range_conservative(
        &self,
        lo: usize,
        hi: usize,
        gray_stack: &mut Vec<*mut ObjHeader>,
    ) {
        let word = std::mem::size_of::<usize>();
        let mut p = lo.next_multiple_of(word);
        while p + word <= hi {
            let w = unsafe { std::ptr::read_volatile(p as *const usize) };
            let mut candidate = self.containing_allocation(w);
            if candidate.is_none() && word == 8 {
                let v = Value::from_bits(w as u64);
                if v.is_object() {
                    if let Some(obj) = v.as_object() {
                        candidate = self.containing_allocation(obj as usize);
                    }
                }
            }
            if let Some(h) = candidate {
                mark_value(Value::object(h as *mut u8), gray_stack);
            }
            p += word;
        }
    }

    /// Collect with precise `roots` plus conservative word scans of
    /// `ranges` (native stack windows, register spills).
    pub fn collect_with_ranges(&mut self, roots: &[Value], ranges: &[(usize, usize)]) {
        let start = Instant::now();
        let mut gray_stack: Vec<*mut ObjHeader> = Vec::with_capacity(1024);
        for &root in roots {
            mark_value(root, &mut gray_stack);
        }
        for &(lo, hi) in ranges {
            if lo < hi {
                self.scan_range_conservative(lo, hi, &mut gray_stack);
            }
        }
        process_gray_stack(&mut gray_stack);
        self.finish_collection(start);
    }

    fn finish_collection(&mut self, start: Instant) {
        // The current regions are swept like any other lines; a fresh
        // region is taken on the next allocation.
        self.small = Region::default();
        self.medium = Region::default();

        let quiet = self.last_collect.elapsed() >= HEARTBEAT;
        let live = self.sweep();
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
        self.stats.major_collections += 1;
        self.stats.gc_time_ns += start.elapsed().as_nanos() as u64;
    }

    fn unlink_intern(&mut self, header: *mut ObjHeader) {
        unsafe {
            if (*header).obj_type != ObjType::String {
                return;
            }
            let s = &*(header as *mut ObjString);
            if let Some(ptrs) = self.intern_table.get_mut(&s.hash) {
                ptrs.retain(|&p| p != header as *mut ObjString);
                if ptrs.is_empty() {
                    self.intern_table.remove(&s.hash);
                }
            }
        }
    }

    // -- Sweep --------------------------------------------------------------

    /// Free dead objects, rebuild the free-line runs, and return the
    /// live bytes.
    fn sweep(&mut self) -> usize {
        self.recycle_spans.clear();
        let mut live_bytes = 0usize;
        let mut freed_bytes = 0usize;
        let mut freed_count = 0usize;
        let mut live_count = 0usize;
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
                    live_count += 1;
                    let first_line = q / QUANTA_PER_LINE;
                    let last_line = (q + quanta - 1) / QUANTA_PER_LINE;
                    for l in first_line..=last_line {
                        line_live[l / 64] |= 1u64 << (l % 64);
                    }
                } else {
                    if matches!(
                        unsafe { (*header).obj_type },
                        ObjType::Class | ObjType::Closure | ObjType::Fn | ObjType::Module
                    ) {
                        self.freed_code_objects = true;
                    }
                    self.unlink_intern(header);
                    unsafe { drop_in_place_by_type(header) };
                    self.objects[q0 + q] = 0;
                    if code == SPAN_OBJECT {
                        self.alloc_sizes[b * LINES_PER_BLOCK + q / QUANTA_PER_LINE] = 0;
                    }
                    freed_bytes += quanta * QUANTUM;
                    freed_count += 1;
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
        self.stats.objects_freed += freed_count;
        self.stats.total_freed += freed_bytes;
        self.object_count = live_count;
        live_bytes
    }
}

/// Free blocks kept resident so a burst after an idle period does not
/// pay page faults immediately.
const RESIDENT_FLOAT: usize = 16;

impl ImmixGc {
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

impl Drop for ImmixGc {
    fn drop(&mut self) {
        let mut all = Vec::new();
        self.for_each_object(|h| all.push(h));
        for h in all {
            unsafe { drop_in_place_by_type(h) };
        }
        // Chunks unmap on their own drop.
    }
}

// ---------------------------------------------------------------------------
// GcAllocator implementation
// ---------------------------------------------------------------------------

impl GcAllocator for ImmixGc {
    fn alloc_string(&mut self, s: String) -> *mut ObjString {
        self.alloc(ObjString::new(s))
    }
    fn alloc_list(&mut self) -> *mut ObjList {
        self.alloc(ObjList::new())
    }
    fn alloc_map(&mut self) -> *mut ObjMap {
        self.alloc(ObjMap::new())
    }
    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> *mut ObjRange {
        self.alloc(ObjRange::new(from, to, inclusive))
    }
    fn alloc_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> *mut ObjTypedArray {
        self.alloc(ObjTypedArray::new(count, kind))
    }
    fn alloc_simd(&mut self, kind: SimdKind, lanes: [u32; 4]) -> *mut ObjSimd {
        self.alloc(ObjSimd::new(kind, lanes))
    }
    fn alloc_fn(
        &mut self,
        name: SymbolId,
        arity: u8,
        upvalue_count: u16,
        fn_id: u32,
    ) -> *mut ObjFn {
        self.alloc(ObjFn::new(name, arity, upvalue_count, fn_id))
    }
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn alloc_closure(&mut self, function: *mut ObjFn) -> *mut ObjClosure {
        let uv_count = if function.is_null() {
            0
        } else {
            unsafe { (*function).upvalue_count as usize }
        };
        self.alloc(ObjClosure::new(function, uv_count))
    }
    fn alloc_upvalue(&mut self, location: *mut Value) -> *mut ObjUpvalue {
        self.alloc(ObjUpvalue::new(location))
    }
    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        self.alloc(ObjFiber::new())
    }
    fn alloc_class(&mut self, name: SymbolId, superclass: *mut ObjClass) -> *mut ObjClass {
        self.alloc(ObjClass::new(name, superclass))
    }
    /// Instance header and field array are one allocation; the fields
    /// start right after the header and are not freed on drop.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn alloc_instance(&mut self, class: *mut ObjClass) -> *mut ObjInstance {
        let num_fields = if class.is_null() {
            0
        } else {
            unsafe { (*class).num_fields as usize }
        };
        let header_size = std::mem::size_of::<ObjInstance>();
        let total = header_size + num_fields * std::mem::size_of::<Value>();
        if total > BLOCK_SIZE {
            return self.alloc(ObjInstance::new(class));
        }
        let p = self.alloc_raw(total) as *mut ObjInstance;
        let fields = if num_fields > 0 {
            let f = unsafe { (p as *mut u8).add(header_size) as *mut Value };
            for i in 0..num_fields {
                unsafe { f.add(i).write(Value::null()) };
            }
            f
        } else {
            std::ptr::null_mut()
        };
        unsafe {
            p.write(ObjInstance::new_with_fields(class, num_fields as u32, fields));
        }
        p
    }
    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        self.alloc(ObjForeign::new(data))
    }
    fn alloc_module(&mut self, name: SymbolId) -> *mut ObjModule {
        self.alloc(ObjModule::new(name))
    }

    fn intern_string(&mut self, s: String) -> *mut ObjString {
        let hash = fnv1a_hash_bytes(s.as_bytes());
        if let Some(ptrs) = self.intern_table.get(&hash) {
            for &ptr in ptrs {
                if unsafe { (*ptr).value == s } {
                    return ptr;
                }
            }
        }
        let ptr = self.alloc_string(s);
        self.intern_table.entry(hash).or_default().push(ptr);
        ptr
    }

    #[inline(always)]
    fn write_barrier(&mut self, _source: *mut ObjHeader, _value: Value) {}

    fn collect(&mut self, roots: &mut [Value]) {
        let start = Instant::now();
        let mut gray_stack: Vec<*mut ObjHeader> = Vec::with_capacity(1024);
        for &root in roots.iter() {
            mark_value(root, &mut gray_stack);
        }
        process_gray_stack(&mut gray_stack);
        self.finish_collection(start);
    }

    fn should_collect(&self) -> bool {
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

    fn stats(&self) -> &GcStats {
        &self.stats
    }
}

fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots_of(ptrs: &[*mut ObjHeader]) -> Vec<Value> {
        ptrs.iter().map(|&p| Value::object(p as *mut u8)).collect()
    }

    #[test]
    fn small_objects_never_straddle_a_line() {
        let mut gc = ImmixGc::new();
        let mut last: usize = 0;
        for _ in 0..10_000 {
            let p = gc.alloc_range(0.0, 1.0, false) as usize;
            let size = std::mem::size_of::<ObjRange>().next_multiple_of(QUANTUM);
            assert_eq!(p / LINE_SIZE, (p + size - 1) / LINE_SIZE, "straddles a line");
            assert!(p != last);
            last = p;
        }
    }

    #[test]
    fn collect_frees_unreachable_and_keeps_roots() {
        let mut gc = ImmixGc::new();
        let keep = gc.alloc_list() as *mut ObjHeader;
        for _ in 0..50_000 {
            gc.alloc_string("garbage".to_string());
        }
        let before = gc.object_count;
        let mut roots = roots_of(&[keep]);
        gc.collect(&mut roots);
        assert_eq!(gc.object_count, 1);
        assert!(gc.stats.objects_freed >= before - 1);
        assert_eq!(unsafe { (*keep).obj_type }, ObjType::List);
        // Reuse the freed lines.
        for _ in 0..50_000 {
            gc.alloc_string("again".to_string());
        }
        gc.collect(&mut roots);
        assert_eq!(gc.object_count, 1);
    }

    #[test]
    fn spans_own_whole_lines() {
        let mut gc = ImmixGc::new();
        let f = gc.alloc_fiber() as usize;
        assert_eq!(f % LINE_SIZE, 0);
        let s = gc.alloc_string("x".to_string()) as usize;
        let fiber_lines = std::mem::size_of::<ObjFiber>().div_ceil(LINE_SIZE);
        assert!(s >= f + fiber_lines * LINE_SIZE, "small object packed into a span's tail line");
    }

    #[test]
    fn instance_fields_are_inline() {
        let mut gc = ImmixGc::new();
        let class = gc.alloc_class(SymbolId::from_raw(0), std::ptr::null_mut());
        unsafe { (*class).num_fields = 3 };
        let inst = gc.alloc_instance(class);
        unsafe {
            assert!(!(*inst).fields_owned);
            assert_eq!(
                (*inst).fields as usize,
                inst as usize + std::mem::size_of::<ObjInstance>()
            );
            (*inst).set_field(2, Value::num(7.0));
            assert_eq!((*inst).get_field(2).unwrap().as_num(), Some(7.0));
        }
        let mut roots = roots_of(&[class as *mut ObjHeader, inst as *mut ObjHeader]);
        gc.collect(&mut roots);
        assert_eq!(unsafe { (*inst).get_field(2).unwrap().as_num() }, Some(7.0));
    }

    #[test]
    fn interned_strings_survive_and_dead_ones_unlink() {
        let mut gc = ImmixGc::new();
        let a = gc.intern_string("hello".to_string());
        let b = gc.intern_string("hello".to_string());
        assert_eq!(a, b);
        let _dead = gc.intern_string("bye".to_string());
        let mut roots = roots_of(&[a as *mut ObjHeader]);
        gc.collect(&mut roots);
        assert_eq!(gc.intern_string("hello".to_string()), a);
        let bye_hash = fnv1a_hash_bytes(b"bye");
        assert!(
            !gc.intern_table.contains_key(&bye_hash),
            "dead interned string still in the table"
        );
    }

    #[test]
    fn interior_pointers_resolve_to_their_allocation() {
        let mut gc = ImmixGc::new();
        let small = gc.alloc_range(0.0, 1.0, false) as usize;
        let fiber = gc.alloc_fiber() as usize;
        let size = std::mem::size_of::<ObjRange>();
        for off in [0, 8, size - 1] {
            assert_eq!(gc.containing_allocation(small + off).map(|h| h as usize), Some(small));
        }
        let fsize = std::mem::size_of::<ObjFiber>();
        for off in [0, 200, fsize - 1] {
            assert_eq!(gc.containing_allocation(fiber + off).map(|h| h as usize), Some(fiber));
        }
        // A span owns its reserved lines end to end; addresses outside
        // the heap resolve to nothing.
        let span_end = fiber + fsize.div_ceil(LINE_SIZE) * LINE_SIZE;
        assert_eq!(gc.containing_allocation(span_end - 1).map(|h| h as usize), Some(fiber));
        assert_eq!(gc.containing_allocation(0x1000), None);
        assert_eq!(gc.containing_allocation(small.wrapping_sub(1)).map(|h| h as usize), None);
    }

    #[test]
    fn conservative_scan_keeps_boxed_raw_and_interior_words() {
        let mut gc = ImmixGc::new();
        let boxed = gc.alloc_list() as *mut ObjHeader;
        let raw = gc.alloc_map() as *mut ObjHeader;
        let interior = gc.alloc_fiber() as *mut ObjHeader;
        let dead = gc.alloc_string("dead".to_string()) as *mut ObjHeader;
        let words: [usize; 4] = [
            Value::object(boxed as *mut u8).to_bits() as usize,
            raw as usize,
            interior as usize + 100,
            0xdead_beef,
        ];
        let lo = words.as_ptr() as usize;
        let hi = lo + std::mem::size_of_val(&words);
        gc.collect_with_ranges(&[], &[(lo, hi)]);
        let mut live = Vec::new();
        gc.for_each_object(|h| live.push(h));
        assert!(live.contains(&boxed));
        assert!(live.contains(&raw));
        assert!(live.contains(&interior));
        assert!(!live.contains(&dead));
    }

    #[test]
    fn handed_back_blocks_are_reclaimed_before_reuse() {
        let mut gc = ImmixGc::new();
        for _ in 0..200_000 {
            gc.alloc_string("g".to_string());
        }
        let mut roots = Vec::new();
        gc.collect(&mut roots);
        assert!(gc.free_blocks.len() > RESIDENT_FLOAT);
        gc.hand_back_free_blocks();
        let handed: usize = gc.handed_back.iter().filter(|&&h| h).count();
        assert_eq!(handed, gc.free_blocks.len() - RESIDENT_FLOAT);
        // Allocate through every handed-back block and verify the
        // memory is usable and the ledger clears.
        for _ in 0..200_000 {
            let p = gc.alloc_string("again".to_string());
            assert_eq!(unsafe { &(*p).value }, "again");
        }
        let still: usize = gc
            .handed_back
            .iter()
            .zip(gc.in_use.iter())
            .filter(|(&h, &u)| h && u)
            .count();
        assert_eq!(still, 0, "an in-use block is still marked handed back");
    }

    #[test]
    fn for_each_fiber_finds_fibers() {
        let mut gc = ImmixGc::new();
        let f1 = gc.alloc_fiber();
        let _ = gc.alloc_list();
        let f2 = gc.alloc_fiber();
        let mut seen = Vec::new();
        gc.for_each_fiber(|f| seen.push(f));
        assert_eq!(seen, vec![f1, f2]);
    }
}
