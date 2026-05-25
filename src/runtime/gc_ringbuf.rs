//! Persistent ring buffer of recent GC operations, backed by an
//! `mmap`'d file. The kernel preserves the file's contents across
//! process termination (SIGSEGV / abort / whatever) so we can read
//! the last N operations even when our SIGSEGV handler can't run
//! (macOS converts EXC_BAD_ACCESS to Mach exceptions before POSIX
//! signal delivery, intercepting our POSIX handler).
//!
//! Recording is opt-in via `WLIFT_GC_RINGBUF=<path>`. Pass a file path
//! the runtime will mmap-write into; post-mortem, read it with the
//! `gc_ringbuf::dump_file` helper in tests.

use core::sync::atomic::{AtomicU64, Ordering};

pub const RING_SIZE: usize = 65536;

/// Event kinds. Stable u64 codes — never renumber, only append. The
/// post-mortem dumper at the bottom maps these back to names.
#[repr(u64)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Ev {
    GcBegin = 1,
    GcEnd = 2,
    PromoteBegin = 10,
    PromoteEnd = 11,
    Drop = 12,
    MapDrainBegin = 20,
    MapKeyForward = 21,
    MapKeyInsert = 22,
    MapDrainEnd = 23,
    NurseryReset = 30,
    UpdateObjBegin = 40,
    UpdateObjEnd = 41,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Event {
    pub seq: u64,
    pub kind: u64,
    pub args: [u64; 6],
}

#[repr(C, align(64))]
pub struct Ring {
    pub head: AtomicU64,
    pub events: [Event; RING_SIZE],
}

static mut RING_PTR: *mut Ring = std::ptr::null_mut();
static ENABLED: AtomicU64 = AtomicU64::new(0);

/// Call once at VM init (idempotent). Reads `WLIFT_GC_RINGBUF`. If
/// set to a value other than "0", treats it as a file path, mmap()s a
/// fixed-size region for the ring, and enables recording.
#[cfg(target_family = "unix")]
pub fn init_if_enabled() {
    static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let path = match std::env::var("WLIFT_GC_RINGBUF") {
            Ok(p) if !p.is_empty() && p != "0" => p,
            _ => return,
        };
        unsafe {
            let cpath = std::ffi::CString::new(path.as_str()).unwrap();
            let fd = libc::open(
                cpath.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o644,
            );
            if fd < 0 {
                eprintln!("[gc-ringbuf] open({path}) failed");
                return;
            }
            let size = std::mem::size_of::<Ring>();
            if libc::ftruncate(fd, size as i64) < 0 {
                eprintln!("[gc-ringbuf] ftruncate failed");
                libc::close(fd);
                return;
            }
            let addr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if addr == libc::MAP_FAILED {
                eprintln!("[gc-ringbuf] mmap failed");
                libc::close(fd);
                return;
            }
            // Zero the region (mmap'd file may have stale contents
            // from a prior run).
            libc::memset(addr, 0, size);
            RING_PTR = addr as *mut Ring;
            ENABLED.store(1, Ordering::Relaxed);
            libc::close(fd);
            eprintln!(
                "[gc-ringbuf] enabled, mmap'd at {:p}, file={}, size={} bytes",
                RING_PTR, path, size
            );
        }
    });
}

#[cfg(not(target_family = "unix"))]
pub fn init_if_enabled() {}

#[inline(always)]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed) != 0
}

#[inline]
pub fn record(kind: Ev, args: [u64; 6]) {
    if !enabled() {
        return;
    }
    unsafe {
        if RING_PTR.is_null() {
            return;
        }
        let ring = &*RING_PTR;
        let seq = ring.head.fetch_add(1, Ordering::Relaxed);
        let idx = (seq as usize) % RING_SIZE;
        let slot = (&ring.events as *const _ as *mut Event).add(idx);
        (*slot).seq = seq;
        (*slot).kind = kind as u64;
        (*slot).args = args;
    }
}

fn kind_name(k: u64) -> &'static str {
    match k {
        1 => "GcBegin",
        2 => "GcEnd",
        10 => "PromoteBegin",
        11 => "PromoteEnd",
        12 => "Drop",
        20 => "MapDrainBegin",
        21 => "MapKeyForward",
        22 => "MapKeyInsert",
        23 => "MapDrainEnd",
        30 => "NurseryReset",
        40 => "UpdateObjBegin",
        41 => "UpdateObjEnd",
        _ => "Unknown",
    }
}

/// Post-mortem helper: load the mmap'd file and dump events in
/// chronological order. Sorts by stored seq (not by head counter),
/// so partial writes / unsynced slots don't show stale stragglers.
pub fn dump_file(path: &str) -> std::io::Result<()> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    if buf.len() < std::mem::size_of::<Ring>() {
        return Err(std::io::Error::other("ring file too small"));
    }
    let ring: &Ring = unsafe { &*(buf.as_ptr() as *const Ring) };
    let head = ring.head.load(Ordering::Relaxed);
    let mut events: Vec<Event> = ring
        .events
        .iter()
        .filter(|e| e.kind != 0)
        .copied()
        .collect();
    events.sort_by_key(|e| e.seq);
    println!(
        "# gc-ringbuf dump: head={head}, slots_used={}",
        events.len()
    );
    for e in &events {
        println!(
            "[{:>10}] {:<18} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
            e.seq,
            kind_name(e.kind),
            e.args[0],
            e.args[1],
            e.args[2],
            e.args[3],
            e.args[4],
            e.args[5]
        );
    }
    Ok(())
}
