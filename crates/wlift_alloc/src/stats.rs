//! Opt-in allocator statistics behind the `stats` feature.
//! Tracks total bytes allocated / freed / peak live. Off by
//! default to keep the fast path branch-free.

use std::sync::atomic::{AtomicUsize, Ordering};

pub static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
pub static FREE_BYTES: AtomicUsize = AtomicUsize::new(0);
pub static LIVE_PEAK: AtomicUsize = AtomicUsize::new(0);

pub fn record_alloc(n: usize) {
    let prev = ALLOC_BYTES.fetch_add(n, Ordering::Relaxed);
    let live = prev + n - FREE_BYTES.load(Ordering::Relaxed);
    LIVE_PEAK.fetch_max(live, Ordering::Relaxed);
}

pub fn record_free(n: usize) {
    FREE_BYTES.fetch_add(n, Ordering::Relaxed);
}
