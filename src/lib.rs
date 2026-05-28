// The wasm build cfg-gates large chunks of the runtime to host-only
// (the JIT pipeline, cranelift backend, libloading plugin loader,
// hatch packaging, the threaded interpreter). What's left is the
// portable parser + BC interpreter + core types — but the gated
// items still parse, and rustc's dead-code / unused-imports lints
// don't follow `cfg(feature = "host")` boundaries through every
// caller chain. The result is a long tail of "function never used",
// "import never used", and "variable assigned but unread" lints
// that all trace back to `#[cfg(feature = "host")]` somewhere
// upstream of the use site. Silencing them globally on the wasm
// arm keeps `cargo clippy --target wasm32-unknown-unknown -- -D
// warnings` green without scattering per-item allow attributes
// across two dozen files.
#![cfg_attr(
    not(feature = "host"),
    allow(dead_code, unused_imports, unused_variables, unreachable_patterns)
)]

// Global allocator override is opt-in via the `wlift_alloc`
// feature, which the wlift / hatch binaries enable but plugins
// (which enable `host` for struct layout compatibility) do not.
// Plugins keep the system allocator at compile time and route
// cross-FFI buffer ownership through the `wlift_malloc` C
// symbols exported from the binary — single instance regardless
// of how each cdylib was linked. See
// `crates/wlift_alloc/src/lib.rs` for the design notes.
// Global-allocator selection. jemalloc wins if both `jemalloc` and
// `wlift_alloc` are enabled — battle-tested production allocator
// with eager page release. `wlift_alloc` is the in-tree fallback;
// the system allocator is the implicit default when neither flag
// is on.
#[cfg(all(feature = "jemalloc", not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "wlift_alloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: wlift_alloc::Wlift = wlift_alloc::Wlift;

// Explicit System global allocator on Unix when no other allocator
// is selected. Without an explicit `#[global_allocator]`, Rust's
// default kicks in — and on Linux x86_64/aarch64 under heavy
// alloc/free churn (the krio fiber yield loop) the default path
// retains pages monotonically, blowing through fly.io's 768 MB
// cap in tens of seconds. Declaring `System` explicitly here is
// functionally identical to the default but routes every alloc
// through libc malloc directly, and empirically holds RssAnon
// flat at ~700 KB over the same workload. Reasons aren't fully
// understood (possibly a Rust-stdlib thread-local arena path
// that this declaration bypasses); the fix is reliable and zero
// overhead.
#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "jemalloc"),
    not(feature = "wlift_alloc"),
    not(feature = "alloc_trace"),
))]
#[global_allocator]
static GLOBAL: std::alloc::System = std::alloc::System;

// Diagnostic counting allocator, opt-in via the `alloc_trace` feature.
// Wraps the system allocator with atomic counters; `alloc_trace_snapshot()`
// returns (allocs, frees, live_bytes). Used by the yield-leak repro to
// attribute allocations to specific runtime call paths.
#[cfg(all(
    feature = "alloc_trace",
    not(target_arch = "wasm32"),
    not(feature = "jemalloc"),
    not(feature = "wlift_alloc"),
))]
mod alloc_trace_impl {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    pub static ALLOCS: AtomicUsize = AtomicUsize::new(0);
    pub static FREES: AtomicUsize = AtomicUsize::new(0);
    pub static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
    pub struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            LIVE_BYTES.fetch_add(l.size(), Ordering::Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            FREES.fetch_add(1, Ordering::Relaxed);
            LIVE_BYTES.fetch_sub(l.size(), Ordering::Relaxed);
            unsafe { System.dealloc(p, l) }
        }
    }
}
#[cfg(all(
    feature = "alloc_trace",
    not(target_arch = "wasm32"),
    not(feature = "jemalloc"),
    not(feature = "wlift_alloc"),
))]
#[global_allocator]
static GLOBAL: alloc_trace_impl::Counting = alloc_trace_impl::Counting;

#[cfg(all(
    feature = "alloc_trace",
    not(target_arch = "wasm32"),
    not(feature = "jemalloc"),
    not(feature = "wlift_alloc"),
))]
pub fn alloc_trace_snapshot() -> (usize, usize, usize) {
    use std::sync::atomic::Ordering;
    (
        alloc_trace_impl::ALLOCS.load(Ordering::Relaxed),
        alloc_trace_impl::FREES.load(Ordering::Relaxed),
        alloc_trace_impl::LIVE_BYTES.load(Ordering::Relaxed),
    )
}

pub mod ast;
pub mod capi;
pub mod codegen;
pub mod plugin_abi;
pub mod diagnostics;
pub mod docs;
pub mod intern;
pub mod mir;
pub mod parse;
pub mod portable_time;
pub mod runtime;
pub mod sema;
pub mod serialize;

// Hatch packaging + registry + service code reaches for `tempfile`,
// `zstd`, `ureq` and other host-only deps. Wasm builds skip the
// whole layer; the wasm interpreter consumes wlbc bytes directly via
// `vm.compile_source_to_blob` / `vm.interpret_hatch` (the runtime
// itself is portable).
// `hatch` itself is target-agnostic — the load/emit/manifest
// types compile everywhere. Build paths
// (`build_from_source_tree*`, the registry/runner crates) stay
// host-gated since they pull in `std::fs`, `git`, `curl`, etc.
pub mod hatch;
#[cfg(feature = "host")]
pub mod hatch_registry;
#[cfg(feature = "host")]
pub mod hatch_runner;
#[cfg(feature = "host")]
pub mod hatch_service;
