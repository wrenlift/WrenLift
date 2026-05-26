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

pub mod ast;
pub mod capi;
pub mod codegen;
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
