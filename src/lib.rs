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

// `mimalloc` as a `#[global_allocator]` was declared here in
// commit ac97d2e to address Linux x86_64 OOM cycles caused by
// glibc's per-thread arena pathology (~399mb anon-rss vs
// ~150-180mb on darwin's libmalloc). Reverted because
// cdylib plugins under `plugins/wlift_*` enable
// `wren_lift/host` for `VM` struct layout compatibility, and
// `dep:mimalloc` rode along through `host` — every plugin's
// link unit then carried its own static `mimalloc` instance.
// Each cdylib + the binary held independent heaps, and any
// cross-FFI buffer growth (e.g. `(*result_ptr).add(map)` from
// plugin code allocating an `ObjList::elements` buffer that
// the host's GC sweep later frees) corrupted on the
// dealloc → silent SIGSEGV / abort under load. Production
// keeps `MALLOC_ARENA_MAX=2` + a 768mb VM budget on Fly to
// bound the per-arena commit instead. A future allocator
// swap needs to avoid the dual-static problem — either the
// `mimalloc/override` feature (libc symbol interposition,
// which broke macOS dev builds when last attempted) or
// shipping mimalloc as a shared library that the binary
// dynamically links and plugins resolve through the dynamic
// linker.

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
