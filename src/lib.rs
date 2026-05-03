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

// No `#[global_allocator]` override here: cdylib plugins enable
// `wren_lift/host` for `VM` struct layout compatibility, so any
// allocator pulled in via `host` would be statically linked into
// every plugin as well. Each cdylib + the binary would then hold
// independent heaps, and cross-FFI buffer growth (e.g. plugin
// code allocating an `ObjList::elements` that the host's GC
// later frees) would corrupt on dealloc. Linux production keeps
// `MALLOC_ARENA_MAX=2` + a 768mb VM budget to bound glibc's
// per-thread arena commit instead. A future allocator swap needs
// to avoid the dual-static problem — either libc symbol
// interposition or a shared-library allocator that the binary
// and plugins resolve through the dynamic linker.

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
