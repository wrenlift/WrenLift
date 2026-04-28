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

pub mod ast;
pub mod capi;
pub mod codegen;
pub mod diagnostics;
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
