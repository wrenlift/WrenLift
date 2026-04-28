pub mod core;
pub mod engine;
pub mod fiber;
pub mod gc;
pub mod gc_arena;
pub mod gc_marksweep;
pub mod gc_trait;
pub mod object;
pub mod object_layout;
pub mod value;
pub mod vm;
pub mod vm_interp;

// `foreign` (libloading + plugin ABI) and `tier` (beadie thread-
// pool tier-up broker) are host-only. The wasm build provides
// drop-in stubs at the same module paths so callers in `vm.rs`,
// `engine.rs`, and friends don't need site-by-site cfg-gating.
#[cfg(feature = "host")]
pub mod foreign;
#[cfg(not(feature = "host"))]
#[path = "foreign_wasm.rs"]
pub mod foreign;

#[cfg(feature = "host")]
pub mod tier;
#[cfg(not(feature = "host"))]
#[path = "tier_wasm.rs"]
pub mod tier;
