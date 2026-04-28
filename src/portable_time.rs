//! Cross-target alias for the few `std::time` types the runtime
//! actually uses.
//!
//! `std::time::Instant::now()` and `SystemTime::now()` both compile
//! to `wasm32-unknown-unknown` but **panic at first use** there —
//! the toolchain has no monotonic / wall-clock source by default.
//! `web-time` swaps in `performance.now()` / `Date.now()` so the
//! same calls work in browsers and `wasm-bindgen`-based hosts.
//!
//! The rule for runtime code: call `crate::portable_time::Instant`
//! / `SystemTime` / `UNIX_EPOCH` instead of `std::time::*`. Native
//! builds get the std types; wasm builds get the `web-time` types
//! that don't panic.
//!
//! `Duration` is unchanged across both — `std::time::Duration` is
//! a pure value type with no platform syscalls — so we just
//! re-export it from `std` for ergonomics.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "wasm32")]
pub use web_time::{Instant, SystemTime, UNIX_EPOCH};

pub use std::time::Duration;
