//! Single source of truth for the runtime's prelude name list.
//!
//! Sema's [`resolve_with_prelude`] (and downstream callers like
//! the LSP's `publish_diagnostics` and the playground's
//! `lint_wren`) pre-intern these symbols so a bare `System.print`
//! doesn't surface as "undefined" before the host has had a
//! chance to install the matching globals. Every name here is
//! either:
//!
//! * a Wren core class registered by `runtime::core::initialize`
//!   (`Object`, `Class`, `Bool`, `Num`, …), or
//! * a wasm-runtime browser bridge installed by the
//!   `wlift_prelude` import the playground prepends to user
//!   source (`Future`, `Browser`, …).
//!
//! Every consumer should import this constant verbatim — keeping
//! the list in one file is the only way to prevent the LSP and
//! the playground from drifting apart, which previously caused
//! browser-bridge names to surface as "undefined" diagnostics
//! only inside the desktop editor.
//!
//! [`resolve_with_prelude`]: super::resolve::resolve_with_prelude

pub const CORE_PRELUDE_NAMES: &[&str] = &[
    // Core classes (vm.rs::core::initialize).
    "Object",
    "Class",
    "Bool",
    "Num",
    "String",
    "List",
    "Map",
    "Range",
    "Null",
    "Fn",
    "Fiber",
    "System",
    "Sequence",
    "ByteArray",
    "Int32Array",
    "Float32Array",
    "Float64Array",
    "Simd",
    "Simd4f",
    "Simd4i",
];

pub const PRELUDE_NAMES: &[&str] = &[
    // Core classes (vm.rs::core::initialize).
    "Object",
    "Class",
    "Bool",
    "Num",
    "String",
    "List",
    "Map",
    "Range",
    "Null",
    "Fn",
    "Fiber",
    "System",
    "Sequence",
    "ByteArray",
    "Int32Array",
    "Float32Array",
    "Float64Array",
    "Simd",
    "Simd4f",
    "Simd4i",
    // Browser bridges injected by the wasm runtime's
    // `PRELUDE_IMPORT`.
    "Future",
    "Browser",
    "WebSocket",
    "Dom",
    "LocalStorage",
    "SessionStorage",
];
