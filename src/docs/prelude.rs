//! Built-in / prelude documentation.
//!
//! The runtime's core classes (`Num`, `String`, `List`, `Map`,
//! `Fiber`, `System`, `Fn`) live in Rust, so the doc collector
//! can't see them by walking source. We side-step by maintaining
//! `///`-doc-only Wren stubs at `src/runtime/prelude/*.wren`,
//! parsing them through the same pipeline, and exposing the
//! resulting `ModuleDoc`s as a static lookup.
//!
//! Adding a new built-in class:
//!
//! 1. Drop a `<name>.wren` next to the existing stubs. Use
//!    `///` on every method whose docs you want to surface; the
//!    bodies don't matter — the collector reads comments,
//!    not bodies.
//! 2. Reference the file in the `STUBS` table below.
//! 3. The hover / completion paths already consult
//!    `prelude_docs()` — no additional wiring needed.

use std::sync::OnceLock;

use super::{collect_module, ModuleDoc};

/// Prelude class stubs baked into the binary at compile time.
/// `(module_name, source_text)` tuples.
const STUBS: &[(&str, &str)] = &[
    // Core (always-on prelude). Order is intentional: classes
    // earlier in the list win when an unrestricted member
    // lookup matches more than one — putting `Object` first
    // means generic methods like `toString` / `==` resolve to
    // their root-class definitions.
    ("Object",       include_str!("../runtime/prelude/object.wren")),
    ("Class",        include_str!("../runtime/prelude/class.wren")),
    ("Bool",         include_str!("../runtime/prelude/bool.wren")),
    ("Null",         include_str!("../runtime/prelude/null.wren")),
    ("System",       include_str!("../runtime/prelude/system.wren")),
    ("Num",          include_str!("../runtime/prelude/num.wren")),
    ("String",       include_str!("../runtime/prelude/string.wren")),
    ("Range",        include_str!("../runtime/prelude/range.wren")),
    ("Sequence",     include_str!("../runtime/prelude/sequence.wren")),
    ("List",         include_str!("../runtime/prelude/list.wren")),
    ("Map",          include_str!("../runtime/prelude/map.wren")),
    ("Fiber",        include_str!("../runtime/prelude/fiber.wren")),
    ("Fn",           include_str!("../runtime/prelude/fn.wren")),
    ("TypedArrays",  include_str!("../runtime/prelude/typed_arrays.wren")),
    // Wasm-runtime prelude — the BROWSER_PRELUDE classes the
    // runtime auto-imports into every wasm `run()` invocation.
    // On native these aren't present; surfacing their docs in
    // the host LSP is harmless (the user won't reach them at
    // runtime there).
    ("Browser",      include_str!("../runtime/prelude/browser.wren")),
    ("Dom",          include_str!("../runtime/prelude/dom.wren")),
    ("Future",       include_str!("../runtime/prelude/future.wren")),
    ("WebSocket",    include_str!("../runtime/prelude/websocket.wren")),
    ("Storage",      include_str!("../runtime/prelude/storage.wren")),
];

/// Lazily-parsed prelude doc model. First access pays the parse
/// cost once; subsequent calls return the cached `Vec`.
pub fn prelude_docs() -> &'static [ModuleDoc] {
    static CACHE: OnceLock<Vec<ModuleDoc>> = OnceLock::new();
    CACHE.get_or_init(|| {
        STUBS
            .iter()
            .map(|(name, src)| {
                let pr = crate::parse::parser::parse(src);
                collect_module(*name, src, &pr.module, &pr.docs, &pr.interner)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_stub_cleanly() {
        let docs = prelude_docs();
        assert!(!docs.is_empty(), "no prelude stubs registered");
        for module in docs {
            assert!(
                !module.classes.is_empty(),
                "prelude stub `{}` has no classes — author error",
                module.name
            );
        }
    }

    #[test]
    fn system_has_print() {
        let docs = prelude_docs();
        let system = docs.iter().find(|m| m.name == "System").unwrap();
        let class = system.classes.iter().find(|c| c.name == "System").unwrap();
        let print_doc = class
            .members
            .iter()
            .find(|m| m.name == "print")
            .expect("System.print stub present");
        assert!(
            print_doc.doc.contains("Print"),
            "System.print docs missing"
        );
    }
}
