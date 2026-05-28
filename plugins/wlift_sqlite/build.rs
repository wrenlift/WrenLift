//! Plugin link configuration. The cdylib references the host's
//! `wlift_plugin_*` symbols via `wlift_abi`'s extern blocks; those
//! symbols don't exist in any link-time library — they live in the
//! host process and the dynamic linker resolves them at `dlopen`.
//!
//! Each platform needs a flag telling the linker to leave those
//! references undefined for runtime resolution instead of failing
//! the link.

fn main() {
    // Host build outputs only — when cross-compiling to wasm the
    // plugin is rlib'd into wlift_wasm and the wasm linker resolves
    // statically against wlift_wasm's exports, no flag needed.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
        return;
    }
    if target.contains("apple") {
        // macOS / iOS / etc. Apple ld(64) flag.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    } else if target.contains("windows") {
        // MSVC link.exe + lld-link both need `/FORCE:UNRESOLVED`
        // to permit undefined symbols in a .dll. The runtime
        // dynamic linker on Windows resolves these against the
        // host .exe via the standard import-table mechanism — we
        // emit a manual import library in a follow-up if
        // /FORCE:UNRESOLVED proves too coarse.
        println!("cargo:rustc-cdylib-link-arg=/FORCE:UNRESOLVED");
    } else {
        // Linux / *BSD / SysV-style ld. The default behaviour
        // (`--unresolved-symbols=report-all`) rejects unresolved
        // refs in shared libraries; explicit override needed.
        println!("cargo:rustc-cdylib-link-arg=-Wl,--unresolved-symbols=ignore-in-object-files");
    }
}
