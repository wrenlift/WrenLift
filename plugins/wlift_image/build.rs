//! Plugin link configuration — see wlift_sqlite/build.rs for the
//! full rationale. The cdylib references the host's `wlift_plugin_*`
//! symbols via `wlift_abi`'s extern blocks; those symbols don't
//! exist at link time. Each platform needs a flag telling the
//! linker to leave the references undefined for runtime resolution.

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
        // rust-lld in `wasm` flavour rejects undefined symbols by
        // default. Plugin cdylibs target wasm32 too (because of
        // `crate-type = ["cdylib", "rlib"]`), and the `wlift_plugin_*`
        // externs from `wlift_abi` only exist in the host (`wren_lift` /
        // `wlift_wasm`) — when the plugin is statically linked into
        // `wlift_wasm` the per-plugin standalone cdylib output is
        // unused anyway, but lld still wants to produce it. Accept
        // them as wasm imports and let the host satisfy them at the
        // wlift_wasm link step.
        println!("cargo:rustc-cdylib-link-arg=--import-undefined");
        return;
    }
    if target.contains("apple") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    } else if target.contains("windows") {
        println!("cargo:rustc-cdylib-link-arg=/FORCE:UNRESOLVED");
    } else {
        println!("cargo:rustc-cdylib-link-arg=-Wl,--unresolved-symbols=ignore-in-object-files");
    }
}
