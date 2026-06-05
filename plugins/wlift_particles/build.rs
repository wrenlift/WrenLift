//! Plugin link configuration. Same shape as every other plugin in
//! the workspace — see wlift_noise/build.rs for the rationale. The
//! cdylib references the host's `wlift_plugin_*` symbols via
//! `wlift_abi`'s extern blocks; those symbols are resolved at dlopen
//! against the host process, so we tell the linker to leave them
//! undefined at build time.

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
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
