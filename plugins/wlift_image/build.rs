//! Plugin link configuration — see wlift_sqlite/build.rs for the
//! full rationale. The cdylib references the host's `wlift_plugin_*`
//! symbols via `wlift_abi`'s extern blocks; those symbols don't
//! exist at link time. Each platform needs a flag telling the
//! linker to leave the references undefined for runtime resolution.

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
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
