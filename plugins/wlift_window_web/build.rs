//! Plugin link configuration — see wlift_sqlite/build.rs for rationale.

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
