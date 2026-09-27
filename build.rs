//! Exports only the plugin ABI from the executables.
//!
//! Plugin cdylibs resolve the host's `wren*` / `wlift_*` C API against
//! the executable at dlopen time, and the e2e tests dlsym their own
//! `wrenlift_*` symbols through `Library::this()`. Listing those names
//! instead of exporting every global keeps the rest of the symbol
//! table local, so the linker drops what nothing references.

fn main() {
    // `stack_fibers`: fibers get a stack of their own, krio's natively
    // and ash's guest fibers in the wasm AOT runtime.
    println!("cargo::rustc-check-cfg=cfg(stack_fibers)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let feature = |f: &str| std::env::var_os(format!("CARGO_FEATURE_{f}")).is_some();
    if feature("HOST") || (arch == "wasm32" && feature("AOT_RUNTIME")) {
        println!("cargo::rustc-cfg=stack_fibers");
    }

    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let arg = if os == "macos" {
        format!("-Wl,-exported_symbols_list,{dir}/tools/link/exports.macos")
    } else if family == "unix" {
        format!("-Wl,--dynamic-list={dir}/tools/link/exports.linux")
    } else {
        return;
    };
    println!("cargo:rerun-if-changed=tools/link/exports.macos");
    println!("cargo:rerun-if-changed=tools/link/exports.linux");
    println!("cargo:rustc-link-arg-bins={arg}");
    println!("cargo:rustc-link-arg-tests={arg}");
}
