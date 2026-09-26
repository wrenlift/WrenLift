//! Helpers the wasm tests share: where the runtime object, rust-lld
//! and wasi-libc are.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// `wlift_runtime.o` beside this build, or where `WLIFT_RUNTIME` says.
pub fn runtime_object() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("WLIFT_RUNTIME") {
        let p = PathBuf::from(explicit);
        return p.is_file().then_some(p);
    }
    // The test binary sits under target/<profile>/deps; the object sits
    // under target/<profile>/wasm32-wasip1.
    let exe = std::env::current_exe().ok()?;
    exe.ancestors()
        .map(|dir| dir.join("wasm32-wasip1").join("wlift_runtime.o"))
        .find(|p| p.is_file())
}

/// rust-lld from the toolchain building this test.
pub fn rust_lld() -> Option<PathBuf> {
    let out = |args: &[&str]| {
        let o = Command::new("rustc").args(args).output().ok()?;
        String::from_utf8(o.stdout).ok()
    };
    let sysroot = out(&["--print", "sysroot"])?;
    let host = out(&["-vV"])?
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))?;
    let lld = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin/rust-lld");
    lld.is_file().then_some(lld)
}

/// The wasi-libc library directory the object was prelinked against.
pub fn wasi_lib_dir() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = std::env::var_os("WASI_SYSROOT")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    roots.extend(
        [
            "/opt/homebrew/opt/wasi-libc/share/wasi-sysroot",
            "/usr/local/opt/wasi-libc/share/wasi-sysroot",
            "/opt/wasi-sdk/share/wasi-sysroot",
            "/usr/share/wasi-sysroot",
        ]
        .map(PathBuf::from),
    );
    roots
        .into_iter()
        .map(|r| r.join("lib/wasm32-wasip1"))
        .find(|d| d.join("crt1-reactor.o").is_file())
}
