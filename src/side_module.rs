//! Native libraries for wasm AOT programs.
//!
//! A library for a compiled wasm program is a `dylink.0` side module: it
//! imports the program's memory and function table, so it works on the
//! program's heap, and its host loads it beside the program at start-up.
//! `hatch plugin build --target wasm32-wasip1` builds one from a plugin
//! crate; `wlift --aot` links a package's archive into one when that is
//! what the package ships.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The target a side module is built for.
pub const TRIPLE: &str = "wasm32-wasip1";

/// Whether `bytes` is a side module: `dylink.0` is the first section
/// a linker writes into one.
pub fn is_side_module(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\0asm")
        && bytes[..bytes.len().min(64)]
            .windows(b"dylink.0".len())
            .any(|w| w == b"dylink.0")
}

/// Whether `bytes` is a static archive.
pub fn is_archive(bytes: &[u8]) -> bool {
    bytes.starts_with(b"!<arch>\n")
}

/// Every `#!symbol = "..."` name in the `.wren` files under `dir`: the
/// functions the package's foreign classes bind.
pub fn wren_symbols(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "wren")
                && let Ok(text) = std::fs::read_to_string(&path)
            {
                out.extend(text.lines().filter_map(symbol_attribute));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The name in a `#!symbol = "name"` line.
fn symbol_attribute(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("#!symbol")?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start().strip_prefix('"')?;
    Some(rest[..rest.find('"')?].to_string())
}

/// Build the crate `krate` of the cargo workspace `source` as a
/// position-independent static archive for [`TRIPLE`]; its path.
///
/// The shipped standard library is not position-independent, so this
/// rebuilds it with `-Z build-std`, which takes a nightly toolchain
/// with `rust-src`: `+nightly` unless `RUSTUP_TOOLCHAIN` names one. A C
/// dependency builds against `WASI_SYSROOT` when that is set.
pub fn build_archive(source: &Path, krate: &str) -> Result<PathBuf, String> {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("target"));
    let mut cmd = Command::new("cargo");
    if std::env::var_os("RUSTUP_TOOLCHAIN").is_none() {
        cmd.arg("+nightly");
    }
    cmd.args([
        "rustc",
        "-p",
        krate,
        "--release",
        "--lib",
        "--target",
        TRIPLE,
    ])
    .args([
        "--crate-type",
        "staticlib",
        "-Z",
        "build-std=std,panic_abort",
    ])
    .arg("--target-dir")
    .arg(&target_dir)
    .current_dir(source);
    let rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
    let flags = format!("{rustflags} -C relocation-model=pic -C target-feature=+mutable-globals");
    cmd.env("RUSTFLAGS", flags.trim());
    if let Some(sysroot) = std::env::var_os("WASI_SYSROOT")
        && std::env::var_os("CFLAGS_wasm32_wasip1").is_none()
    {
        cmd.env(
            "CFLAGS_wasm32_wasip1",
            format!(
                "--target=wasm32-wasi --sysroot={} -fPIC",
                Path::new(&sysroot).display()
            ),
        );
    }
    let status = cmd
        .status()
        .map_err(|e| format!("running cargo in {}: {e}", source.display()))?;
    if !status.success() {
        return Err(format!(
            "building {krate} for {TRIPLE} failed. It needs a nightly toolchain with \
             rust-src and the target: rustup toolchain install nightly --component rust-src \
             --target {TRIPLE}"
        ));
    }
    let archive = target_dir
        .join(TRIPLE)
        .join("release")
        .join(format!("lib{}.a", krate.replace('-', "_")));
    if archive.is_file() {
        Ok(archive)
    } else {
        Err(format!("cargo produced no {}", archive.display()))
    }
}

/// Link `archive` into the side module `output`, exporting each of
/// `exports` it defines.
pub fn link(archive: &Path, exports: &[String], output: &Path) -> Result<(), String> {
    let lld = wasm_linker().ok_or_else(|| {
        "linking a side module needs rust-lld (the Rust toolchain's) or wasm-ld; \
         set WLIFT_WASM_LD to one"
            .to_string()
    })?;
    let mut cmd = Command::new(&lld);
    if lld.file_stem().is_some_and(|s| s == "rust-lld") {
        cmd.args(["-flavor", "wasm"]);
    }
    // Undefined data as well as functions are imported: Rust's std takes
    // the address of `errno`.
    cmd.args([
        "--experimental-pic",
        "-shared",
        "--no-entry",
        "--gc-sections",
        "--unresolved-symbols=import-dynamic",
    ]);
    cmd.args(exports.iter().map(|e| format!("--export-if-defined={e}")));
    cmd.arg("--whole-archive")
        .arg(archive)
        .arg("--no-whole-archive")
        .arg("-o")
        .arg(output);
    let out = cmd
        .output()
        .map_err(|e| format!("running {}: {e}", lld.display()))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    let bytes = std::fs::read(output).map_err(|e| e.to_string())?;
    if !is_side_module(&bytes) {
        return Err(format!("{} is not a side module", output.display()));
    }
    Ok(())
}

/// `WLIFT_WASM_LD`, else the Rust toolchain's rust-lld, else wasm-ld on
/// the `PATH`.
pub fn wasm_linker() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("WLIFT_WASM_LD") {
        return Some(PathBuf::from(p));
    }
    let run = |args: &[&str]| {
        Command::new("rustc")
            .args(args)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
    };
    let lld = run(&["--print", "sysroot"])
        .zip(run(&["-vV"]))
        .and_then(|(sysroot, info)| {
            let host = info.lines().find_map(|l| l.strip_prefix("host: "))?;
            let p = Path::new(sysroot.trim())
                .join("lib/rustlib")
                .join(host)
                .join("bin/rust-lld");
            p.is_file().then_some(p)
        });
    lld.or_else(|| {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|d| d.join("wasm-ld"))
                .find(|p| p.is_file())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_symbol_attribute_names_its_function() {
        assert_eq!(
            symbol_attribute(r#"  #!symbol = "wlift_noise_simplex2""#),
            Some("wlift_noise_simplex2".to_string())
        );
        assert_eq!(symbol_attribute(r#"#!native = "wlift_noise""#), None);
        assert_eq!(
            symbol_attribute("  foreign static simplex2(x, y, seed)"),
            None
        );
    }
}
