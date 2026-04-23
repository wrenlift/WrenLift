//! Integration tests for the embedder-facing `HatchRunner`.
//!
//! Each test builds a fresh `.hatch` artifact on the fly (`hatch
//! build`), feeds it to the runner through one of the public APIs,
//! and then executes user code that imports the package.

use std::path::PathBuf;
use std::process::Command;

use wren_lift::hatch_runner::{HatchRunner, RunnerError};
use wren_lift::runtime::vm::VMConfig;

/// Repo-relative path to the workspace root.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Shell out to `hatch build` on a packaged hatch source dir and
/// return the resulting `.hatch` bytes. Uses the in-repo binary
/// so we don't depend on a globally-installed CLI.
fn build_hatch(package_dir: &str) -> Vec<u8> {
    let root = repo_root();
    let hatch_bin = root.join("target").join("release").join("hatch");
    assert!(
        hatch_bin.exists(),
        "hatch binary not built — run `cargo build --release --bin hatch` first ({})",
        hatch_bin.display()
    );
    let pkg = root.join("hatch").join("packages").join(package_dir);
    // Write the artifact into a unique tempfile so parallel tests
    // don't race on a shared output path.
    let out_dir = tempfile::tempdir().expect("tempdir for build output");
    let out_path = out_dir.path().join("pkg.hatch");
    let output = Command::new(&hatch_bin)
        .current_dir(&pkg)
        .arg("build")
        .arg("--out")
        .arg(&out_path)
        .output()
        .expect("hatch build failed to spawn");
    assert!(
        output.status.success(),
        "hatch build failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read(&out_path).expect("failed to read built hatch")
}

/// Grab a fresh VM with output capture enabled.
fn runner_capturing() -> HatchRunner {
    let mut runner = HatchRunner::with_config(VMConfig::default());
    runner.vm_mut().output_buffer = Some(String::new());
    runner
}

#[test]
fn install_bytes_then_run_importing_code() {
    let bytes = build_hatch("hatch-math");
    let mut runner = runner_capturing();
    runner
        .install_bytes_tagged("@hatch:math", &bytes)
        .expect("install_bytes");
    assert!(runner.is_installed("@hatch:math"));

    runner
        .run_source(
            "main",
            r#"
            import "@hatch:math" for Vec3
            var v = Vec3.new(3, 4, 0)
            System.print(v.length)
            "#,
        )
        .expect("run_source");
    let out = runner.vm_mut().take_output();
    assert!(out.contains("5"), "expected '5' in output, got {:?}", out);
}

#[test]
fn install_many_resolves_through_search_path() {
    // Build both packages and drop them in a shared dir, then
    // install both via the search-path + name flow.
    let math = build_hatch("hatch-math");
    let fp = build_hatch("hatch-fp");

    let tmp = tempfile::tempdir().unwrap();
    let math_path = tmp.path().join("@hatch:math-0.1.0.hatch");
    let fp_path = tmp.path().join("@hatch:fp-0.1.0.hatch");
    std::fs::write(&math_path, math).unwrap();
    std::fs::write(&fp_path, fp).unwrap();

    let mut runner = runner_capturing();
    runner.add_search_path(tmp.path()).unwrap();
    runner
        .install_many(&["@hatch:math", "@hatch:fp"])
        .expect("install_many");

    assert!(runner.is_installed("@hatch:math"));
    assert!(runner.is_installed("@hatch:fp"));

    runner
        .run_source(
            "main",
            r#"
            import "@hatch:math" for Vec3, Math
            import "@hatch:fp"   for Pipe

            var sum = Pipe.of([Vec3.new(1, 0, 0), Vec3.new(0, 2, 0), Vec3.new(0, 0, 3)])
              .map(Fn.new {|v| v.length })
              .sum
            // lengths are 1 + 2 + 3 = 6
            System.print(Math.approxEq(sum, 6))
            "#,
        )
        .expect("cross-package run");
    let out = runner.vm_mut().take_output();
    assert!(out.contains("true"), "got {:?}", out);
}

#[test]
fn install_all_found_picks_up_everything_in_path() {
    let math = build_hatch("hatch-math");
    let fp = build_hatch("hatch-fp");

    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("@hatch:math-0.1.0.hatch"), math).unwrap();
    std::fs::write(tmp.path().join("@hatch:fp-0.1.0.hatch"), fp).unwrap();

    let mut runner = runner_capturing();
    runner.add_search_path(tmp.path()).unwrap();
    let n = runner.install_all_found().expect("install_all_found");
    assert_eq!(n, 2);
    assert!(runner.is_installed("@hatch:math"));
    assert!(runner.is_installed("@hatch:fp"));
}

#[test]
fn install_same_name_twice_is_noop() {
    let bytes = build_hatch("hatch-math");
    let mut runner = HatchRunner::new();
    runner
        .install_bytes_tagged("@hatch:math", &bytes)
        .expect("first install");
    // Second install with the same name shouldn't reparse the
    // bytes; the tracking list prevents duplicate work.
    runner
        .install("@hatch:math")
        .expect("second install should no-op");
    assert_eq!(runner.installed_packages().len(), 1);
}

#[test]
fn unknown_package_returns_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    // Point the ambient cache somewhere we control (and empty).
    std::env::set_var("HATCH_CACHE_DIR", tmp.path());
    let mut runner = HatchRunner::new();
    let err = runner.install("@hatch:nope-not-real").unwrap_err();
    assert!(matches!(err, RunnerError::NotFound(_)));
}
