//! End-to-end AOT runtime tests — compile a Wren source string,
//! link the produced object against `libwren_lift.a`, run the
//! resulting executable, and assert on stdout / exit. Validates
//! the lowering, bootstrap, and runtime helpers as a single
//! pipeline rather than just inspecting the emitted symbol table.
//!
//! The harness reuses [`wren_lift::codegen::aot::link_executable`]
//! and [`locate_runtime_staticlib`] so the link contract stays
//! identical to what the `wlift --aot` CLI runs.
//!
//! Skipped on Windows (different linker driver) and on builds
//! without the `aot` feature.
//!
//! Prerequisites: the runtime staticlib must exist, e.g. via
//!   cargo build --release --features aot
//! before running these tests. The harness errors out with a
//! one-line hint if the staticlib can't be found.
#![cfg(all(feature = "aot", any(target_os = "macos", target_os = "linux")))]

use std::path::{Path, PathBuf};
use std::process::Command;

use wren_lift::codegen::aot::{
    compile_walk_to_object_with_manifest, link_executable, locate_runtime_staticlib,
    walk_imports, AotBundleMeta,
};

struct AotRun {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

fn staticlib() -> PathBuf {
    locate_runtime_staticlib().unwrap_or_else(|| {
        panic!(
            "could not locate libwren_lift.a; run `cargo build --release --features aot` \
             or set WLIFT_STATICLIB to the absolute path of the staticlib"
        );
    })
}

/// Compile + link + run a single Wren source string. Routes
/// through the same multi-module pipeline the CLI uses (so the
/// produced `.o` carries the Cranelift-emitted `main` entry the
/// linker needs); the source is just written to a one-file
/// tempdir and walked.
fn compile_link_run(source: &str) -> AotRun {
    compile_link_run_files("entry", &[("entry", source)])
}

/// Multi-module variant: write each `(name, contents)` pair into
/// a tempdir as `<name>.wren`, walk imports rooted at `entry`,
/// emit + link, run.
fn compile_link_run_files(entry: &str, files: &[(&str, &str)]) -> AotRun {
    use std::fs;

    let tmp = tempfile::Builder::new()
        .prefix("wlift_aot_test_")
        .tempdir()
        .expect("tempdir");
    for (name, contents) in files {
        fs::write(tmp.path().join(format!("{}.wren", name)), contents)
            .expect("write source file");
    }
    let entry_path = tmp.path().join(format!("{}.wren", entry));
    let walk = walk_imports(&entry_path).expect("walk_imports");
    let obj = tmp.path().join("program.o");
    compile_walk_to_object_with_manifest(&walk.modules, &AotBundleMeta::default(), &obj)
        .expect("compile_walk_to_object_with_manifest");
    let exe = tmp.path().join("program");
    link_executable(&obj, &staticlib(), &exe).expect("link_executable");
    run(&exe)
}

fn run(exe: &Path) -> AotRun {
    let output = Command::new(exe).output().expect("execute aot binary");
    AotRun {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

// ---------------------------------------------------------------------------
// Phase 1 tests — broad smoke coverage of the AOT lowering paths
// ---------------------------------------------------------------------------

#[test]
fn print_literal() {
    let r = compile_link_run("System.print(42)\n");
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "42\n");
}

#[test]
fn arithmetic() {
    let r = compile_link_run("System.print(1 + 2 * 3 - 4)\n");
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "3\n");
}

#[test]
fn module_var_roundtrip() {
    // Exercises the `wlift_modvars_main` data section: a top-level
    // var written + read back through GetModuleVar / SetModuleVar.
    let r = compile_link_run("var x = 7\nx = x * 6\nSystem.print(x)\n");
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "42\n");
}

#[test]
fn class_method_call() {
    // `wren_call_N` against an AOT-installed class method via
    // wlift_aot_install_class.
    let r = compile_link_run(
        "class App {\n  static run() { return 21 + 21 }\n}\nSystem.print(App.run())\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "42\n");
}

#[test]
fn cha_direct_dispatch() {
    // Single-impl signature → AOT-CHA emits a class-checked
    // direct call into the method body, bypassing wren_call_N.
    let r = compile_link_run(
        "class Box {\n  \
            construct new(v) { _v = v }\n  \
            value() { return _v }\n\
         }\n\
         var b = Box.new(42)\n\
         System.print(b.value())\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "42\n");
}

#[test]
fn static_field_roundtrip() {
    // Regression test for the step 4a fix: a class method that
    // reads + writes a static field, called twice. Pre-fix this
    // would print null (helper read JitContext.defining_class
    // which wlift_aot_enter never populates) and silently drop
    // every write.
    let r = compile_link_run(
        "class Counter {\n  \
            static incr() {\n    \
                __n = (__n == null ? 0 : __n) + 1\n    \
                return __n\n  \
            }\n\
         }\n\
         Counter.incr()\n\
         System.print(Counter.incr())\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "2\n");
}

#[test]
fn multi_module() {
    // Cross-module class import: helper module exposes a class,
    // entry imports it and invokes a method. Validates per-module
    // wlift_modvars_<n> + the import-binding fixup that copies
    // the helper's class slot into the entry module's modvars.
    let r = compile_link_run_files(
        "entry",
        &[
            (
                "helper",
                "class Helper {\n  static answer() { return 42 }\n}\n",
            ),
            (
                "entry",
                "import \"./helper\" for Helper\nSystem.print(Helper.answer())\n",
            ),
        ],
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "42\n");
}

#[test]
fn closure_upvalue() {
    // Closure capturing a local + reading it back: exercises
    // the AOT inline upvalue read against the function-scoped
    // `closure_ptr` Variable (loaded once at entry from
    // `JitContext.closure`, then consulted per-access without
    // a helper call).
    let r = compile_link_run(
        "var n = 40\n\
         var add2 = Fn.new { n + 2 }\n\
         System.print(add2.call())\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "42\n");
}

#[test]
fn closure_upvalue_mutation() {
    // Closure reads an upvalue, mutates it, reads it back —
    // exercises both inline GetUpvalue and SetUpvalue
    // (including the wren_write_barrier call) under AOT.
    // Repeated invocations must observe the carried-over
    // upvalue state.
    //
    // Caveat: top-level `var n` is a *module variable*, not a
    // captured local — MIR lowers `n` references to
    // GetModuleVar / SetModuleVar, not GetUpvalue / SetUpvalue.
    // The genuine upvalue path runs through
    // `closure_captures_function_local` below.
    let r = compile_link_run(
        "var n = 0\n\
         var bump = Fn.new { n = n + 1 }\n\
         bump.call()\n\
         bump.call()\n\
         bump.call()\n\
         System.print(n)\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "3\n");
}

#[test]
fn closure_captures_function_local() {
    // Real upvalue: a function-local `n` captured by an inner
    // closure. The factory returns the closure; the outer
    // local goes out of scope, so the upvalue must carry the
    // closed-over value through `ObjUpvalue`'s `closed` field
    // and indirect through `location`. AOT lowers GetUpvalue /
    // SetUpvalue inline against the closure's `upvalues` Vec
    // data pointer (not the Vec field itself — Rust's current
    // Vec layout puts capacity at offset 0, the data pointer
    // at offset 8, length at offset 16). Pre-fix this test
    // SIGSEGVed at the first upvalue read because the lowering
    // dereferenced the capacity field as a pointer.
    let r = compile_link_run(
        "class Counter {\n  \
            static make() {\n    \
                var n = 0\n    \
                return Fn.new {\n      \
                    n = n + 1\n      \
                    return n\n    \
                }\n  \
            }\n\
         }\n\
         var bump = Counter.make()\n\
         bump.call()\n\
         bump.call()\n\
         System.print(bump.call())\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "3\n");
}
