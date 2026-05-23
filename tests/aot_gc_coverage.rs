//! Phase 4 — per-pattern AOT GC stress tests.
//!
//! Each test compiles a tiny Wren program that exercises a
//! specific MIR-instruction class (allocators, mutators,
//! call chains, fibers, …) in a tight loop, then runs the
//! resulting binary under:
//!   - `WLIFT_AOT_GC=1`              — opt into the AOT GC path
//!   - `WLIFT_GC_STRESS=1`           — force collection at every
//!                                     `finish_alloc` so any
//!                                     stale-reference bug
//!                                     surfaces immediately
//!   - `WLIFT_VALIDATE_BARRIERS=1`   — pre-collect remembered-set
//!                                     check in both directions
//!                                     (missed barriers + stale
//!                                     sources)
//!
//! A clean exit means every alloc-site survives a GC fired at
//! that exact instruction's safepoint, with full barrier
//! coverage. Any panic / segfault / non-zero exit identifies the
//! offending lowering or runtime helper.
//!
//! These tests target the latent bug exposed by the larger
//! nursery in `b4f6775` (sustained mix died on
//! `trace_object+1112` after a few requests). Stress mode
//! ensures the bug surfaces in <1 s of test time instead of
//! after ~100 site requests.
//!
//! Skipped on Windows (different linker driver) and on builds
//! without the `aot` feature.
#![cfg(all(feature = "aot", any(target_os = "macos", target_os = "linux")))]

use std::path::{Path, PathBuf};
use std::process::Command;

use wren_lift::codegen::aot::{
    compile_walk_to_object_with_manifest, link_executable, locate_runtime_staticlib, walk_imports,
    AotBundleMeta,
};

fn staticlib() -> PathBuf {
    locate_runtime_staticlib().unwrap_or_else(|| {
        panic!(
            "could not locate libwren_lift.a; run `cargo build --release --features aot` \
             or set WLIFT_STATICLIB to the absolute path of the staticlib"
        );
    })
}

/// Compile + link + run with AOT-GC stress envs preset. Returns
/// `(exit_code, stdout, stderr)`. A successful run is `exit=0`
/// with no validator-panic substring in stderr.
fn run_under_stress(source: &str) -> (i32, String, String) {
    use std::fs;

    let tmp = tempfile::Builder::new()
        .prefix("wlift_aot_gc_cov_")
        .tempdir()
        .expect("tempdir");
    fs::write(tmp.path().join("entry.wren"), source).expect("write source file");
    let entry_path = tmp.path().join("entry.wren");
    let walk = walk_imports(&entry_path).expect("walk_imports");
    let obj = tmp.path().join("program.o");
    compile_walk_to_object_with_manifest(&walk.modules, &AotBundleMeta::default(), &obj)
        .expect("compile_walk_to_object_with_manifest");
    let exe = tmp.path().join("program");
    link_executable(&obj, &staticlib(), &exe).expect("link_executable");

    let output = Command::new(&exe)
        .env("WLIFT_AOT_GC", "1")
        .env("WLIFT_GC_STRESS", "1")
        .env("WLIFT_VALIDATE_BARRIERS", "1")
        .output()
        .expect("execute aot binary");

    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Helper: assert clean exit + no validator-panic substring.
fn assert_clean(label: &str, exit: i32, stderr: &str) {
    assert_eq!(
        exit, 0,
        "{label}: non-zero exit ({exit}). stderr:\n{stderr}"
    );
    for bad in [
        "WRITE BARRIER BUG",
        "STALE-SOURCE",
        "ALIAS-DETECT",
        "STACKMAP COVERAGE GAP",
        "fatal runtime error",
        "panicked at",
    ] {
        assert!(
            !stderr.contains(bad),
            "{label}: stderr contains `{bad}`. full stderr:\n{stderr}"
        );
    }
}

// ---------------------------------------------------------------------------
// String allocators — `string + string` produces a new ObjString each iteration.
// ---------------------------------------------------------------------------

#[test]
fn stress_string_concat() {
    let source = r#"
var acc = ""
for (i in 1..200) {
    acc = "a" + "b"
}
System.print(acc.count)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("string_concat", exit, &stderr);
}

// ---------------------------------------------------------------------------
// List allocators + element writes — exercises `wren_make_list_*` + `wren_list_add`.
// ---------------------------------------------------------------------------

#[test]
fn stress_list_add_loop() {
    let source = r#"
var xs = []
for (i in 1..200) {
    xs.add("item-" + i.toString)
}
System.print(xs.count)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("list_add_loop", exit, &stderr);
}

#[test]
fn stress_list_subscript_set() {
    let source = r#"
var xs = ["x", "x", "x", "x", "x"]
for (i in 1..200) {
    xs[0] = "y-" + i.toString
}
System.print(xs[0])
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("list_subscript_set", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Map allocators + element writes — the suspected residual bug class.
// `update_old_gen_pointers_inline` hashes MapKey during the drain+reinsert
// loop; a stale string key would crash here.
// ---------------------------------------------------------------------------

#[test]
fn stress_map_set_string_keys() {
    let source = r#"
var m = {}
for (i in 1..200) {
    m["key-" + i.toString] = i
}
System.print(m.count)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("map_set_string_keys", exit, &stderr);
}

#[test]
fn stress_map_set_and_remove() {
    let source = r#"
var m = {}
for (i in 1..200) {
    var k = "k-" + i.toString
    m[k] = i
    if (i > 50) {
        m.remove("k-" + (i - 50).toString)
    }
}
System.print(m.count)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("map_set_and_remove", exit, &stderr);
}

#[test]
fn stress_map_iterate_after_grow() {
    let source = r#"
var m = {}
for (i in 1..100) {
    m["k-" + i.toString] = "v-" + i.toString
}
var total = 0
for (key in m.keys) {
    total = total + 1
}
System.print(total)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("map_iterate_after_grow", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Range allocators — `wren_make_range`.
// ---------------------------------------------------------------------------

#[test]
fn stress_range_iteration() {
    let source = r#"
var total = 0
for (j in 1..50) {
    for (i in 1..100) {
        total = total + i
    }
}
System.print(total)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("range_iteration", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Closure construction — `wren_make_closure_N`.
// ---------------------------------------------------------------------------

#[test]
fn stress_closure_construction() {
    let source = r#"
var acc = 0
for (i in 1..100) {
    var n = i
    var fn = Fn.new { n * 2 }
    acc = acc + fn.call
}
System.print(acc)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("closure_construction", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Instance allocation — `wren_alloc_instance`.
// ---------------------------------------------------------------------------

#[test]
fn stress_instance_allocation() {
    let source = r#"
class Box {
    construct new(v) { _v = v }
    v { _v }
}
var sum = 0
for (i in 1..200) {
    var b = Box.new(i)
    sum = sum + b.v
}
System.print(sum)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("instance_allocation", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Minimal repro for the constructor-receiver-stale-across-alloc bug.
// First field store runs fine; the second runs through an allocating
// runtime helper (string concat) and on return the receiver pointer
// in a callee-saved register may have moved.
// ---------------------------------------------------------------------------

#[test]
fn stress_constructor_field_after_alloc() {
    let source = r#"
class Item {
    construct new(n) {
        _n = n
        _label = "item-" + n.toString
    }
    label { _label }
}
var it = Item.new(7)
System.print(it.label)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("constructor_field_after_alloc", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Mixed allocation — closer to real-world site code. Exercises the
// nested-call path (constructor body → wren_call_0 → toString) that
// surfaced the Pass-1 covered_fps bug: the AOT path's `push_jit_frame`
// uses the MIR func_id, but AOT metadata is registered under a
// separate AOT func_id. Pass 1 silently failed the metadata lookup
// while leaving the constructor's fp marked covered, so Pass 2 also
// skipped the constructor. The receiver promoted from nursery to
// old-gen never got its spill slot written back; the next nursery
// reset reused the stale address for the concat-result string, and
// the second field store dereferenced what it thought was the
// receiver's `fields` ptr at a String header.
#[test]
fn stress_mixed_alloc_chain() {
    let source = r#"
class Item {
    construct new(n) {
        _n = n
        _label = "item-" + n.toString
    }
    label { _label }
    n { _n }
}
var index = {}
for (i in 1..150) {
    var it = Item.new(i)
    index[it.label] = it.n
}
System.print(index.count)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("mixed_alloc_chain", exit, &stderr);
}

// ---------------------------------------------------------------------------
// Fiber allocation + yield/resume — exercises `mir_frames`/`aot_frames`
// fields whose barriers are covered by always-trace-fibers OR by the
// per-write barriers on `setup_fiber_from_closure`.
// ---------------------------------------------------------------------------

#[test]
fn stress_fiber_spawn_and_call() {
    let source = r#"
var sum = 0
for (i in 1..50) {
    var f = Fiber.new { i * 3 }
    sum = sum + f.call
}
System.print(sum)
"#;
    let (exit, _stdout, stderr) = run_under_stress(source);
    assert_clean("fiber_spawn_and_call", exit, &stderr);
}
