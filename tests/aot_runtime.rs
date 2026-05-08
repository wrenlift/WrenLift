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
    compile_walk_to_object_with_manifest, link_executable, locate_runtime_staticlib, walk_imports,
    AotBundleMeta,
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
        fs::write(tmp.path().join(format!("{}.wren", name)), contents).expect("write source file");
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
fn simd_float32_interop() {
    let r = compile_link_run(
        "var floats = Float32Array.fromList([1, 2, 3, 4])\n\
         var v = Simd4f.load(floats, 0) * Simd4f.splat(2)\n\
         v.store(floats, 0)\n\
         System.print(floats.toList)\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "[2, 4, 6, 8]\n");
}

#[test]
fn simd_int32_interop() {
    let r = compile_link_run(
        "var ints = Int32Array.fromList([1, 2, 3, 4])\n\
         var v = Simd4i.load(ints, 0) + Simd4i.splat(5)\n\
         v.store(ints, 0)\n\
         System.print(ints.toList)\n\
         System.print((v > Simd4i.splat(7)).bitmask)\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "[6, 7, 8, 9]\n12\n");
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

// ---------------------------------------------------------------------------
// Phase 2 tests — broader correctness coverage (polymorphic CHA,
// inheritance, instance fields, exceptions, fibers, lists)
// ---------------------------------------------------------------------------

#[test]
fn instance_field_roundtrip() {
    // Instance fields go through Instance.fields[idx] (raw
    // *mut Value at +INSTANCE_FIELDS). Validates that the
    // Cranelift-emitted GetField / SetField lowering reads the
    // right offset and that the GC keeps fields alive across
    // method calls.
    let r = compile_link_run(
        "class Point {\n  \
            construct new(x, y) {\n    \
                _x = x\n    \
                _y = y\n  \
            }\n  \
            x { _x }\n  \
            y { _y }\n  \
            move(dx, dy) {\n    \
                _x = _x + dx\n    \
                _y = _y + dy\n  \
            }\n\
         }\n\
         var p = Point.new(1, 2)\n\
         p.move(10, 20)\n\
         System.print(\"%(p.x),%(p.y)\")\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "11,22\n");
}

#[test]
fn polymorphic_dispatch() {
    // Two classes implement the same `kind()` signature — CHA
    // sees 2 impls and emits a class-dispatch tree (chained
    // class checks per impl), falling back to wren_call_N when
    // no class matches. Exercises the multi-impl fast path and
    // the iteration that mixes both classes through one
    // virtual call site.
    let r = compile_link_run(
        "class Cat {\n  \
            construct new() {}\n  \
            kind() { return \"cat\" }\n\
         }\n\
         class Dog {\n  \
            construct new() {}\n  \
            kind() { return \"dog\" }\n\
         }\n\
         var animals = [Cat.new(), Dog.new(), Cat.new()]\n\
         var counts = { \"cat\": 0, \"dog\": 0 }\n\
         for (a in animals) { counts[a.kind()] = counts[a.kind()] + 1 }\n\
         System.print(\"%(counts[\"cat\"]),%(counts[\"dog\"])\")\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "2,1\n");
}

#[test]
fn inheritance_super_call() {
    // Subclass calls super to extend a parent method. Uses
    // the wren_super_call_N helper path under AOT; validates
    // that AOT-installed classes wire parent_slot correctly
    // and the runtime can walk the inheritance chain back to
    // the parent's method.
    let r = compile_link_run(
        "class Greeter {\n  \
            construct new() {}\n  \
            greet(name) { return \"Hello, \" + name }\n\
         }\n\
         class Shouter is Greeter {\n  \
            construct new() {}\n  \
            greet(name) { return super.greet(name) + \"!\" }\n\
         }\n\
         System.print(Shouter.new().greet(\"world\"))\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "Hello, world!\n");
}

#[test]
fn list_iteration() {
    // Lists go through the runtime's ObjList + for-in protocol
    // (iterator() / iteratorValue()). Exercises the IC fast
    // path on `[_]` subscript and `count` getter, plus the
    // for-in lowering.
    let r = compile_link_run(
        "var sum = 0\n\
         for (x in [1, 2, 3, 4, 5]) sum = sum + x\n\
         System.print(sum)\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "15\n");
}

#[test]
fn nested_closures_share_upvalues() {
    // A canonical "make multiple closures over the same local"
    // case: getter + setter pair share one ObjUpvalue. Both
    // closures' upvalues Vec contains the same upvalue
    // pointer — the inline lowering must compute the same
    // location_ptr from each.
    let r = compile_link_run(
        "class Cell {\n  \
            static make(initial) {\n    \
                var x = initial\n    \
                return [\n      \
                    Fn.new { x },\n      \
                    Fn.new {|v| x = v }\n    \
                ]\n  \
            }\n\
         }\n\
         var pair = Cell.make(5)\n\
         var get = pair[0]\n\
         var set = pair[1]\n\
         System.print(get.call())\n\
         set.call(99)\n\
         System.print(get.call())\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "5\n99\n");
}

#[test]
fn fiber_yield_state_machine() {
    // Pure-AOT Fiber.yield support — no interpreter fallback.
    // The closure body is lowered with the stackless 2-arg
    // poll signature; each yield advances the state ID and
    // stamps `kind=Yield`, the dispatcher reads back the kind
    // to suspend the fiber and surface the value to the
    // caller. Three `f.call()`s drain the body's three states:
    // 10, 20, 30 (the final return).
    let r = compile_link_run(
        "var f = Fiber.new {\n  \
           System.print(\"step 1\")\n  \
           Fiber.yield(10)\n  \
           System.print(\"step 2\")\n  \
           Fiber.yield(20)\n  \
           System.print(\"step 3\")\n  \
           return 30\n\
         }\n\
         System.print(\"a=%(f.call())\")\n\
         System.print(\"b=%(f.call())\")\n\
         System.print(\"c=%(f.call())\")\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "step 1\na=10\nstep 2\nb=20\nstep 3\nc=30\n");
}

#[test]
fn fiber_yield_in_loop_with_branch_and_live_loop_var() {
    // v2-cap1 + cap-2: yield inside an `if` inside a `while`,
    // with the loop counter `i` live across the suspension.
    // Exercises:
    //   - tail-duplication of the post-yield code path so the
    //     non-yielding side (i odd) still flows through the
    //     original block,
    //   - save/load of `i` (bb1's block param) via the fiber's
    //     saved-slot table,
    //   - the back-edge branch arg correctly threading the
    //     loaded value back into the loop header.
    // Expected: yields at i=0 and i=2, then returns "done"
    // when i hits 4 and exits the while.
    let r = compile_link_run(
        "var f = Fiber.new {\n  \
           var i = 0\n  \
           while (i < 4) {\n    \
             if (i % 2 == 0) {\n      \
               Fiber.yield(i)\n    \
             }\n    \
             i = i + 1\n  \
           }\n  \
           return \"done\"\n\
         }\n\
         while (true) {\n  \
           var v = f.call()\n  \
           if (f.isDone) {\n    \
             System.print(\"final=%(v)\")\n    \
             break\n  \
           }\n  \
           System.print(\"yielded %(v)\")\n\
         }\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "yielded 0\nyielded 2\nfinal=done\n");
}

#[test]
fn cross_function_fiber_yield_propagation() {
    // v2-cap3: a class method that yields, called from inside
    // a fiber body. The yield must propagate up through the
    // cross-fn poll path so the caller's fiber.call observes
    // the yield, and a subsequent fiber.call resumes the
    // method at its post-yield state. Two `c.step()`
    // invocations chain yield->return->yield->return:
    //
    //   c.step() #1: _v=1, yield 1.
    //   resume:      _v=2, return 2.
    //   c.step() #2: _v=3, yield 3.
    //   resume:      _v=4, return 4.
    //
    // The closure body's value flows through, so the fiber
    // observes 1, 3, 4 across three calls.
    let r = compile_link_run(
        "class Counter {\n  \
           construct new() { _v = 0 }\n  \
           step() {\n    \
             _v = _v + 1\n    \
             Fiber.yield(_v)\n    \
             _v = _v + 1\n    \
             return _v\n  \
           }\n\
         }\n\
         var c = Counter.new()\n\
         var f = Fiber.new {\n  \
           c.step()\n  \
           c.step()\n\
         }\n\
         System.print(\"a=%(f.call())\")\n\
         System.print(\"b=%(f.call())\")\n\
         System.print(\"c=%(f.call())\")\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "a=1\nb=3\nc=4\n");
}

#[test]
fn nested_class_method_calls() {
    // Non-trivial method chain with intermediate object
    // construction. Exercises the IC + GC interaction across
    // a hot loop that allocates per-iteration — if the AOT
    // path forgets to root values across allocation safepoints
    // this is where it breaks.
    let r = compile_link_run(
        "class Wrap {\n  \
            construct new(v) { _v = v }\n  \
            doubled() { return Wrap.new(_v * 2) }\n  \
            value { _v }\n\
         }\n\
         var w = Wrap.new(1)\n\
         for (i in 0...10) w = w.doubled()\n\
         System.print(w.value)\n",
    );
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    assert_eq!(r.stdout, "1024\n");
}
