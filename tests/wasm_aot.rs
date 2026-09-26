//! Wren compiled to wasm32 through LLVM, linked in process against the
//! prelinked runtime object and run under wasmtime.
//!
//! Skipped when the runtime object is absent (see
//! `tools/build_wasm_runtime.sh`).
#![cfg(all(feature = "aot", feature = "llvm"))]

mod common;

use std::path::Path;
use std::process::Command;

use wren_lift::codegen::aot::{AotBundleMeta, walk_imports};
use wren_lift::codegen::llvm_aot::{LlvmTarget, compile_modules_to_llvm_object, link_wasm};

/// How the program object and the runtime object become one module.
#[derive(Clone, Copy)]
enum Link {
    /// wlift's own, in process.
    Wlift,
    /// rust-lld, as a reference; `None` when there is none.
    Lld,
}

/// Compile `files` (the first is the entry) for wasm32, link them `how`
/// and run the program; its exit code and stdout. `None` when the
/// runtime object, or rust-lld for `Link::Lld`, is missing.
fn run_linked(files: &[(&str, &str)], how: Link, env: &[(&str, &str)]) -> Option<(i32, String)> {
    let Some(runtime) = common::runtime_object() else {
        eprintln!("no runtime object; skipping");
        return None;
    };
    let dir = tempfile::Builder::new()
        .prefix("wlift_wasm_aot_")
        .tempdir()
        .expect("tempdir");
    for (name, source) in files {
        std::fs::write(dir.path().join(format!("{name}.wren")), source).expect("write source");
    }
    let entry = dir.path().join(format!("{}.wren", files[0].0));
    let walk = walk_imports(&entry).expect("walk_imports");
    let object = dir.path().join("program.o");
    let target = LlvmTarget::new("wasm32-wasip1", None, None);
    compile_modules_to_llvm_object(&walk.modules, &AotBundleMeta::default(), &target, &object)
        .expect("compile to a wasm32 object");

    let wasm = dir.path().join("program.wasm");
    match how {
        Link::Wlift => link_wasm(&object, &runtime, &wasm).expect("linking"),
        Link::Lld => {
            let Some(lld) = common::rust_lld() else {
                eprintln!("no rust-lld; skipping");
                return None;
            };
            let Some(libs) = common::wasi_lib_dir() else {
                eprintln!("no wasi-libc; skipping");
                return None;
            };
            // wasm-ld runs the constructors from crt1's `_start`.
            let out = Command::new(&lld)
                .args(["-flavor", "wasm"])
                .arg(libs.join("crt1-command.o"))
                .arg(&object)
                .arg(&runtime)
                .arg(format!("-L{}", libs.display()))
                .arg("-lc")
                .arg("-o")
                .arg(&wasm)
                .output()
                .expect("running rust-lld");
            assert!(
                out.status.success(),
                "rust-lld failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    Some(run_module(&wasm, env))
}

/// Run a WASI command module with `env`; its exit code and stdout.
fn run_module(wasm: &Path, env: &[(&str, &str)]) -> (i32, String) {
    use wasmtime::{Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1::{self, WasiP1Ctx};

    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm).expect("loading the program");
    let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(1 << 20);
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .stdout(stdout.clone())
        .inherit_stderr()
        .envs(env)
        .build_p1();
    let mut store = Store::new(&engine, wasi);
    let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
    preview1::add_to_linker_sync(&mut linker, |s| s).expect("wasi imports");
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiating the program");
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .expect("_start");
    let code = match start.call(&mut store, ()) {
        Ok(()) => 0,
        Err(err) => match err.downcast_ref::<wasmtime_wasi::I32Exit>() {
            Some(exit) => exit.0,
            None => panic!("the program trapped: {err:?}"),
        },
    };
    let text = String::from_utf8(stdout.contents().to_vec()).expect("utf-8 output");
    (code, text)
}

fn run_wasm(files: &[(&str, &str)]) -> Option<(i32, String)> {
    run_linked(files, Link::Wlift, &[])
}

fn expect(files: &[(&str, &str)], want: &str) {
    let Some((code, out)) = run_wasm(files) else {
        return;
    };
    assert_eq!(code, 0, "exit code; output:\n{out}");
    assert_eq!(out, want);
}

#[test]
fn arithmetic_strings_and_control_flow() {
    expect(
        &[(
            "main",
            r#"
var a = 6
var b = 7
System.print(a * b)
System.print("x" + "y")
System.print("%(a) and %(b)")
var s = 0
for (i in 1..10) {
  if (i % 2 == 0) s = s + i
}
System.print(s)
var n = 0
while (n < 5) n = n + 1
System.print(n)
"#,
        )],
        "42\nxy\n6 and 7\n30\n5\n",
    );
}

#[test]
fn closures_lists_and_maps() {
    expect(
        &[(
            "main",
            r#"
var counter = Fn.new {
  var n = 0
  return Fn.new { n = n + 1 }
}.call()
counter.call()
counter.call()
System.print(counter.call())
var xs = [3, 1, 2]
xs.add(5)
System.print(xs)
System.print(xs.count)
var sum = 0
for (x in xs) sum = sum + x
System.print(sum)
var m = {"a": 1, "b": 2}
m["c"] = 3
System.print(m["a"] + m["c"])
System.print(xs.map { |x| x * 10 }.toList)
"#,
        )],
        "3\n[3, 1, 2, 5]\n4\n11\n4\n[30, 10, 20, 50]\n",
    );
}

#[test]
fn classes_fields_statics_and_super() {
    expect(
        &[(
            "main",
            r#"
class Shape {
  construct new(name) {
    _name = name
    __made = (__made == null ? 0 : __made) + 1
  }
  name { _name }
  area { 0 }
  describe { "%(name) of area %(area)" }
  static made { __made }
}
class Rect is Shape {
  construct new(w, h) {
    super("rect")
    _w = w
    _h = h
  }
  area { _w * _h }
  describe { "[" + super.describe + "]" }
}
var r = Rect.new(3, 4)
System.print(r.area)
System.print(r.describe)
System.print(Shape.new("dot").describe)
System.print(Shape.made)
System.print(r is Shape)
"#,
        )],
        "12\n[rect of area 12]\ndot of area 0\n2\ntrue\n",
    );
}

#[test]
fn a_module_imports_another() {
    expect(
        &[
            (
                "main",
                r#"
import "./geo" for Point
var p = Point.new(1, 2) + Point.new(3, 4)
System.print(p.toString)
"#,
            ),
            (
                "geo",
                r#"
class Point {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  y { _y }
  +(o) { Point.new(_x + o.x, _y + o.y) }
  toString { "(%(_x), %(_y))" }
}
"#,
            ),
        ],
        "(4, 6)\n",
    );
}

#[test]
fn fibers_that_finish_and_many_allocations() {
    expect(
        &[(
            "main",
            r#"
System.print(Fiber.new { 5 }.call())
var f = Fiber.new { 21 * 2 }
System.print(f.call())
System.print(f.isDone)
var keep = []
for (i in 0...200000) {
  var s = "s%(i)"
  if (i % 1000 == 0) keep.add(s)
}
System.print(keep.count)
System.print(keep[199])
"#,
        )],
        "5\n42\ntrue\n200\ns199000\n",
    );
}

/// Calls go straight to the implementation for the receiver's class, or
/// the class itself for a static method; any other receiver, a subclass
/// that inherits, or a class from another module goes through dispatch.
#[test]
fn direct_calls_pick_the_receivers_implementation() {
    expect(
        &[
            (
                "main",
                r#"
import "other" for Box
class Crate {
  construct new(v) { _v = v }
  v { _v }
  name { "crate" }
  static make(v) { Crate.new(v + 1) }
  static fib(n) { n < 2 ? n : fib(n - 1) + fib(n - 2) }
}
class Sub is Crate {
  construct new(v) { super(v) }
  name { "sub" }
}
class Plain is Crate {
  construct new(v) { super(v) }
}
var items = [Crate.new(1), Sub.new(2), Plain.new(3), Box.new(4)]
for (item in items) System.print("%(item.name) %(item.v)")
var f = Fiber.new { items[0].make(1) }
f.try()
System.print(f.error)
System.print(Crate.make(1).v)
System.print(Box.make(1).v)
System.print(Crate.fib(15))
"#,
            ),
            (
                "other",
                r#"
class Box {
  construct new(v) { _v = v }
  v { "box %(_v)" }
  name { "box" }
  static make(v) { Box.new(v * 10) }
}
"#,
            ),
        ],
        "crate 1\nsub 2\ncrate 3\nbox box 4\nCrate does not implement 'make(_)'\n2\nbox 10\n610\n",
    );
}

/// Compiled frames keep their values in the shadow stack, where the
/// collector finds them: collecting at every allocation loses none.
#[test]
fn a_collection_keeps_what_compiled_frames_hold() {
    let Some((code, out)) = run_linked(
        &[(
            "main",
            r#"
class Tree {
  construct new(left, right) {
    _left = left
    _right = right
  }
  check { _left == null ? 1 : 1 + _left.check + _right.check }
  static build(depth) {
    if (depth == 0) return Tree.new(null, null)
    return Tree.new(build(depth - 1), build(depth - 1))
  }
}
var label = "tree"
var tree = Tree.build(6)
var parts = []
for (i in 0...50) {
  var s = "%(label)-%(i)"
  parts.add(s + "!" + Tree.build(3).check.toString)
}
System.print(tree.check)
System.print(parts[0])
System.print(parts[49])
"#,
        )],
        Link::Wlift,
        &[("WLIFT_GC_STRESS", "1")],
    ) else {
        return;
    };
    assert_eq!((code, out.as_str()), (0, "127\ntree-0!15\ntree-49!15\n"));
}

#[test]
fn an_uncaught_error_ends_the_program_with_70() {
    let Some((code, out)) = run_wasm(&[(
        "main",
        "System.print(1)\nFiber.abort(\"boom\")\nSystem.print(2)\n",
    )]) else {
        return;
    };
    assert_eq!((code, out.as_str()), (70, "1\n"));
}

/// A yield inside compiled code needs a fiber stack, which wasm does not
/// have yet: it raises instead of running on past the yield.
#[test]
fn a_yield_in_compiled_code_raises() {
    let Some((code, out)) = run_wasm(&[(
        "main",
        "var f = Fiber.new {\n  Fiber.yield(1)\n  System.print(\"resumed\")\n}\nf.call()\nSystem.print(\"after\")\n",
    )]) else {
        return;
    };
    assert_eq!((code, out.as_str()), (70, ""));
}

/// The features of an object compiled for `target`, as its
/// `target_features` section lists them.
fn object_features(target: &LlvmTarget) -> Vec<String> {
    let dir = tempfile::Builder::new()
        .prefix("wlift_wasm_features_")
        .tempdir()
        .expect("tempdir");
    let entry = dir.path().join("main.wren");
    std::fs::write(
        &entry,
        "var s = 0\nfor (i in 0...8) s = s + i\nSystem.print(s)\n",
    )
    .expect("write source");
    let walk = walk_imports(&entry).expect("walk_imports");
    let object = dir.path().join("program.o");
    compile_modules_to_llvm_object(&walk.modules, &AotBundleMeta::default(), target, &object)
        .expect("compile");
    let bytes = std::fs::read(&object).expect("read object");
    let mut features = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let Ok(wasmparser::Payload::CustomSection(c)) = payload
            && c.name() == "target_features"
        {
            let data = c.data();
            let mut r = wasmparser::BinaryReader::new(data, 0);
            let n = r.read_var_u32().unwrap();
            for _ in 0..n {
                let prefix = r.read_u8().unwrap() as char;
                let name = r.read_string().unwrap();
                features.push(format!("{prefix}{name}"));
            }
        }
    }
    features
}

#[test]
fn the_requested_target_reaches_the_object() {
    let default = object_features(&LlvmTarget::new("wasm32-wasip1", None, None));
    for f in ["+bulk-memory", "+nontrapping-fptoint", "+sign-ext"] {
        assert!(default.iter().any(|x| x == f), "{f} in {default:?}");
    }
    assert!(!default.iter().any(|x| x == "+simd128"), "{default:?}");

    let simd = LlvmTarget::new(
        "wasm32-wasip1",
        None,
        Some(&format!(
            "{},+simd128",
            wren_lift::codegen::llvm_aot::WASM32_FEATURES
        )),
    );
    let with_simd = object_features(&simd);
    assert!(with_simd.iter().any(|x| x == "+simd128"), "{with_simd:?}");
}

#[test]
fn export_checks_what_it_declares() {
    expect(
        &[(
            "main",
            r#"class Tally {
  #export = "new(t: Num)"
  construct new(t) { _t = t }
  #export = "bump(x: Num) -> Num"
  bump(x) {
    _t = _t + x
    return _t
  }
  #export = "total -> Num"
  total { _t }
  #export = "name(s: String) -> String"
  static name(s) { s + "!" }
  #export = "bad -> Num"
  static bad { "no" }
}
var t = Tally.new(0)
for (i in 0...20000) t.bump(1)
System.print(t.total)
System.print(Tally.name("hi"))
System.print(Fiber.new { t.bump("x") }.try())
System.print(Fiber.new { Tally.new(null) }.try())
System.print(Fiber.new { Tally.name(3) }.try())
System.print(Fiber.new { Tally.bad }.try())
System.print(t.total)
"#,
        )],
        "20000\nhi!\nbump(_) expects Num for `x`\nnew(_) expects Num for `t`\nname(_) expects String for `s`\nbad returns Num\n20000\n",
    );
}

#[test]
fn ranges_count_in_their_own_direction() {
    expect(
        &[(
            "main",
            r#"var c = 0
for (i in 100000...0) c = c + 1
System.print(c)
var n = 0
var d = 0
for (i in 300000...n) d = d + i
System.print(d)
var up = 0
var m = 200000
for (i in 0...m) up = up + i
System.print(up)
var e = 0
for (i in 5...5) e = e + 1
System.print(e)
var f = 0
for (i in 0.5...100000) f = f + 1
System.print(f)
var g = 0
for (i in 100000...0.5) g = g + 1
System.print(g)
var nest = 0
for (a in 0...300) for (b in 300...a) nest = nest + 1
System.print(nest)
var neg = 0
var lim = -50000
for (i in 0...lim) neg = neg + i
System.print(neg)
"#,
        )],
        "100000\n45000150000\n19999900000\n0\n100000\n100000\n45150\n-1249975000\n",
    );
}

/// wlift's linker and rust-lld make modules that behave the same.
#[test]
fn the_linker_agrees_with_rust_lld() {
    let program: &[(&str, &str)] = &[
        (
            "main",
            r#"
import "./shapes" for Rect
var xs = []
for (i in 0...50) xs.add(Rect.new(i, i + 1))
var total = 0
for (r in xs) total = total + r.area
System.print(total)
var f = Fiber.new { "fib %(Rect.fib(20))" }
System.print(f.call())
System.print({"k": [1, 2, 3]}["k"].count)
for (i in 3...0) System.print(i)
"#,
        ),
        (
            "shapes",
            r#"
class Rect {
  construct new(w, h) {
    _w = w
    _h = h
  }
  area { _w * _h }
  static fib(n) { n < 2 ? n : fib(n - 1) + fib(n - 2) }
}
"#,
        ),
    ];
    let (Some(ours), Some(lld)) = (
        run_linked(program, Link::Wlift, &[]),
        run_linked(program, Link::Lld, &[]),
    ) else {
        return;
    };
    assert_eq!(ours, lld);
    assert_eq!(ours, (0, "41650\nfib 6765\n3\n3\n2\n1\n".to_string()));
}
