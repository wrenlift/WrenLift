//! Wren compiled to wasm32 through LLVM, linked in process against the
//! prelinked runtime object and run under wasmtime.
//!
//! Skipped when the runtime object is absent (see
//! the README's "From source").
#![cfg(all(feature = "aot", feature = "llvm"))]

mod common;

use std::path::Path;
use std::process::Command;

use wren_lift::codegen::aot::{AotBundleMeta, walk_imports};
use wren_lift::codegen::llvm_aot::{
    AotEntry, LlvmTarget, compile_modules_to_llvm_object, compile_modules_to_llvm_object_as,
    link_wasm, place_wasm_libraries,
};

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

/// Run a WASI command module with `env`, and the directory it is in as
/// its working directory; its exit code and stdout.
fn run_module(wasm: &Path, env: &[(&str, &str)]) -> (i32, String) {
    use wasmtime::{Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1::{self, WasiP1Ctx};
    use wasmtime_wasi::{DirPerms, FilePerms};

    let engine = Engine::default();
    let module = Module::from_file(&engine, wasm).expect("loading the program");
    let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(1 << 20);
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .stdout(stdout.clone())
        .inherit_stderr()
        .envs(env)
        .preopened_dir(
            wasm.parent().expect("program dir"),
            ".",
            DirPerms::all(),
            FilePerms::all(),
        )
        .expect("preopening the program's directory")
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

/// Compile `files` for wasm32 with `libraries` (name, side module text)
/// placed beside the output, and run it the way an Ash host does: load
/// each side module into the program's memory and table, then start.
fn run_with_libraries(files: &[(&str, &str)], libraries: &[(&str, &str)]) -> Option<(i32, String)> {
    let runtime = common::runtime_object()?;
    let dir = tempfile::Builder::new()
        .prefix("wlift_wasm_libs_")
        .tempdir()
        .expect("tempdir");
    for (name, source) in files {
        std::fs::write(dir.path().join(format!("{name}.wren")), source).expect("write source");
    }
    let walk = walk_imports(&dir.path().join(format!("{}.wren", files[0].0))).expect("walk");
    let object = dir.path().join("program.o");
    let target = LlvmTarget::new("wasm32-wasip1", None, None);
    compile_modules_to_llvm_object(&walk.modules, &AotBundleMeta::default(), &target, &object)
        .expect("compile to a wasm32 object");
    let out = dir.path().join("out").join("program.wasm");
    std::fs::create_dir(out.parent().unwrap()).expect("output dir");
    let modules: Vec<(String, Vec<u8>)> = libraries
        .iter()
        .map(|(lib, text)| {
            (
                lib.to_string(),
                wat::parse_str(text).expect("side module text"),
            )
        })
        .collect();
    place_wasm_libraries(&modules, &Default::default(), &out).expect("placing libraries");
    link_wasm(&object, &runtime, &out).expect("linking");
    Some(host::run(&out))
}

/// A host for a wasm AOT program with native libraries, after Ash's
/// (`ash_wasm_runtime`'s `native/dylink.rs`): side modules beside the
/// program are loaded into its memory and table before it starts, and it
/// reaches them through `ash_host_dlopen` / `ash_host_dlsym`.
mod host {
    use std::collections::HashMap;
    use std::path::Path;

    use wasmtime::{
        Caller, Engine, Extern, Global, GlobalType, Instance, Linker, Module, Mutability, Ref,
        Store, Table, Val, ValType,
    };
    use wasmtime_wasi::preview1::{self, WasiP1Ctx};

    pub struct Host {
        wasi: WasiP1Ctx,
        libraries: HashMap<String, Instance>,
        table: Option<Table>,
        /// `(library, symbol)` to the table slot handed out for it.
        resolved: HashMap<(String, String), i32>,
    }

    fn guest_str(caller: &mut Caller<'_, Host>, ptr: i32, len: i32) -> String {
        let Some(Extern::Memory(memory)) = caller.get_export("memory") else {
            return String::new();
        };
        let bytes = &memory.data(&caller)[ptr as usize..(ptr + len) as usize];
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// A table slot holding `f`.
    fn slot(store: &mut Store<Host>, table: Table, f: wasmtime::Func) -> i32 {
        let index = table.size(&mut *store) as i32;
        table
            .grow(&mut *store, 1, Ref::Func(Some(f)))
            .expect("growing the table");
        index
    }

    /// Load the side module `bytes` into `main`'s memory and table as
    /// `lib`, and run its initializers.
    pub fn load(
        store: &mut Store<Host>,
        linker: &Linker<Host>,
        main: Instance,
        lib: &str,
        bytes: &[u8],
    ) -> Instance {
        let side = ash_wasm_link::read_side_module(bytes)
            .expect("reading dylink.0")
            .expect("a side module");
        let module = Module::new(store.engine(), bytes).expect("compiling a side module");
        let memory = main
            .get_export(&mut *store, "memory")
            .expect("program memory");
        let table = main
            .get_table(&mut *store, "__indirect_function_table")
            .expect("the program exports its table to host libraries");
        let memory_base = if side.memory_size > 0 {
            let malloc = main
                .get_typed_func::<i32, i32>(&mut *store, "malloc")
                .expect("the program exports malloc");
            malloc
                .call(&mut *store, side.memory_size as i32)
                .expect("malloc")
        } else {
            0
        };
        let table_base = table.size(&mut *store) as i32;
        table
            .grow(&mut *store, side.table_size as u64, Ref::Func(None))
            .expect("table");
        let constant = |store: &mut Store<Host>, v: i32| {
            Global::new(
                &mut *store,
                GlobalType::new(ValType::I32, Mutability::Const),
                Val::I32(v),
            )
            .expect("global")
        };
        let memory_base = constant(store, memory_base);
        let table_base = constant(store, table_base);
        let mut imports = Vec::new();
        let mut got = Vec::new();
        for import in module.imports() {
            if import.module() == "GOT.mem" || import.module() == "GOT.func" {
                let g = Global::new(
                    &mut *store,
                    GlobalType::new(ValType::I32, Mutability::Var),
                    Val::I32(0),
                )
                .expect("global");
                got.push((import.module() == "GOT.func", import.name().to_string(), g));
                imports.push(Extern::Global(g));
                continue;
            }
            let found = match (import.module(), import.name()) {
                ("env", "memory") => Some(memory.clone()),
                ("env", "__indirect_function_table") => Some(Extern::Table(table)),
                ("env", "__memory_base") => Some(Extern::Global(memory_base)),
                ("env", "__table_base") => Some(Extern::Global(table_base)),
                ("env", name) => main
                    .get_export(&mut *store, name)
                    .or_else(|| linker.get(&mut *store, "env", name)),
                (module, name) => linker.get(&mut *store, module, name),
            };
            imports.push(
                found.unwrap_or_else(|| {
                    panic!("{lib} imports {}::{}", import.module(), import.name())
                }),
            );
        }
        let instance = Instance::new(&mut *store, &module, &imports).expect("instantiating");
        for (function, name, g) in got {
            let value = [instance, main].iter().find_map(|owner| {
                match owner.get_export(&mut *store, &name)? {
                    Extern::Global(a) if !function => a.get(&mut *store).i32(),
                    Extern::Func(f) if function => Some(slot(store, table, f)),
                    _ => None,
                }
            });
            g.set(&mut *store, Val::I32(value.expect("a GOT entry")))
                .expect("GOT");
        }
        for init in ["__wasm_apply_data_relocs", "__wasm_call_ctors"] {
            if let Ok(f) = instance.get_typed_func::<(), ()>(&mut *store, init) {
                f.call(&mut *store, ()).expect(init);
            }
        }
        store.data_mut().table = Some(table);
        store.data_mut().libraries.insert(lib.to_string(), instance);
        instance
    }

    /// A store and linker for a program whose stdout goes to `stdout`,
    /// with `ash_host_dlopen` / `ash_host_dlsym` over the libraries
    /// loaded into it.
    pub fn host(
        engine: &Engine,
        stdout: &wasmtime_wasi::pipe::MemoryOutputPipe,
    ) -> (Store<Host>, Linker<Host>) {
        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .stdout(stdout.clone())
            .inherit_stderr()
            .build_p1();
        let store = Store::new(
            engine,
            Host {
                wasi,
                libraries: HashMap::new(),
                table: None,
                resolved: HashMap::new(),
            },
        );
        let mut linker: Linker<Host> = Linker::new(engine);
        preview1::add_to_linker_sync(&mut linker, |h| &mut h.wasi).expect("wasi imports");
        linker
            .func_wrap(
                "env",
                "ash_host_dlopen",
                |mut caller: Caller<'_, Host>, name: i32, len: i32| -> i32 {
                    let name = guest_str(&mut caller, name, len);
                    caller.data().libraries.contains_key(&name) as i32
                },
            )
            .expect("dlopen");
        linker
            .func_wrap(
                "env",
                "ash_host_dlsym",
                |mut caller: Caller<'_, Host>,
                 lib: i32,
                 lib_len: i32,
                 sym: i32,
                 sym_len: i32|
                 -> i32 {
                    let key = (
                        guest_str(&mut caller, lib, lib_len),
                        guest_str(&mut caller, sym, sym_len),
                    );
                    if let Some(&index) = caller.data().resolved.get(&key) {
                        return index;
                    }
                    let (Some(instance), Some(table)) = (
                        caller.data().libraries.get(&key.0).copied(),
                        caller.data().table,
                    ) else {
                        return 0;
                    };
                    let Some(f) = instance.get_func(&mut caller, &key.1) else {
                        return 0;
                    };
                    let index = table.size(&caller) as i32;
                    table
                        .grow(&mut caller, 1, Ref::Func(Some(f)))
                        .expect("growing the table");
                    caller.data_mut().resolved.insert(key, index);
                    index
                },
            )
            .expect("dlsym");
        (store, linker)
    }

    pub fn run(program: &Path) -> (i32, String) {
        let engine = Engine::default();
        let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(1 << 20);
        let (mut store, linker) = host(&engine, &stdout);
        let module = Module::from_file(&engine, program).expect("loading the program");
        let main = linker
            .instantiate(&mut store, &module)
            .expect("instantiating the program");
        let dir = program.parent().expect("program dir");
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .expect("program dir")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p != program && p.extension().is_some_and(|e| e == "wasm"))
            .collect();
        paths.sort();
        for path in paths {
            let bytes = std::fs::read(&path).expect("reading a library");
            let lib = path.file_stem().unwrap().to_string_lossy().into_owned();
            load(&mut store, &linker, main, &lib, &bytes);
        }
        let start = main
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

/// The pure-computation built-in modules run in the wasm runtime.
#[test]
fn hash_crypto_and_zip_modules() {
    expect(
        &[(
            "main",
            r#"
import "hash" for HashCore
import "crypto" for CryptoCore
import "zip" for ZipCore
System.print(HashCore.sha256Hex("abc"))
System.print(HashCore.hmacSha256Hex("key", "msg"))
System.print(HashCore.base64Encode("hello"))
System.print(CryptoCore.randomBytes(16).count)
var key = CryptoCore.aesGcmKey()
var nonce = CryptoCore.aesGcmNonce()
var sealed = CryptoCore.aesGcmEncrypt(key, nonce, "secret", null)
System.print(CryptoCore.aesGcmDecrypt(key, nonce, sealed, null).count)
System.print(CryptoCore.argon2Verify("pw", CryptoCore.argon2Hash("pw")))
var z = ZipCore.write({"a.txt": "hello zip"}, "deflate")
System.print(ZipCore.entries(z))
System.print(ZipCore.read(z, "a.txt").count)
"#,
        )],
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n\
         2d93cbc1be167bcb1637a4a23cbff01a7878f0c50ee833954ea5221bb1b8c628\n\
         aGVsbG8=\n16\n6\ntrue\n[a.txt]\n9\n",
    );
}

/// fs and os run over WASI: files in the directories the host opened,
/// and the host's environment.
#[test]
fn fs_and_os_modules() {
    let Some((code, out)) = run_linked(
        &[(
            "main",
            r#"
import "fs" for FS
import "os" for OS
FS.mkdirs("scratch/a/b")
FS.writeText("scratch/a/b/note.txt", "hello fs")
System.print(FS.readText("scratch/a/b/note.txt"))
System.print([FS.isDir("scratch/a"), FS.isFile("scratch/a/b/note.txt"), FS.size("scratch/a/b/note.txt")])
FS.rename("scratch/a/b/note.txt", "scratch/a/moved.txt")
System.print(FS.listDir("scratch/a"))
FS.removeTree("scratch")
System.print(FS.exists("scratch"))
System.print(OS.platform)
System.print(OS.env("WLIFT_PROBE"))
OS.setEnv("WLIFT_PROBE", "set")
System.print(OS.env("WLIFT_PROBE"))
OS.exit(3)
"#,
        )],
        Link::Wlift,
        &[("WLIFT_PROBE", "from host")],
    ) else {
        return;
    };
    assert_eq!(
        (code, out.as_str()),
        (
            3,
            "hello fs\n[true, true, 8]\n[b, moved.txt]\nfalse\nwasi\nfrom host\nset\n"
        )
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

/// A library program runs in the VM its host makes, registered from a
/// static constructor, and each `#export` member is an external symbol
/// under caribou's link rule taking and returning NaN-boxed values.
#[test]
fn a_library_runs_in_its_hosts_vm_and_exports_its_members() {
    use wasmtime::{Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1::{self, WasiP1Ctx};

    let Some(runtime) = common::runtime_object() else {
        eprintln!("no runtime object; skipping");
        return;
    };
    let dir = tempfile::Builder::new()
        .prefix("wlift_wasm_library_")
        .tempdir()
        .expect("tempdir");
    let entry = dir.path().join("tally.wren");
    std::fs::write(
        &entry,
        r#"
class Tally {
  #export = "new(start: Num)"
  construct new(start) { _total = start }

  #export = "add(n: Num) -> Num"
  add(n) {
    _total = _total + n
    return _total
  }

  #export = "total -> Num"
  total { _total }

  #export = "double(n: Num) -> Num"
  static double(n) { n * 2 }

  #export = "fail()"
  static fail() { Fiber.abort("boom") }

  #export = "loud(start: Num)"
  static loud(start) { Loud.new(start) }
}

// Through Tally's exported add, an instance of this reaches its own.
class Loud is Tally {
  #export = "new(start: Num)"
  construct new(start) { super(start) }
  add(n) { super.add(n * 10) }
}
System.print("module ran")
"#,
    )
    .expect("write source");
    let mut walk = walk_imports(&entry).expect("walk");
    walk.modules.last_mut().unwrap().request_name = "demo/tally".to_string();
    let object = dir.path().join("tally.o");
    let target = LlvmTarget::new("wasm32-wasip1", None, None);
    compile_modules_to_llvm_object_as(
        &walk.modules,
        &AotBundleMeta::default(),
        &target,
        AotEntry::Library,
        &object,
    )
    .expect("compile a library");

    let symbol = |kind: &str, name: &str, arity: usize| {
        format!(
            "caribou_4wren_12demo_2ftally_5Tally_{kind}{}{name}_{arity}",
            name.len()
        )
    };
    let loud_new = "caribou_4wren_12demo_2ftally_4Loud_c3new_1".to_string();
    let (new, add, total, double, fail, loud) = (
        symbol("c", "new", 1),
        symbol("m", "add", 1),
        symbol("g", "total", 0),
        symbol("t", "double", 1),
        symbol("t", "fail", 0),
        symbol("t", "loud", 1),
    );
    let read = |path: &Path| {
        let bytes = std::fs::read(path).expect("read object");
        ash_wasm_link::read(&path.display().to_string(), &bytes).expect("parse object")
    };
    let options = ash_wasm_link::LinkOptions {
        roots: [
            "wlift_aot_new_vm",
            "wlift_aot_run_programs",
            "wlift_aot_take_error",
        ]
        .iter()
        .map(|s| s.to_string())
        .chain([&new, &add, &total, &double, &fail, &loud, &loud_new].map(|s| s.clone()))
        .collect(),
        ..Default::default()
    };
    let module = ash_wasm_link::link(vec![read(&object), read(&runtime)], &options).expect("link");

    let engine = Engine::default();
    let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(1 << 16);
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .stdout(stdout.clone())
        .inherit_stderr()
        .build_p1();
    let mut store: Store<WasiP1Ctx> = Store::new(&engine, wasi);
    let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
    preview1::add_to_linker_sync(&mut linker, |s| s).expect("wasi imports");
    let module = Module::new(&engine, &module).expect("load the library");
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate");
    let f = |store: &mut Store<WasiP1Ctx>, name: &str| {
        instance
            .get_func(&mut *store, name)
            .unwrap_or_else(|| panic!("{name} exported"))
    };

    let vm = f(&mut store, "wlift_aot_new_vm")
        .typed::<(), i32>(&store)
        .unwrap()
        .call(&mut store, ())
        .expect("new vm");
    let run = f(&mut store, "wlift_aot_run_programs")
        .typed::<i32, i32>(&store)
        .unwrap();
    assert_eq!(run.call(&mut store, vm).expect("run"), 0);
    assert_eq!(
        String::from_utf8(stdout.contents().to_vec()).unwrap(),
        "module ran\n"
    );

    let num = |x: f64| x.to_bits() as i64;
    let back = |v: i64| f64::from_bits(v as u64);
    let tally = f(&mut store, &new)
        .typed::<i64, i64>(&store)
        .unwrap()
        .call(&mut store, num(10.0))
        .expect("construct");
    let add = f(&mut store, &add)
        .typed::<(i64, i64), i64>(&store)
        .unwrap();
    assert_eq!(back(add.call(&mut store, (tally, num(5.0))).unwrap()), 15.0);
    assert_eq!(back(add.call(&mut store, (tally, num(2.5))).unwrap()), 17.5);
    let total = f(&mut store, &total).typed::<i64, i64>(&store).unwrap();
    assert_eq!(back(total.call(&mut store, tally).unwrap()), 17.5);
    let double = f(&mut store, &double).typed::<i64, i64>(&store).unwrap();
    assert_eq!(back(double.call(&mut store, num(21.0)).unwrap()), 42.0);

    let take_error = f(&mut store, "wlift_aot_take_error")
        .typed::<i32, i32>(&store)
        .unwrap();
    assert_eq!(take_error.call(&mut store, vm).unwrap(), 0);
    f(&mut store, &fail)
        .typed::<(), i64>(&store)
        .unwrap()
        .call(&mut store, ())
        .expect("a raising member returns");
    assert_eq!(take_error.call(&mut store, vm).unwrap(), 70);
    assert_eq!(
        take_error.call(&mut store, vm).unwrap(),
        0,
        "taking the error clears it"
    );
    assert_eq!(back(double.call(&mut store, num(4.0)).unwrap()), 8.0);

    let loud = f(&mut store, &loud)
        .typed::<i64, i64>(&store)
        .unwrap()
        .call(&mut store, num(1.0))
        .expect("a subclass instance");
    assert_eq!(
        back(add.call(&mut store, (loud, num(2.0))).unwrap()),
        21.0,
        "the override runs"
    );
    assert_eq!(back(total.call(&mut store, loud).unwrap()), 21.0);

    // A constructor that calls super is made through dispatch.
    let quiet = f(&mut store, &loud_new)
        .typed::<i64, i64>(&store)
        .unwrap()
        .call(&mut store, num(5.0))
        .expect("a subclass constructed through its export");
    assert_eq!(back(add.call(&mut store, (quiet, num(1.0))).unwrap()), 15.0);
}

const COUNTER_V1: &str = r#"
import "./helper" for Helper

class Counter {
  #export = "new(start: Num)"
  construct new(start) { _n = start }

  #export = "step() -> Num"
  step() {
    _n = _n + 1
    return _n
  }

  #export = "version() -> Num"
  static version() { 1 }
}
var Made = Counter.new(100)
System.print("counter 1")
"#;

const COUNTER_V2: &str = r#"
import "./helper" for Helper

class Counter {
  #export = "new(start: Num)"
  construct new(start) { _n = start }

  #export = "step() -> Num"
  step() {
    _n = _n + Helper.bump
    return _n
  }

  #export = "version() -> Num"
  static version() { 2 }
}
var Made = Counter.new(200)
System.print("counter 2 made %(Made.step())")
"#;

/// A module compiled again while its program runs replaces its classes'
/// methods in place: an instance made before the reload runs the new
/// ones, the module body runs again, and its imports reach the running
/// program's modules.
#[test]
fn a_reloaded_module_gives_its_classes_new_methods() {
    use wasmtime::{Engine, Module};
    use wren_lift::codegen::llvm_aot::{
        compile_modules_to_llvm_object_with, link_wasm_reload, reload_host_exports,
    };

    let Some(runtime) = common::runtime_object() else {
        eprintln!("no runtime object; skipping");
        return;
    };
    if wren_lift::side_module::wasm_linker().is_none() {
        eprintln!("no wasm linker; skipping");
        return;
    }
    let dir = tempfile::Builder::new()
        .prefix("wlift_wasm_reload_")
        .tempdir()
        .expect("tempdir");
    let entry = dir.path().join("counter.wren");
    std::fs::write(
        dir.path().join("helper.wren"),
        "class Helper {\n  static bump { 10 }\n}\n",
    )
    .expect("write helper");
    let target = LlvmTarget::new("wasm32-wasip1", None, None);
    let compile = |source: &str, entry_kind: AotEntry, object: &Path| {
        std::fs::write(&entry, source).expect("write counter");
        let mut walk = walk_imports(&entry).expect("walk");
        walk.modules.last_mut().unwrap().request_name = "demo/counter".to_string();
        compile_modules_to_llvm_object_with(
            &walk.modules,
            &AotBundleMeta::default(),
            &target,
            entry_kind,
            true,
            object,
        )
        .expect("compile");
    };

    let program = dir.path().join("counter.o");
    compile(COUNTER_V1, AotEntry::Library, &program);
    let symbol = |kind: &str, name: &str, arity: usize| {
        format!(
            "caribou_4wren_14demo_2fcounter_7Counter_{kind}{}{name}_{arity}",
            name.len()
        )
    };
    let (new, step, version) = (
        symbol("c", "new", 1),
        symbol("m", "step", 0),
        symbol("t", "version", 0),
    );
    let read = |path: &Path| {
        let bytes = std::fs::read(path).expect("read object");
        ash_wasm_link::read(&path.display().to_string(), &bytes).expect("parse object")
    };
    let (functions, data) = reload_host_exports();
    let options = ash_wasm_link::LinkOptions {
        roots: [
            "wlift_aot_new_vm",
            "wlift_aot_run_programs",
            "wlift_aot_take_error",
        ]
        .iter()
        .map(|s| s.to_string())
        .chain([&new, &step, &version].map(|s| s.clone()))
        .collect(),
        hdll_imports: functions
            .into_iter()
            .chain(["malloc".to_string()])
            .collect(),
        hdll_data: data,
        ..Default::default()
    };
    let linked = ash_wasm_link::link(vec![read(&program), read(&runtime)], &options).expect("link");

    let reload_object = dir.path().join("reload.o");
    compile(COUNTER_V2, AotEntry::Reload, &reload_object);
    let reload_module = dir.path().join("reload.wasm");
    link_wasm_reload(&reload_object, &reload_module).expect("link the reload");

    let engine = Engine::default();
    let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(1 << 16);
    let (mut store, linker) = host::host(&engine, &stdout);
    let module = Module::new(&engine, &linked).expect("load the program");
    let main = linker
        .instantiate(&mut store, &module)
        .expect("instantiate");
    let call = |store: &mut wasmtime::Store<host::Host>, name: &str, args: &[i64]| -> i64 {
        let f = main
            .get_func(&mut *store, name)
            .unwrap_or_else(|| panic!("{name} exported"));
        let args: Vec<wasmtime::Val> = args.iter().map(|&a| wasmtime::Val::I64(a)).collect();
        let mut out = [wasmtime::Val::I64(0)];
        f.call(&mut *store, &args, &mut out).expect(name);
        out[0].unwrap_i64()
    };
    let vm = main
        .get_typed_func::<(), i32>(&mut store, "wlift_aot_new_vm")
        .unwrap()
        .call(&mut store, ())
        .expect("new vm");
    let run = main
        .get_typed_func::<i32, i32>(&mut store, "wlift_aot_run_programs")
        .unwrap();
    assert_eq!(run.call(&mut store, vm).expect("run"), 0);

    let num = |x: f64| x.to_bits() as i64;
    let back = |v: i64| f64::from_bits(v as u64);
    let before = call(&mut store, &new, &[num(0.0)]);
    assert_eq!(back(call(&mut store, &step, &[before])), 1.0);
    assert_eq!(back(call(&mut store, &version, &[])), 1.0);

    let bytes = std::fs::read(&reload_module).expect("read the reload");
    let reload = host::load(&mut store, &linker, main, "reload", &bytes);
    let reload_run = reload
        .get_typed_func::<i32, i32>(&mut store, "wlift_reload_run")
        .expect("the reload exports its run");
    assert_eq!(reload_run.call(&mut store, vm).expect("reload"), 0);

    assert_eq!(
        back(call(&mut store, &step, &[before])),
        11.0,
        "the new step"
    );
    assert_eq!(back(call(&mut store, &version, &[])), 2.0);
    let after = call(&mut store, &new, &[num(5.0)]);
    assert_eq!(back(call(&mut store, &step, &[after])), 15.0);
    assert_eq!(
        String::from_utf8(stdout.contents().to_vec()).unwrap(),
        "counter 1\ncounter 2 made 210\n"
    );
}

/// A native library as a `dylink.0` side module, the shape an Ash host
/// loads: it imports the program's memory and table and the C API it
/// calls, and works on the program's heap directly.
const CALC_LIBRARY: &str = r#"
(module
  (@custom "dylink.0" (before first) "\01\04\10\00\00\00")
  (import "env" "memory" (memory 0))
  (import "env" "__indirect_function_table" (table 0 funcref))
  (import "env" "__memory_base" (global $base i32))
  (import "env" "__table_base" (global $table_base i32))
  (import "env" "wrenGetSlotDouble" (func $get (param i32 i32) (result f64)))
  (import "env" "wrenSetSlotDouble" (func $set (param i32 i32 f64)))
  (import "env" "wrenGetSlotString" (func $get_str (param i32 i32) (result i32)))
  (import "env" "wrenSetSlotString" (func $set_str (param i32 i32 i32)))
  (data (global.get $base) "hello, library\00")
  (func (export "wlift_calc_add") (param $vm i32)
    (call $set (local.get $vm) (i32.const 0)
      (f64.add (call $get (local.get $vm) (i32.const 1))
               (call $get (local.get $vm) (i32.const 2)))))
  (func (export "wlift_calc_greet") (param $vm i32)
    (call $set_str (local.get $vm) (i32.const 0) (global.get $base)))
  (func (export "wlift_calc_echo") (param $vm i32)
    (call $set_str (local.get $vm) (i32.const 0)
      (call $get_str (local.get $vm) (i32.const 1)))))
"#;

const CALC_PROGRAM: &str = r#"
#!native = "wlift_calc"
foreign class Calc {
  #!symbol = "wlift_calc_add"
  foreign static add(a, b)
  #!symbol = "wlift_calc_greet"
  foreign static greet
  #!symbol = "wlift_calc_echo"
  foreign static echo(text)
}
System.print(Calc.add(2, 3))
System.print(Calc.greet)
System.print(Calc.echo("wren"))
"#;

/// A foreign class binds to the side module loaded beside the program,
/// whose functions read and write the program's own memory.
#[test]
fn a_foreign_class_calls_the_library_beside_it() {
    let Some(result) =
        run_with_libraries(&[("main", CALC_PROGRAM)], &[("wlift_calc", CALC_LIBRARY)])
    else {
        return;
    };
    assert_eq!(result, (0, "5\nhello, library\nwren\n".to_string()));
}

const DEPTH_LIBRARY: &str = r#"
(module
  (@custom "dylink.0" (before first) "\01\04\00\00\00\00")
  (import "env" "memory" (memory 0))
  (import "env" "__indirect_function_table" (table 0 funcref))
  (import "env" "__memory_base" (global $base i32))
  (import "env" "__table_base" (global $table_base i32))
  (import "env" "wrenSetSlotDouble" (func $set (param i32 i32 f64)))
  (import "env" "wlift_runtime_callout_depth" (func $depth (result i32)))
  (func (export "wlift_depth_now") (param $vm i32)
    (call $set (local.get $vm) (i32.const 0)
      (f64.convert_i32_s (call $depth)))))
"#;

const DEPTH_PROGRAM: &str = r#"
#!native = "wlift_depth"
foreign class Depth {
  #!symbol = "wlift_depth_now"
  foreign static now
}
System.print(Depth.now)
System.print(Fn.new { Depth.now }.call())
System.print(Fn.new { Fn.new { Depth.now }.call() }.call())
System.print([1].map {|x| Depth.now }.toList)
System.print(Depth.now)
"#;

/// A native reached through a runtime call into compiled code sees that
/// call counted, and the count is back to 0 once it returns.
#[test]
fn a_call_into_compiled_code_is_counted_while_it_runs() {
    let Some(result) = run_with_libraries(
        &[("main", DEPTH_PROGRAM)],
        &[("wlift_depth", DEPTH_LIBRARY)],
    ) else {
        return;
    };
    assert_eq!(result, (0, "0\n1\n2\n[1]\n0\n".to_string()));
}

/// Without the library, the program stops before running any code.
#[test]
fn a_foreign_class_without_its_library_ends_the_program() {
    let Some((code, out)) = run_with_libraries(&[("main", CALC_PROGRAM)], &[]) else {
        return;
    };
    assert_eq!((code, out.as_str()), (70, ""));
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
