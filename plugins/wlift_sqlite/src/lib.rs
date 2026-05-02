//! SQLite plugin for WrenLift. Built as a `cdylib` for native
//! hosts (the runtime `dlopen`s it from `@hatch:sqlite`'s
//! `NativeLib` section) and as an `rlib` for the wasm static-link
//! path (linked into `wlift_wasm` so the browser playground can
//! `import "@hatch:sqlite"` without a separate `.wasm` artifact).
//!
//! Two backends sit behind a thin module-level switch:
//!
//!   * Native: [`backend_native`] uses `rusqlite` with bundled
//!     libsqlite3. `Connection::open_with_flags`, prepared
//!     statements, the `Statement::raw_*` family.
//!
//!   * Wasm: [`backend_wasm`] uses `sqlite-wasm-rs` — the official
//!     SQLite-Wasm C build wrapped as `wasm32-unknown-unknown`
//!     bindings. Drives `sqlite3_prepare_v2` / `sqlite3_step` /
//!     `sqlite3_finalize` directly. Memory VFS only (the playground
//!     doesn't need OPFS persistence yet).
//!
//! Both backends register the same `(connection_id, sql, params)`
//! shape against the foreign-method registry, so the Wren-side
//! `SqliteCore` class is target-independent.
//!
//! # Safety
//!
//! Every `wlift_sqlite_*` entry point is called by the runtime
//! with a valid `*mut VM` that stays live for the duration of
//! the call. The runtime is also responsible for providing
//! well-formed slot arguments — we validate them defensively
//! but trust that the pointer itself dereferences.

#![allow(clippy::missing_safety_doc)]

use wren_lift::runtime::object::{
    NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType, ObjTypedArray, TypedArrayKind,
};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

#[cfg(not(target_arch = "wasm32"))]
mod backend_native;
#[cfg(not(target_arch = "wasm32"))]
use backend_native as backend;

#[cfg(target_arch = "wasm32")]
mod backend_wasm;
#[cfg(target_arch = "wasm32")]
use backend_wasm as backend;

/// Plugin ABI handshake — see wlift_gpu::wlift_plugin_abi_version.
///
/// Native-only. The host's `libloading::dlsym` path uses it to
/// reject a dylib built against a different `wren_lift` than the
/// host. Statically-linked wasm plugins can't drift versions
/// (everything's compiled from the same tree), and re-exporting
/// the symbol from every linked rlib produces wasm-linker
/// duplicate-symbol errors.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    wren_lift::runtime::foreign::WLIFT_PLUGIN_ABI_VERSION
}

// --- Backend-independent value model ---------------------------

/// Tagged-union mirror of the five SQLite storage classes. Owned
/// strings / blobs so the backend's row buffers can be dropped
/// before we walk the result back into Wren values.
#[derive(Clone)]
pub(crate) enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

pub(crate) enum Params {
    None,
    Positional(Vec<SqlValue>),
    Named(Vec<(String, SqlValue)>),
}

// --- Slot helpers ----------------------------------------------
//
// Foreign C functions receive only `*mut VM`. Arguments arrive on
// the VM's `api_stack` (slot 0 = receiver / return value; slots
// 1..N = positional args). `VM` implements `NativeContext` so we
// can reach into its allocator for alloc_string / alloc_list /
// alloc_map, and surface errors via `runtime_error`.

unsafe fn slot(vm: *mut VM, index: usize) -> Value {
    unsafe {
        let stack = &(*vm).api_stack;
        stack.get(index).copied().unwrap_or(Value::null())
    }
}

unsafe fn set_return(vm: *mut VM, v: Value) {
    unsafe {
        let stack = &mut (*vm).api_stack;
        if stack.is_empty() {
            stack.push(v);
        } else {
            stack[0] = v;
        }
    }
}

unsafe fn ctx(vm: *mut VM) -> &'static mut VM {
    unsafe { &mut *vm }
}

// --- Type coercion ---------------------------------------------

unsafe fn string_of(v: Value) -> Option<String> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::String {
        return None;
    }
    let s = ptr as *const ObjString;
    Some(unsafe { (*s).as_str().to_string() })
}

fn id_of(vm: &mut VM, v: Value, label: &str) -> Option<u64> {
    match v.as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Some(n as u64),
        _ => {
            vm.runtime_error(format!("{}: id must be a non-negative integer.", label));
            None
        }
    }
}

fn wren_to_sql(v: Value, label: &str) -> Result<SqlValue, String> {
    if v.is_null() {
        return Ok(SqlValue::Null);
    }
    if let Some(b) = v.as_bool() {
        return Ok(SqlValue::Integer(if b { 1 } else { 0 }));
    }
    if let Some(n) = v.as_num() {
        if n.is_finite() && n == n.trunc() && (i64::MIN as f64..=i64::MAX as f64).contains(&n) {
            return Ok(SqlValue::Integer(n as i64));
        }
        if n.is_nan() || n.is_infinite() {
            return Err(format!("{}: NaN/Infinity can't be bound.", label));
        }
        return Ok(SqlValue::Real(n));
    }
    if !v.is_object() {
        return Err(format!("{}: unsupported parameter type.", label));
    }
    let ptr = v.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    match unsafe { (*header).obj_type } {
        ObjType::String => {
            let s = ptr as *const ObjString;
            Ok(SqlValue::Text(unsafe { (*s).as_str().to_string() }))
        }
        // ByteArray is the canonical blob carrier — `sql_to_wren`
        // returns one for every BLOB column read, so a round-trip
        // (`db.query(...)["b"]` → bind back) doesn't need a manual
        // List<Num> conversion in Wren.
        ObjType::TypedArray => {
            let arr = ptr as *const ObjTypedArray;
            if unsafe { (*arr).kind_tag() } != TypedArrayKind::U8 {
                return Err(format!(
                    "{}: BLOB parameter must be a ByteArray (got {:?}).",
                    label,
                    unsafe { (*arr).kind_tag() }
                ));
            }
            let bytes = unsafe { (*arr).as_bytes() }.to_vec();
            Ok(SqlValue::Blob(bytes))
        }
        // Backwards-compatible List<Num> path. Pre-ByteArray
        // versions of `@hatch:sqlite` accepted blobs as a Wren
        // List of integers in 0..=255; existing call sites
        // (`db.execute("INSERT … blob = ?", [[0xde, 0xad]])`) keep
        // working unchanged. New code should prefer ByteArray.
        ObjType::List => {
            let lst = ptr as *const ObjList;
            let (count, data) = unsafe { ((*lst).count as usize, (*lst).elements) };
            let mut bytes = Vec::with_capacity(count);
            for i in 0..count {
                let elem = unsafe { *data.add(i) };
                let n = elem
                    .as_num()
                    .ok_or_else(|| format!("{}: BLOB entries must be numbers.", label))?;
                if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
                    return Err(format!(
                        "{}: BLOB entries must be integers in 0..=255.",
                        label
                    ));
                }
                bytes.push(n as u8);
            }
            Ok(SqlValue::Blob(bytes))
        }
        _ => Err(format!("{}: unsupported parameter type.", label)),
    }
}

fn sql_to_wren(ctx: &mut dyn NativeContext, v: SqlValue) -> Value {
    match v {
        SqlValue::Null => Value::null(),
        SqlValue::Integer(i) => Value::num(i as f64),
        SqlValue::Real(f) => Value::num(f),
        SqlValue::Text(s) => ctx.alloc_string(s),
        // BLOB → ByteArray. Half the memory of a List<Num> per
        // element (1 byte vs 8-byte boxed Value), zero per-element
        // GC overhead, and a `Response.bytes(...)`-friendly shape
        // for callers that want to forward the column verbatim.
        // `wren_to_sql` accepts both ByteArray and the legacy
        // List<Num> on the bind side so existing code that builds
        // blobs as Wren lists keeps working.
        SqlValue::Blob(b) => {
            let value = ctx.alloc_typed_array(b.len() as u32, TypedArrayKind::U8);
            let arr_ptr = value.as_object().unwrap() as *mut ObjTypedArray;
            if !b.is_empty() {
                unsafe { (*arr_ptr).as_bytes_mut().copy_from_slice(&b) };
            }
            value
        }
    }
}

fn collect_params(v: Value, label: &str) -> Result<Params, String> {
    if v.is_null() {
        return Ok(Params::None);
    }
    if !v.is_object() {
        return Err(format!("{}: params must be a List, Map, or null.", label));
    }
    let ptr = v.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    match unsafe { (*header).obj_type } {
        ObjType::List => {
            let lst = ptr as *const ObjList;
            let (count, data) = unsafe { ((*lst).count as usize, (*lst).elements) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let elem = unsafe { *data.add(i) };
                out.push(wren_to_sql(elem, label)?);
            }
            Ok(Params::Positional(out))
        }
        ObjType::Map => {
            let m = ptr as *const ObjMap;
            let entries: Vec<(Value, Value)> =
                unsafe { (*m).entries.iter().map(|(k, v)| (k.0, *v)).collect() };
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let key = match unsafe { string_of(k) } {
                    Some(s) => s,
                    None => {
                        return Err(format!("{}: named parameter keys must be strings.", label));
                    }
                };
                let key = if key.starts_with(':') || key.starts_with('@') || key.starts_with('$') {
                    key
                } else {
                    format!(":{}", key)
                };
                out.push((key, wren_to_sql(v, label)?));
            }
            Ok(Params::Named(out))
        }
        _ => Err(format!("{}: params must be a List, Map, or null.", label)),
    }
}

// --- Foreign C entry points ------------------------------------
//
// The runtime resolves these by name. Each one routes through the
// `backend` alias above, so the same Wren-side `SqliteCore` class
// works on host and wasm.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_open(vm: *mut VM) {
    unsafe {
        let path_val = slot(vm, 1);
        let path = match string_of(path_val) {
            Some(s) => s,
            None => {
                ctx(vm).runtime_error("Sqlite.open: path must be a string.".to_string());
                set_return(vm, Value::null());
                return;
            }
        };
        match backend::open(&path) {
            Ok(id) => set_return(vm, Value::num(id as f64)),
            Err(e) => {
                ctx(vm).runtime_error(format!("Sqlite.open: {}", e));
                set_return(vm, Value::null());
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_close(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.close") else {
            set_return(vm, Value::null());
            return;
        };
        backend::close(id);
        set_return(vm, Value::null());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_execute(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let sql_val = slot(vm, 2);
        let params_val = slot(vm, 3);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.execute") else {
            set_return(vm, Value::null());
            return;
        };
        let sql = match string_of(sql_val) {
            Some(s) => s,
            None => {
                ctx(vm).runtime_error("Sqlite.execute: sql must be a string.".to_string());
                set_return(vm, Value::null());
                return;
            }
        };
        let params = match collect_params(params_val, "Sqlite.execute") {
            Ok(p) => p,
            Err(e) => {
                ctx(vm).runtime_error(e);
                set_return(vm, Value::null());
                return;
            }
        };
        match backend::execute(id, &sql, &params) {
            Ok(rows) => set_return(vm, Value::num(rows as f64)),
            Err(e) => {
                ctx(vm).runtime_error(format!("Sqlite.execute: {}", e));
                set_return(vm, Value::null());
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_query(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let sql_val = slot(vm, 2);
        let params_val = slot(vm, 3);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.query") else {
            set_return(vm, Value::null());
            return;
        };
        let sql = match string_of(sql_val) {
            Some(s) => s,
            None => {
                ctx(vm).runtime_error("Sqlite.query: sql must be a string.".to_string());
                set_return(vm, Value::null());
                return;
            }
        };
        let params = match collect_params(params_val, "Sqlite.query") {
            Ok(p) => p,
            Err(e) => {
                ctx(vm).runtime_error(e);
                set_return(vm, Value::null());
                return;
            }
        };

        // Materialize rows BEFORE alloc_string / alloc_map so a
        // GC inside the allocator can't reach into a borrowed
        // backend statement.
        let (col_names, rows) = match backend::query(id, &sql, &params) {
            Ok(r) => r,
            Err(e) => {
                ctx(vm).runtime_error(format!("Sqlite.query: {}", e));
                set_return(vm, Value::null());
                return;
            }
        };

        // GC rooting: the previous implementation accumulated
        // `Value`s in a plain `Vec<Value>` on the host stack —
        // that Vec isn't a GC root, and any allocator call inside
        // the loop (`alloc_map`, `alloc_string`, `sql_to_wren`)
        // could trigger a collection that freed the partially-
        // built maps before they were handed back to Wren. Symptom
        // in production: catalog refresh allocates ~hundreds of
        // Maps in a tight loop; the GC threshold trips mid-rebuild;
        // a subsequent `byName` query lands on freed pointers and
        // SIGSEGVs (`hatch run … [catalog] refreshed (39 packages)
        // … zsh: segmentation fault`).
        //
        // Strategy: build the result list FIRST as the receiver/
        // return slot (api_stack[0], scanned by the GC's root set
        // — see `vm.rs` `roots.extend_from_slice(&self.api_stack)`)
        // and append each freshly-allocated map to it BEFORE
        // populating its fields. Once the map is reachable through
        // the result list's element array, subsequent
        // `alloc_string` / `sql_to_wren` calls can collect freely
        // without freeing the in-progress map — the list's element
        // array is itself reachable through the root, so the map
        // is transitively rooted.
        let context = ctx(vm);
        let result = context.alloc_list(Vec::new());
        set_return(vm, result);
        let result_ptr = result.as_object().unwrap() as *mut ObjList;

        for row_vals in rows {
            let map = context.alloc_map();
            let map_ptr = map.as_object().unwrap() as *mut ObjMap;
            // Root the map via the result list before populating
            // its fields. `map` as a local Rust binding isn't a
            // GC root; the list's element array is.
            (*result_ptr).add(map);
            for (i, val) in row_vals.into_iter().enumerate() {
                let key = context.alloc_string(col_names[i].clone());
                let wv = sql_to_wren(context, val);
                (*map_ptr).set(key, wv);
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_last_insert_rowid(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.lastInsertRowid") else {
            set_return(vm, Value::null());
            return;
        };
        match backend::last_insert_rowid(id) {
            Some(n) => set_return(vm, Value::num(n as f64)),
            None => {
                ctx(vm).runtime_error(format!(
                    "Sqlite.lastInsertRowid: unknown connection id {}.",
                    id
                ));
                set_return(vm, Value::null());
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_changes(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.changes") else {
            set_return(vm, Value::null());
            return;
        };
        match backend::changes(id) {
            Some(n) => set_return(vm, Value::num(n as f64)),
            None => {
                ctx(vm).runtime_error(format!("Sqlite.changes: unknown connection id {}.", id));
                set_return(vm, Value::null());
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wlift_sqlite_in_transaction(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.inTransaction") else {
            set_return(vm, Value::null());
            return;
        };
        match backend::in_transaction(id) {
            Some(b) => set_return(vm, Value::bool(b)),
            None => {
                ctx(vm).runtime_error(format!(
                    "Sqlite.inTransaction: unknown connection id {}.",
                    id
                ));
                set_return(vm, Value::null());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Static-link symbol registry (wasm only)
//
// On `wasm32-*`, plugins ship as Rust crates statically linked into
// `wlift_wasm` rather than as separate `.wasm` cdylibs — the wasm
// runtime has no `dlsym`, so foreign-method exports register
// themselves into a static table at host-init.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub fn register_static_symbols() {
    use wren_lift::runtime::foreign::register_plugin_symbol_unsafe;
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_open",              wlift_sqlite_open);
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_close",             wlift_sqlite_close);
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_execute",           wlift_sqlite_execute);
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_query",             wlift_sqlite_query);
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_last_insert_rowid", wlift_sqlite_last_insert_rowid);
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_changes",           wlift_sqlite_changes);
    register_plugin_symbol_unsafe("wlift_sqlite", "wlift_sqlite_in_transaction",    wlift_sqlite_in_transaction);
}
