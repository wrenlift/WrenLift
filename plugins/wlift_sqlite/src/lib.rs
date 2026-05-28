//! SQLite plugin for WrenLift. Built as a `cdylib` for native
//! hosts (the runtime `dlopen`s it from `@hatch:sqlite`'s
//! `NativeLib` section) and as an `rlib` for the wasm static-link
//! path (linked into `wlift_wasm` so the browser playground can
//! `import "@hatch:sqlite"` without a separate `.wasm` artifact).
//!
//! Two backends sit behind a thin module-level switch:
//!
//!   * Native: `backend_native` uses `rusqlite` with bundled
//!     libsqlite3. `Connection::open_with_flags`, prepared
//!     statements, the `Statement::raw_*` family.
//!
//!   * Wasm: `backend_wasm` uses `sqlite-wasm-rs` — the official
//!     SQLite-Wasm C build wrapped as `wasm32-unknown-unknown`
//!     bindings.
//!
//! All host interaction goes through `wlift_abi`: a header-only
//! Rust crate that declares the `wlift_plugin_*` C surface the
//! host binary exports. No `wren_lift` Rust internals reach this
//! cdylib, so a `VM` / `GcImpl` / `ObjList` layout change in the
//! host can't silently SIGSEGV plugin code anymore.
//!
//! # Safety
//!
//! Every `wlift_sqlite_*` entry point is called by the runtime
//! with a valid `*mut WrenVm` that stays live for the duration of
//! the call. The runtime is also responsible for providing
//! well-formed slot arguments — we validate them defensively
//! but trust that the pointer itself dereferences inside the
//! host's `wlift_plugin_*` helpers.

#![allow(clippy::missing_safety_doc)]

use wlift_abi::{
    alloc_list, alloc_map, alloc_string, alloc_typed_array, list_add, list_count, list_get,
    map_iter, map_set, obj_type, runtime_error, set_return, slot, string_str, typed_array_bytes,
    typed_array_bytes_mut, typed_array_kind, ObjType, TypedArrayKind, Value, WrenVm,
};

#[cfg(not(target_arch = "wasm32"))]
mod backend_native;
#[cfg(not(target_arch = "wasm32"))]
use backend_native as backend;

#[cfg(target_arch = "wasm32")]
mod backend_wasm;
#[cfg(target_arch = "wasm32")]
use backend_wasm as backend;

/// Plugin ABI handshake. The host's plugin loader calls this
/// at `dlopen` and compares against its own `PLUGIN_ABI_VERSION`;
/// mismatch aborts cleanly with a "rebuild against wren_lift ≥ X"
/// message instead of the silent SIGSEGV that struct-layout drift
/// used to produce.
///
/// Native-only — statically-linked wasm plugins can't drift versions
/// (everything's compiled from the same tree).
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    wlift_abi::ABI_VERSION
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

// --- Type coercion ---------------------------------------------

fn string_of(v: Value) -> Option<String> {
    string_str(v).map(|s| s.to_string())
}

fn id_of(vm: *mut WrenVm, v: Value, label: &str) -> Option<u64> {
    match v.as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Some(n as u64),
        _ => {
            runtime_error(
                vm,
                &format!("{}: id must be a non-negative integer.", label),
            );
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
    match obj_type(v) {
        Some(ObjType::String) => Ok(SqlValue::Text(string_of(v).unwrap_or_default())),
        // ByteArray is the canonical blob carrier — `sql_to_wren`
        // returns one for every BLOB column read, so a round-trip
        // (`db.query(...)["b"]` → bind back) doesn't need a manual
        // List<Num> conversion in Wren.
        Some(ObjType::TypedArray) => {
            if typed_array_kind(v) != Some(TypedArrayKind::U8) {
                return Err(format!(
                    "{}: BLOB parameter must be a ByteArray (got {:?}).",
                    label,
                    typed_array_kind(v),
                ));
            }
            let bytes = typed_array_bytes(v).map(|b| b.to_vec()).unwrap_or_default();
            Ok(SqlValue::Blob(bytes))
        }
        // Backwards-compatible List<Num> path. Pre-ByteArray
        // versions of `@hatch:sqlite` accepted blobs as a Wren
        // List of integers in 0..=255; existing call sites
        // (`db.execute("INSERT … blob = ?", [[0xde, 0xad]])`) keep
        // working unchanged. New code should prefer ByteArray.
        Some(ObjType::List) => {
            let count = list_count(v);
            let mut bytes = Vec::with_capacity(count as usize);
            for i in 0..count {
                let elem = list_get(v, i);
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

fn sql_to_wren(vm: *mut WrenVm, v: SqlValue) -> Value {
    match v {
        SqlValue::Null => Value::NULL,
        SqlValue::Integer(i) => Value::num(i as f64),
        SqlValue::Real(f) => Value::num(f),
        SqlValue::Text(s) => alloc_string(vm, &s),
        // BLOB → ByteArray. Half the memory of a List<Num> per
        // element (1 byte vs 8-byte boxed Value), zero per-element
        // GC overhead, and a `Response.bytes(...)`-friendly shape
        // for callers that want to forward the column verbatim.
        SqlValue::Blob(b) => {
            let arr = alloc_typed_array(vm, b.len() as u32, TypedArrayKind::U8);
            if !b.is_empty() {
                if let Some(buf) = typed_array_bytes_mut(arr) {
                    buf.copy_from_slice(&b);
                }
            }
            arr
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
    match obj_type(v) {
        Some(ObjType::List) => {
            let count = list_count(v);
            let mut out = Vec::with_capacity(count as usize);
            for i in 0..count {
                out.push(wren_to_sql(list_get(v, i), label)?);
            }
            Ok(Params::Positional(out))
        }
        Some(ObjType::Map) => {
            // Snapshot entries before any allocation so iteration
            // can't be invalidated by GC moves later in the loop.
            let entries: Vec<(Value, Value)> = map_iter(v).collect();
            let mut out = Vec::with_capacity(entries.len());
            for (k, val) in entries {
                let key = match string_of(k) {
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
                out.push((key, wren_to_sql(val, label)?));
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

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_open(vm: *mut WrenVm) {
    let path_val = slot(vm, 1);
    let path = match string_of(path_val) {
        Some(s) => s,
        None => {
            runtime_error(vm, "Sqlite.open: path must be a string.");
            set_return(vm, Value::NULL);
            return;
        }
    };
    match backend::open(&path) {
        Ok(id) => set_return(vm, Value::num(id as f64)),
        Err(e) => {
            runtime_error(vm, &format!("Sqlite.open: {}", e));
            set_return(vm, Value::NULL);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_close(vm: *mut WrenVm) {
    let id_val = slot(vm, 1);
    let Some(id) = id_of(vm, id_val, "Sqlite.close") else {
        set_return(vm, Value::NULL);
        return;
    };
    backend::close(id);
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_execute(vm: *mut WrenVm) {
    let id_val = slot(vm, 1);
    let sql_val = slot(vm, 2);
    let params_val = slot(vm, 3);
    let Some(id) = id_of(vm, id_val, "Sqlite.execute") else {
        set_return(vm, Value::NULL);
        return;
    };
    let sql = match string_of(sql_val) {
        Some(s) => s,
        None => {
            runtime_error(vm, "Sqlite.execute: sql must be a string.");
            set_return(vm, Value::NULL);
            return;
        }
    };
    let params = match collect_params(params_val, "Sqlite.execute") {
        Ok(p) => p,
        Err(e) => {
            runtime_error(vm, &e);
            set_return(vm, Value::NULL);
            return;
        }
    };
    match backend::execute(id, &sql, &params) {
        Ok(rows) => set_return(vm, Value::num(rows as f64)),
        Err(e) => {
            runtime_error(vm, &format!("Sqlite.execute: {}", e));
            set_return(vm, Value::NULL);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_query(vm: *mut WrenVm) {
    let id_val = slot(vm, 1);
    let sql_val = slot(vm, 2);
    let params_val = slot(vm, 3);
    let Some(id) = id_of(vm, id_val, "Sqlite.query") else {
        set_return(vm, Value::NULL);
        return;
    };
    let sql = match string_of(sql_val) {
        Some(s) => s,
        None => {
            runtime_error(vm, "Sqlite.query: sql must be a string.");
            set_return(vm, Value::NULL);
            return;
        }
    };
    let params = match collect_params(params_val, "Sqlite.query") {
        Ok(p) => p,
        Err(e) => {
            runtime_error(vm, &e);
            set_return(vm, Value::NULL);
            return;
        }
    };

    // Materialize rows BEFORE alloc_string / alloc_map so a GC
    // inside the allocator can't reach into a borrowed backend
    // statement.
    let (col_names, rows) = match backend::query(id, &sql, &params) {
        Ok(r) => r,
        Err(e) => {
            runtime_error(vm, &format!("Sqlite.query: {}", e));
            set_return(vm, Value::NULL);
            return;
        }
    };

    // GC rooting: build the result list FIRST as the receiver/
    // return slot (api_stack[0], scanned by the GC's root set) and
    // append each freshly-allocated map to it via `list_add` BEFORE
    // populating the map's fields. Once a map is reachable through
    // the result list, subsequent `alloc_string` / `sql_to_wren`
    // calls can collect freely — the list's element array is
    // itself reachable through the root, so the map is
    // transitively rooted.
    //
    // The `wlift_plugin_list_add` host helper handles the GC-safe
    // append (the old `(*list).add(v)` direct field write had the
    // same semantics, just via private Rust internals). We re-read
    // the list from api_stack[0] each iteration in case a previous
    // allocation promoted it across the nursery/old-gen boundary.
    let result = alloc_list(vm, 0);
    set_return(vm, result);

    for row_vals in rows {
        let map = alloc_map(vm);
        // alloc_map may have moved `result` — refresh from slot 0.
        let result_list = slot(vm, 0);
        list_add(vm, result_list, map);
        let last_idx = list_count(result_list) - 1;
        for (i, val) in row_vals.into_iter().enumerate() {
            let key = alloc_string(vm, &col_names[i]);
            let wv = sql_to_wren(vm, val);
            // Both alloc_string and sql_to_wren may have moved the
            // list and the in-progress map. Re-derive the map from
            // the list's last element each iteration.
            let result_list = slot(vm, 0);
            let map_v = list_get(result_list, last_idx);
            map_set(vm, map_v, key, wv);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_last_insert_rowid(vm: *mut WrenVm) {
    let id_val = slot(vm, 1);
    let Some(id) = id_of(vm, id_val, "Sqlite.lastInsertRowid") else {
        set_return(vm, Value::NULL);
        return;
    };
    match backend::last_insert_rowid(id) {
        Some(n) => set_return(vm, Value::num(n as f64)),
        None => {
            runtime_error(
                vm,
                &format!("Sqlite.lastInsertRowid: unknown connection id {}.", id),
            );
            set_return(vm, Value::NULL);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_changes(vm: *mut WrenVm) {
    let id_val = slot(vm, 1);
    let Some(id) = id_of(vm, id_val, "Sqlite.changes") else {
        set_return(vm, Value::NULL);
        return;
    };
    match backend::changes(id) {
        Some(n) => set_return(vm, Value::num(n as f64)),
        None => {
            runtime_error(
                vm,
                &format!("Sqlite.changes: unknown connection id {}.", id),
            );
            set_return(vm, Value::NULL);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_sqlite_in_transaction(vm: *mut WrenVm) {
    let id_val = slot(vm, 1);
    let Some(id) = id_of(vm, id_val, "Sqlite.inTransaction") else {
        set_return(vm, Value::NULL);
        return;
    };
    match backend::in_transaction(id) {
        Some(b) => set_return(vm, Value::bool(b)),
        None => {
            runtime_error(
                vm,
                &format!("Sqlite.inTransaction: unknown connection id {}.", id),
            );
            set_return(vm, Value::NULL);
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
    unsafe {
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_open",
            wlift_sqlite_open as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_close",
            wlift_sqlite_close as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_execute",
            wlift_sqlite_execute as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_query",
            wlift_sqlite_query as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_last_insert_rowid",
            wlift_sqlite_last_insert_rowid as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_changes",
            wlift_sqlite_changes as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_sqlite",
            "wlift_sqlite_in_transaction",
            wlift_sqlite_in_transaction as *const (),
        );
    }
}
