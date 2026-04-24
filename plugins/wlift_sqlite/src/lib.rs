//! SQLite plugin for WrenLift. Built as a `cdylib`; the resulting
//! `libwlift_sqlite.dylib` (or `.so` / `.dll`) is bundled into
//! `@hatch:sqlite` as a `NativeLib` section. At install time the
//! runtime extracts the dylib to a temp dir, `dlopen`s it, and
//! binds each `foreign` method on `SqliteCore` to the matching
//! symbol below.
//!
//! The plugin uses `wren_lift` as a path dependency, so Wren's
//! `Value` and `VM` types are ABI-identical between the main
//! binary and this dylib. Global state is the SQLite connection
//! registry — private to this dylib and fine that way (no
//! sharing across platform boundaries).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use rusqlite::{types::Value as SqlValue, types::ValueRef, Connection, OpenFlags};

use wren_lift::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

// --- Registry --------------------------------------------------

fn registry() -> &'static Mutex<HashMap<u64, Connection>> {
    static REG: OnceLock<Mutex<HashMap<u64, Connection>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
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
        (&(*vm).api_stack)
            .get(index)
            .copied()
            .unwrap_or(Value::null())
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

fn sql_to_wren(ctx: &mut dyn NativeContext, v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::null(),
        ValueRef::Integer(i) => Value::num(i as f64),
        ValueRef::Real(f) => Value::num(f),
        ValueRef::Text(bytes) => {
            let s = String::from_utf8_lossy(bytes).into_owned();
            ctx.alloc_string(s)
        }
        ValueRef::Blob(bytes) => {
            let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
            ctx.alloc_list(elements)
        }
    }
}

enum Params {
    None,
    Positional(Vec<SqlValue>),
    Named(Vec<(String, SqlValue)>),
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

fn bind_params(stmt: &mut rusqlite::Statement<'_>, params: &Params) -> rusqlite::Result<()> {
    match params {
        Params::None => Ok(()),
        Params::Positional(vals) => {
            for (i, v) in vals.iter().enumerate() {
                stmt.raw_bind_parameter(i + 1, v)?;
            }
            Ok(())
        }
        Params::Named(vals) => {
            for (name, v) in vals {
                match stmt.parameter_index(name)? {
                    Some(idx) => stmt.raw_bind_parameter(idx, v)?,
                    None => {
                        return Err(rusqlite::Error::InvalidParameterName(name.clone()));
                    }
                }
            }
            Ok(())
        }
    }
}

// --- Foreign C entry points ------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_open(vm: *mut VM) {
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
        let conn = if path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open_with_flags(
                &path,
                OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
            )
        };
        match conn {
            Ok(c) => {
                let id = next_id();
                registry().lock().unwrap().insert(id, c);
                set_return(vm, Value::num(id as f64));
            }
            Err(e) => {
                ctx(vm).runtime_error(format!("Sqlite.open: {}", e));
                set_return(vm, Value::null());
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_close(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.close") else {
            set_return(vm, Value::null());
            return;
        };
        registry().lock().unwrap().remove(&id);
        set_return(vm, Value::null());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_execute(vm: *mut VM) {
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
        let mut reg = registry().lock().unwrap();
        let Some(conn) = reg.get_mut(&id) else {
            ctx(vm).runtime_error(format!("Sqlite.execute: unknown connection id {}.", id));
            set_return(vm, Value::null());
            return;
        };
        let mut stmt = match conn.prepare_cached(&sql) {
            Ok(s) => s,
            Err(e) => {
                ctx(vm).runtime_error(format!("Sqlite.execute: prepare: {}", e));
                set_return(vm, Value::null());
                return;
            }
        };
        if let Err(e) = bind_params(&mut stmt, &params) {
            ctx(vm).runtime_error(format!("Sqlite.execute: bind: {}", e));
            set_return(vm, Value::null());
            return;
        }
        match stmt.raw_execute() {
            Ok(rows) => set_return(vm, Value::num(rows as f64)),
            Err(e) => {
                ctx(vm).runtime_error(format!("Sqlite.execute: {}", e));
                set_return(vm, Value::null());
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_query(vm: *mut VM) {
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

        // Build up the row list by collecting rows into a Vec of
        // (col_names, values) pairs FIRST, then walk it to alloc
        // into Wren objects. This avoids holding the connection
        // mutex while calling alloc_string / alloc_map, which
        // might trigger a GC that reaches into wren_lift state.
        let rows_result: Result<(Vec<String>, Vec<Vec<SqlValue>>), String> = (|| {
            let mut reg = registry().lock().unwrap();
            let conn = reg
                .get_mut(&id)
                .ok_or_else(|| format!("Sqlite.query: unknown connection id {}.", id))?;
            let mut stmt = conn
                .prepare_cached(&sql)
                .map_err(|e| format!("Sqlite.query: prepare: {}", e))?;
            bind_params(&mut stmt, &params).map_err(|e| format!("Sqlite.query: bind: {}", e))?;
            let col_count = stmt.column_count();
            let col_names: Vec<String> = (0..col_count)
                .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                .collect();
            let mut rows_out: Vec<Vec<SqlValue>> = Vec::new();
            let mut rows = stmt.raw_query();
            while let Some(row) = rows.next().map_err(|e| format!("Sqlite.query: {}", e))? {
                let mut row_vals = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    let vref = row.get_ref(i).map_err(|e| format!("Sqlite.query: {}", e))?;
                    // Clone out of the borrow so we can drop the row.
                    let v: SqlValue = match vref {
                        ValueRef::Null => SqlValue::Null,
                        ValueRef::Integer(i) => SqlValue::Integer(i),
                        ValueRef::Real(f) => SqlValue::Real(f),
                        ValueRef::Text(b) => {
                            SqlValue::Text(String::from_utf8_lossy(b).into_owned())
                        }
                        ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
                    };
                    row_vals.push(v);
                }
                rows_out.push(row_vals);
            }
            Ok((col_names, rows_out))
        })();

        let (col_names, rows) = match rows_result {
            Ok(r) => r,
            Err(e) => {
                ctx(vm).runtime_error(e);
                set_return(vm, Value::null());
                return;
            }
        };

        let context = ctx(vm);
        let mut out_values: Vec<Value> = Vec::with_capacity(rows.len());
        for row_vals in rows {
            let map = context.alloc_map();
            let map_ptr = map.as_object().unwrap() as *mut ObjMap;
            for (i, val) in row_vals.into_iter().enumerate() {
                let key = context.alloc_string(col_names[i].clone());
                // `sql_to_wren` wants a ValueRef, but we've already
                // cloned into owned SqlValue — inline the conversion.
                let wv = match val {
                    SqlValue::Null => Value::null(),
                    SqlValue::Integer(i) => Value::num(i as f64),
                    SqlValue::Real(f) => Value::num(f),
                    SqlValue::Text(s) => context.alloc_string(s),
                    SqlValue::Blob(b) => {
                        let elems: Vec<Value> = b.iter().map(|&x| Value::num(x as f64)).collect();
                        context.alloc_list(elems)
                    }
                };
                let _ = sql_to_wren; // keep import live for future helpers
                (*map_ptr).set(key, wv);
            }
            out_values.push(map);
        }
        let result = context.alloc_list(out_values);
        set_return(vm, result);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_last_insert_rowid(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.lastInsertRowid") else {
            set_return(vm, Value::null());
            return;
        };
        let reg = registry().lock().unwrap();
        let Some(conn) = reg.get(&id) else {
            ctx(vm).runtime_error(format!(
                "Sqlite.lastInsertRowid: unknown connection id {}.",
                id
            ));
            set_return(vm, Value::null());
            return;
        };
        set_return(vm, Value::num(conn.last_insert_rowid() as f64));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_changes(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.changes") else {
            set_return(vm, Value::null());
            return;
        };
        let reg = registry().lock().unwrap();
        let Some(conn) = reg.get(&id) else {
            ctx(vm).runtime_error(format!("Sqlite.changes: unknown connection id {}.", id));
            set_return(vm, Value::null());
            return;
        };
        set_return(vm, Value::num(conn.changes() as f64));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wlift_sqlite_in_transaction(vm: *mut VM) {
    unsafe {
        let id_val = slot(vm, 1);
        let Some(id) = id_of(ctx(vm), id_val, "Sqlite.inTransaction") else {
            set_return(vm, Value::null());
            return;
        };
        let reg = registry().lock().unwrap();
        let Some(conn) = reg.get(&id) else {
            ctx(vm).runtime_error(format!(
                "Sqlite.inTransaction: unknown connection id {}.",
                id
            ));
            set_return(vm, Value::null());
            return;
        };
        set_return(vm, Value::bool(!conn.is_autocommit()));
    }
}
