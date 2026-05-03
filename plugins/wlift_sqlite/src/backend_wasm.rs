//! Wasm sqlite backend — `sqlite-wasm-rs`'s raw C bindings.
//!
//! sqlite-wasm-rs ships SQLite compiled to `wasm32-unknown-unknown`
//! with `-DSQLITE_THREADSAFE=0` and a memory VFS. The browser
//! playground is single-threaded, so the connection registry uses
//! a Mutex purely for API parity with the native backend; lock
//! contention is impossible.
//!
//! `*mut sqlite3` is `!Send + !Sync` by default; we wrap it in a
//! `Conn` newtype with unsafe Send/Sync impls. Sound here because
//! the wasm runtime is single-threaded — the wrapper never
//! actually crosses a thread boundary, the bounds just satisfy
//! `Mutex<HashMap<u64, Conn>>`.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use sqlite_wasm_rs as ffi;

use crate::{Params, SqlValue};

#[repr(transparent)]
struct Conn(*mut ffi::sqlite3);
unsafe impl Send for Conn {}
unsafe impl Sync for Conn {}

fn registry() -> &'static Mutex<HashMap<u64, Conn>> {
    static REG: OnceLock<Mutex<HashMap<u64, Conn>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

unsafe fn err_msg(db: *mut ffi::sqlite3) -> String {
    let raw = unsafe { ffi::sqlite3_errmsg(db) };
    if raw.is_null() {
        return "(no message)".into();
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(raw) };
    cstr.to_string_lossy().into_owned()
}

unsafe fn finalize(stmt: *mut ffi::sqlite3_stmt) {
    if !stmt.is_null() {
        unsafe { ffi::sqlite3_finalize(stmt) };
    }
}

unsafe fn prepare(db: *mut ffi::sqlite3, sql: &str) -> Result<*mut ffi::sqlite3_stmt, String> {
    let cs = CString::new(sql).map_err(|_| "sql contains an interior NUL byte".to_string())?;
    let mut stmt: *mut ffi::sqlite3_stmt = ptr::null_mut();
    let rc = unsafe {
        ffi::sqlite3_prepare_v2(db, cs.as_ptr(), -1, &mut stmt as *mut _, ptr::null_mut())
    };
    if rc != ffi::SQLITE_OK {
        return Err(format!("prepare: {}", unsafe { err_msg(db) }));
    }
    Ok(stmt)
}

// SQLITE_TRANSIENT tells SQLite to copy text/blob payloads
// instead of holding the caller's pointer. The constant is
// `((sqlite3_destructor_type)-1)`; in Rust we forge it as a
// function-pointer with -1 cast.
fn sqlite_transient() -> ffi::sqlite3_destructor_type {
    Some(unsafe { std::mem::transmute::<isize, unsafe extern "C" fn(*mut c_void)>(-1) })
}

unsafe fn bind_one(stmt: *mut ffi::sqlite3_stmt, idx: c_int, v: &SqlValue) -> Result<(), String> {
    let rc = match v {
        SqlValue::Null => unsafe { ffi::sqlite3_bind_null(stmt, idx) },
        SqlValue::Integer(i) => unsafe { ffi::sqlite3_bind_int64(stmt, idx, *i) },
        SqlValue::Real(f) => unsafe { ffi::sqlite3_bind_double(stmt, idx, *f) },
        SqlValue::Text(s) => {
            let bytes = s.as_bytes();
            unsafe {
                ffi::sqlite3_bind_text(
                    stmt,
                    idx,
                    bytes.as_ptr() as *const c_char,
                    bytes.len() as c_int,
                    sqlite_transient(),
                )
            }
        }
        SqlValue::Blob(b) => unsafe {
            ffi::sqlite3_bind_blob(
                stmt,
                idx,
                b.as_ptr() as *const c_void,
                b.len() as c_int,
                sqlite_transient(),
            )
        },
    };
    if rc != ffi::SQLITE_OK {
        return Err(format!("bind index {}: rc={}", idx, rc));
    }
    Ok(())
}

unsafe fn bind_params(stmt: *mut ffi::sqlite3_stmt, params: &Params) -> Result<(), String> {
    match params {
        Params::None => Ok(()),
        Params::Positional(vals) => {
            for (i, v) in vals.iter().enumerate() {
                unsafe { bind_one(stmt, (i + 1) as c_int, v)? };
            }
            Ok(())
        }
        Params::Named(vals) => {
            for (name, v) in vals {
                let cname = CString::new(name.as_str())
                    .map_err(|_| "param name contains NUL".to_string())?;
                let idx = unsafe { ffi::sqlite3_bind_parameter_index(stmt, cname.as_ptr()) };
                if idx == 0 {
                    return Err(format!("unknown named parameter: {}", name));
                }
                unsafe { bind_one(stmt, idx, v)? };
            }
            Ok(())
        }
    }
}

unsafe fn read_column(stmt: *mut ffi::sqlite3_stmt, i: c_int) -> SqlValue {
    let ty = unsafe { ffi::sqlite3_column_type(stmt, i) };
    match ty {
        ffi::SQLITE_NULL => SqlValue::Null,
        ffi::SQLITE_INTEGER => SqlValue::Integer(unsafe { ffi::sqlite3_column_int64(stmt, i) }),
        ffi::SQLITE_FLOAT => SqlValue::Real(unsafe { ffi::sqlite3_column_double(stmt, i) }),
        ffi::SQLITE_TEXT => {
            let bytes = unsafe { ffi::sqlite3_column_bytes(stmt, i) };
            let ptr = unsafe { ffi::sqlite3_column_text(stmt, i) };
            if ptr.is_null() || bytes <= 0 {
                SqlValue::Text(String::new())
            } else {
                let slice = unsafe { std::slice::from_raw_parts(ptr, bytes as usize) };
                SqlValue::Text(String::from_utf8_lossy(slice).into_owned())
            }
        }
        ffi::SQLITE_BLOB => {
            let bytes = unsafe { ffi::sqlite3_column_bytes(stmt, i) };
            let ptr = unsafe { ffi::sqlite3_column_blob(stmt, i) } as *const u8;
            if ptr.is_null() || bytes <= 0 {
                SqlValue::Blob(Vec::new())
            } else {
                let slice = unsafe { std::slice::from_raw_parts(ptr, bytes as usize) };
                SqlValue::Blob(slice.to_vec())
            }
        }
        _ => SqlValue::Null,
    }
}

pub(crate) fn open(path: &str) -> Result<u64, String> {
    // Memory VFS only — the playground doesn't need OPFS yet, and
    // `:memory:` round-trips identically to native rusqlite.
    let cs = CString::new(path).map_err(|_| "path contains NUL".to_string())?;
    let mut db: *mut ffi::sqlite3 = ptr::null_mut();
    let rc = unsafe {
        ffi::sqlite3_open_v2(
            cs.as_ptr(),
            &mut db as *mut _,
            ffi::SQLITE_OPEN_READWRITE | ffi::SQLITE_OPEN_CREATE,
            ptr::null(),
        )
    };
    if rc != ffi::SQLITE_OK {
        let msg = if db.is_null() {
            format!("sqlite3_open_v2 rc={}", rc)
        } else {
            let m = unsafe { err_msg(db) };
            unsafe { ffi::sqlite3_close(db) };
            m
        };
        return Err(msg);
    }
    let id = next_id();
    registry().lock().unwrap().insert(id, Conn(db));
    Ok(id)
}

pub(crate) fn close(id: u64) {
    if let Some(Conn(db)) = registry().lock().unwrap().remove(&id) {
        unsafe { ffi::sqlite3_close(db) };
    }
}

pub(crate) fn execute(id: u64, sql: &str, params: &Params) -> Result<usize, String> {
    let reg = registry().lock().unwrap();
    let Conn(db) = reg
        .get(&id)
        .ok_or_else(|| format!("unknown connection id {}.", id))?;
    let db = *db;
    let stmt = unsafe { prepare(db, sql)? };
    let result = (|| unsafe {
        bind_params(stmt, params)?;
        loop {
            let rc = ffi::sqlite3_step(stmt);
            match rc {
                ffi::SQLITE_DONE => break,
                ffi::SQLITE_ROW => continue, // ignore rows from execute()
                _ => return Err(format!("step: {}", err_msg(db))),
            }
        }
        Ok(ffi::sqlite3_changes(db) as usize)
    })();
    unsafe { finalize(stmt) };
    result
}

pub(crate) fn query(
    id: u64,
    sql: &str,
    params: &Params,
) -> Result<(Vec<String>, Vec<Vec<SqlValue>>), String> {
    let reg = registry().lock().unwrap();
    let Conn(db) = reg
        .get(&id)
        .ok_or_else(|| format!("unknown connection id {}.", id))?;
    let db = *db;
    let stmt = unsafe { prepare(db, sql)? };
    let result = (|| unsafe {
        bind_params(stmt, params)?;
        let col_count = ffi::sqlite3_column_count(stmt);
        let mut col_names = Vec::with_capacity(col_count as usize);
        for i in 0..col_count {
            let raw = ffi::sqlite3_column_name(stmt, i);
            let name = if raw.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(raw).to_string_lossy().into_owned()
            };
            col_names.push(name);
        }
        let mut rows_out: Vec<Vec<SqlValue>> = Vec::new();
        loop {
            let rc = ffi::sqlite3_step(stmt);
            match rc {
                ffi::SQLITE_ROW => {
                    let mut row_vals = Vec::with_capacity(col_count as usize);
                    for i in 0..col_count {
                        row_vals.push(read_column(stmt, i));
                    }
                    rows_out.push(row_vals);
                }
                ffi::SQLITE_DONE => break,
                _ => return Err(format!("step: {}", err_msg(db))),
            }
        }
        Ok((col_names, rows_out))
    })();
    unsafe { finalize(stmt) };
    result
}

pub(crate) fn last_insert_rowid(id: u64) -> Option<i64> {
    let reg = registry().lock().unwrap();
    let Conn(db) = reg.get(&id)?;
    Some(unsafe { ffi::sqlite3_last_insert_rowid(*db) })
}

pub(crate) fn changes(id: u64) -> Option<u64> {
    let reg = registry().lock().unwrap();
    let Conn(db) = reg.get(&id)?;
    Some(unsafe { ffi::sqlite3_changes(*db) } as u64)
}

pub(crate) fn in_transaction(id: u64) -> Option<bool> {
    let reg = registry().lock().unwrap();
    let Conn(db) = reg.get(&id)?;
    Some(unsafe { ffi::sqlite3_get_autocommit(*db) } == 0)
}
