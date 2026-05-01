//! Native sqlite backend — `rusqlite` with bundled libsqlite3.
//!
//! Owns a per-process `HashMap<u64, Connection>` keyed by the
//! u64 connection ids the plugin hands out to Wren. Statements
//! are recompiled per call (`prepare_cached` keeps the second
//! call fast); the registry mutex is held for the duration of
//! each operation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use rusqlite::{types::Value as RusqValue, types::ValueRef, Connection, OpenFlags};

use crate::{Params, SqlValue};

fn registry() -> &'static Mutex<HashMap<u64, Connection>> {
    static REG: OnceLock<Mutex<HashMap<u64, Connection>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

fn to_rusq(v: &SqlValue) -> RusqValue {
    match v {
        SqlValue::Null => RusqValue::Null,
        SqlValue::Integer(i) => RusqValue::Integer(*i),
        SqlValue::Real(f) => RusqValue::Real(*f),
        SqlValue::Text(s) => RusqValue::Text(s.clone()),
        SqlValue::Blob(b) => RusqValue::Blob(b.clone()),
    }
}

fn from_value_ref(v: ValueRef<'_>) -> SqlValue {
    match v {
        ValueRef::Null => SqlValue::Null,
        ValueRef::Integer(i) => SqlValue::Integer(i),
        ValueRef::Real(f) => SqlValue::Real(f),
        ValueRef::Text(b) => SqlValue::Text(String::from_utf8_lossy(b).into_owned()),
        ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
    }
}

fn bind_params(stmt: &mut rusqlite::Statement<'_>, params: &Params) -> rusqlite::Result<()> {
    match params {
        Params::None => Ok(()),
        Params::Positional(vals) => {
            for (i, v) in vals.iter().enumerate() {
                stmt.raw_bind_parameter(i + 1, to_rusq(v))?;
            }
            Ok(())
        }
        Params::Named(vals) => {
            for (name, v) in vals {
                match stmt.parameter_index(name)? {
                    Some(idx) => stmt.raw_bind_parameter(idx, to_rusq(v))?,
                    None => return Err(rusqlite::Error::InvalidParameterName(name.clone())),
                }
            }
            Ok(())
        }
    }
}

pub(crate) fn open(path: &str) -> Result<u64, String> {
    let conn = if path == ":memory:" {
        Connection::open_in_memory()
    } else {
        Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
    }
    .map_err(|e| e.to_string())?;
    let id = next_id();
    registry().lock().unwrap().insert(id, conn);
    Ok(id)
}

pub(crate) fn close(id: u64) {
    registry().lock().unwrap().remove(&id);
}

pub(crate) fn execute(id: u64, sql: &str, params: &Params) -> Result<usize, String> {
    let mut reg = registry().lock().unwrap();
    let conn = reg
        .get_mut(&id)
        .ok_or_else(|| format!("unknown connection id {}.", id))?;
    let mut stmt = conn
        .prepare_cached(sql)
        .map_err(|e| format!("prepare: {}", e))?;
    bind_params(&mut stmt, params).map_err(|e| format!("bind: {}", e))?;
    stmt.raw_execute().map_err(|e| e.to_string())
}

pub(crate) fn query(
    id: u64,
    sql: &str,
    params: &Params,
) -> Result<(Vec<String>, Vec<Vec<SqlValue>>), String> {
    let mut reg = registry().lock().unwrap();
    let conn = reg
        .get_mut(&id)
        .ok_or_else(|| format!("unknown connection id {}.", id))?;
    let mut stmt = conn
        .prepare_cached(sql)
        .map_err(|e| format!("prepare: {}", e))?;
    bind_params(&mut stmt, params).map_err(|e| format!("bind: {}", e))?;
    let col_count = stmt.column_count();
    let col_names: Vec<String> = (0..col_count)
        .map(|i| stmt.column_name(i).unwrap_or("").to_string())
        .collect();
    let mut rows_out: Vec<Vec<SqlValue>> = Vec::new();
    let mut rows = stmt.raw_query();
    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let mut row_vals = Vec::with_capacity(col_count);
        for i in 0..col_count {
            let vref = row.get_ref(i).map_err(|e| e.to_string())?;
            row_vals.push(from_value_ref(vref));
        }
        rows_out.push(row_vals);
    }
    Ok((col_names, rows_out))
}

pub(crate) fn last_insert_rowid(id: u64) -> Option<i64> {
    let reg = registry().lock().unwrap();
    reg.get(&id).map(|c| c.last_insert_rowid())
}

pub(crate) fn changes(id: u64) -> Option<u64> {
    let reg = registry().lock().unwrap();
    reg.get(&id).map(|c| c.changes())
}

pub(crate) fn in_transaction(id: u64) -> Option<bool> {
    let reg = registry().lock().unwrap();
    reg.get(&id).map(|c| !c.is_autocommit())
}
