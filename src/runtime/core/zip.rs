//! Optional `zip` module — read/write ZIP archives entirely in
//! memory. Paired with @hatch:zip on the Wren side.
//!
//! Byte conventions match the rest of the stdlib: inputs are
//! String (UTF-8) or List<Num in 0..=255>, outputs are always
//! List<Num>. Archives themselves are handled as byte lists so
//! callers can pair this with `FS.readBytes` / `FS.writeBytes`
//! (or with an in-memory pipeline) without touching the FS.
//!
//! Backed by the Rust `zip` crate with `deflate` + `zstd`.

use std::io::{Cursor, Read, Write};

use zip::read::ZipArchive;
use zip::write::{SimpleFileOptions, ZipWriter};
use zip::CompressionMethod;

use super::bytes_from_byte_list as bytes_from_list;
use super::bytes_from_value;
use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

fn bytes_to_list(ctx: &mut dyn NativeContext, bytes: &[u8]) -> Value {
    let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
    ctx.alloc_list(elements)
}

// --- Compression mapping ---------------------------------------

fn method_from_str(s: &str) -> Option<CompressionMethod> {
    match s {
        "store" | "stored" | "none" => Some(CompressionMethod::Stored),
        "deflate" | "deflated" => Some(CompressionMethod::Deflated),
        "zstd" | "zstandard" => Some(CompressionMethod::Zstd),
        _ => None,
    }
}

fn method_to_str(m: CompressionMethod) -> &'static str {
    match m {
        CompressionMethod::Stored => "store",
        CompressionMethod::Deflated => "deflate",
        CompressionMethod::Zstd => "zstd",
        _ => "other",
    }
}

// --- Read side --------------------------------------------------

fn zip_entries(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(archive_bytes) = bytes_from_list(ctx, args[1], "Zip.entries") else {
        return Value::null();
    };
    let cursor = Cursor::new(archive_bytes);
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            ctx.runtime_error(format!("Zip.entries: {}", e));
            return Value::null();
        }
    };
    let mut names: Vec<Value> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let file = match archive.by_index(i) {
            Ok(f) => f,
            Err(e) => {
                ctx.runtime_error(format!("Zip.entries: {}", e));
                return Value::null();
            }
        };
        names.push(ctx.alloc_string(file.name().to_string()));
    }
    ctx.alloc_list(names)
}

fn zip_info(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(archive_bytes) = bytes_from_list(ctx, args[1], "Zip.info") else {
        return Value::null();
    };
    let cursor = Cursor::new(archive_bytes);
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            ctx.runtime_error(format!("Zip.info: {}", e));
            return Value::null();
        }
    };
    let mut entries: Vec<Value> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let file = match archive.by_index(i) {
            Ok(f) => f,
            Err(e) => {
                ctx.runtime_error(format!("Zip.info: {}", e));
                return Value::null();
            }
        };
        let map_value = ctx.alloc_map();
        let map_ptr = map_value.as_object().unwrap() as *mut ObjMap;

        let k_name = ctx.alloc_string("name".to_string());
        let v_name = ctx.alloc_string(file.name().to_string());
        unsafe { (*map_ptr).set(k_name, v_name) };

        let k_size = ctx.alloc_string("size".to_string());
        let v_size = Value::num(file.size() as f64);
        unsafe { (*map_ptr).set(k_size, v_size) };

        let k_csize = ctx.alloc_string("compressedSize".to_string());
        let v_csize = Value::num(file.compressed_size() as f64);
        unsafe { (*map_ptr).set(k_csize, v_csize) };

        let k_isdir = ctx.alloc_string("isDir".to_string());
        let v_isdir = Value::bool(file.is_dir());
        unsafe { (*map_ptr).set(k_isdir, v_isdir) };

        let k_method = ctx.alloc_string("method".to_string());
        let v_method = ctx.alloc_string(method_to_str(file.compression()).to_string());
        unsafe { (*map_ptr).set(k_method, v_method) };

        entries.push(map_value);
    }
    ctx.alloc_list(entries)
}

fn zip_read_entry(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(archive_bytes) = bytes_from_list(ctx, args[1], "Zip.read") else {
        return Value::null();
    };
    let Some(name) = super::validate_string(ctx, args[2], "Zip.read (name)") else {
        return Value::null();
    };
    let cursor = Cursor::new(archive_bytes);
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            ctx.runtime_error(format!("Zip.read: {}", e));
            return Value::null();
        }
    };
    let mut file = match archive.by_name(&name) {
        Ok(f) => f,
        // Entry not found → null rather than abort, so callers can
        // treat it like a Map miss without error-handling scaffolding.
        Err(zip::result::ZipError::FileNotFound) => return Value::null(),
        Err(e) => {
            ctx.runtime_error(format!("Zip.read: {}", e));
            return Value::null();
        }
    };
    let mut out = Vec::with_capacity(file.size() as usize);
    if let Err(e) = file.read_to_end(&mut out) {
        ctx.runtime_error(format!("Zip.read: {}", e));
        return Value::null();
    }
    bytes_to_list(ctx, &out)
}

fn zip_read_all(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(archive_bytes) = bytes_from_list(ctx, args[1], "Zip.readAll") else {
        return Value::null();
    };
    let cursor = Cursor::new(archive_bytes);
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(e) => {
            ctx.runtime_error(format!("Zip.readAll: {}", e));
            return Value::null();
        }
    };
    // Preallocate each entry's bytes outside the Map construction so
    // we can `ctx.alloc_*` between reads without holding a pointer
    // into the archive across GC-possible allocations.
    let mut collected: Vec<(String, Vec<u8>)> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let mut file = match archive.by_index(i) {
            Ok(f) => f,
            Err(e) => {
                ctx.runtime_error(format!("Zip.readAll: {}", e));
                return Value::null();
            }
        };
        // Skip directory entries — they have no payload and would
        // collide with the file-name-as-key convention if a zip had
        // both `foo/` and `foo/bar`.
        if file.is_dir() {
            continue;
        }
        let name = file.name().to_string();
        let mut buf = Vec::with_capacity(file.size() as usize);
        if let Err(e) = file.read_to_end(&mut buf) {
            ctx.runtime_error(format!("Zip.readAll: {}", e));
            return Value::null();
        }
        collected.push((name, buf));
    }
    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;
    for (name, bytes) in collected {
        let key = ctx.alloc_string(name);
        let val = bytes_to_list(ctx, &bytes);
        unsafe { (*out_ptr).set(key, val) };
    }
    out
}

// --- Write side -------------------------------------------------

fn entries_from_value(
    ctx: &mut dyn NativeContext,
    value: Value,
    label: &str,
) -> Option<Vec<(String, Vec<u8>)>> {
    if !value.is_object() {
        ctx.runtime_error(format!("{}: expected a Map or List.", label));
        return None;
    }
    let ptr = value.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    let obj_type = unsafe { (*header).obj_type };
    match obj_type {
        ObjType::Map => {
            let m = ptr as *const ObjMap;
            // Snapshot entries so we don't hold the pointer while
            // recursing into `bytes_from_value` (which may GC).
            let raw: Vec<(Value, Value)> =
                unsafe { (*m).entries.iter().map(|(k, v)| (k.0, *v)).collect() };
            let mut out = Vec::with_capacity(raw.len());
            for (k, v) in raw {
                let name = match key_as_string(k) {
                    Some(s) => s,
                    None => {
                        ctx.runtime_error(format!(
                            "{}: entry names must be strings.",
                            label
                        ));
                        return None;
                    }
                };
                let bytes = bytes_from_value(ctx, v, label)?;
                out.push((name, bytes));
            }
            Some(out)
        }
        ObjType::List => {
            // List of [name, bytes] pairs preserves caller order.
            let lst = ptr as *const ObjList;
            let (count, data) = unsafe { ((*lst).count as usize, (*lst).elements) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let pair = unsafe { *data.add(i) };
                let Some(pair_ptr) = pair.as_object() else {
                    ctx.runtime_error(format!(
                        "{}: each entry must be a [name, bytes] list.",
                        label
                    ));
                    return None;
                };
                let pair_header = pair_ptr as *const ObjHeader;
                if unsafe { (*pair_header).obj_type } != ObjType::List {
                    ctx.runtime_error(format!(
                        "{}: each entry must be a [name, bytes] list.",
                        label
                    ));
                    return None;
                }
                let pair_list = pair_ptr as *const ObjList;
                let (pcount, pdata) =
                    unsafe { ((*pair_list).count as usize, (*pair_list).elements) };
                if pcount != 2 {
                    ctx.runtime_error(format!(
                        "{}: each entry must be [name, bytes] (2 elements).",
                        label
                    ));
                    return None;
                }
                let name_val = unsafe { *pdata.add(0) };
                let bytes_val = unsafe { *pdata.add(1) };
                let name = match key_as_string(name_val) {
                    Some(s) => s,
                    None => {
                        ctx.runtime_error(format!(
                            "{}: entry name must be a string.",
                            label
                        ));
                        return None;
                    }
                };
                let bytes = bytes_from_value(ctx, bytes_val, label)?;
                out.push((name, bytes));
            }
            Some(out)
        }
        _ => {
            ctx.runtime_error(format!("{}: expected a Map or List.", label));
            None
        }
    }
}

fn key_as_string(v: Value) -> Option<String> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::String {
            return None;
        }
        let s = ptr as *const ObjString;
        Some((*s).as_str().to_string())
    }
}

fn zip_write(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(entries) = entries_from_value(ctx, args[1], "Zip.write") else {
        return Value::null();
    };
    // Third arg is a method-name string (null → default = deflate).
    let method = if args[2].is_null() {
        CompressionMethod::Deflated
    } else {
        let Some(name) = super::validate_string(ctx, args[2], "Zip.write (method)") else {
            return Value::null();
        };
        match method_from_str(&name) {
            Some(m) => m,
            None => {
                ctx.runtime_error(format!(
                    "Zip.write: unknown compression method '{}' (try 'store', 'deflate', 'zstd').",
                    name
                ));
                return Value::null();
            }
        }
    };

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ZipWriter::new(Cursor::new(&mut buf));
        let options: SimpleFileOptions = SimpleFileOptions::default()
            .compression_method(method)
            .unix_permissions(0o644);
        for (name, bytes) in entries {
            if let Err(e) = writer.start_file(&name, options) {
                ctx.runtime_error(format!("Zip.write: {}", e));
                return Value::null();
            }
            if let Err(e) = writer.write_all(&bytes) {
                ctx.runtime_error(format!("Zip.write: {}", e));
                return Value::null();
            }
        }
        if let Err(e) = writer.finish() {
            ctx.runtime_error(format!("Zip.write: {}", e));
            return Value::null();
        }
    }
    bytes_to_list(ctx, &buf)
}

// --- Registration -----------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("ZipCore", vm.object_class);

    vm.primitive_static(class, "entries(_)", zip_entries);
    vm.primitive_static(class, "info(_)", zip_info);
    vm.primitive_static(class, "read(_,_)", zip_read_entry);
    vm.primitive_static(class, "readAll(_)", zip_read_all);
    vm.primitive_static(class, "write(_,_)", zip_write);

    class
}
