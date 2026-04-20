//! Optional `fs` module — exposes a `FS` class with the primitive
//! filesystem operations Wren's core lacks. `@hatch:fs` wraps these
//! with path normalization, error-kind classification, and other
//! ergonomic touches.
//!
//! Loaded on-demand when `import "fs" for FS` is encountered.
//!
//! Everything here is synchronous — Wren has no async runtime, and
//! cooperative yielding across blocking I/O is the caller's job
//! (spawn a Fiber, Fiber.yield, etc.).

use std::fs;
use std::path::Path;

use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Strings ------------------------------------------------------------

/// FS.readText(path) → String. Aborts the fiber on I/O error.
fn fs_read_text(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    match fs::read_to_string(&path) {
        Ok(s) => ctx.alloc_string(s),
        Err(e) => {
            ctx.runtime_error(format!("FS.readText: {}: {}", path, e));
            Value::null()
        }
    }
}

/// FS.writeText(path, content). Creates the file if absent,
/// truncates otherwise. Aborts on error.
fn fs_write_text(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    let Some(content) = super::validate_string(ctx, args[2], "Content") else {
        return Value::null();
    };
    if let Err(e) = fs::write(&path, content.as_bytes()) {
        ctx.runtime_error(format!("FS.writeText: {}: {}", path, e));
    }
    Value::null()
}

// --- Bytes --------------------------------------------------------------

/// FS.readBytes(path) → List of Num (0–255). Useful for binary
/// files where UTF-8 decoding would mangle the bytes.
fn fs_read_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    match fs::read(&path) {
        Ok(bytes) => {
            let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
            ctx.alloc_list(elements)
        }
        Err(e) => {
            ctx.runtime_error(format!("FS.readBytes: {}: {}", path, e));
            Value::null()
        }
    }
}

/// FS.writeBytes(path, list). Each list entry must be an integer
/// in 0..=255; non-integers abort.
fn fs_write_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    let list_val = args[2];
    let list_ptr = match list_val.as_object() {
        Some(p) => p as *const crate::runtime::object::ObjList,
        None => {
            ctx.runtime_error("FS.writeBytes: Bytes must be a list.".to_string());
            return Value::null();
        }
    };
    let (count, ptr) = unsafe { ((*list_ptr).count as usize, (*list_ptr).elements) };
    let mut bytes = Vec::with_capacity(count);
    for i in 0..count {
        let v = unsafe { *ptr.add(i) };
        let n = match v.as_num() {
            Some(n) => n,
            None => {
                ctx.runtime_error(
                    "FS.writeBytes: every byte must be a number in 0..=255.".to_string(),
                );
                return Value::null();
            }
        };
        if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
            ctx.runtime_error(
                "FS.writeBytes: every byte must be an integer in 0..=255.".to_string(),
            );
            return Value::null();
        }
        bytes.push(n as u8);
    }
    if let Err(e) = fs::write(&path, &bytes) {
        ctx.runtime_error(format!("FS.writeBytes: {}: {}", path, e));
    }
    Value::null()
}

// --- Metadata -----------------------------------------------------------

fn fs_exists(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = validate_path(args[1]) else {
        return Value::bool(false);
    };
    Value::bool(Path::new(&path).exists())
}

fn fs_is_file(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = validate_path(args[1]) else {
        return Value::bool(false);
    };
    Value::bool(Path::new(&path).is_file())
}

fn fs_is_dir(_ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = validate_path(args[1]) else {
        return Value::bool(false);
    };
    Value::bool(Path::new(&path).is_dir())
}

/// FS.size(path) → Num. File byte count. Aborts if the entry is
/// missing or unreadable.
fn fs_size(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    match fs::metadata(&path) {
        Ok(md) => Value::num(md.len() as f64),
        Err(e) => {
            ctx.runtime_error(format!("FS.size: {}: {}", path, e));
            Value::null()
        }
    }
}

// --- Directories --------------------------------------------------------

/// FS.listDir(path) → List of String (entry basenames, sorted).
fn fs_list_dir(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    let read = match fs::read_dir(&path) {
        Ok(r) => r,
        Err(e) => {
            ctx.runtime_error(format!("FS.listDir: {}: {}", path, e));
            return Value::null();
        }
    };
    let mut names: Vec<String> = Vec::new();
    for entry in read.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            names.push(name.to_string());
        }
    }
    // Sorted output so callers don't depend on platform-specific
    // ordering (ext4 insertion order, HFS+ case-insensitive, …).
    names.sort();
    let values: Vec<Value> = names.into_iter().map(|n| ctx.alloc_string(n)).collect();
    ctx.alloc_list(values)
}

fn fs_mkdir(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    if let Err(e) = fs::create_dir(&path) {
        ctx.runtime_error(format!("FS.mkdir: {}: {}", path, e));
    }
    Value::null()
}

fn fs_mkdirs(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    if let Err(e) = fs::create_dir_all(&path) {
        ctx.runtime_error(format!("FS.mkdirs: {}: {}", path, e));
    }
    Value::null()
}

/// FS.remove(path). Removes a single file, symlink, or empty dir.
/// Use `removeTree` for recursive deletion.
fn fs_remove(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    let p = Path::new(&path);
    let result = if p.is_dir() && !p.is_symlink() {
        fs::remove_dir(p)
    } else {
        fs::remove_file(p)
    };
    if let Err(e) = result {
        ctx.runtime_error(format!("FS.remove: {}: {}", path, e));
    }
    Value::null()
}

fn fs_remove_tree(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(path) = super::validate_string(ctx, args[1], "Path") else {
        return Value::null();
    };
    if let Err(e) = fs::remove_dir_all(&path) {
        ctx.runtime_error(format!("FS.removeTree: {}: {}", path, e));
    }
    Value::null()
}

fn fs_rename(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(from) = super::validate_string(ctx, args[1], "From") else {
        return Value::null();
    };
    let Some(to) = super::validate_string(ctx, args[2], "To") else {
        return Value::null();
    };
    if let Err(e) = fs::rename(&from, &to) {
        ctx.runtime_error(format!("FS.rename: {} -> {}: {}", from, to, e));
    }
    Value::null()
}

// --- Process paths ------------------------------------------------------

fn fs_cwd(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    match std::env::current_dir() {
        Ok(p) => ctx.alloc_string(p.to_string_lossy().into_owned()),
        Err(e) => {
            ctx.runtime_error(format!("FS.cwd: {}", e));
            Value::null()
        }
    }
}

fn fs_home(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    match std::env::var_os("HOME") {
        Some(h) => ctx.alloc_string(h.to_string_lossy().into_owned()),
        None => {
            ctx.runtime_error("FS.home: HOME not set.".to_string());
            Value::null()
        }
    }
}

fn fs_tmp_dir(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let p = std::env::temp_dir();
    ctx.alloc_string(p.to_string_lossy().into_owned())
}

// --- Helpers ------------------------------------------------------------

/// Like `validate_string` but for predicates that want a `false`
/// result instead of a runtime error on non-string input.
fn validate_path(value: Value) -> Option<String> {
    if !value.is_object() {
        return None;
    }
    let ptr = value.as_object()?;
    let header = ptr as *const crate::runtime::object::ObjHeader;
    unsafe {
        if (*header).obj_type != crate::runtime::object::ObjType::String {
            return None;
        }
    }
    let s = ptr as *const crate::runtime::object::ObjString;
    Some(unsafe { (*s).as_str().to_string() })
}

// --- Registration -------------------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("FS", vm.object_class);

    vm.primitive_static(class, "readText(_)", fs_read_text);
    vm.primitive_static(class, "writeText(_,_)", fs_write_text);
    vm.primitive_static(class, "readBytes(_)", fs_read_bytes);
    vm.primitive_static(class, "writeBytes(_,_)", fs_write_bytes);

    vm.primitive_static(class, "exists(_)", fs_exists);
    vm.primitive_static(class, "isFile(_)", fs_is_file);
    vm.primitive_static(class, "isDir(_)", fs_is_dir);
    vm.primitive_static(class, "size(_)", fs_size);

    vm.primitive_static(class, "listDir(_)", fs_list_dir);
    vm.primitive_static(class, "mkdir(_)", fs_mkdir);
    vm.primitive_static(class, "mkdirs(_)", fs_mkdirs);
    vm.primitive_static(class, "remove(_)", fs_remove);
    vm.primitive_static(class, "removeTree(_)", fs_remove_tree);
    vm.primitive_static(class, "rename(_,_)", fs_rename);

    vm.primitive_static(class, "cwd", fs_cwd);
    vm.primitive_static(class, "home", fs_home);
    vm.primitive_static(class, "tmpDir", fs_tmp_dir);

    class
}
