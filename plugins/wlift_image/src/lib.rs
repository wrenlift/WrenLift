//! Image plugin for WrenLift. Decodes PNG / JPG / BMP / WebP
//! into RGBA8 pixel buffers; encodes the same. Built as a cdylib
//! and bundled into @hatch:image — game + tooling pipelines all
//! flow through the same decoded representation.
//!
//! All host interaction goes through `wlift_abi`, the header-only
//! ABI shim crate. The cdylib has no `wren_lift` Rust dep — a
//! `VM` / `Obj*` layout change in the host can't silently SIGSEGV
//! this plugin anymore.

#![allow(clippy::missing_safety_doc)]

use std::io::Cursor;

use wlift_abi::{
    alloc_map, alloc_string, alloc_typed_array, list_count, list_get, map_set, obj_type,
    runtime_error, set_return, slot, string_str, typed_array_bytes, typed_array_bytes_mut,
    typed_array_kind, ObjType, TypedArrayKind, Value, WrenVm,
};

/// Plugin ABI handshake. Native-only: statically-linked wasm
/// plugins can't drift versions (everything's compiled from the
/// same tree), and re-exporting the symbol from every linked
/// rlib produces wasm-linker duplicate-symbol errors.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    wlift_abi::ABI_VERSION
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn string_of(v: Value) -> Option<String> {
    string_str(v).map(|s| s.to_string())
}

/// Read a `List<Num>` or `ByteArray` into a `Vec<u8>`.
fn read_byte_buffer(vm: *mut WrenVm, v: Value, label: &str) -> Option<Vec<u8>> {
    if !v.is_object() {
        runtime_error(vm, &format!("{}: expected a List<Num> or ByteArray.", label));
        return None;
    }
    match obj_type(v) {
        Some(ObjType::List) => {
            let n = list_count(v);
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                let entry = list_get(v, i);
                let byte = match entry.as_num() {
                    Some(v) if (0.0..=255.0).contains(&v) && v.fract() == 0.0 => v as u8,
                    _ => {
                        runtime_error(vm, &format!("{}: byte {} not in 0..=255.", label, i));
                        return None;
                    }
                };
                out.push(byte);
            }
            Some(out)
        }
        Some(ObjType::TypedArray) => {
            if typed_array_kind(v) == Some(TypedArrayKind::U8) {
                Some(typed_array_bytes(v).map(|b| b.to_vec()).unwrap_or_default())
            } else {
                runtime_error(vm, &format!("{}: typed array must be ByteArray (u8).", label));
                None
            }
        }
        _ => {
            runtime_error(vm, &format!("{}: expected a List<Num> or ByteArray.", label));
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------
//
// `wlift_image_decode(bytes) -> Map { "width", "height", "pixels" }`
//
// `pixels` is a Wren ByteArray of RGBA8 in row-major order. Returning
// a typed array keeps the memory pinned in one allocation; callers
// that need it as a generic List can convert via `pixels.toList`.

#[no_mangle]
pub unsafe extern "C" fn wlift_image_decode(vm: *mut WrenVm) {
    let bytes = match read_byte_buffer(vm, slot(vm, 1), "Image.decode") {
        Some(b) => b,
        None => return,
    };

    let img = match image::load_from_memory(&bytes) {
        Ok(i) => i.to_rgba8(),
        Err(e) => {
            runtime_error(vm, &format!("Image.decode: {}", e));
            return;
        }
    };
    let (w, h) = img.dimensions();
    let pixels = img.into_raw();

    // GC rooting strategy: every allocation goes through the host's
    // `finish_alloc`, which pushes the freshly-allocated Value onto
    // the GC's `JIT_ROOTS_STORE`. Locals here borrow rooted values
    // until the foreign call returns. Build the map first as the
    // return-slot root, then fill it — every key/value alloc stays
    // reachable through finish_alloc plus the map's own field
    // pointers once installed.

    let map = alloc_map(vm);
    set_return(vm, map);

    // Pixels ByteArray. Allocated AFTER the map exists so a GC fired
    // during alloc_typed_array can't free the unrooted map; the
    // return-slot write above keeps `map` reachable.
    let arr = alloc_typed_array(vm, pixels.len() as u32, TypedArrayKind::U8);
    if let Some(buf) = typed_array_bytes_mut(arr) {
        let n = buf.len().min(pixels.len());
        buf[..n].copy_from_slice(&pixels[..n]);
    }
    // Install pixels first — once it's in the map, the only root
    // we need to maintain is the map itself.
    let kp = alloc_string(vm, "pixels");
    map_set(vm, map, kp, arr);

    let kw = alloc_string(vm, "width");
    map_set(vm, map, kw, Value::num(w as f64));

    let kh = alloc_string(vm, "height");
    map_set(vm, map, kh, Value::num(h as f64));
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------
//
// `wlift_image_encode(format, width, height, pixels) -> ByteArray`
//
// `format` is "png" | "jpeg" | "bmp". `pixels` is RGBA8 row-major;
// dimensions are explicit so the caller can encode partial views
// without reshaping. Returns a fresh ByteArray.

fn encode_format_from_str(s: &str) -> Option<image::ImageFormat> {
    match s {
        "png" => Some(image::ImageFormat::Png),
        "jpeg" | "jpg" => Some(image::ImageFormat::Jpeg),
        "bmp" => Some(image::ImageFormat::Bmp),
        _ => None,
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_encode(vm: *mut WrenVm) {
    let fmt_str = match string_of(slot(vm, 1)) {
        Some(s) => s,
        None => {
            runtime_error(vm, "Image.encode: format must be a string.");
            return;
        }
    };
    let fmt = match encode_format_from_str(&fmt_str) {
        Some(f) => f,
        None => {
            runtime_error(
                vm,
                &format!("Image.encode: unknown format '{}' (png|jpeg|bmp).", fmt_str),
            );
            return;
        }
    };
    let width = match slot(vm, 2).as_num() {
        Some(n) if n.is_finite() && n > 0.0 => n as u32,
        _ => {
            runtime_error(vm, "Image.encode: width must be a positive integer.");
            return;
        }
    };
    let height = match slot(vm, 3).as_num() {
        Some(n) if n.is_finite() && n > 0.0 => n as u32,
        _ => {
            runtime_error(vm, "Image.encode: height must be a positive integer.");
            return;
        }
    };
    let pixels = match read_byte_buffer(vm, slot(vm, 4), "Image.encode") {
        Some(b) => b,
        None => return,
    };
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.len() != expected {
        runtime_error(
            vm,
            &format!(
                "Image.encode: expected {} bytes (w*h*4), got {}.",
                expected,
                pixels.len()
            ),
        );
        return;
    }
    let buf = match image::RgbaImage::from_raw(width, height, pixels) {
        Some(b) => b,
        None => {
            runtime_error(vm, "Image.encode: pixel buffer could not be wrapped.");
            return;
        }
    };
    let mut out: Vec<u8> = Vec::new();
    if let Err(e) = buf.write_to(&mut Cursor::new(&mut out), fmt) {
        runtime_error(vm, &format!("Image.encode: {}", e));
        return;
    }

    let arr = alloc_typed_array(vm, out.len() as u32, TypedArrayKind::U8);
    if let Some(dst) = typed_array_bytes_mut(arr) {
        let n = dst.len().min(out.len());
        dst[..n].copy_from_slice(&out[..n]);
    }
    set_return(vm, arr);
}

// ---------------------------------------------------------------------------
// Static plugin registration (wasm path)
// ---------------------------------------------------------------------------
//
// On `wasm32-*` the runtime can't `dlopen` a cdylib; plugins ship
// as Rust crates that the runtime statically links in, then call
// this once at host-init time to publish their symbols to the
// foreign-method registry. Host builds resolve via `dlsym` and
// never call this.

#[cfg(target_arch = "wasm32")]
pub fn register_static_symbols() {
    unsafe {
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_decode",
            wlift_image_decode as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_encode",
            wlift_image_encode as *const (),
        );
    }
}
