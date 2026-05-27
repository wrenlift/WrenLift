//! Image plugin for WrenLift. Decodes PNG / JPG / BMP / WebP
//! into RGBA8 pixel buffers; encodes the same. Built as a cdylib
//! and bundled into @hatch:image — game + tooling pipelines all
//! flow through the same decoded representation.
//!
//! All functions return / accept Wren values plainly; the Image
//! class on the Wren side wraps `(width, height, pixels)` triples
//! so callers can pass them straight to @hatch:gpu's
//! `Device.uploadImage(image)` helper.

#![allow(clippy::missing_safety_doc)]

use std::io::Cursor;

use wren_lift::runtime::object::{
    NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType, ObjTypedArray, TypedArrayKind,
};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

/// Plugin ABI handshake — see wlift_gpu::wlift_plugin_abi_version.
///
/// Native-only. The host's `libloading::dlsym` path uses it to
/// reject a dylib built against a different `wren_lift` than the
/// host. Statically-linked wasm plugins can't drift versions
/// (everything's compiled from the same tree), and re-exporting
/// the symbol from every linked rlib produces wasm-linker
/// duplicate-symbol errors once a second plugin (physics / sqlite)
/// joins the static-link list.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn wlift_plugin_abi_version() -> u32 {
    wren_lift::runtime::foreign::WLIFT_PLUGIN_ABI_VERSION
}

// ---------------------------------------------------------------------------
// Slot helpers (mirror the other plugins)
// ---------------------------------------------------------------------------

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

unsafe fn ctx<'a>(vm: *mut VM) -> &'a mut VM {
    unsafe { &mut *vm }
}

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

/// Read a `List<Num>` or `ByteArray` into a `Vec<u8>`.
unsafe fn read_byte_buffer(vm: &mut VM, v: Value, label: &str) -> Option<Vec<u8>> {
    if !v.is_object() {
        vm.runtime_error(format!("{}: expected a List<Num> or ByteArray.", label));
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    match unsafe { (*header).obj_type } {
        ObjType::List => {
            let list = ptr as *const ObjList;
            let n = unsafe { (*list).count } as usize;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let entry = unsafe { *(*list).elements.add(i) };
                let v = match entry.as_num() {
                    Some(v) if (0.0..=255.0).contains(&v) && v.fract() == 0.0 => v as u8,
                    _ => {
                        vm.runtime_error(format!("{}: byte {} not in 0..=255.", label, i));
                        return None;
                    }
                };
                out.push(v);
            }
            Some(out)
        }
        ObjType::TypedArray => {
            let arr = ptr as *const ObjTypedArray;
            if unsafe { (*arr).kind_tag() } == TypedArrayKind::U8 {
                Some(unsafe { (*arr).as_bytes() }.to_vec())
            } else {
                vm.runtime_error(format!("{}: typed array must be ByteArray (u8).", label));
                None
            }
        }
        _ => {
            vm.runtime_error(format!("{}: expected a List<Num> or ByteArray.", label));
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
// `pixels` is a Wren ByteArray of RGBA8 in row-major order
// (image's `to_rgba8()`). Returning a typed array keeps the
// memory pinned in one allocation; callers that need it as a
// generic List can convert via `pixels.toList`.

#[no_mangle]
pub unsafe extern "C" fn wlift_image_decode(vm: *mut VM) {
    unsafe {
        let bytes = match read_byte_buffer(ctx(vm), slot(vm, 1), "Image.decode") {
            Some(b) => b,
            None => return,
        };

        let img = match image::load_from_memory(&bytes) {
            Ok(i) => i.to_rgba8(),
            Err(e) => {
                ctx(vm).runtime_error(format!("Image.decode: {}", e));
                return;
            }
        };
        let (w, h) = img.dimensions();
        let pixels = img.into_raw();

        // Drop into a Wren ByteArray so the Wren wrapper can hand
        // it straight to `Device.uploadImage` without copying back
        // through a List<Num>.
        let context = ctx(vm);
        let arr = context.alloc_typed_array(pixels.len() as u32, TypedArrayKind::U8);
        // Write pixels into the array via the typed-array API. We
        // cast back through the registered class to grab the raw
        // memory.
        if let Some(ptr) = arr.as_object() {
            let header = ptr as *const ObjHeader;
            if (*header).obj_type == ObjType::TypedArray {
                let ta = ptr as *mut ObjTypedArray;
                let dst = (*ta).as_bytes_mut();
                let n = dst.len().min(pixels.len());
                dst[..n].copy_from_slice(&pixels[..n]);
            }
        }

        // GC rooting: alloc the map first and drop it into slot 0
        // (the canonical return slot, scanned as a GC root via
        // `vm.api_stack`). The pre-allocated `arr` ByteArray is
        // unrooted in its local Rust binding, so we stash it on
        // `api_stack` as a temp root before any further alloc —
        // otherwise `alloc_string` for the keys could trigger a
        // collection that frees `arr` before we install it into
        // the map. Each key is `set` into the map immediately
        // after allocation; the map's hash table holds the key
        // string once written, so no separate roothold needed.
        // Direct `(*map_ptr).set` instead of `call_method_on(_,
        // "[_]=(_)", ...)` because method dispatch runs more
        // allocator-touching code between key alloc and value
        // commit.
        (*vm).api_stack.push(arr);
        let arr_slot = (*vm).api_stack.len() - 1;
        let map = context.alloc_map();
        set_return(vm, map);
        let map_ptr = map.as_object().unwrap() as *mut ObjMap;
        let kw = context.alloc_string("width".to_string());
        (*map_ptr).set(kw, Value::num(w as f64));
        let kh = context.alloc_string("height".to_string());
        (*map_ptr).set(kh, Value::num(h as f64));
        let kp = context.alloc_string("pixels".to_string());
        let arr_rooted = {
            let stack = &(*vm).api_stack;
            stack[arr_slot]
        };
        (*map_ptr).set(kp, arr_rooted);
        (*vm).api_stack.pop();
    }
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
pub unsafe extern "C" fn wlift_image_encode(vm: *mut VM) {
    unsafe {
        let fmt_str = match string_of(slot(vm, 1)) {
            Some(s) => s,
            None => {
                ctx(vm).runtime_error("Image.encode: format must be a string.".to_string());
                return;
            }
        };
        let fmt = match encode_format_from_str(&fmt_str) {
            Some(f) => f,
            None => {
                ctx(vm).runtime_error(format!(
                    "Image.encode: unknown format '{}' (png|jpeg|bmp).",
                    fmt_str
                ));
                return;
            }
        };
        let width = match slot(vm, 2).as_num() {
            Some(n) if n.is_finite() && n > 0.0 => n as u32,
            _ => {
                ctx(vm)
                    .runtime_error("Image.encode: width must be a positive integer.".to_string());
                return;
            }
        };
        let height = match slot(vm, 3).as_num() {
            Some(n) if n.is_finite() && n > 0.0 => n as u32,
            _ => {
                ctx(vm)
                    .runtime_error("Image.encode: height must be a positive integer.".to_string());
                return;
            }
        };
        let pixels = match read_byte_buffer(ctx(vm), slot(vm, 4), "Image.encode") {
            Some(b) => b,
            None => return,
        };
        let expected = (width as usize) * (height as usize) * 4;
        if pixels.len() != expected {
            ctx(vm).runtime_error(format!(
                "Image.encode: expected {} bytes (w*h*4), got {}.",
                expected,
                pixels.len()
            ));
            return;
        }
        let buf = match image::RgbaImage::from_raw(width, height, pixels) {
            Some(b) => b,
            None => {
                ctx(vm)
                    .runtime_error("Image.encode: pixel buffer could not be wrapped.".to_string());
                return;
            }
        };
        let mut out: Vec<u8> = Vec::new();
        if let Err(e) = buf.write_to(&mut Cursor::new(&mut out), fmt) {
            ctx(vm).runtime_error(format!("Image.encode: {}", e));
            return;
        }

        // Hand back a ByteArray with the encoded bytes.
        let context = ctx(vm);
        let arr = context.alloc_typed_array(out.len() as u32, TypedArrayKind::U8);
        if let Some(ptr) = arr.as_object() {
            let header = ptr as *const ObjHeader;
            if (*header).obj_type == ObjType::TypedArray {
                let ta = ptr as *mut ObjTypedArray;
                let dst = (*ta).as_bytes_mut();
                let n = dst.len().min(out.len());
                dst[..n].copy_from_slice(&out[..n]);
            }
        }
        set_return(vm, arr);
    }
}

// ---------------------------------------------------------------------------
// Static plugin registration (wasm path)
// ---------------------------------------------------------------------------
//
// On `wasm32-unknown-unknown` the runtime can't `dlopen` a cdylib;
// plugins ship as Rust crates that the runtime statically links
// in, then call this function once at host-init time to publish
// their symbols to the foreign-method registry. The host build
// uses `dlsym` instead, so this entire block is a no-op there.
//
// Hosts that include this crate (today: `wlift_wasm`) call
// `wlift_image::register_static_symbols()` from their `_wasm_init`
// shim — same place `console_error_panic_hook::set_once()` lives.

/// Register every `#[no_mangle] wlift_image_*` foreign-method
/// export under the library name `"wlift_image"`. Idempotent —
/// re-calling overwrites the same `(library, symbol)` keys.
///
/// Only exists on `wasm32-*` targets, where `wren_lift`'s
/// `foreign` module routes to the static registry. Host builds
/// resolve these symbols via `dlsym` and never call this; the
/// function isn't compiled there at all.
#[cfg(target_arch = "wasm32")]
pub fn register_static_symbols() {
    use wren_lift::runtime::foreign::register_plugin_symbol_unsafe;
    // SAFETY: each export is `unsafe extern "C" fn(*mut VM)` with
    // the same ABI the host build dispatches via `dlsym`; see the
    // safety contract on `register_plugin_symbol_unsafe`.
    register_plugin_symbol_unsafe("wlift_image", "wlift_image_decode", wlift_image_decode);
    register_plugin_symbol_unsafe("wlift_image", "wlift_image_encode", wlift_image_encode);
}
