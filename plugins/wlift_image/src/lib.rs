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
    alloc_map, alloc_string, alloc_typed_array, list_count, list_get, map_set, obj_type, push_root,
    reload_root, roots_restore, roots_snapshot, runtime_error, set_return, slot, string_str,
    typed_array_bytes, typed_array_bytes_mut, typed_array_kind, ObjType, TypedArrayKind, Value,
    WrenVm,
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
unsafe fn read_byte_buffer(vm: *mut WrenVm, v: Value, label: &str) -> Option<Vec<u8>> {
    if !v.is_object() {
        runtime_error(
            vm,
            &format!("{}: expected a List<Num> or ByteArray.", label),
        );
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
                runtime_error(
                    vm,
                    &format!("{}: typed array must be ByteArray (u8).", label),
                );
                None
            }
        }
        _ => {
            runtime_error(
                vm,
                &format!("{}: expected a List<Num> or ByteArray.", label),
            );
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

    // The map and pixel array survive across multiple subsequent
    // allocator calls; a nursery GC during any of them forwards the
    // underlying objects, so any plain Rust local holding their
    // pre-GC NaN-boxed bits becomes a zombie. Use the JIT-roots
    // stack: snapshot on entry, push every survivor, reload before
    // use, restore on exit.

    let snap = roots_snapshot(vm);
    let map = alloc_map(vm);
    let map_r = push_root(vm, map);
    // Slot 0 is the return slot — keeps `map` rooted past the
    // foreign-call boundary even after `roots_restore`.
    set_return(vm, map);

    let arr = alloc_typed_array(vm, pixels.len() as u32, TypedArrayKind::U8);
    let arr_r = push_root(vm, arr);
    if let Some(buf) = typed_array_bytes_mut(reload_root(vm, arr_r)) {
        let n = buf.len().min(pixels.len());
        buf[..n].copy_from_slice(&pixels[..n]);
    }
    let kp = alloc_string(vm, "pixels");
    map_set(vm, reload_root(vm, map_r), kp, reload_root(vm, arr_r));

    let kw = alloc_string(vm, "width");
    map_set(vm, reload_root(vm, map_r), kw, Value::num(w as f64));

    let kh = alloc_string(vm, "height");
    map_set(vm, reload_root(vm, map_r), kh, Value::num(h as f64));

    roots_restore(vm, snap);
}

// ---------------------------------------------------------------------------
// Async decode (worker-thread JobRegistry)
// ---------------------------------------------------------------------------
//
// Synchronous `Image.decode` keeps the VM thread parked for the full
// PNG decode — multi-MB textures can pin the loop for seconds and
// freeze the window. The async path stages the decode onto a worker
// thread (one `std::thread::spawn` per job, bounded by typical
// asset loads of 10–500 images) and gives the caller a numeric job
// id to poll each frame.
//
// Job lifecycle:
//   decode_begin(bytes) -> id   Pending →  Ready{w,h,pixels} | Failed(msg)
//   poll(id) -> 0 | 1 | 2       0=Pending, 1=Ready, 2=Failed (Map-encoded
//                               so the Wren side reads a Num)
//   take(id) -> Map             drains a Ready slot, raises if not Ready
//   take_error(id) -> String    drains a Failed slot, raises if not Failed
//   cancel(id) -> null          drains either state without raising
//   inflight() -> Num           diagnostic — count of live entries
//
// The worker thread owns the decode (image::load_from_memory + .to_rgba8 +
// .into_raw) and writes the result back under JOBS.lock(). It never
// touches *mut WrenVm — VM-side allocation happens on the VM thread,
// inside `take`, mirroring the synchronous decode's GC-roots pattern.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

#[derive(Debug)]
enum JobState {
    Pending,
    Ready { w: u32, h: u32, pixels: Vec<u8> },
    Failed(String),
}

static JOBS: Lazy<Mutex<HashMap<u32, JobState>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static NEXT_JOB: AtomicU32 = AtomicU32::new(1);

fn alloc_job_id() -> u32 {
    // u32 wraps after ~4B jobs. For game asset loads (~10-500 per
    // session) the wrap is unreachable; document the cap. A
    // long-running server would want AtomicU64.
    NEXT_JOB.fetch_add(1, Ordering::Relaxed)
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_decode(job_id: u32, bytes: Vec<u8>) {
    std::thread::spawn(move || {
        // image::load_from_memory is Send-safe but can panic on
        // malformed input. Catch the unwind so a worker panic
        // doesn't poison JOBS.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            image::load_from_memory(&bytes).map(|i| i.to_rgba8())
        }));
        let state = match result {
            Ok(Ok(img)) => {
                let (w, h) = img.dimensions();
                JobState::Ready {
                    w,
                    h,
                    pixels: img.into_raw(),
                }
            }
            Ok(Err(e)) => JobState::Failed(format!("Image.decodeBegin: {}", e)),
            Err(_) => JobState::Failed("Image.decodeBegin: decoder panicked".to_string()),
        };
        // If the entry has already been cancelled the id is gone —
        // drop silently. Else overwrite Pending with the result.
        let mut jobs = JOBS.lock().unwrap();
        if jobs.contains_key(&job_id) {
            jobs.insert(job_id, state);
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn spawn_decode(job_id: u32, bytes: Vec<u8>) {
    // wasm32 has no std::thread::spawn. Run the decode synchronously
    // on the calling fiber so the API contract is preserved; the
    // browser-bridge fiber-yield path is the right asynchrony for
    // wasm and is tracked separately. The decode being inline here
    // means web callers see the same blocking time as if they had
    // called Image.decode directly — the value of decodeBegin on
    // wasm is purely API compatibility with native code paths.
    let state = match image::load_from_memory(&bytes) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            JobState::Ready {
                w,
                h,
                pixels: rgba.into_raw(),
            }
        }
        Err(e) => JobState::Failed(format!("Image.decodeBegin: {}", e)),
    };
    JOBS.lock().unwrap().insert(job_id, state);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_decode_begin(vm: *mut WrenVm) {
    let bytes = match read_byte_buffer(vm, slot(vm, 1), "Image.decodeBegin") {
        Some(b) => b,
        None => return,
    };
    let id = alloc_job_id();
    JOBS.lock().unwrap().insert(id, JobState::Pending);
    spawn_decode(id, bytes);
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_poll(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n.is_finite() && n >= 0.0 => n as u32,
        _ => {
            runtime_error(vm, "Image.pollDecode: id must be a non-negative number.");
            return;
        }
    };
    let jobs = JOBS.lock().unwrap();
    let code: f64 = match jobs.get(&id) {
        Some(JobState::Pending) => 0.0,
        Some(JobState::Ready { .. }) => 1.0,
        Some(JobState::Failed(_)) => 2.0,
        // Unknown id reads as "Failed" so the caller routes through
        // take_error; this catches double-drains + use-after-cancel
        // with one error path.
        None => 2.0,
    };
    set_return(vm, Value::num(code));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_take(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n.is_finite() && n >= 0.0 => n as u32,
        _ => {
            runtime_error(vm, "Image.takeDecode: id must be a non-negative number.");
            return;
        }
    };
    // Atomically remove only if Ready. If Pending we leave it so the
    // worker can still complete; if Failed we leave it so the caller
    // can route via takeDecodeError.
    let drained = {
        let mut jobs = JOBS.lock().unwrap();
        match jobs.get(&id) {
            Some(JobState::Ready { .. }) => jobs.remove(&id),
            _ => None,
        }
    };
    let (w, h, pixels) = match drained {
        Some(JobState::Ready { w, h, pixels }) => (w, h, pixels),
        _ => {
            runtime_error(
                vm,
                "Image.takeDecode: job is not Ready (still pending, failed, or already drained).",
            );
            return;
        }
    };

    // Allocator chain mirrors wlift_image_decode's GC-roots pattern.
    let snap = roots_snapshot(vm);
    let map = alloc_map(vm);
    let map_r = push_root(vm, map);
    set_return(vm, map);

    let arr = alloc_typed_array(vm, pixels.len() as u32, TypedArrayKind::U8);
    let arr_r = push_root(vm, arr);
    if let Some(buf) = typed_array_bytes_mut(reload_root(vm, arr_r)) {
        let n = buf.len().min(pixels.len());
        buf[..n].copy_from_slice(&pixels[..n]);
    }
    let kp = alloc_string(vm, "pixels");
    map_set(vm, reload_root(vm, map_r), kp, reload_root(vm, arr_r));

    let kw = alloc_string(vm, "width");
    map_set(vm, reload_root(vm, map_r), kw, Value::num(w as f64));

    let kh = alloc_string(vm, "height");
    map_set(vm, reload_root(vm, map_r), kh, Value::num(h as f64));

    roots_restore(vm, snap);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_take_error(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n.is_finite() && n >= 0.0 => n as u32,
        _ => {
            runtime_error(
                vm,
                "Image.takeDecodeError: id must be a non-negative number.",
            );
            return;
        }
    };
    let drained = {
        let mut jobs = JOBS.lock().unwrap();
        match jobs.get(&id) {
            Some(JobState::Failed(_)) => jobs.remove(&id),
            _ => None,
        }
    };
    let msg = match drained {
        Some(JobState::Failed(m)) => m,
        _ => "Image.takeDecodeError: job is not Failed (still pending, ready, or already drained)."
            .to_string(),
    };
    let s = alloc_string(vm, &msg);
    set_return(vm, s);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_cancel(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n.is_finite() && n >= 0.0 => n as u32,
        _ => {
            runtime_error(vm, "Image.cancelDecode: id must be a non-negative number.");
            return;
        }
    };
    JOBS.lock().unwrap().remove(&id);
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_image_inflight(vm: *mut WrenVm) {
    let n = JOBS.lock().unwrap().len();
    set_return(vm, Value::num(n as f64));
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
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_decode_begin",
            wlift_image_decode_begin as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_poll",
            wlift_image_poll as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_take",
            wlift_image_take as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_take_error",
            wlift_image_take_error as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_cancel",
            wlift_image_cancel as *const (),
        );
        wlift_abi::register_symbol(
            "wlift_image",
            "wlift_image_inflight",
            wlift_image_inflight as *const (),
        );
    }
}
