//! `wlift_gpu_web` — thin C-ABI WebGPU plugin for the WrenLift
//! game framework.
//!
//! Architecture
//! ------------
//!
//! This is `wlift_window_web`'s pattern applied to GPU work:
//!
//!   * No `wgpu` crate. No `wasm-bindgen`. No `web-sys`.
//!   * Every capability is a wasm import provided by the JS-side
//!     loader, which talks to `navigator.gpu` directly.
//!   * Shaders pass through as raw WGSL strings — the browser
//!     compiles them.
//!
//! The MVP surface here covers "clear the canvas to a color"
//! — five foreign methods. Vertex buffers / pipelines / shaders
//! are added in the next layer (3b).
//!
//! Resource handles
//! ----------------
//!
//! Every WebGPU object the bridge owns (GPUDevice, GPUSurface,
//! per-frame state) is referenced by a `u32` handle. The plugin
//! treats handles as opaque integers; the JS side maintains the
//! actual `Map<handle, GPUObject>`.
//!
//! Async device init
//! -----------------
//!
//! `navigator.gpu.requestAdapter()` is async, but Wren-side
//! foreign methods are synchronous. The JS loader pre-resolves
//! the device at plugin-install time (the install step is itself
//! async); `Gpu.init()` returns the cached result.

#![allow(clippy::missing_safety_doc)]

use core::ffi::c_void;

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    // Wren C API passthrough — same set wlift_window_web uses.
    fn wrenGetSlotDouble(vm: *mut c_void, slot: i32) -> f64;
    fn wrenSetSlotDouble(vm: *mut c_void, slot: i32, value: f64);
    fn wrenSetSlotBool(vm: *mut c_void, slot: i32, value: bool);
    fn wrenSetSlotNull(vm: *mut c_void, slot: i32);
}

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    // GPU bridge — JS-side owns the GPUDevice / GPUSurface /
    // per-frame state. Plugin only sees integer handles.
    //
    // Convention: bridge imports use the `host_*` prefix to
    // distinguish them from plugin exports (`wlift_<lib>_*`).
    // Same naming wlift_window_web uses (`wlift_dom_*` over
    // there).

    /// 1 if a GPU device is ready (adapter+device acquired at
    /// plugin install). 0 if WebGPU is unavailable or init
    /// failed; the plugin surfaces that as `false` from
    /// `Gpu.init()`.
    fn host_gpu_ready() -> i32;

    /// Attach a WebGPU rendering context to a canvas (the
    /// `canvas_handle` is the integer handle wlift_window_web
    /// returned). Configures the context with the device's
    /// preferred format. Returns a `surface` handle (≥1) or 0
    /// on failure.
    fn host_gpu_attach_canvas(canvas_handle: u32) -> u32;

    /// Acquire the next frame's `GPUTexture` from the surface,
    /// open a command encoder, and stash both keyed by the
    /// returned `frame` handle. Returns 0 on failure (surface
    /// not configured / lost).
    fn host_gpu_begin_frame(surface_handle: u32) -> u32;

    /// Record a single clear-color render pass against the
    /// frame's texture and end the pass. Components are 0..1.
    fn host_gpu_clear(frame_handle: u32, r: f32, g: f32, b: f32, a: f32);

    /// Submit the frame's encoder to the device queue and
    /// release the per-frame state. The browser presents the
    /// canvas automatically once the queue settles.
    fn host_gpu_end_frame(frame_handle: u32);
}

// ---------------------------------------------------------------
// Foreign methods
// ---------------------------------------------------------------

/// `Gpu.init_()` — returns Bool. `true` if WebGPU is available
/// and the adapter+device pair was acquired at install time.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_init(vm: *mut c_void) {
    let ok = unsafe { host_gpu_ready() } != 0;
    unsafe { wrenSetSlotBool(vm, 0, ok) };
}

/// `Gpu.attachCanvas_(canvasHandle)` — returns Num (the surface
/// handle) or `-1` on failure.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_attach_canvas(vm: *mut c_void) {
    let canvas = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let surface = unsafe { host_gpu_attach_canvas(canvas) };
    let result = if surface == 0 { -1.0 } else { surface as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.beginFrame_(surfaceHandle)` — returns Num (the frame
/// handle) or `-1` on failure.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_begin_frame(vm: *mut c_void) {
    let surface = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let frame = unsafe { host_gpu_begin_frame(surface) };
    let result = if frame == 0 { -1.0 } else { frame as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.clear_(frameHandle, r, g, b, a)` — records and ends a
/// clear-color render pass. Returns null.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_clear(vm: *mut c_void) {
    let frame = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let r = unsafe { wrenGetSlotDouble(vm, 2) } as f32;
    let g = unsafe { wrenGetSlotDouble(vm, 3) } as f32;
    let b = unsafe { wrenGetSlotDouble(vm, 4) } as f32;
    let a = unsafe { wrenGetSlotDouble(vm, 5) } as f32;
    unsafe { host_gpu_clear(frame, r, g, b, a) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

/// `Gpu.endFrame_(frameHandle)` — submits the queue, presents.
/// Returns null.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_end_frame(vm: *mut c_void) {
    let frame = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    unsafe { host_gpu_end_frame(frame) };
    unsafe { wrenSetSlotNull(vm, 0) };
}
