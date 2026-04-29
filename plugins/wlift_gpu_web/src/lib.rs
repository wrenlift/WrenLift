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
    fn wrenGetSlotBytes(vm: *mut c_void, slot: i32, length: *mut i32) -> *const u8;
}

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    // Slot bridges for descriptor JSON / WGSL passing. The host
    // owns the cross-memory copy.
    fn wlift_get_slot_str(vm: *mut c_void, slot: i32, out: *mut u8, max: i32) -> i32;
    fn wlift_set_slot_str(vm: *mut c_void, slot: i32, ptr: *const u8, len: i32);
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

    // ---- Buffers ----------------------------------------------
    fn host_gpu_create_buffer(size: u32, usage_bits: u32) -> u32;
    /// Bridge reads bytes directly from the host-side Wren slot —
    /// no plugin-memory boundary. Pass the VM + slot containing
    /// a Wren String / Bytes value.
    fn host_gpu_buffer_write(vm: *mut c_void, buffer_handle: u32, offset: u32, slot: i32);
    fn host_gpu_destroy_buffer(handle: u32);

    /// Wren-side ergonomics — given a List of Nums, pack as a
    /// `Float32Array` and create a buffer in one shot. Avoids
    /// the byte-packing dance Wren would otherwise need.
    fn host_gpu_create_buffer_from_f32(vm: *mut c_void, slot: i32, usage_bits: u32) -> u32;
    fn host_gpu_buffer_write_f32(vm: *mut c_void, buffer_handle: u32, offset: u32, slot: i32);
    fn host_gpu_buffer_write_u32(vm: *mut c_void, buffer_handle: u32, offset: u32, slot: i32);

    // ---- Shaders ----------------------------------------------
    /// WGSL source comes from the slot — same slot-based pattern
    /// as buffer_write.
    fn host_gpu_create_shader(vm: *mut c_void, slot: i32) -> u32;

    // ---- Pipelines / bind groups (descriptor JSON in slots) ---
    fn host_gpu_create_bind_group_layout(vm: *mut c_void, slot: i32) -> u32;
    fn host_gpu_create_bind_group(vm: *mut c_void, slot: i32) -> u32;
    fn host_gpu_create_pipeline(vm: *mut c_void, slot: i32) -> u32;

    // ---- Render pass commands --------------------------------
    /// Begin a render pass. `desc_slot` carries an optional JSON
    /// descriptor; pass `-1` to use defaults (clear to opaque
    /// black, single color attachment from the frame texture).
    fn host_gpu_render_pass_begin(
        vm: *mut c_void,
        frame_handle: u32,
        desc_slot: i32,
    ) -> u32;
    fn host_gpu_render_pass_set_pipeline(pass: u32, pipeline: u32);
    fn host_gpu_render_pass_set_bind_group(pass: u32, group_index: u32, bg: u32);
    fn host_gpu_render_pass_set_vertex_buffer(pass: u32, slot: u32, buffer: u32);
    fn host_gpu_render_pass_set_index_buffer(pass: u32, buffer: u32, format32: i32);
    fn host_gpu_render_pass_draw(
        pass: u32,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    );
    fn host_gpu_render_pass_draw_indexed(
        pass: u32,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    );
    fn host_gpu_render_pass_end(pass: u32);

    // ---- Textures + samplers ----------------------------------
    /// JSON descriptor in `slot`: { width, height, format,
    /// usage, dimension?, mipLevelCount?, sampleCount? }.
    /// Returns texture handle (≥1) or 0.
    fn host_gpu_create_texture(vm: *mut c_void, slot: i32) -> u32;
    /// Build a `GPUTextureView`. `desc_slot` is optional — pass
    /// `-1` for the default view (covers the whole texture).
    fn host_gpu_create_texture_view(
        vm: *mut c_void,
        texture_handle: u32,
        desc_slot: i32,
    ) -> u32;
    /// Upload pixel bytes via `device.queue.writeTexture`.
    /// `bytes_slot` carries a Wren String/Bytes; `desc_slot`
    /// carries the layout / origin descriptor JSON.
    fn host_gpu_queue_write_texture(
        vm: *mut c_void,
        texture_handle: u32,
        bytes_slot: i32,
        desc_slot: i32,
    ) -> u32;
    /// JSON descriptor (or `-1` for defaults). Returns sampler
    /// handle (≥1) or 0.
    fn host_gpu_create_sampler(vm: *mut c_void, desc_slot: i32) -> u32;

    // ---- Generic destroy --------------------------------------
    fn host_gpu_destroy(handle: u32);
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

// ---------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------

/// `Gpu.createBuffer_(size, usageBits)` — returns Num (handle) or
/// `-1`. `usageBits` follows GPUBufferUsage flags (1=MAP_READ,
/// 2=MAP_WRITE, 4=COPY_SRC, 8=COPY_DST, 16=INDEX, 32=VERTEX,
/// 64=UNIFORM, 128=STORAGE, 256=INDIRECT, 512=QUERY_RESOLVE).
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_buffer(vm: *mut c_void) {
    let size = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let usage = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let h = unsafe { host_gpu_create_buffer(size, usage) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.bufferWrite_(handle, offset, bytes)` — bytes is a Wren
/// String / Bytes value. Returns null.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let offset = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    unsafe { host_gpu_buffer_write(vm, handle, offset, 3) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

/// `Gpu.destroyBuffer_(handle)` — releases the GPUBuffer.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_destroy_buffer(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    unsafe { host_gpu_destroy_buffer(handle) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

/// `Gpu.createBufferFromFloats_(floats, usageBits)` — packs a
/// Wren `List<Num>` as `Float32Array` bytes and creates a
/// GPUBuffer in one call. The bridge does the f32 conversion;
/// the Wren side just hands over a list. Returns Num (handle)
/// or `-1`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_buffer_from_floats(vm: *mut c_void) {
    let usage = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let h = unsafe { host_gpu_create_buffer_from_f32(vm, 1, usage) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.bufferWriteUints_(buffer, offset, uints)` — writes a
/// Wren `List<Num>` to an existing buffer as u32 bytes (each
/// value truncates to 32 bits). Returns null.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_uints(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let offset = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    unsafe { host_gpu_buffer_write_u32(vm, handle, offset, 3) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

/// `Gpu.bufferWriteFloats_(buffer, offset, floats)` — writes a
/// Wren `List<Num>` to an existing buffer as f32 bytes. Returns
/// null.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_floats(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let offset = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    unsafe { host_gpu_buffer_write_f32(vm, handle, offset, 3) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

// ---------------------------------------------------------------
// Shaders / pipelines / bind groups
// ---------------------------------------------------------------

/// `Gpu.createShader_(wgslSource)` — compiles a `GPUShaderModule`
/// from a WGSL string. Returns Num (handle) or `-1`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_shader(vm: *mut c_void) {
    let h = unsafe { host_gpu_create_shader(vm, 1) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.createBindGroupLayout_(descriptorJson)` — descriptor:
///   { "entries": [
///       { "binding": Num, "visibility": "vertex|fragment|compute",
///         "buffer": { "type": "uniform"|"storage"|"read-only-storage" },
///         "sampler": {...}, "texture": {...}, "storageTexture": {...}
///       }
///   ] }
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_bind_group_layout(vm: *mut c_void) {
    let h = unsafe { host_gpu_create_bind_group_layout(vm, 1) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.createBindGroup_(descriptorJson)` — descriptor:
///   { "layout": Num,
///     "entries": [
///       { "binding": Num,
///         "resource": { "kind": "buffer", "buffer": Num, "offset": Num, "size": Num }
///       }
///     ] }
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_bind_group(vm: *mut c_void) {
    let h = unsafe { host_gpu_create_bind_group(vm, 1) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.createPipeline_(descriptorJson)` — descriptor:
///   { "layouts": ["auto" or [Num, Num, ...]],
///     "vertex":   { "shader": Num, "entry": "vs_main", "buffers": [...] },
///     "fragment": { "shader": Num, "entry": "fs_main",
///                   "targets": [{ "format": "preferred"|"rgba8unorm"|... }] },
///     "primitive": { "topology": "triangle-list"|... } }
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_pipeline(vm: *mut c_void) {
    let h = unsafe { host_gpu_create_pipeline(vm, 1) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

// ---------------------------------------------------------------
// Render pass — explicit pass for shader-based draws (the simple
// `Gpu.clear_` shortcut above is a single-purpose convenience).
// ---------------------------------------------------------------

/// `Gpu.renderPassBegin_(frame, descriptorJson)` — `descriptorJson`
/// can be an empty string (or null) to use sensible defaults
/// (single colour attachment, clear to opaque black). Otherwise:
///   { "colorAttachments": [
///       { "clearValue": {"r":..,"g":..,"b":..,"a":..},
///         "loadOp": "clear"|"load",
///         "storeOp": "store"|"discard" }
///   ] }
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_begin(vm: *mut c_void) {
    let frame = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let h = unsafe { host_gpu_render_pass_begin(vm, frame, 2) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_set_pipeline(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let pipeline = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    unsafe { host_gpu_render_pass_set_pipeline(pass, pipeline) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_set_bind_group(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let group = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let bg = unsafe { wrenGetSlotDouble(vm, 3) } as u32;
    unsafe { host_gpu_render_pass_set_bind_group(pass, group, bg) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_set_vertex_buffer(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let slot_idx = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let buffer = unsafe { wrenGetSlotDouble(vm, 3) } as u32;
    unsafe { host_gpu_render_pass_set_vertex_buffer(pass, slot_idx, buffer) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

/// `Gpu.renderPassSetIndexBuffer_(pass, buffer, format32)` —
/// `format32` is true for `uint32`, false (default) for `uint16`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_set_index_buffer(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let buffer = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let format32 = unsafe { wrenGetSlotDouble(vm, 3) } as i32;
    unsafe { host_gpu_render_pass_set_index_buffer(pass, buffer, format32) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_draw(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let vertex_count = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let instance_count = unsafe { wrenGetSlotDouble(vm, 3) } as u32;
    let first_vertex = unsafe { wrenGetSlotDouble(vm, 4) } as u32;
    let first_instance = unsafe { wrenGetSlotDouble(vm, 5) } as u32;
    unsafe { host_gpu_render_pass_draw(pass, vertex_count, instance_count, first_vertex, first_instance) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_draw_indexed(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let index_count = unsafe { wrenGetSlotDouble(vm, 2) } as u32;
    let instance_count = unsafe { wrenGetSlotDouble(vm, 3) } as u32;
    let first_index = unsafe { wrenGetSlotDouble(vm, 4) } as u32;
    let base_vertex = unsafe { wrenGetSlotDouble(vm, 5) } as i32;
    let first_instance = unsafe { wrenGetSlotDouble(vm, 6) } as u32;
    unsafe {
        host_gpu_render_pass_draw_indexed(
            pass,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    };
    unsafe { wrenSetSlotNull(vm, 0) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pass_end(vm: *mut c_void) {
    let pass = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    unsafe { host_gpu_render_pass_end(pass) };
    unsafe { wrenSetSlotNull(vm, 0) };
}

// ---------------------------------------------------------------
// Textures + samplers
// ---------------------------------------------------------------

/// `Gpu.createTexture_(descriptorJson)` — JSON descriptor:
///   { "width": Num, "height": Num,
///     "format": "rgba8unorm"|"bgra8unorm"|"depth24plus"|...,
///     "usage": Num    // GPUTextureUsage bitset:
///                     //   COPY_SRC=1, COPY_DST=2,
///                     //   TEXTURE_BINDING=4,
///                     //   STORAGE_BINDING=8,
///                     //   RENDER_ATTACHMENT=16
///     "dimension"?: "1d"|"2d"|"3d",
///     "depth"?: Num,
///     "mipLevelCount"?: Num,
///     "sampleCount"?: Num }
/// Returns Num (handle) or `-1`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_texture(vm: *mut c_void) {
    let h = unsafe { host_gpu_create_texture(vm, 1) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.createTextureView_(textureHandle, descriptorJson)` —
/// pass `null` for the descriptor to use the texture's default
/// view (covers the whole resource with the original format).
/// Returns Num (handle) or `-1`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_texture_view(vm: *mut c_void) {
    let texture = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    // Wren passes the descriptor as either a string (slot 2) or
    // `null`. For null, signal "default view" by passing slot 0
    // (the JS bridge skips JSON decode when slot is 0/falsy).
    let h = unsafe { host_gpu_create_texture_view(vm, texture, 2) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

/// `Gpu.queueWriteTexture_(textureHandle, bytes, descriptorJson)` —
/// upload tightly-packed pixel bytes. Descriptor:
///   { "bytesPerRow": Num,
///     "rowsPerImage"?: Num,
///     "origin"?: { x, y, z },
///     "mipLevel"?: Num, "aspect"?: "all"|"depth-only"|"stencil-only",
///     "width"?, "height"?, "depth"? } // copy size; defaults to texture size
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_queue_write_texture(vm: *mut c_void) {
    let texture = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    let ok = unsafe { host_gpu_queue_write_texture(vm, texture, 2, 3) };
    unsafe { wrenSetSlotDouble(vm, 0, ok as f64) };
}

/// `Gpu.createSampler_(descriptorJson)` — pass `null` for a
/// default sampler (linear / clamp-to-edge across the board).
/// Descriptor:
///   { "magFilter"?: "linear"|"nearest",
///     "minFilter"?, "mipmapFilter"?,
///     "addressModeU"?: "repeat"|"clamp-to-edge"|"mirror-repeat",
///     "addressModeV"?, "addressModeW"?,
///     "lodMinClamp"?, "lodMaxClamp"?,
///     "compare"?, "maxAnisotropy"? }
/// Returns Num (handle) or `-1`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_create_sampler(vm: *mut c_void) {
    let h = unsafe { host_gpu_create_sampler(vm, 1) };
    let result = if h == 0 { -1.0 } else { h as f64 };
    unsafe { wrenSetSlotDouble(vm, 0, result) };
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_destroy(vm: *mut c_void) {
    let handle = unsafe { wrenGetSlotDouble(vm, 1) } as u32;
    unsafe { host_gpu_destroy(handle) };
    unsafe { wrenSetSlotNull(vm, 0) };
}
