//! GPU plugin for WrenLift. Built as a `cdylib` and bundled into
//! the `@hatch:gpu` package as a `NativeLib` section. The runtime
//! `dlopen`s the dylib at install time and binds each `foreign`
//! method on the package's foreign classes to the `wlift_gpu_*`
//! `extern "C"` symbols below.
//!
//! # Architecture
//!
//! Every GPU object the user touches (device, buffer, texture,
//! shader, pipeline, encoder, render pass) lives in a per-type
//! registry inside this dylib, keyed by a `u64` id. Wren-side
//! foreign methods receive ids on the api-stack and dispatch
//! against the registry. This keeps the FFI surface trivially
//! ABI-safe — no shared `repr(C)` layouts to reason about, no
//! cross-dylib drop ordering — at the cost of a `HashMap` lookup
//! per call. Profile if it shows up.
//!
//! # Current scope
//!
//! Headless rendering: device + adapter, buffer, texture (as
//! render attachment or readback source), shader module, render
//! pipeline, command encoder, render pass, sampler, bind group +
//! layout. Windowing / surfaces are not exposed yet — render to a
//! texture and read pixels back through `Buffer.readBytes` for
//! off-screen pipelines and tests.
//!
//! # Safety
//!
//! Same model as `wlift_sqlite`: every entry point gets a `*mut
//! VM` valid for the duration of the call, slot arguments are
//! validated defensively but the pointer itself is trusted.

#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use wren_lift::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
//
// One shared registry per resource type. wgpu's smart pointers
// (Device / Buffer / etc.) handle lifetime inside their own
// reference-counted graph; ours just maps `u64 → Arc<Resource>`
// so Wren can pass a stable id around. Drop happens when Wren
// calls `<Resource>.destroy(id)` and we remove the entry.

struct Devices {
    /// Per-device record: the wgpu Device + its Queue.
    devices: HashMap<u64, DeviceRecord>,
}

struct Buffers {
    /// Buffer + (size in bytes, owning device id) for diagnostics.
    /// wgpu::Buffer is itself ref-counted internally; we hold one
    /// strong handle per Wren id.
    buffers: HashMap<u64, BufferRecord>,
}

struct BufferRecord {
    buffer: wgpu::Buffer,
    size: u64,
    device_id: u64,
}

struct Shaders {
    shaders: HashMap<u64, wgpu::ShaderModule>,
}

struct Textures {
    textures: HashMap<u64, TextureRecord>,
}

struct TextureRecord {
    texture: wgpu::Texture,
    format: wgpu::TextureFormat,
    /// Width / height / owning-device id are diagnostic metadata
    /// for now — kept around so future paths (sanity-check on
    /// copy_texture_to_buffer extents, cross-device validation)
    /// don't have to plumb them back through.
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    #[allow(dead_code)]
    device_id: u64,
}

struct Views {
    views: HashMap<u64, wgpu::TextureView>,
}

struct Samplers {
    samplers: HashMap<u64, wgpu::Sampler>,
}

struct BindGroupLayouts {
    layouts: HashMap<u64, wgpu::BindGroupLayout>,
}

struct PipelineLayouts {
    layouts: HashMap<u64, wgpu::PipelineLayout>,
}

struct BindGroups {
    groups: HashMap<u64, wgpu::BindGroup>,
}

struct RenderPipelines {
    pipelines: HashMap<u64, wgpu::RenderPipeline>,
}

struct Surfaces {
    /// Surface + (latest configured size, format) for diagnostics
    /// and to drive `getCurrentTexture` allocations. wgpu::Surface
    /// is `Surface<'static>` because we built it via
    /// `create_surface_unsafe` from raw handles — the underlying
    /// window lives in the embedder (winit-backed @hatch:window or
    /// a custom host); their lifetime is the user's responsibility.
    surfaces: HashMap<u64, SurfaceRecord>,
}

struct SurfaceRecord {
    surface: wgpu::Surface<'static>,
    /// One adapter handle is needed at configure time so we can
    /// resolve preferred formats / capabilities. Stored alongside
    /// to avoid reaching back into the device registry on the hot
    /// configure path.
    #[allow(dead_code)]
    device_id: u64,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

/// Registry of in-flight `SurfaceTexture` frames acquired via
/// `Surface.acquire()`. Holds both the SurfaceTexture (so it
/// stays alive across the render pass) and a TextureView built
/// from it. The view's id can be used in render-pass color
/// attachments exactly like a normal TextureView. `frame.present`
/// drops the entry, which consumes the SurfaceTexture and
/// schedules the swap.
struct SurfaceFrames {
    frames: HashMap<u64, SurfaceFrameRecord>,
}

struct SurfaceFrameRecord {
    surface_texture: wgpu::SurfaceTexture,
    view_id: u64,
}

/// CommandEncoder + finished CommandBuffer registry.
///
/// Render passes aren't exposed as long-lived Wren objects — their
/// lifetime is tied to the encoder via wgpu's borrow rules. The
/// Wren wrapper builds a command list, then a single
/// `encoder_record_pass` foreign call begins the pass, replays the
/// commands, and ends it inside a tight Rust scope.
struct Encoders {
    encoders: HashMap<u64, EncoderState>,
}

enum EncoderState {
    Open {
        encoder: wgpu::CommandEncoder,
        device_id: u64,
    },
    /// `device_id` here is preserved so `Queue.submit` can
    /// (eventually) sanity-check that the encoder was opened on
    /// the device it's now being submitted to.
    Finished {
        buffer: wgpu::CommandBuffer,
        #[allow(dead_code)]
        device_id: u64,
    },
}

struct DeviceRecord {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Held so the adapter doesn't drop while the device is alive
    /// (wgpu's lifetime graph keeps the device working anyway, but
    /// we hold this for diagnostics — `adapter.get_info()` etc.).
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    /// The Instance the device was created from. Surfaces have to
    /// be built off the same Instance, so retain it here for the
    /// `device.createSurface(handle)` path to reach.
    instance: wgpu::Instance,
}

fn devices() -> &'static Mutex<Devices> {
    static REG: OnceLock<Mutex<Devices>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Devices {
            devices: HashMap::new(),
        })
    })
}

fn buffers() -> &'static Mutex<Buffers> {
    static REG: OnceLock<Mutex<Buffers>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Buffers {
            buffers: HashMap::new(),
        })
    })
}

fn shaders() -> &'static Mutex<Shaders> {
    static REG: OnceLock<Mutex<Shaders>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Shaders {
            shaders: HashMap::new(),
        })
    })
}

fn textures() -> &'static Mutex<Textures> {
    static REG: OnceLock<Mutex<Textures>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Textures {
            textures: HashMap::new(),
        })
    })
}

fn views() -> &'static Mutex<Views> {
    static REG: OnceLock<Mutex<Views>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Views {
            views: HashMap::new(),
        })
    })
}

fn samplers() -> &'static Mutex<Samplers> {
    static REG: OnceLock<Mutex<Samplers>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Samplers {
            samplers: HashMap::new(),
        })
    })
}

fn bind_group_layouts() -> &'static Mutex<BindGroupLayouts> {
    static REG: OnceLock<Mutex<BindGroupLayouts>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(BindGroupLayouts {
            layouts: HashMap::new(),
        })
    })
}

fn pipeline_layouts() -> &'static Mutex<PipelineLayouts> {
    static REG: OnceLock<Mutex<PipelineLayouts>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(PipelineLayouts {
            layouts: HashMap::new(),
        })
    })
}

fn bind_groups() -> &'static Mutex<BindGroups> {
    static REG: OnceLock<Mutex<BindGroups>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(BindGroups {
            groups: HashMap::new(),
        })
    })
}

fn render_pipelines() -> &'static Mutex<RenderPipelines> {
    static REG: OnceLock<Mutex<RenderPipelines>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(RenderPipelines {
            pipelines: HashMap::new(),
        })
    })
}

fn encoders() -> &'static Mutex<Encoders> {
    static REG: OnceLock<Mutex<Encoders>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Encoders {
            encoders: HashMap::new(),
        })
    })
}

fn surfaces() -> &'static Mutex<Surfaces> {
    static REG: OnceLock<Mutex<Surfaces>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Surfaces {
            surfaces: HashMap::new(),
        })
    })
}

fn surface_frames() -> &'static Mutex<SurfaceFrames> {
    static REG: OnceLock<Mutex<SurfaceFrames>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(SurfaceFrames {
            frames: HashMap::new(),
        })
    })
}

// ---------------------------------------------------------------------------
// Format / usage decoders
// ---------------------------------------------------------------------------

fn texture_format_from_str(s: &str) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    Some(match s {
        "rgba8unorm" => F::Rgba8Unorm,
        "rgba8unorm-srgb" => F::Rgba8UnormSrgb,
        "bgra8unorm" => F::Bgra8Unorm,
        "bgra8unorm-srgb" => F::Bgra8UnormSrgb,
        "r8unorm" => F::R8Unorm,
        "r16float" => F::R16Float,
        "r32float" => F::R32Float,
        "rg8unorm" => F::Rg8Unorm,
        "rg16float" => F::Rg16Float,
        "rg32float" => F::Rg32Float,
        "rgba16float" => F::Rgba16Float,
        "rgba32float" => F::Rgba32Float,
        "depth32float" => F::Depth32Float,
        "depth24plus" => F::Depth24Plus,
        "depth24plus-stencil8" => F::Depth24PlusStencil8,
        _ => return None,
    })
}

fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> u32 {
    use wgpu::TextureFormat as F;
    match format {
        F::R8Unorm => 1,
        F::Rg8Unorm | F::R16Float => 2,
        F::Rgba8Unorm
        | F::Rgba8UnormSrgb
        | F::Bgra8Unorm
        | F::Bgra8UnormSrgb
        | F::R32Float
        | F::Rg16Float => 4,
        F::Rg32Float | F::Rgba16Float => 8,
        F::Rgba32Float => 16,
        F::Depth32Float => 4,
        // 24-bit depth formats can't be copy-src readable in wgpu;
        // callers shouldn't try to read them back. Return 4 as a
        // placeholder so size math doesn't divide-by-zero.
        F::Depth24Plus | F::Depth24PlusStencil8 => 4,
        _ => 4,
    }
}

unsafe fn texture_usage_from_list(vm: &mut VM, v: Value) -> Option<wgpu::TextureUsages> {
    let list = match unsafe { list_view(v) } {
        Some(l) => l,
        None => {
            vm.runtime_error(
                "Texture.create: descriptor `usage` must be a list of strings.".to_string(),
            );
            return None;
        }
    };
    let mut flags = wgpu::TextureUsages::empty();
    for i in 0..list.count as usize {
        let s = match unsafe { string_of(list_get(list, i)) } {
            Some(s) => s,
            None => {
                vm.runtime_error(
                    "Texture.create: every `usage` entry must be a string.".to_string(),
                );
                return None;
            }
        };
        flags |= match s.as_str() {
            "render-attachment" | "renderAttachment" => wgpu::TextureUsages::RENDER_ATTACHMENT,
            "texture-binding" | "textureBinding" => wgpu::TextureUsages::TEXTURE_BINDING,
            "storage-binding" | "storageBinding" => wgpu::TextureUsages::STORAGE_BINDING,
            "copy-src" | "copySrc" => wgpu::TextureUsages::COPY_SRC,
            "copy-dst" | "copyDst" => wgpu::TextureUsages::COPY_DST,
            other => {
                vm.runtime_error(format!("Texture.create: unknown usage flag '{}'.", other));
                return None;
            }
        };
    }
    Some(flags)
}

fn filter_from_str(s: &str) -> wgpu::FilterMode {
    match s {
        "linear" => wgpu::FilterMode::Linear,
        _ => wgpu::FilterMode::Nearest,
    }
}

fn address_mode_from_str(s: &str) -> wgpu::AddressMode {
    match s {
        "repeat" => wgpu::AddressMode::Repeat,
        "mirror-repeat" | "mirrorRepeat" => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Slot helpers (mirror wlift_sqlite)
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

// ---------------------------------------------------------------------------
// Type coercion helpers
// ---------------------------------------------------------------------------

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

/// Read a string-keyed entry out of a Wren `Map` value.
unsafe fn map_get(v: Value, key: &str) -> Option<Value> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Map {
        return None;
    }
    let map = ptr as *const ObjMap;
    // ObjMap stores keys as Wren Values; iterate to find a matching
    // string. Tiny maps in practice (descriptor configs).
    unsafe {
        for (k, val) in (*map).entries.iter() {
            if let Some(s) = string_of(k.0) {
                if s == key {
                    return Some(*val);
                }
            }
        }
    }
    None
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

unsafe fn list_view<'a>(v: Value) -> Option<&'a ObjList> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::List {
        return None;
    }
    Some(unsafe { &*(ptr as *const ObjList) })
}

unsafe fn list_get(list: &ObjList, i: usize) -> Value {
    debug_assert!(i < list.count as usize);
    unsafe { *list.elements.add(i) }
}

/// Decode a List<String> of usage flags into a `wgpu::BufferUsages`
/// bitmask. Unknown entries surface as a runtime error so typos
/// don't silently fail at submit time.
unsafe fn buffer_usage_from_list(vm: &mut VM, v: Value) -> Option<wgpu::BufferUsages> {
    let list = match unsafe { list_view(v) } {
        Some(l) => l,
        None => {
            vm.runtime_error(
                "Buffer.create: descriptor `usage` must be a list of strings.".to_string(),
            );
            return None;
        }
    };
    let mut flags = wgpu::BufferUsages::empty();
    for i in 0..list.count as usize {
        let item = unsafe { list_get(list, i) };
        let s = match unsafe { string_of(item) } {
            Some(s) => s,
            None => {
                vm.runtime_error(
                    "Buffer.create: every `usage` entry must be a string.".to_string(),
                );
                return None;
            }
        };
        let bit = match s.as_str() {
            "vertex" => wgpu::BufferUsages::VERTEX,
            "index" => wgpu::BufferUsages::INDEX,
            "uniform" => wgpu::BufferUsages::UNIFORM,
            "storage" => wgpu::BufferUsages::STORAGE,
            "indirect" => wgpu::BufferUsages::INDIRECT,
            "copy-src" | "copySrc" => wgpu::BufferUsages::COPY_SRC,
            "copy-dst" | "copyDst" => wgpu::BufferUsages::COPY_DST,
            "map-read" | "mapRead" => wgpu::BufferUsages::MAP_READ,
            "map-write" | "mapWrite" => wgpu::BufferUsages::MAP_WRITE,
            "query-resolve" | "queryResolve" => wgpu::BufferUsages::QUERY_RESOLVE,
            other => {
                vm.runtime_error(format!("Buffer.create: unknown usage flag '{}'.", other));
                return None;
            }
        };
        flags |= bit;
    }
    Some(flags)
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

fn backends_from_descriptor(desc: Value) -> wgpu::Backends {
    unsafe {
        let s = match map_get(desc, "backends").and_then(|v| string_of(v)) {
            Some(s) => s,
            None => return wgpu::Backends::PRIMARY,
        };
        match s.as_str() {
            "primary" => wgpu::Backends::PRIMARY,
            "secondary" => wgpu::Backends::SECONDARY,
            "all" => wgpu::Backends::all(),
            "vulkan" => wgpu::Backends::VULKAN,
            "metal" => wgpu::Backends::METAL,
            "dx12" => wgpu::Backends::DX12,
            "gl" | "gles" => wgpu::Backends::GL,
            _ => wgpu::Backends::PRIMARY,
        }
    }
}

// ---------------------------------------------------------------------------
// wlift_gpu_request_device(descriptor) -> id
// ---------------------------------------------------------------------------
//
// One-shot: builds an Instance, requests an Adapter, requests a
// Device+Queue, stashes the trio under a fresh id, and returns
// the id to Wren as a Num.
//
// Headless: no `compatible_surface` on adapter request. wgpu picks
// the best adapter for the requested backends. Future windowed
// path will overload `request_device(desc)` with a Surface in the
// descriptor.
//
// `descriptor` is a Wren Map; supported keys (all optional):
//   - "backends": "primary" | "all" | "metal" | "vulkan" | "dx12" | "gl"
//   - "powerPreference": "low-power" | "high-performance"
//   - "label": String — passed to wgpu for diagnostics

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_request_device(vm: *mut VM) {
    unsafe {
        let desc = slot(vm, 1);
        let backends = backends_from_descriptor(desc);

        let power_pref = match map_get(desc, "powerPreference").and_then(|v| string_of(v)) {
            Some(s) if s == "low-power" => wgpu::PowerPreference::LowPower,
            Some(s) if s == "high-performance" => wgpu::PowerPreference::HighPerformance,
            _ => wgpu::PowerPreference::default(),
        };
        let label = map_get(desc, "label").and_then(|v| string_of(v));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Some(a) => a,
                None => {
                    ctx(vm).runtime_error(
                        "Gpu.requestDevice: no compatible adapter found.".to_string(),
                    );
                    return;
                }
            };

        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: label.as_deref(),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )) {
            Ok(pair) => pair,
            Err(e) => {
                ctx(vm).runtime_error(format!("Gpu.requestDevice: {}", e));
                return;
            }
        };

        let id = next_id();
        devices().lock().unwrap().devices.insert(
            id,
            DeviceRecord {
                device,
                queue,
                adapter,
                instance,
            },
        );
        set_return(vm, Value::num(id as f64));
    }
}

// ---------------------------------------------------------------------------
// wlift_gpu_device_destroy(id)
// ---------------------------------------------------------------------------
//
// Drops the device + queue + adapter. Idempotent — calling on an
// already-removed id is a no-op (no error).

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_device_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Device.destroy") {
            Some(i) => i,
            None => return,
        };
        devices().lock().unwrap().devices.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// wlift_gpu_device_info(id) -> Map
// ---------------------------------------------------------------------------
//
// Returns adapter + device metadata as a Wren Map. Useful for
// diagnostics, CI assertions, and the spec test's "is this a real
// device" sanity check.

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_device_info(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Device.info") {
            Some(i) => i,
            None => return,
        };
        let info = {
            let reg = devices().lock().unwrap();
            let rec = match reg.devices.get(&id) {
                Some(r) => r,
                None => {
                    ctx(vm).runtime_error("Device.info: unknown device id.".to_string());
                    return;
                }
            };
            rec.adapter.get_info()
        };

        let context = ctx(vm);
        let result = context.alloc_map();
        let name = context.alloc_string(info.name.clone());
        let backend = context.alloc_string(format!("{:?}", info.backend).to_lowercase());
        let device_type = context.alloc_string(format!("{:?}", info.device_type).to_lowercase());

        // Push values into the Map. ObjMap exposes a Wren-side
        // `[k]=v` setter, but from Rust we go through the public
        // `call_method_on` helper to drive it.
        let key_name = context.alloc_string("name".to_string());
        let key_backend = context.alloc_string("backend".to_string());
        let key_device_type = context.alloc_string("deviceType".to_string());
        let _ = context.call_method_on(result, "[_]=(_)", &[key_name, name]);
        let _ = context.call_method_on(result, "[_]=(_)", &[key_backend, backend]);
        let _ = context.call_method_on(result, "[_]=(_)", &[key_device_type, device_type]);
        set_return(vm, result);
    }
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------
//
// Descriptor:
//   {
//     "size":  Num,             // bytes, must be > 0 and <= u32::MAX for now
//     "usage": List<String>,    // see buffer_usage_from_list
//     "label": String?,         // diagnostics
//   }

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createBuffer") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let size = match map_get(desc, "size").and_then(|v| v.as_num()) {
            Some(n) if n.is_finite() && n > 0.0 && n.fract() == 0.0 => n as u64,
            _ => {
                ctx(vm).runtime_error(
                    "Buffer.create: descriptor `size` must be a positive integer.".to_string(),
                );
                return;
            }
        };
        // wgpu requires buffer sizes be aligned to 4 — surface the
        // misalignment instead of silently rounding.
        if size % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            ctx(vm).runtime_error(format!(
                "Buffer.create: size {} not aligned to {} bytes.",
                size,
                wgpu::COPY_BUFFER_ALIGNMENT,
            ));
            return;
        }

        let usage = match map_get(desc, "usage") {
            Some(v) => match buffer_usage_from_list(ctx(vm), v) {
                Some(u) => u,
                None => return,
            },
            None => {
                ctx(vm).runtime_error(
                    "Buffer.create: descriptor must include a `usage` list.".to_string(),
                );
                return;
            }
        };

        let label = map_get(desc, "label").and_then(|v| string_of(v));

        let buffer = {
            let reg = devices().lock().unwrap();
            let dev = match reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("Buffer.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: label.as_deref(),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let id = next_id();
        buffers().lock().unwrap().buffers.insert(
            id,
            BufferRecord {
                buffer,
                size,
                device_id,
            },
        );
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Buffer.destroy") {
            Some(i) => i,
            None => return,
        };
        buffers().lock().unwrap().buffers.remove(&id);
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_size(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Buffer.size") {
            Some(i) => i,
            None => return,
        };
        let size = buffers()
            .lock()
            .unwrap()
            .buffers
            .get(&id)
            .map(|b| b.size)
            .unwrap_or(0);
        set_return(vm, Value::num(size as f64));
    }
}

// ---------------------------------------------------------------------------
// Buffer writes — float / uint scalar lists
// ---------------------------------------------------------------------------
//
// Walks a `List<Num>` and writes the values into the buffer at
// `offset`. Always one `queue.write_buffer` call regardless of
// list length — the f64→f32 / f64→u32 conversion happens in this
// function so per-element overhead stays negligible.

unsafe fn write_buffer_with(
    vm: *mut VM,
    label: &str,
    converter: fn(f64) -> Vec<u8>,
    bytes_per_element: usize,
) {
    unsafe {
        let buffer_id = match id_of(ctx(vm), slot(vm, 1), label) {
            Some(i) => i,
            None => return,
        };
        let offset = match slot(vm, 2).as_num() {
            Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as u64,
            _ => {
                ctx(vm).runtime_error(format!("{}: offset must be a non-negative integer.", label));
                return;
            }
        };
        let list = match list_view(slot(vm, 3)) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error(format!("{}: data must be a list of numbers.", label));
                return;
            }
        };

        let count = list.count as usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(count * bytes_per_element);
        for i in 0..count {
            let n = match list_get(list, i).as_num() {
                Some(n) => n,
                None => {
                    ctx(vm).runtime_error(format!(
                        "{}: every element must be a number (index {}).",
                        label, i
                    ));
                    return;
                }
            };
            bytes.extend_from_slice(&converter(n));
        }

        let buf_reg = buffers().lock().unwrap();
        let buf = match buf_reg.buffers.get(&buffer_id) {
            Some(b) => b,
            None => {
                ctx(vm).runtime_error(format!("{}: unknown buffer id.", label));
                return;
            }
        };
        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&buf.device_id) {
            Some(d) => d,
            None => {
                ctx(vm).runtime_error(format!("{}: device dropped before write.", label));
                return;
            }
        };
        dev.queue.write_buffer(&buf.buffer, offset, &bytes);
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_floats(vm: *mut VM) {
    unsafe {
        write_buffer_with(
            vm,
            "Buffer.writeFloats",
            |n| (n as f32).to_le_bytes().to_vec(),
            4,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_uints(vm: *mut VM) {
    unsafe {
        write_buffer_with(
            vm,
            "Buffer.writeUints",
            |n| {
                let v = if n.is_finite() && n >= 0.0 {
                    (n as u32).to_le_bytes()
                } else {
                    [0u8; 4]
                };
                v.to_vec()
            },
            4,
        );
    }
}

// ---------------------------------------------------------------------------
// Buffer batched math uploads
// ---------------------------------------------------------------------------
//
// Each variant takes a Wren List of math objects and packs every
// element's `.data` list as f32 into a single buffered write.
// The expected element types are:
//
//   writeMat4s  — Mat4   (16 f32, row-major; user passes a layout-
//                         aware shader that transposes if needed)
//   writeVec3s  — Vec3   (3 f32, no padding — caller pads if the
//                         shader expects std140-aligned vec3s)
//   writeVec4s  — Vec4   (4 f32)
//   writeQuats  — Quat   (4 f32, w-x-y-z order — matches Quat.data)

/// Pack a `List<List<Num>>` of math components into a contiguous
/// `Vec<u8>` of f32, writing the result into the buffer.
///
/// The outer Wren list contains one inner list per element; each
/// inner list MUST already be the math object's `.data` (i.e. the
/// Wren wrapper extracted it before crossing the FFI boundary).
/// Going through Wren-side extraction sidesteps the GC-during-
/// `call_method_on` hazard — no callbacks back into the VM during
/// the conversion loop, so the outer list's `elements` pointer
/// stays valid for the whole pass.
unsafe fn write_buffer_math_batch(vm: *mut VM, label: &str, floats_per_element: usize) {
    unsafe {
        let buffer_id = match id_of(ctx(vm), slot(vm, 1), label) {
            Some(i) => i,
            None => return,
        };
        let offset = match slot(vm, 2).as_num() {
            Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as u64,
            _ => {
                ctx(vm).runtime_error(format!("{}: offset must be a non-negative integer.", label));
                return;
            }
        };
        let outer = match list_view(slot(vm, 3)) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error(format!("{}: data must be a list of lists.", label));
                return;
            }
        };

        let count = outer.count as usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(count * floats_per_element * 4);

        for i in 0..count {
            let inner_v = list_get(outer, i);
            let inner = match list_view(inner_v) {
                Some(l) => l,
                None => {
                    ctx(vm).runtime_error(format!(
                        "{}: element {} must be a list (use `obj.data`).",
                        label, i
                    ));
                    return;
                }
            };
            if inner.count as usize != floats_per_element {
                ctx(vm).runtime_error(format!(
                    "{}: element {} has {} components, expected {}.",
                    label, i, inner.count, floats_per_element
                ));
                return;
            }
            for j in 0..floats_per_element {
                let n = match list_get(inner, j).as_num() {
                    Some(n) => n,
                    None => {
                        ctx(vm).runtime_error(format!(
                            "{}: element {} component {} not a number.",
                            label, i, j
                        ));
                        return;
                    }
                };
                bytes.extend_from_slice(&(n as f32).to_le_bytes());
            }
        }

        let buf_reg = buffers().lock().unwrap();
        let buf = match buf_reg.buffers.get(&buffer_id) {
            Some(b) => b,
            None => {
                ctx(vm).runtime_error(format!("{}: unknown buffer id.", label));
                return;
            }
        };
        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&buf.device_id) {
            Some(d) => d,
            None => {
                ctx(vm).runtime_error(format!("{}: device dropped before write.", label));
                return;
            }
        };
        dev.queue.write_buffer(&buf.buffer, offset, &bytes);
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_mat4s(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeMat4s", 16) }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_vec3s(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeVec3s", 3) }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_vec4s(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeVec4s", 4) }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_write_quats(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeQuats", 4) }
}

// ---------------------------------------------------------------------------
// ShaderModule
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_shader_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createShaderModule") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);
        let code = match map_get(desc, "code").and_then(|v| string_of(v)) {
            Some(c) => c,
            None => {
                ctx(vm).runtime_error(
                    "ShaderModule.create: descriptor must include a `code` string.".to_string(),
                );
                return;
            }
        };
        let label = map_get(desc, "label").and_then(|v| string_of(v));

        let module = {
            let reg = devices().lock().unwrap();
            let dev = match reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("ShaderModule.create: unknown device id.".to_string());
                    return;
                }
            };
            // wgpu emits compile errors via the device's error
            // callback rather than the create_shader_module return
            // value. We surface them through the validation path
            // attached to the device; cleaner per-error mapping
            // (line + column inside the WGSL source) is a planned
            // follow-up.
            dev.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: label.as_deref(),
                    source: wgpu::ShaderSource::Wgsl(code.into()),
                })
        };

        let id = next_id();
        shaders().lock().unwrap().shaders.insert(id, module);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_shader_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "ShaderModule.destroy") {
            Some(i) => i,
            None => return,
        };
        shaders().lock().unwrap().shaders.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// Texture + TextureView
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_texture_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createTexture") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let width = match map_get(desc, "width").and_then(|v| v.as_num()) {
            Some(n) if n.is_finite() && n > 0.0 && n.fract() == 0.0 => n as u32,
            _ => {
                ctx(vm).runtime_error(
                    "Texture.create: `width` must be a positive integer.".to_string(),
                );
                return;
            }
        };
        let height = match map_get(desc, "height").and_then(|v| v.as_num()) {
            Some(n) if n.is_finite() && n > 0.0 && n.fract() == 0.0 => n as u32,
            _ => {
                ctx(vm).runtime_error(
                    "Texture.create: `height` must be a positive integer.".to_string(),
                );
                return;
            }
        };
        let depth = map_get(desc, "depth")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(1);
        let format = match map_get(desc, "format").and_then(|v| string_of(v)) {
            Some(s) => match texture_format_from_str(&s) {
                Some(f) => f,
                None => {
                    ctx(vm).runtime_error(format!("Texture.create: unknown format '{}'.", s));
                    return;
                }
            },
            None => {
                ctx(vm).runtime_error(
                    "Texture.create: descriptor must include a `format` string.".to_string(),
                );
                return;
            }
        };
        let usage = match map_get(desc, "usage") {
            Some(v) => match texture_usage_from_list(ctx(vm), v) {
                Some(u) => u,
                None => return,
            },
            None => {
                ctx(vm).runtime_error(
                    "Texture.create: descriptor must include a `usage` list.".to_string(),
                );
                return;
            }
        };
        let label = map_get(desc, "label").and_then(|v| string_of(v));
        let sample_count = map_get(desc, "sampleCount")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(1);

        let texture = {
            let reg = devices().lock().unwrap();
            let dev = match reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("Texture.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device.create_texture(&wgpu::TextureDescriptor {
                label: label.as_deref(),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: depth,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        let id = next_id();
        textures().lock().unwrap().textures.insert(
            id,
            TextureRecord {
                texture,
                format,
                width,
                height,
                device_id,
            },
        );
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_texture_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Texture.destroy") {
            Some(i) => i,
            None => return,
        };
        textures().lock().unwrap().textures.remove(&id);
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_texture_create_view(vm: *mut VM) {
    unsafe {
        let texture_id = match id_of(ctx(vm), slot(vm, 1), "Texture.createView") {
            Some(i) => i,
            None => return,
        };
        let view = {
            let reg = textures().lock().unwrap();
            let rec = match reg.textures.get(&texture_id) {
                Some(r) => r,
                None => {
                    ctx(vm).runtime_error("Texture.createView: unknown texture id.".to_string());
                    return;
                }
            };
            rec.texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let id = next_id();
        views().lock().unwrap().views.insert(id, view);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_view_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "TextureView.destroy") {
            Some(i) => i,
            None => return,
        };
        views().lock().unwrap().views.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_sampler_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createSampler") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let mag = map_get(desc, "magFilter")
            .and_then(|v| string_of(v))
            .map(|s| filter_from_str(&s))
            .unwrap_or(wgpu::FilterMode::Linear);
        let min = map_get(desc, "minFilter")
            .and_then(|v| string_of(v))
            .map(|s| filter_from_str(&s))
            .unwrap_or(wgpu::FilterMode::Linear);
        let mip = map_get(desc, "mipmapFilter")
            .and_then(|v| string_of(v))
            .map(|s| filter_from_str(&s))
            .unwrap_or(wgpu::FilterMode::Nearest);
        let au = map_get(desc, "addressModeU")
            .and_then(|v| string_of(v))
            .map(|s| address_mode_from_str(&s))
            .unwrap_or(wgpu::AddressMode::ClampToEdge);
        let av = map_get(desc, "addressModeV")
            .and_then(|v| string_of(v))
            .map(|s| address_mode_from_str(&s))
            .unwrap_or(wgpu::AddressMode::ClampToEdge);
        let aw = map_get(desc, "addressModeW")
            .and_then(|v| string_of(v))
            .map(|s| address_mode_from_str(&s))
            .unwrap_or(wgpu::AddressMode::ClampToEdge);
        let label = map_get(desc, "label").and_then(|v| string_of(v));

        let sampler = {
            let reg = devices().lock().unwrap();
            let dev = match reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("Sampler.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device.create_sampler(&wgpu::SamplerDescriptor {
                label: label.as_deref(),
                address_mode_u: au,
                address_mode_v: av,
                address_mode_w: aw,
                mag_filter: mag,
                min_filter: min,
                mipmap_filter: mip,
                ..Default::default()
            })
        };
        let id = next_id();
        samplers().lock().unwrap().samplers.insert(id, sampler);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_sampler_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Sampler.destroy") {
            Some(i) => i,
            None => return,
        };
        samplers().lock().unwrap().samplers.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// BindGroupLayout
// ---------------------------------------------------------------------------
//
// Descriptor:
//   {
//     "entries": [
//       { "binding": Num,
//         "visibility": List<String>, // "vertex" | "fragment" | "compute"
//         "kind": "uniform" | "storage" | "read-only-storage" |
//                 "sampler" | "texture" | "storage-texture",
//         // sampler/texture extras as needed; this currently
//         // covers uniform + texture + sampler (filtering)
//         "viewDimension": "2d" (texture only),
//         "sampleType":   "float" | "depth" (texture only),
//       },
//       ...
//     ]
//   }

unsafe fn shader_stages_from_list(vm: &mut VM, v: Value) -> Option<wgpu::ShaderStages> {
    let list = match unsafe { list_view(v) } {
        Some(l) => l,
        None => {
            vm.runtime_error("`visibility` must be a list of strings.".to_string());
            return None;
        }
    };
    let mut s = wgpu::ShaderStages::empty();
    for i in 0..list.count as usize {
        let str_v = match unsafe { string_of(list_get(list, i)) } {
            Some(s) => s,
            None => {
                vm.runtime_error("`visibility` entries must be strings.".to_string());
                return None;
            }
        };
        s |= match str_v.as_str() {
            "vertex" => wgpu::ShaderStages::VERTEX,
            "fragment" => wgpu::ShaderStages::FRAGMENT,
            "compute" => wgpu::ShaderStages::COMPUTE,
            other => {
                vm.runtime_error(format!("Unknown shader stage '{}'.", other));
                return None;
            }
        };
    }
    Some(s)
}

unsafe fn binding_type_from_entry(vm: &mut VM, entry: Value) -> Option<wgpu::BindingType> {
    let kind = match map_get(entry, "kind").and_then(|v| string_of(v)) {
        Some(s) => s,
        None => {
            vm.runtime_error("BindGroupLayout entry: missing `kind` string.".to_string());
            return None;
        }
    };
    Some(match kind.as_str() {
        "uniform" => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        "storage" => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        "read-only-storage" => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        "sampler" => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        "texture" => {
            let sample_type = match map_get(entry, "sampleType").and_then(|v| string_of(v)) {
                Some(s) if s == "depth" => wgpu::TextureSampleType::Depth,
                Some(s) if s == "uint" => wgpu::TextureSampleType::Uint,
                Some(s) if s == "sint" => wgpu::TextureSampleType::Sint,
                _ => wgpu::TextureSampleType::Float { filterable: true },
            };
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        }
        other => {
            vm.runtime_error(format!("Unknown bind group entry kind '{}'.", other));
            return None;
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_bind_group_layout_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createBindGroupLayout") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);
        let label = map_get(desc, "label").and_then(|v| string_of(v));

        let entries_v = match map_get(desc, "entries") {
            Some(v) => v,
            None => {
                ctx(vm).runtime_error(
                    "BindGroupLayout.create: descriptor must include `entries`.".to_string(),
                );
                return;
            }
        };
        let entries_list = match list_view(entries_v) {
            Some(l) => l,
            None => {
                ctx(vm)
                    .runtime_error("BindGroupLayout.create: `entries` must be a list.".to_string());
                return;
            }
        };

        let mut entries: Vec<wgpu::BindGroupLayoutEntry> =
            Vec::with_capacity(entries_list.count as usize);
        for i in 0..entries_list.count as usize {
            let entry = list_get(entries_list, i);
            let binding = match map_get(entry, "binding").and_then(|v| v.as_num()) {
                Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as u32,
                _ => {
                    ctx(vm).runtime_error(format!(
                        "BindGroupLayout entry {}: `binding` must be a non-negative integer.",
                        i
                    ));
                    return;
                }
            };
            let visibility = match map_get(entry, "visibility") {
                Some(v) => match shader_stages_from_list(ctx(vm), v) {
                    Some(s) => s,
                    None => return,
                },
                None => {
                    ctx(vm).runtime_error(format!(
                        "BindGroupLayout entry {}: missing `visibility`.",
                        i
                    ));
                    return;
                }
            };
            let ty = match binding_type_from_entry(ctx(vm), entry) {
                Some(t) => t,
                None => return,
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty,
                count: None,
            });
        }

        let layout = {
            let reg = devices().lock().unwrap();
            let dev = match reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("BindGroupLayout.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: label.as_deref(),
                    entries: &entries,
                })
        };
        let id = next_id();
        bind_group_layouts()
            .lock()
            .unwrap()
            .layouts
            .insert(id, layout);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_bind_group_layout_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "BindGroupLayout.destroy") {
            Some(i) => i,
            None => return,
        };
        bind_group_layouts().lock().unwrap().layouts.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// PipelineLayout
// ---------------------------------------------------------------------------
//
// Descriptor:
//   {
//     "bindGroupLayouts": [layoutId, layoutId, ...],
//     "label": String?
//   }

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_pipeline_layout_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createPipelineLayout") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let label = map_get(desc, "label").and_then(|v| string_of(v));
        let bgl_ids_v = match map_get(desc, "bindGroupLayouts") {
            Some(v) => v,
            None => {
                ctx(vm).runtime_error(
                    "PipelineLayout.create: descriptor must include `bindGroupLayouts`."
                        .to_string(),
                );
                return;
            }
        };
        let bgl_list = match list_view(bgl_ids_v) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error(
                    "PipelineLayout.create: `bindGroupLayouts` must be a list.".to_string(),
                );
                return;
            }
        };

        let bgl_reg = bind_group_layouts().lock().unwrap();
        let mut refs: Vec<&wgpu::BindGroupLayout> = Vec::with_capacity(bgl_list.count as usize);
        for i in 0..bgl_list.count as usize {
            let id = match list_get(bgl_list, i).as_num() {
                Some(n) if n.is_finite() && n >= 0.0 => n as u64,
                _ => {
                    drop(bgl_reg);
                    ctx(vm).runtime_error(format!(
                        "PipelineLayout.create: bindGroupLayouts[{}] must be an id.",
                        i
                    ));
                    return;
                }
            };
            let bgl = match bgl_reg.layouts.get(&id) {
                Some(l) => l,
                None => {
                    drop(bgl_reg);
                    ctx(vm).runtime_error(format!(
                        "PipelineLayout.create: unknown bind group layout id {}.",
                        id
                    ));
                    return;
                }
            };
            refs.push(bgl);
        }

        let layout = {
            let dev_reg = devices().lock().unwrap();
            let dev = match dev_reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    drop(bgl_reg);
                    ctx(vm).runtime_error("PipelineLayout.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: label.as_deref(),
                    bind_group_layouts: &refs,
                    push_constant_ranges: &[],
                })
        };
        drop(bgl_reg);

        let id = next_id();
        pipeline_layouts()
            .lock()
            .unwrap()
            .layouts
            .insert(id, layout);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_pipeline_layout_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "PipelineLayout.destroy") {
            Some(i) => i,
            None => return,
        };
        pipeline_layouts().lock().unwrap().layouts.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// BindGroup
// ---------------------------------------------------------------------------
//
// Descriptor:
//   {
//     "layout": layoutId,
//     "entries": [
//       { "binding": Num, "buffer":  bufferId, "offset"?: Num, "size"?: Num },
//       { "binding": Num, "sampler": samplerId },
//       { "binding": Num, "view":    textureViewId },
//     ],
//     "label": String?
//   }

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_bind_group_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createBindGroup") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let layout_id = match map_get(desc, "layout").and_then(|v| v.as_num()) {
            Some(n) if n.is_finite() && n >= 0.0 => n as u64,
            _ => {
                ctx(vm).runtime_error(
                    "BindGroup.create: descriptor must include a numeric `layout` id.".to_string(),
                );
                return;
            }
        };

        let label = map_get(desc, "label").and_then(|v| string_of(v));
        let entries_v = match map_get(desc, "entries") {
            Some(v) => v,
            None => {
                ctx(vm).runtime_error(
                    "BindGroup.create: descriptor must include `entries`.".to_string(),
                );
                return;
            }
        };
        let entries_list = match list_view(entries_v) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error("BindGroup.create: `entries` must be a list.".to_string());
                return;
            }
        };

        // Walk entries, capturing (binding, kind, ids) — actual
        // wgpu::BindingResource refs need lifetimes against the
        // registry locks held during create_bind_group, so we
        // resolve them in a second pass below.
        enum EntryKind {
            Buffer {
                id: u64,
                offset: u64,
                size: Option<u64>,
            },
            Sampler {
                id: u64,
            },
            View {
                id: u64,
            },
        }
        let mut decoded: Vec<(u32, EntryKind)> = Vec::with_capacity(entries_list.count as usize);
        for i in 0..entries_list.count as usize {
            let entry = list_get(entries_list, i);
            let binding = match map_get(entry, "binding").and_then(|v| v.as_num()) {
                Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as u32,
                _ => {
                    ctx(vm).runtime_error(format!(
                        "BindGroup entry {}: `binding` must be a non-negative integer.",
                        i
                    ));
                    return;
                }
            };
            if let Some(b) = map_get(entry, "buffer").and_then(|v| v.as_num()) {
                let bid = b as u64;
                let offset = map_get(entry, "offset")
                    .and_then(|v| v.as_num())
                    .map(|n| n as u64)
                    .unwrap_or(0);
                let size = map_get(entry, "size")
                    .and_then(|v| v.as_num())
                    .map(|n| n as u64);
                decoded.push((
                    binding,
                    EntryKind::Buffer {
                        id: bid,
                        offset,
                        size,
                    },
                ));
            } else if let Some(sid) = map_get(entry, "sampler").and_then(|v| v.as_num()) {
                decoded.push((binding, EntryKind::Sampler { id: sid as u64 }));
            } else if let Some(vid) = map_get(entry, "view").and_then(|v| v.as_num()) {
                decoded.push((binding, EntryKind::View { id: vid as u64 }));
            } else {
                ctx(vm).runtime_error(format!(
                    "BindGroup entry {}: must include one of `buffer`, `sampler`, `view`.",
                    i
                ));
                return;
            }
        }

        // Resolve resource refs and create the bind group while
        // every needed registry is locked. The Mutex guards live
        // exactly long enough for create_bind_group to consume the
        // refs.
        let bgl_reg = bind_group_layouts().lock().unwrap();
        let layout = match bgl_reg.layouts.get(&layout_id) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error("BindGroup.create: unknown layout id.".to_string());
                return;
            }
        };
        let buf_reg = buffers().lock().unwrap();
        let smp_reg = samplers().lock().unwrap();
        let view_reg = views().lock().unwrap();

        let mut entries_built: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(decoded.len());
        for (binding, kind) in &decoded {
            let resource = match kind {
                EntryKind::Buffer { id, offset, size } => {
                    let buf = match buf_reg.buffers.get(id) {
                        Some(b) => b,
                        None => {
                            ctx(vm)
                                .runtime_error("BindGroup.create: unknown buffer id.".to_string());
                            return;
                        }
                    };
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buf.buffer,
                        offset: *offset,
                        size: size.and_then(std::num::NonZeroU64::new),
                    })
                }
                EntryKind::Sampler { id } => {
                    let smp = match smp_reg.samplers.get(id) {
                        Some(s) => s,
                        None => {
                            ctx(vm)
                                .runtime_error("BindGroup.create: unknown sampler id.".to_string());
                            return;
                        }
                    };
                    wgpu::BindingResource::Sampler(smp)
                }
                EntryKind::View { id } => {
                    let view = match view_reg.views.get(id) {
                        Some(v) => v,
                        None => {
                            ctx(vm).runtime_error("BindGroup.create: unknown view id.".to_string());
                            return;
                        }
                    };
                    wgpu::BindingResource::TextureView(view)
                }
            };
            entries_built.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource,
            });
        }

        let bg = {
            let dev_reg = devices().lock().unwrap();
            let dev = match dev_reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("BindGroup.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: label.as_deref(),
                layout,
                entries: &entries_built,
            })
        };
        drop(view_reg);
        drop(smp_reg);
        drop(buf_reg);
        drop(bgl_reg);

        let id = next_id();
        bind_groups().lock().unwrap().groups.insert(id, bg);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_bind_group_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "BindGroup.destroy") {
            Some(i) => i,
            None => return,
        };
        bind_groups().lock().unwrap().groups.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// RenderPipeline
// ---------------------------------------------------------------------------
//
// Descriptor (large but flat — no nested DSL beyond two layers):
//
//   {
//     "layout": pipelineLayoutId | "auto",
//     "vertex": {
//       "module":     shaderId,
//       "entryPoint": String,
//       "buffers":    [
//         { "arrayStride": Num,
//           "stepMode":    "vertex" | "instance" (default "vertex"),
//           "attributes":  [
//             { "shaderLocation": Num,
//               "offset":         Num,
//               "format":         "float32x2" | "float32x3" | "float32x4" |
//                                 "uint32" | "uint32x2" | "uint32x3" |
//                                 "uint32x4" | "sint32" | "sint32x2" | ... }
//           ]
//         }, ...
//       ]
//     },
//     "fragment": {
//       "module":     shaderId,
//       "entryPoint": String,
//       "targets": [
//         { "format": String, "blend"?: { ... }, "writeMask"?: Num },
//         ...
//       ]
//     },
//     "primitive": {
//       "topology":  "triangle-list" (default) | "triangle-strip" |
//                    "line-list" | "line-strip" | "point-list",
//       "cullMode":  "none" (default) | "front" | "back",
//       "frontFace": "ccw" (default) | "cw"
//     },
//     "depthStencil"?: {
//       "format":         "depth32float" | "depth24plus" | ...,
//       "depthWriteEnabled": Bool (default true),
//       "depthCompare":   "less" (default) | "less-equal" | "always" | ...
//     },
//     "label": String?
//   }

fn vertex_format_from_str(s: &str) -> Option<wgpu::VertexFormat> {
    use wgpu::VertexFormat as V;
    Some(match s {
        "float32" => V::Float32,
        "float32x2" => V::Float32x2,
        "float32x3" => V::Float32x3,
        "float32x4" => V::Float32x4,
        "uint32" => V::Uint32,
        "uint32x2" => V::Uint32x2,
        "uint32x3" => V::Uint32x3,
        "uint32x4" => V::Uint32x4,
        "sint32" => V::Sint32,
        "sint32x2" => V::Sint32x2,
        "sint32x3" => V::Sint32x3,
        "sint32x4" => V::Sint32x4,
        "uint8x2" => V::Uint8x2,
        "uint8x4" => V::Uint8x4,
        "unorm8x2" => V::Unorm8x2,
        "unorm8x4" => V::Unorm8x4,
        _ => return None,
    })
}

fn topology_from_str(s: &str) -> wgpu::PrimitiveTopology {
    match s {
        "triangle-strip" => wgpu::PrimitiveTopology::TriangleStrip,
        "line-list" => wgpu::PrimitiveTopology::LineList,
        "line-strip" => wgpu::PrimitiveTopology::LineStrip,
        "point-list" => wgpu::PrimitiveTopology::PointList,
        _ => wgpu::PrimitiveTopology::TriangleList,
    }
}

fn compare_from_str(s: &str) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match s {
        "never" => C::Never,
        "less" => C::Less,
        "equal" => C::Equal,
        "less-equal" => C::LessEqual,
        "greater" => C::Greater,
        "not-equal" => C::NotEqual,
        "greater-equal" => C::GreaterEqual,
        _ => C::Always,
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pipeline_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createRenderPipeline") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);
        let label = map_get(desc, "label").and_then(|v| string_of(v));

        // -- Layout: "auto" | id ----------------------------------
        enum LayoutChoice {
            Auto,
            Id(u64),
        }
        let layout_choice = match map_get(desc, "layout") {
            Some(v) => {
                if let Some(s) = string_of(v) {
                    if s == "auto" {
                        LayoutChoice::Auto
                    } else {
                        ctx(vm).runtime_error(format!(
                            "RenderPipeline.create: layout '{}' not recognized; use 'auto' or a layout id.",
                            s
                        ));
                        return;
                    }
                } else if let Some(n) = v.as_num() {
                    LayoutChoice::Id(n as u64)
                } else {
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: `layout` must be 'auto' or a layout id."
                            .to_string(),
                    );
                    return;
                }
            }
            None => LayoutChoice::Auto,
        };

        // -- Vertex stage ----------------------------------------
        let vertex_v = match map_get(desc, "vertex") {
            Some(v) => v,
            None => {
                ctx(vm).runtime_error(
                    "RenderPipeline.create: descriptor must include `vertex`.".to_string(),
                );
                return;
            }
        };
        let v_module_id = match map_get(vertex_v, "module").and_then(|v| v.as_num()) {
            Some(n) => n as u64,
            None => {
                ctx(vm).runtime_error("RenderPipeline.create: vertex.module missing.".to_string());
                return;
            }
        };
        let v_entry = match map_get(vertex_v, "entryPoint").and_then(|v| string_of(v)) {
            Some(s) => s,
            None => {
                ctx(vm)
                    .runtime_error("RenderPipeline.create: vertex.entryPoint missing.".to_string());
                return;
            }
        };

        // Vertex buffer layouts. Each layout owns its attributes
        // Vec; both have to outlive the wgpu pipeline create call.
        struct OwnedLayout {
            stride: u64,
            step_mode: wgpu::VertexStepMode,
            attrs: Vec<wgpu::VertexAttribute>,
        }
        let mut owned_layouts: Vec<OwnedLayout> = Vec::new();
        if let Some(buffers_v) = map_get(vertex_v, "buffers") {
            let buffers_list = match list_view(buffers_v) {
                Some(l) => l,
                None => {
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: vertex.buffers must be a list.".to_string(),
                    );
                    return;
                }
            };
            for i in 0..buffers_list.count as usize {
                let layout = list_get(buffers_list, i);
                let stride = match map_get(layout, "arrayStride").and_then(|v| v.as_num()) {
                    Some(n) => n as u64,
                    None => {
                        ctx(vm).runtime_error(format!(
                            "vertex.buffers[{}]: `arrayStride` missing.",
                            i
                        ));
                        return;
                    }
                };
                let step_mode = match map_get(layout, "stepMode").and_then(|v| string_of(v)) {
                    Some(s) if s == "instance" => wgpu::VertexStepMode::Instance,
                    _ => wgpu::VertexStepMode::Vertex,
                };
                let attrs_v = match map_get(layout, "attributes") {
                    Some(v) => v,
                    None => {
                        ctx(vm)
                            .runtime_error(format!("vertex.buffers[{}]: `attributes` missing.", i));
                        return;
                    }
                };
                let attrs_list = match list_view(attrs_v) {
                    Some(l) => l,
                    None => {
                        ctx(vm).runtime_error(format!(
                            "vertex.buffers[{}]: `attributes` must be a list.",
                            i
                        ));
                        return;
                    }
                };
                let mut attrs: Vec<wgpu::VertexAttribute> = Vec::new();
                for j in 0..attrs_list.count as usize {
                    let attr = list_get(attrs_list, j);
                    let location = match map_get(attr, "shaderLocation").and_then(|v| v.as_num()) {
                        Some(n) => n as u32,
                        None => {
                            ctx(vm).runtime_error(format!(
                                "vertex.buffers[{}].attributes[{}]: `shaderLocation` missing.",
                                i, j
                            ));
                            return;
                        }
                    };
                    let offset = map_get(attr, "offset")
                        .and_then(|v| v.as_num())
                        .map(|n| n as u64)
                        .unwrap_or(0);
                    let format = match map_get(attr, "format").and_then(|v| string_of(v)) {
                        Some(s) => match vertex_format_from_str(&s) {
                            Some(f) => f,
                            None => {
                                ctx(vm).runtime_error(format!(
                                    "vertex.buffers[{}].attributes[{}]: unknown format '{}'.",
                                    i, j, s
                                ));
                                return;
                            }
                        },
                        None => {
                            ctx(vm).runtime_error(format!(
                                "vertex.buffers[{}].attributes[{}]: `format` missing.",
                                i, j
                            ));
                            return;
                        }
                    };
                    attrs.push(wgpu::VertexAttribute {
                        shader_location: location,
                        offset,
                        format,
                    });
                }
                owned_layouts.push(OwnedLayout {
                    stride,
                    step_mode,
                    attrs,
                });
            }
        }

        // -- Fragment stage --------------------------------------
        // Optional but almost always present (color output).
        let fragment_v = map_get(desc, "fragment");
        struct FragInfo {
            module_id: u64,
            entry: String,
            targets: Vec<Option<wgpu::ColorTargetState>>,
        }
        let frag_info = if let Some(fv) = fragment_v {
            let module_id = match map_get(fv, "module").and_then(|v| v.as_num()) {
                Some(n) => n as u64,
                None => {
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: fragment.module missing.".to_string(),
                    );
                    return;
                }
            };
            let entry = match map_get(fv, "entryPoint").and_then(|v| string_of(v)) {
                Some(s) => s,
                None => {
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: fragment.entryPoint missing.".to_string(),
                    );
                    return;
                }
            };
            let targets_v = match map_get(fv, "targets") {
                Some(v) => v,
                None => {
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: fragment.targets missing.".to_string(),
                    );
                    return;
                }
            };
            let targets_list = match list_view(targets_v) {
                Some(l) => l,
                None => {
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: fragment.targets must be a list.".to_string(),
                    );
                    return;
                }
            };
            let mut targets: Vec<Option<wgpu::ColorTargetState>> = Vec::new();
            for i in 0..targets_list.count as usize {
                let t = list_get(targets_list, i);
                let format = match map_get(t, "format").and_then(|v| string_of(v)) {
                    Some(s) => match texture_format_from_str(&s) {
                        Some(f) => f,
                        None => {
                            ctx(vm).runtime_error(format!(
                                "fragment.targets[{}]: unknown format '{}'.",
                                i, s
                            ));
                            return;
                        }
                    },
                    None => {
                        ctx(vm)
                            .runtime_error(format!("fragment.targets[{}]: `format` missing.", i));
                        return;
                    }
                };
                targets.push(Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }));
            }
            Some(FragInfo {
                module_id,
                entry,
                targets,
            })
        } else {
            None
        };

        // -- Primitive --------------------------------------------
        let primitive = if let Some(pv) = map_get(desc, "primitive") {
            let topology = map_get(pv, "topology")
                .and_then(|v| string_of(v))
                .map(|s| topology_from_str(&s))
                .unwrap_or(wgpu::PrimitiveTopology::TriangleList);
            let cull_mode = match map_get(pv, "cullMode").and_then(|v| string_of(v)) {
                Some(s) if s == "front" => Some(wgpu::Face::Front),
                Some(s) if s == "back" => Some(wgpu::Face::Back),
                _ => None,
            };
            let front_face = match map_get(pv, "frontFace").and_then(|v| string_of(v)) {
                Some(s) if s == "cw" => wgpu::FrontFace::Cw,
                _ => wgpu::FrontFace::Ccw,
            };
            wgpu::PrimitiveState {
                topology,
                cull_mode,
                front_face,
                ..Default::default()
            }
        } else {
            wgpu::PrimitiveState::default()
        };

        // -- DepthStencil ----------------------------------------
        let depth_stencil = if let Some(dv) = map_get(desc, "depthStencil") {
            let format = match map_get(dv, "format").and_then(|v| string_of(v)) {
                Some(s) => match texture_format_from_str(&s) {
                    Some(f) => f,
                    None => {
                        ctx(vm).runtime_error(format!("depthStencil.format: unknown '{}'.", s));
                        return;
                    }
                },
                None => {
                    ctx(vm).runtime_error("depthStencil: `format` missing.".to_string());
                    return;
                }
            };
            let depth_write_enabled = map_get(dv, "depthWriteEnabled")
                .map(|v| {
                    // Treat any non-falsy bool as true. Wren has
                    // exactly two boolean singletons; missing key
                    // defaults to true (the common case).
                    !v.is_falsy()
                })
                .unwrap_or(true);
            let depth_compare = map_get(dv, "depthCompare")
                .and_then(|v| string_of(v))
                .map(|s| compare_from_str(&s))
                .unwrap_or(wgpu::CompareFunction::Less);
            Some(wgpu::DepthStencilState {
                format,
                depth_write_enabled,
                depth_compare,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        } else {
            None
        };

        // -- Build the pipeline ----------------------------------
        let shader_reg = shaders().lock().unwrap();
        let v_module = match shader_reg.shaders.get(&v_module_id) {
            Some(m) => m,
            None => {
                drop(shader_reg);
                ctx(vm)
                    .runtime_error("RenderPipeline.create: unknown vertex shader id.".to_string());
                return;
            }
        };
        let f_module_opt = match &frag_info {
            Some(f) => match shader_reg.shaders.get(&f.module_id) {
                Some(m) => Some(m),
                None => {
                    drop(shader_reg);
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: unknown fragment shader id.".to_string(),
                    );
                    return;
                }
            },
            None => None,
        };
        let pl_reg = pipeline_layouts().lock().unwrap();
        let layout_ref = match &layout_choice {
            LayoutChoice::Auto => None,
            LayoutChoice::Id(id) => match pl_reg.layouts.get(id) {
                Some(l) => Some(l),
                None => {
                    drop(pl_reg);
                    drop(shader_reg);
                    ctx(vm).runtime_error(
                        "RenderPipeline.create: unknown pipeline layout id.".to_string(),
                    );
                    return;
                }
            },
        };

        // Build vertex buffer layout refs from owned data — the
        // attrs vecs are addressed via `&attrs[..]` so they live
        // for the duration of the create call.
        let vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = owned_layouts
            .iter()
            .map(|ol| wgpu::VertexBufferLayout {
                array_stride: ol.stride,
                step_mode: ol.step_mode,
                attributes: &ol.attrs,
            })
            .collect();

        let pipeline = {
            let dev_reg = devices().lock().unwrap();
            let dev = match dev_reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    drop(pl_reg);
                    drop(shader_reg);
                    ctx(vm).runtime_error("RenderPipeline.create: unknown device id.".to_string());
                    return;
                }
            };

            dev.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: label.as_deref(),
                    layout: layout_ref,
                    vertex: wgpu::VertexState {
                        module: v_module,
                        entry_point: &v_entry,
                        compilation_options: Default::default(),
                        buffers: &vertex_buffer_layouts,
                    },
                    fragment: frag_info.as_ref().map(|f| wgpu::FragmentState {
                        module: f_module_opt.unwrap(),
                        entry_point: &f.entry,
                        compilation_options: Default::default(),
                        targets: &f.targets,
                    }),
                    primitive,
                    depth_stencil,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
        };
        drop(pl_reg);
        drop(shader_reg);

        let id = next_id();
        render_pipelines()
            .lock()
            .unwrap()
            .pipelines
            .insert(id, pipeline);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_render_pipeline_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "RenderPipeline.destroy") {
            Some(i) => i,
            None => return,
        };
        render_pipelines().lock().unwrap().pipelines.remove(&id);
        set_return(vm, Value::null());
    }
}

// ---------------------------------------------------------------------------
// CommandEncoder + RenderPass (record-and-replay)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_encoder_create(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createCommandEncoder") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);
        let label = map_get(desc, "label").and_then(|v| string_of(v));
        let encoder = {
            let reg = devices().lock().unwrap();
            let dev = match reg.devices.get(&device_id) {
                Some(d) => d,
                None => {
                    ctx(vm).runtime_error("CommandEncoder.create: unknown device id.".to_string());
                    return;
                }
            };
            dev.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: label.as_deref(),
                })
        };
        let id = next_id();
        encoders()
            .lock()
            .unwrap()
            .encoders
            .insert(id, EncoderState::Open { encoder, device_id });
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_encoder_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "CommandEncoder.destroy") {
            Some(i) => i,
            None => return,
        };
        encoders().lock().unwrap().encoders.remove(&id);
        set_return(vm, Value::null());
    }
}

/// Decode a `loadOp` string into a `wgpu::LoadOp<wgpu::Color>`. The
/// `clearValue` Map / List<Num> on the descriptor supplies the clear
/// color when load == "clear".
unsafe fn load_op_color(attachment: Value) -> wgpu::LoadOp<wgpu::Color> {
    let op = unsafe { map_get(attachment, "loadOp").and_then(|v| string_of(v)) };
    match op.as_deref() {
        Some("load") => wgpu::LoadOp::Load,
        _ => {
            // Default to clear so a missing loadOp is the safe choice.
            let clear_v = unsafe { map_get(attachment, "clearValue") };
            let mut color = wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            };
            if let Some(v) = clear_v {
                if let Some(list) = unsafe { list_view(v) } {
                    let n = list.count as usize;
                    if n > 0 {
                        color.r = unsafe { list_get(list, 0) }.as_num().unwrap_or(0.0);
                    }
                    if n > 1 {
                        color.g = unsafe { list_get(list, 1) }.as_num().unwrap_or(0.0);
                    }
                    if n > 2 {
                        color.b = unsafe { list_get(list, 2) }.as_num().unwrap_or(0.0);
                    }
                    if n > 3 {
                        color.a = unsafe { list_get(list, 3) }.as_num().unwrap_or(1.0);
                    }
                }
            }
            wgpu::LoadOp::Clear(color)
        }
    }
}

unsafe fn store_op(attachment: Value) -> wgpu::StoreOp {
    let op = unsafe { map_get(attachment, "storeOp").and_then(|v| string_of(v)) };
    match op.as_deref() {
        Some("discard") => wgpu::StoreOp::Discard,
        _ => wgpu::StoreOp::Store,
    }
}

/// Record a render pass into the named encoder.
///
/// Wren wrapper drives this via:
/// ```
/// pass = encoder.beginRenderPass({...descriptor...})
/// pass.setPipeline(p); pass.draw(3); ... ; pass.end
/// ```
/// Internally `pass.end` packages every recorded command into the
/// descriptor's `commands` list and calls this single foreign
/// function. Skips the wgpu-side `RenderPass<'a>` lifetime knot.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_encoder_record_pass(vm: *mut VM) {
    unsafe {
        let encoder_id = match id_of(ctx(vm), slot(vm, 1), "RenderPass.end") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        // Color attachments — we own ResolvedColorAttachment so the
        // texture views borrowed from the registry stay alive for
        // the begin_render_pass call.
        let color_v = match map_get(desc, "colorAttachments") {
            Some(v) => v,
            None => {
                ctx(vm).runtime_error(
                    "RenderPass: descriptor must include `colorAttachments`.".to_string(),
                );
                return;
            }
        };
        let color_list = match list_view(color_v) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error("RenderPass: `colorAttachments` must be a list.".to_string());
                return;
            }
        };

        struct Color {
            view_id: u64,
            load: wgpu::LoadOp<wgpu::Color>,
            store: wgpu::StoreOp,
        }
        let mut colors: Vec<Color> = Vec::with_capacity(color_list.count as usize);
        for i in 0..color_list.count as usize {
            let att = list_get(color_list, i);
            let view_id = match map_get(att, "view").and_then(|v| v.as_num()) {
                Some(n) => n as u64,
                None => {
                    ctx(vm).runtime_error(format!(
                        "RenderPass: colorAttachments[{}].view missing.",
                        i
                    ));
                    return;
                }
            };
            colors.push(Color {
                view_id,
                load: load_op_color(att),
                store: store_op(att),
            });
        }

        // Optional depth-stencil attachment.
        struct Depth {
            view_id: u64,
            depth_load: wgpu::LoadOp<f32>,
            depth_store: wgpu::StoreOp,
        }
        let depth = if let Some(d) = map_get(desc, "depthStencilAttachment") {
            let view_id = match map_get(d, "view").and_then(|v| v.as_num()) {
                Some(n) => n as u64,
                None => {
                    ctx(vm).runtime_error(
                        "RenderPass: depthStencilAttachment.view missing.".to_string(),
                    );
                    return;
                }
            };
            let depth_load = match map_get(d, "depthLoadOp")
                .and_then(|v| string_of(v))
                .as_deref()
            {
                Some("load") => wgpu::LoadOp::Load,
                _ => wgpu::LoadOp::Clear(
                    map_get(d, "depthClearValue")
                        .and_then(|v| v.as_num())
                        .map(|n| n as f32)
                        .unwrap_or(1.0),
                ),
            };
            let depth_store = match map_get(d, "depthStoreOp")
                .and_then(|v| string_of(v))
                .as_deref()
            {
                Some("discard") => wgpu::StoreOp::Discard,
                _ => wgpu::StoreOp::Store,
            };
            Some(Depth {
                view_id,
                depth_load,
                depth_store,
            })
        } else {
            None
        };

        // Decode the commands list.
        let commands_v = match map_get(desc, "commands") {
            Some(v) => v,
            None => {
                ctx(vm).runtime_error("RenderPass: missing `commands` list.".to_string());
                return;
            }
        };
        let commands_list = match list_view(commands_v) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error("RenderPass: `commands` must be a list.".to_string());
                return;
            }
        };

        enum Cmd {
            SetPipeline(u64),
            SetVertexBuffer {
                slot: u32,
                buffer: u64,
            },
            SetIndexBuffer {
                buffer: u64,
                format: wgpu::IndexFormat,
            },
            SetBindGroup {
                index: u32,
                group: u64,
            },
            Draw {
                vertex_count: u32,
                instance_count: u32,
            },
            DrawIndexed {
                index_count: u32,
                instance_count: u32,
            },
        }
        let mut decoded: Vec<Cmd> = Vec::with_capacity(commands_list.count as usize);
        for i in 0..commands_list.count as usize {
            let cmd = list_get(commands_list, i);
            let op = match map_get(cmd, "op").and_then(|v| string_of(v)) {
                Some(s) => s,
                None => {
                    ctx(vm)
                        .runtime_error(format!("RenderPass commands[{}]: missing `op` string.", i));
                    return;
                }
            };
            match op.as_str() {
                "setPipeline" => {
                    let pid = map_get(cmd, "pipeline")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u64;
                    decoded.push(Cmd::SetPipeline(pid));
                }
                "setVertexBuffer" => {
                    let s = map_get(cmd, "slot").and_then(|v| v.as_num()).unwrap_or(0.0) as u32;
                    let b = map_get(cmd, "buffer")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u64;
                    decoded.push(Cmd::SetVertexBuffer { slot: s, buffer: b });
                }
                "setIndexBuffer" => {
                    let b = map_get(cmd, "buffer")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u64;
                    let f = match map_get(cmd, "format").and_then(|v| string_of(v)).as_deref() {
                        Some("uint32") => wgpu::IndexFormat::Uint32,
                        _ => wgpu::IndexFormat::Uint16,
                    };
                    decoded.push(Cmd::SetIndexBuffer {
                        buffer: b,
                        format: f,
                    });
                }
                "setBindGroup" => {
                    let idx = map_get(cmd, "index")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u32;
                    let g = map_get(cmd, "group")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u64;
                    decoded.push(Cmd::SetBindGroup {
                        index: idx,
                        group: g,
                    });
                }
                "draw" => {
                    let vc = map_get(cmd, "vertexCount")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u32;
                    let ic = map_get(cmd, "instanceCount")
                        .and_then(|v| v.as_num())
                        .unwrap_or(1.0) as u32;
                    decoded.push(Cmd::Draw {
                        vertex_count: vc,
                        instance_count: ic,
                    });
                }
                "drawIndexed" => {
                    let ic = map_get(cmd, "indexCount")
                        .and_then(|v| v.as_num())
                        .unwrap_or(0.0) as u32;
                    let inst = map_get(cmd, "instanceCount")
                        .and_then(|v| v.as_num())
                        .unwrap_or(1.0) as u32;
                    decoded.push(Cmd::DrawIndexed {
                        index_count: ic,
                        instance_count: inst,
                    });
                }
                other => {
                    ctx(vm).runtime_error(format!(
                        "RenderPass commands[{}]: unknown op '{}'.",
                        i, other
                    ));
                    return;
                }
            }
        }

        // Begin pass + replay + end. Lock all resource registries
        // to keep refs alive for the duration of the render pass.
        let mut enc_reg = encoders().lock().unwrap();
        let enc = match enc_reg.encoders.get_mut(&encoder_id) {
            Some(e) => e,
            None => {
                drop(enc_reg);
                ctx(vm).runtime_error("RenderPass: unknown encoder id.".to_string());
                return;
            }
        };
        let encoder = match enc {
            EncoderState::Open { encoder, .. } => encoder,
            EncoderState::Finished { .. } => {
                drop(enc_reg);
                ctx(vm).runtime_error(
                    "RenderPass: encoder already finished; create a new one.".to_string(),
                );
                return;
            }
        };

        let view_reg = views().lock().unwrap();
        let pipe_reg = render_pipelines().lock().unwrap();
        let buf_reg = buffers().lock().unwrap();
        let bg_reg = bind_groups().lock().unwrap();

        // Build attachment ref structs against the registries.
        let color_ref_specs: Vec<wgpu::RenderPassColorAttachment> = colors
            .iter()
            .filter_map(|c| {
                view_reg
                    .views
                    .get(&c.view_id)
                    .map(|v| wgpu::RenderPassColorAttachment {
                        view: v,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: c.load,
                            store: c.store,
                        },
                    })
            })
            .collect();
        if color_ref_specs.len() != colors.len() {
            drop(bg_reg);
            drop(buf_reg);
            drop(pipe_reg);
            drop(view_reg);
            drop(enc_reg);
            ctx(vm).runtime_error("RenderPass: unknown color attachment view id.".to_string());
            return;
        }
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> =
            color_ref_specs.into_iter().map(Some).collect();

        let depth_attachment = if let Some(d) = &depth {
            match view_reg.views.get(&d.view_id) {
                Some(v) => Some(wgpu::RenderPassDepthStencilAttachment {
                    view: v,
                    depth_ops: Some(wgpu::Operations {
                        load: d.depth_load,
                        store: d.depth_store,
                    }),
                    stencil_ops: None,
                }),
                None => {
                    drop(bg_reg);
                    drop(buf_reg);
                    drop(pipe_reg);
                    drop(view_reg);
                    drop(enc_reg);
                    ctx(vm)
                        .runtime_error("RenderPass: unknown depth attachment view id.".to_string());
                    return;
                }
            }
        } else {
            None
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            for cmd in &decoded {
                match cmd {
                    Cmd::SetPipeline(pid) => {
                        if let Some(p) = pipe_reg.pipelines.get(pid) {
                            pass.set_pipeline(p);
                        }
                    }
                    Cmd::SetVertexBuffer { slot, buffer } => {
                        if let Some(b) = buf_reg.buffers.get(buffer) {
                            pass.set_vertex_buffer(*slot, b.buffer.slice(..));
                        }
                    }
                    Cmd::SetIndexBuffer { buffer, format } => {
                        if let Some(b) = buf_reg.buffers.get(buffer) {
                            pass.set_index_buffer(b.buffer.slice(..), *format);
                        }
                    }
                    Cmd::SetBindGroup { index, group } => {
                        if let Some(g) = bg_reg.groups.get(group) {
                            pass.set_bind_group(*index, g, &[]);
                        }
                    }
                    Cmd::Draw {
                        vertex_count,
                        instance_count,
                    } => {
                        pass.draw(0..*vertex_count, 0..*instance_count);
                    }
                    Cmd::DrawIndexed {
                        index_count,
                        instance_count,
                    } => {
                        pass.draw_indexed(0..*index_count, 0, 0..*instance_count);
                    }
                }
            }
            // pass dropped here, ending the render pass
        }

        drop(bg_reg);
        drop(buf_reg);
        drop(pipe_reg);
        drop(view_reg);
        drop(enc_reg);
        set_return(vm, Value::null());
    }
}

/// `encoder.copyTextureToBuffer(srcTextureId, dstBufferId, descriptor)`
///
/// Descriptor:
///   { "width": u32, "height": u32,
///     "bytesPerRow": u32?,    // computed if absent
///     "rowsPerImage": u32? }
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_encoder_copy_texture_to_buffer(vm: *mut VM) {
    unsafe {
        let encoder_id = match id_of(ctx(vm), slot(vm, 1), "CommandEncoder.copyTextureToBuffer") {
            Some(i) => i,
            None => return,
        };
        let texture_id = match id_of(ctx(vm), slot(vm, 2), "CommandEncoder.copyTextureToBuffer") {
            Some(i) => i,
            None => return,
        };
        let buffer_id = match id_of(ctx(vm), slot(vm, 3), "CommandEncoder.copyTextureToBuffer") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 4);
        let width = map_get(desc, "width")
            .and_then(|v| v.as_num())
            .unwrap_or(0.0) as u32;
        let height = map_get(desc, "height")
            .and_then(|v| v.as_num())
            .unwrap_or(0.0) as u32;

        let mut enc_reg = encoders().lock().unwrap();
        let tex_reg = textures().lock().unwrap();
        let buf_reg = buffers().lock().unwrap();
        let enc = match enc_reg.encoders.get_mut(&encoder_id) {
            Some(EncoderState::Open { encoder, .. }) => encoder,
            _ => {
                drop(buf_reg);
                drop(tex_reg);
                drop(enc_reg);
                ctx(vm)
                    .runtime_error("copyTextureToBuffer: encoder not in Open state.".to_string());
                return;
            }
        };
        let tex = match tex_reg.textures.get(&texture_id) {
            Some(t) => t,
            None => {
                drop(buf_reg);
                drop(tex_reg);
                drop(enc_reg);
                ctx(vm).runtime_error("copyTextureToBuffer: unknown texture id.".to_string());
                return;
            }
        };
        let buf = match buf_reg.buffers.get(&buffer_id) {
            Some(b) => b,
            None => {
                drop(buf_reg);
                drop(tex_reg);
                drop(enc_reg);
                ctx(vm).runtime_error("copyTextureToBuffer: unknown buffer id.".to_string());
                return;
            }
        };

        let bpp = format_bytes_per_pixel(tex.format);
        let bytes_per_row = map_get(desc, "bytesPerRow")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(width * bpp);
        let rows_per_image = map_get(desc, "rowsPerImage")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(height);

        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &tex.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows_per_image),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        drop(buf_reg);
        drop(tex_reg);
        drop(enc_reg);
        set_return(vm, Value::null());
    }
}

/// Finish recording, transition to `Finished` state. Subsequent
/// `submit` calls consume the CommandBuffer.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_encoder_finish(vm: *mut VM) {
    unsafe {
        let encoder_id = match id_of(ctx(vm), slot(vm, 1), "CommandEncoder.finish") {
            Some(i) => i,
            None => return,
        };
        let mut reg = encoders().lock().unwrap();
        let entry = match reg.encoders.remove(&encoder_id) {
            Some(e) => e,
            None => {
                drop(reg);
                ctx(vm).runtime_error("CommandEncoder.finish: unknown encoder id.".to_string());
                return;
            }
        };
        let new_state = match entry {
            EncoderState::Open { encoder, device_id } => EncoderState::Finished {
                buffer: encoder.finish(),
                device_id,
            },
            EncoderState::Finished { .. } => {
                drop(reg);
                ctx(vm).runtime_error("CommandEncoder.finish: already finished.".to_string());
                return;
            }
        };
        reg.encoders.insert(encoder_id, new_state);
        drop(reg);
        set_return(vm, Value::null());
    }
}

/// Submit a list of finished encoder ids to their device's queue.
/// Encoders transition out of the registry as their CommandBuffer
/// is consumed.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_queue_submit(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Queue.submit") {
            Some(i) => i,
            None => return,
        };
        let list = match list_view(slot(vm, 2)) {
            Some(l) => l,
            None => {
                ctx(vm).runtime_error(
                    "Queue.submit: argument must be a list of encoder ids.".to_string(),
                );
                return;
            }
        };
        let mut buffers_to_submit: Vec<wgpu::CommandBuffer> = Vec::new();
        {
            let mut reg = encoders().lock().unwrap();
            for i in 0..list.count as usize {
                let id = match list_get(list, i).as_num() {
                    Some(n) => n as u64,
                    None => {
                        drop(reg);
                        ctx(vm).runtime_error(format!(
                            "Queue.submit: list[{}] must be an encoder id.",
                            i
                        ));
                        return;
                    }
                };
                match reg.encoders.remove(&id) {
                    Some(EncoderState::Finished { buffer, .. }) => {
                        buffers_to_submit.push(buffer);
                    }
                    Some(other) => {
                        // Put it back since we couldn't use it.
                        reg.encoders.insert(id, other);
                        drop(reg);
                        ctx(vm).runtime_error(format!(
                            "Queue.submit: encoder {} not finished — call .finish first.",
                            id
                        ));
                        return;
                    }
                    None => {
                        drop(reg);
                        ctx(vm).runtime_error(format!("Queue.submit: unknown encoder id {}.", id));
                        return;
                    }
                }
            }
        }
        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&device_id) {
            Some(d) => d,
            None => {
                drop(dev_reg);
                ctx(vm).runtime_error("Queue.submit: unknown device id.".to_string());
                return;
            }
        };
        dev.queue.submit(buffers_to_submit);
        drop(dev_reg);
        set_return(vm, Value::null());
    }
}

/// Synchronously read bytes back from a buffer. Maps the buffer
/// for read, blocks via `device.poll(Maintain::Wait)` until the GPU
/// catches up, copies the mapped range into a Wren `List<Num>` (one
/// number per byte), then unmaps. Used by the headless render specs
/// that copy a texture into a readback buffer + verify pixel values.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_buffer_read_bytes(vm: *mut VM) {
    unsafe {
        let buffer_id = match id_of(ctx(vm), slot(vm, 1), "Buffer.readBytes") {
            Some(i) => i,
            None => return,
        };

        // Hold the registry locks while reading so the buffer +
        // device don't get destroyed mid-map.
        let buf_reg = buffers().lock().unwrap();
        let buf = match buf_reg.buffers.get(&buffer_id) {
            Some(b) => b,
            None => {
                drop(buf_reg);
                ctx(vm).runtime_error("Buffer.readBytes: unknown buffer id.".to_string());
                return;
            }
        };
        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&buf.device_id) {
            Some(d) => d,
            None => {
                drop(dev_reg);
                drop(buf_reg);
                ctx(vm).runtime_error("Buffer.readBytes: device dropped.".to_string());
                return;
            }
        };

        let slice = buf.buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        // Block until the GPU finishes pending work and the map
        // completes. Maintain::Wait is the right knob for tests
        // and one-shot CPU readback; an async/fiber-yielding map
        // path is a planned addition for long-running pipelines
        // that shouldn't stall the host.
        dev.device.poll(wgpu::Maintain::Wait);
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                drop(dev_reg);
                drop(buf_reg);
                ctx(vm).runtime_error(format!("Buffer.readBytes: map failed: {}", e));
                return;
            }
            Err(e) => {
                drop(dev_reg);
                drop(buf_reg);
                ctx(vm).runtime_error(format!("Buffer.readBytes: channel: {}", e));
                return;
            }
        }
        let data = slice.get_mapped_range();
        let bytes: Vec<u8> = data.to_vec();
        drop(data);
        buf.buffer.unmap();
        drop(dev_reg);
        drop(buf_reg);

        // Build the result list. Each byte → one Wren Num.
        let context = ctx(vm);
        let elements: Vec<Value> = bytes.into_iter().map(|b| Value::num(b as f64)).collect();
        let list = context.alloc_list(elements);
        set_return(vm, list);
    }
}

// ---------------------------------------------------------------------------
// Surface — bring-your-own-window
// ---------------------------------------------------------------------------
//
// `Device.createSurface(handle)` accepts a platform-tagged Map
// of raw window-handle integers and produces a `wgpu::Surface`
// from them. Lets any embedder — the default winit-backed
// @hatch:window package, an IDE viewport, a future C/host shell
// — bring its own window without dragging winit into this dylib.
//
// Handle Map shape (all keys "Num"; pointers passed as integers
// — native side reinterprets via NonNull):
//
//   AppKit (macOS):     "platform": "appkit",  "ns_view": Num
//   UIKit (iOS):        "platform": "uikit",   "ui_view": Num
//   Win32:              "platform": "win32",   "hwnd": Num,
//                                              "hinstance": Num
//   Xlib (Linux X11):   "platform": "xlib",    "window": Num,
//                                              "display": Num
//   Xcb (Linux X11):    "platform": "xcb",     "window": Num,
//                                              "connection": Num
//   Wayland:            "platform": "wayland", "surface": Num,
//                                              "display": Num
//
// The caller is responsible for keeping the underlying window
// alive — wgpu won't pin it. If the window dies before the
// Surface, configure / acquire surface a runtime error rather
// than crashing.

unsafe fn nonnull_ptr_from_num(v: Value) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
    let n = v.as_num()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    let raw = n as u64 as usize as *mut std::ffi::c_void;
    std::ptr::NonNull::new(raw)
}

unsafe fn decode_window_handle(
    vm: &mut VM,
    desc: Value,
) -> Option<(
    raw_window_handle::RawWindowHandle,
    raw_window_handle::RawDisplayHandle,
)> {
    use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

    let platform = match unsafe { map_get(desc, "platform").and_then(|v| string_of(v)) } {
        Some(s) => s,
        None => {
            vm.runtime_error(
                "Device.createSurface: descriptor must include `platform` string.".to_string(),
            );
            return None;
        }
    };

    match platform.as_str() {
        "appkit" => {
            let view =
                match unsafe { map_get(desc, "ns_view").and_then(|v| nonnull_ptr_from_num(v)) } {
                    Some(p) => p,
                    None => {
                        vm.runtime_error(
                            "Device.createSurface: appkit requires `ns_view` (NSView*) integer."
                                .to_string(),
                        );
                        return None;
                    }
                };
            let mut h = raw_window_handle::AppKitWindowHandle::new(view);
            let _ = &mut h; // suppress potential unused-mut on platforms that don't compile this branch
            Some((
                RawWindowHandle::AppKit(raw_window_handle::AppKitWindowHandle::new(view)),
                RawDisplayHandle::AppKit(raw_window_handle::AppKitDisplayHandle::new()),
            ))
        }
        "uikit" => {
            let view =
                match unsafe { map_get(desc, "ui_view").and_then(|v| nonnull_ptr_from_num(v)) } {
                    Some(p) => p,
                    None => {
                        vm.runtime_error(
                            "Device.createSurface: uikit requires `ui_view` (UIView*) integer."
                                .to_string(),
                        );
                        return None;
                    }
                };
            Some((
                RawWindowHandle::UiKit(raw_window_handle::UiKitWindowHandle::new(view)),
                RawDisplayHandle::UiKit(raw_window_handle::UiKitDisplayHandle::new()),
            ))
        }
        "win32" => {
            let hwnd_n = match unsafe { map_get(desc, "hwnd").and_then(|v| v.as_num()) } {
                Some(n) if n.is_finite() => n as i64,
                _ => {
                    vm.runtime_error(
                        "Device.createSurface: win32 requires `hwnd` integer.".to_string(),
                    );
                    return None;
                }
            };
            let hinstance_n = unsafe {
                map_get(desc, "hinstance")
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.0)
            };
            let hwnd = match std::num::NonZeroIsize::new(hwnd_n as isize) {
                Some(h) => h,
                None => {
                    vm.runtime_error(
                        "Device.createSurface: win32 `hwnd` must be non-zero.".to_string(),
                    );
                    return None;
                }
            };
            let mut wh = raw_window_handle::Win32WindowHandle::new(hwnd);
            wh.hinstance = std::num::NonZeroIsize::new(hinstance_n as isize);
            Some((
                RawWindowHandle::Win32(wh),
                RawDisplayHandle::Windows(raw_window_handle::WindowsDisplayHandle::new()),
            ))
        }
        "xlib" => {
            let window_id = match unsafe { map_get(desc, "window").and_then(|v| v.as_num()) } {
                Some(n) if n.is_finite() => n as u64,
                _ => {
                    vm.runtime_error(
                        "Device.createSurface: xlib requires `window` (XID) integer.".to_string(),
                    );
                    return None;
                }
            };
            let display = unsafe { map_get(desc, "display").and_then(|v| nonnull_ptr_from_num(v)) };
            let visual_id = unsafe {
                map_get(desc, "visual_id")
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.0)
            } as u64;
            let mut wh = raw_window_handle::XlibWindowHandle::new(window_id);
            wh.visual_id = visual_id;
            Some((
                RawWindowHandle::Xlib(wh),
                RawDisplayHandle::Xlib(raw_window_handle::XlibDisplayHandle::new(display, 0)),
            ))
        }
        "xcb" => {
            let window_n = match unsafe { map_get(desc, "window").and_then(|v| v.as_num()) } {
                Some(n) if n.is_finite() && n > 0.0 => n as u32,
                _ => {
                    vm.runtime_error(
                        "Device.createSurface: xcb requires `window` integer (> 0).".to_string(),
                    );
                    return None;
                }
            };
            let window = match std::num::NonZeroU32::new(window_n) {
                Some(w) => w,
                None => return None,
            };
            let connection =
                unsafe { map_get(desc, "connection").and_then(|v| nonnull_ptr_from_num(v)) };
            let visual_n = unsafe {
                map_get(desc, "visual_id")
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.0)
            } as u32;
            let mut wh = raw_window_handle::XcbWindowHandle::new(window);
            wh.visual_id = std::num::NonZeroU32::new(visual_n);
            Some((
                RawWindowHandle::Xcb(wh),
                RawDisplayHandle::Xcb(raw_window_handle::XcbDisplayHandle::new(connection, 0)),
            ))
        }
        "wayland" => {
            let surface =
                match unsafe { map_get(desc, "surface").and_then(|v| nonnull_ptr_from_num(v)) } {
                    Some(p) => p,
                    None => {
                        vm.runtime_error(
                        "Device.createSurface: wayland requires `surface` (wl_surface*) integer."
                            .to_string(),
                    );
                        return None;
                    }
                };
            let display =
                match unsafe { map_get(desc, "display").and_then(|v| nonnull_ptr_from_num(v)) } {
                    Some(p) => p,
                    None => {
                        vm.runtime_error(
                        "Device.createSurface: wayland requires `display` (wl_display*) integer."
                            .to_string(),
                    );
                        return None;
                    }
                };
            Some((
                RawWindowHandle::Wayland(raw_window_handle::WaylandWindowHandle::new(surface)),
                RawDisplayHandle::Wayland(raw_window_handle::WaylandDisplayHandle::new(display)),
            ))
        }
        other => {
            vm.runtime_error(format!(
                "Device.createSurface: unknown platform '{}'.",
                other
            ));
            None
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_surface_create_from_handle(vm: *mut VM) {
    unsafe {
        let device_id = match id_of(ctx(vm), slot(vm, 1), "Device.createSurface") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let (raw_window, raw_display) = match decode_window_handle(ctx(vm), desc) {
            Some(pair) => pair,
            None => return,
        };

        // Build the unsafe target. wgpu 22 has a SAFETY contract:
        // the handles must remain valid for the surface's
        // lifetime; the embedder owns the window so it's their
        // job to enforce that.
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: raw_display,
            raw_window_handle: raw_window,
        };

        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&device_id) {
            Some(d) => d,
            None => {
                drop(dev_reg);
                ctx(vm).runtime_error("Device.createSurface: unknown device id.".to_string());
                return;
            }
        };

        let surface = match dev.instance.create_surface_unsafe(target) {
            Ok(s) => s,
            Err(e) => {
                drop(dev_reg);
                ctx(vm).runtime_error(format!("Device.createSurface: {}", e));
                return;
            }
        };

        // Pick a sensible default format from the surface's
        // capabilities — first sRGB candidate, else first.
        let caps = surface.get_capabilities(&dev.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
        drop(dev_reg);

        let id = next_id();
        surfaces().lock().unwrap().surfaces.insert(
            id,
            SurfaceRecord {
                surface,
                device_id,
                width: 0,
                height: 0,
                format,
            },
        );
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_surface_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Surface.destroy") {
            Some(i) => i,
            None => return,
        };
        surfaces().lock().unwrap().surfaces.remove(&id);
        set_return(vm, Value::null());
    }
}

/// Configure / re-configure the surface. Required before the
/// first `acquire`, and again on every window resize.
///
/// Descriptor:
///   "width":  Num,
///   "height": Num,
///   "format":      String?  (defaults to the picked sRGB or first cap)
///   "presentMode": String?  ("fifo" | "immediate" | "mailbox", default fifo)
///   "alphaMode":   String?  ("auto" | "opaque" | "premultiplied", default auto)

#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_surface_configure(vm: *mut VM) {
    unsafe {
        let surface_id = match id_of(ctx(vm), slot(vm, 1), "Surface.configure") {
            Some(i) => i,
            None => return,
        };
        let desc = slot(vm, 2);

        let width = match map_get(desc, "width").and_then(|v| v.as_num()) {
            Some(n) if n.is_finite() && n > 0.0 => n as u32,
            _ => {
                ctx(vm).runtime_error(
                    "Surface.configure: `width` must be a positive integer.".to_string(),
                );
                return;
            }
        };
        let height = match map_get(desc, "height").and_then(|v| v.as_num()) {
            Some(n) if n.is_finite() && n > 0.0 => n as u32,
            _ => {
                ctx(vm).runtime_error(
                    "Surface.configure: `height` must be a positive integer.".to_string(),
                );
                return;
            }
        };
        let override_format = map_get(desc, "format")
            .and_then(|v| string_of(v))
            .and_then(|s| texture_format_from_str(&s));

        let present_mode = match map_get(desc, "presentMode")
            .and_then(|v| string_of(v))
            .as_deref()
        {
            Some("immediate") => wgpu::PresentMode::Immediate,
            Some("mailbox") => wgpu::PresentMode::Mailbox,
            _ => wgpu::PresentMode::Fifo,
        };
        let alpha_mode = match map_get(desc, "alphaMode")
            .and_then(|v| string_of(v))
            .as_deref()
        {
            Some("opaque") => wgpu::CompositeAlphaMode::Opaque,
            Some("premultiplied") => wgpu::CompositeAlphaMode::PreMultiplied,
            _ => wgpu::CompositeAlphaMode::Auto,
        };

        let mut surf_reg = surfaces().lock().unwrap();
        let rec = match surf_reg.surfaces.get_mut(&surface_id) {
            Some(r) => r,
            None => {
                drop(surf_reg);
                ctx(vm).runtime_error("Surface.configure: unknown surface id.".to_string());
                return;
            }
        };
        let format = override_format.unwrap_or(rec.format);
        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&rec.device_id) {
            Some(d) => d,
            None => {
                drop(dev_reg);
                drop(surf_reg);
                ctx(vm).runtime_error("Surface.configure: surface's device is gone.".to_string());
                return;
            }
        };
        // wgpu panics on configure if the requested dimensions exceed
        // `max_texture_dimension_2d` for the active device. A user
        // resizing a window past that cap (HiDPI displays trip this
        // routinely on low-tier adapters) shouldn't crash the
        // process — clamp to the limit and surface the clamp via a
        // diagnostic write so it isn't silent.
        let max_dim = dev.device.limits().max_texture_dimension_2d.max(1);
        let cw = width.min(max_dim);
        let ch = height.min(max_dim);
        if cw != width || ch != height {
            eprintln!(
                "warning: Surface.configure clamped {}x{} to {}x{} (device max_texture_dimension_2d = {})",
                width, height, cw, ch, max_dim
            );
        }
        rec.surface.configure(
            &dev.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: cw,
                height: ch,
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: vec![],
            },
        );
        rec.width = cw;
        rec.height = ch;
        rec.format = format;
        drop(dev_reg);
        drop(surf_reg);
        set_return(vm, Value::null());
    }
}

/// Acquire the next swap-chain frame. Returns a `Map` with two
/// `Num` ids: `frame` (consumed by `surface_present_frame`) and
/// `view` (a TextureView usable in render-pass attachments).
///
/// On `Outdated` / `Lost` results, surface a runtime error so the
/// Wren caller can re-configure (typically on a window resize).
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_surface_acquire(vm: *mut VM) {
    unsafe {
        let surface_id = match id_of(ctx(vm), slot(vm, 1), "Surface.acquire") {
            Some(i) => i,
            None => return,
        };

        let surface_texture = {
            let surf_reg = surfaces().lock().unwrap();
            let rec = match surf_reg.surfaces.get(&surface_id) {
                Some(r) => r,
                None => {
                    drop(surf_reg);
                    ctx(vm).runtime_error("Surface.acquire: unknown surface id.".to_string());
                    return;
                }
            };
            match rec.surface.get_current_texture() {
                Ok(t) => t,
                Err(e) => {
                    drop(surf_reg);
                    ctx(vm).runtime_error(format!("Surface.acquire: {:?}", e));
                    return;
                }
            }
        };

        // Build a TextureView and stash both alongside the
        // SurfaceTexture so the view borrows safely until present.
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let view_id = next_id();
        views().lock().unwrap().views.insert(view_id, view);

        let frame_id = next_id();
        surface_frames().lock().unwrap().frames.insert(
            frame_id,
            SurfaceFrameRecord {
                surface_texture,
                view_id,
            },
        );

        // Return a Wren Map { "frame": frame_id, "view": view_id }.
        let context = ctx(vm);
        let map = context.alloc_map();
        let key_frame = context.alloc_string("frame".to_string());
        let key_view = context.alloc_string("view".to_string());
        let frame_v = Value::num(frame_id as f64);
        let view_v = Value::num(view_id as f64);
        let _ = context.call_method_on(map, "[_]=(_)", &[key_frame, frame_v]);
        let _ = context.call_method_on(map, "[_]=(_)", &[key_view, view_v]);
        set_return(vm, map);
    }
}

// ---------------------------------------------------------------------------
// queue.writeTexture
// ---------------------------------------------------------------------------
//
// CPU → texture upload. Image decoding lives in @hatch:image
// (the wlift_image plugin); this entry point just takes raw
// pixel bytes and a layout descriptor and routes them through
// `queue.write_texture`.

unsafe fn read_byte_buffer(vm: &mut VM, v: Value) -> Option<Vec<u8>> {
    use wren_lift::runtime::object::{ObjList, ObjType, ObjTypedArray, TypedArrayKind};

    if !v.is_object() {
        vm.runtime_error("Texture.fromImage: bytes must be a List<Num> or ByteArray.".to_string());
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
                        vm.runtime_error(format!("Texture.fromImage: byte {} not in 0..=255.", i));
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
                vm.runtime_error(
                    "Texture.fromImage: typed array must be ByteArray (u8).".to_string(),
                );
                None
            }
        }
        _ => {
            vm.runtime_error(
                "Texture.fromImage: bytes must be a List<Num> or ByteArray.".to_string(),
            );
            None
        }
    }
}

/// Direct byte → texture upload. The Wren caller already knows
/// the layout (width, height, bytes-per-row); this just wraps
/// `queue.write_texture`.
///
/// `descriptor`:
///   "x", "y": Num     // offset within the texture (default 0)
///   "width", "height": Num
///   "bytesPerRow":  Num
///   "rowsPerImage": Num   // optional, defaults to height
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_queue_write_texture(vm: *mut VM) {
    unsafe {
        let texture_id = match id_of(ctx(vm), slot(vm, 1), "Queue.writeTexture") {
            Some(i) => i,
            None => return,
        };
        let bytes = match read_byte_buffer(ctx(vm), slot(vm, 2)) {
            Some(b) => b,
            None => return,
        };
        let desc = slot(vm, 3);
        let width = map_get(desc, "width")
            .and_then(|v| v.as_num())
            .unwrap_or(0.0) as u32;
        let height = map_get(desc, "height")
            .and_then(|v| v.as_num())
            .unwrap_or(0.0) as u32;
        let x = map_get(desc, "x").and_then(|v| v.as_num()).unwrap_or(0.0) as u32;
        let y = map_get(desc, "y").and_then(|v| v.as_num()).unwrap_or(0.0) as u32;
        let bytes_per_row = match map_get(desc, "bytesPerRow").and_then(|v| v.as_num()) {
            Some(n) if n > 0.0 => n as u32,
            _ => {
                ctx(vm).runtime_error(
                    "Queue.writeTexture: descriptor must include `bytesPerRow`.".to_string(),
                );
                return;
            }
        };
        let rows_per_image = map_get(desc, "rowsPerImage")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(height);

        let tex_reg = textures().lock().unwrap();
        let rec = match tex_reg.textures.get(&texture_id) {
            Some(t) => t,
            None => {
                drop(tex_reg);
                ctx(vm).runtime_error("Queue.writeTexture: unknown texture id.".to_string());
                return;
            }
        };
        let dev_reg = devices().lock().unwrap();
        let dev = match dev_reg.devices.get(&rec.device_id) {
            Some(d) => d,
            None => {
                drop(dev_reg);
                drop(tex_reg);
                ctx(vm).runtime_error("Queue.writeTexture: device gone.".to_string());
                return;
            }
        };
        dev.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &rec.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        drop(dev_reg);
        drop(tex_reg);
        set_return(vm, Value::null());
    }
}

/// Present the in-flight frame. Consumes the `SurfaceTexture`
/// (which schedules the swap) and drops the auxiliary `TextureView`.
#[no_mangle]
pub unsafe extern "C" fn wlift_gpu_surface_present_frame(vm: *mut VM) {
    unsafe {
        let frame_id = match id_of(ctx(vm), slot(vm, 1), "SurfaceFrame.present") {
            Some(i) => i,
            None => return,
        };
        let entry = match surface_frames().lock().unwrap().frames.remove(&frame_id) {
            Some(e) => e,
            None => {
                ctx(vm).runtime_error(
                    "SurfaceFrame.present: unknown frame id (already presented?)".to_string(),
                );
                return;
            }
        };
        // Drop the view first so the SurfaceTexture's Drop /
        // present chain isn't confused by an outstanding borrow.
        views().lock().unwrap().views.remove(&entry.view_id);
        entry.surface_texture.present();
        set_return(vm, Value::null());
    }
}
