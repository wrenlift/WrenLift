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
//! # Phase 1 scope
//!
//! Headless: device + adapter, buffer, texture (as render
//! attachment + readback target), shader module, render pipeline,
//! command encoder, render pass, sampler, bind group + layout.
//! No window, no surface — those land in phase 3.
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
    width: u32,
    height: u32,
    device_id: u64,
}

struct Views {
    views: HashMap<u64, wgpu::TextureView>,
}

struct Samplers {
    samplers: HashMap<u64, wgpu::Sampler>,
}

struct DeviceRecord {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Held so the adapter doesn't drop while the device is alive
    /// (wgpu's lifetime graph keeps the device working anyway, but
    /// we hold this for diagnostics — `adapter.get_info()` etc.).
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
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
                vm.runtime_error("Texture.create: every `usage` entry must be a string.".to_string());
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
                vm.runtime_error(format!(
                    "Buffer.create: unknown usage flag '{}'.",
                    other
                ));
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
pub extern "C" fn wlift_gpu_request_device(vm: *mut VM) {
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

        let adapter = match pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )) {
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
pub extern "C" fn wlift_gpu_device_destroy(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_device_info(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_buffer_create(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_buffer_destroy(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_buffer_size(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_buffer_write_floats(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_buffer_write_uints(vm: *mut VM) {
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
unsafe fn write_buffer_math_batch(
    vm: *mut VM,
    label: &str,
    floats_per_element: usize,
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
pub extern "C" fn wlift_gpu_buffer_write_mat4s(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeMat4s", 16) }
}

#[no_mangle]
pub extern "C" fn wlift_gpu_buffer_write_vec3s(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeVec3s", 3) }
}

#[no_mangle]
pub extern "C" fn wlift_gpu_buffer_write_vec4s(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeVec4s", 4) }
}

#[no_mangle]
pub extern "C" fn wlift_gpu_buffer_write_quats(vm: *mut VM) {
    unsafe { write_buffer_math_batch(vm, "Buffer.writeQuats", 4) }
}

// ---------------------------------------------------------------------------
// ShaderModule
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wlift_gpu_shader_create(vm: *mut VM) {
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
                    "ShaderModule.create: descriptor must include a `code` string."
                        .to_string(),
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
            // wgpu emits errors via the device's error callback; for
            // phase 1 we surface them as runtime errors via the
            // device's pollster-blocking validation path. Compile
            // errors panic-via-callback in wgpu so wrap in
            // catch_unwind; cleaner error mapping lands later.
            dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
pub extern "C" fn wlift_gpu_shader_destroy(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_texture_create(vm: *mut VM) {
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
                    ctx(vm).runtime_error(format!(
                        "Texture.create: unknown format '{}'.",
                        s
                    ));
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
pub extern "C" fn wlift_gpu_texture_destroy(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_texture_create_view(vm: *mut VM) {
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
            rec.texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let id = next_id();
        views().lock().unwrap().views.insert(id, view);
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub extern "C" fn wlift_gpu_view_destroy(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_sampler_create(vm: *mut VM) {
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
pub extern "C" fn wlift_gpu_sampler_destroy(vm: *mut VM) {
    unsafe {
        let id = match id_of(ctx(vm), slot(vm, 1), "Sampler.destroy") {
            Some(i) => i,
            None => return,
        };
        samplers().lock().unwrap().samplers.remove(&id);
        set_return(vm, Value::null());
    }
}
