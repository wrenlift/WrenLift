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

use wren_lift::runtime::object::{NativeContext, ObjHeader, ObjMap, ObjString, ObjType};
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
