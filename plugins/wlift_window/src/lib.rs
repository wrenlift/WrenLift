//! Window provider for WrenLift. Built as `cdylib`; bundled into
//! @hatch:window. The runtime dlopens it and binds each `foreign`
//! method on the package's classes to the symbols below.
//!
//! # Architecture
//!
//! winit 0.30+ uses an `ApplicationHandler` callback model. We
//! drive it via `pump_app_events(Some(Duration::ZERO), &mut app)`
//! once per Wren `window.pollEvents` call — the call returns
//! immediately after draining whatever events winit has, and any
//! state changes (close requested, resize) end up on a stash
//! buffer the Wren side reads.
//!
//! Single global event loop + a per-window state record. macOS
//! requires the EventLoop to live on the main thread; Wren runs
//! on the main thread, so this composes naturally.
//!
//! The handle exposed to Wren is platform-tagged with the same
//! shape `wlift_gpu`'s `Device.createSurface` accepts. Custom
//! embedders (host shells, IDE viewports, future C-FFI hosts)
//! can produce the same Map without depending on this crate.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowId};

use wren_lift::runtime::object::{NativeContext, ObjHeader, ObjMap, ObjString, ObjType};
use wren_lift::runtime::value::Value;
use wren_lift::runtime::vm::VM;

// ---------------------------------------------------------------------------
// Slot helpers (mirror wlift_gpu / wlift_sqlite)
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

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------
//
// The EventLoop and per-window WindowState live on the main thread
// only. macOS in particular *requires* the EventLoop on the main
// thread; on every platform winit panics if the EventLoop crosses
// thread boundaries. We use thread-local storage gated behind a
// `Mutex` for the parts that need to be pulled into FFI calls.
//
// The Wren runtime is single-threaded today; if/when fibers grow
// kernel threads, this would need a more careful design.

thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
    static APP: RefCell<App> = RefCell::new(App::new());
}

struct App {
    /// Live windows by id. winit hands each window its own typed
    /// `WindowId`; we store our own monotonic u64 alongside so the
    /// Wren side has a stable handle.
    windows: HashMap<u64, WindowEntry>,
    /// Lookup from winit's WindowId to our id. `WindowEvent` calls
    /// land via winit's id; we map back through this.
    by_winit: HashMap<WindowId, u64>,
    /// Per-window event queues, keyed by our id. Drained on
    /// `pollEvents`.
    queues: HashMap<u64, Vec<EventRecord>>,
    /// Set by `Window.close()` — the next `pump_app_events` will
    /// destroy these on the active event loop.
    pending_close: Vec<u64>,
}

impl App {
    fn new() -> Self {
        Self {
            windows: HashMap::new(),
            by_winit: HashMap::new(),
            queues: HashMap::new(),
            pending_close: Vec::new(),
        }
    }
}

struct WindowEntry {
    window: Window,
    width: u32,
    height: u32,
    close_requested: bool,
}

/// Recorded event ready to be exposed to Wren as a Map.
enum EventRecord {
    CloseRequested,
    Resized { width: u32, height: u32 },
    KeyDown { code: String },
    KeyUp { code: String },
    MouseMoved { x: f64, y: f64 },
    MouseDown { button: String },
    MouseUp { button: String },
}

// ---------------------------------------------------------------------------
// ApplicationHandler — drives event delivery during pump_app_events
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PumpHandler {
    /// New window requests posted between pumps. The first
    /// `resumed` callback gets a chance to actually allocate them
    /// against the ActiveEventLoop.
    new_windows: Vec<(u64, NewWindowRequest)>,
}

struct NewWindowRequest {
    title: String,
    width: u32,
    height: u32,
    resizable: bool,
}

impl PumpHandler {
    /// Drain pending create / close requests against the active
    /// event loop. Called from `new_events` on every pump cycle —
    /// `resumed` only fires once at init, so we can't rely on it
    /// for subsequent window allocations.
    fn drain_pending(&mut self, event_loop: &ActiveEventLoop) {
        let pending = std::mem::take(&mut self.new_windows);
        for (id, req) in pending {
            let attrs = winit::window::Window::default_attributes()
                .with_title(req.title)
                .with_inner_size(winit::dpi::PhysicalSize::new(req.width, req.height))
                .with_resizable(req.resizable);
            match event_loop.create_window(attrs) {
                Ok(window) => {
                    let winit_id = window.id();
                    let entry = WindowEntry {
                        window,
                        width: req.width,
                        height: req.height,
                        close_requested: false,
                    };
                    APP.with(|cell| {
                        let mut app = cell.borrow_mut();
                        app.windows.insert(id, entry);
                        app.by_winit.insert(winit_id, id);
                        app.queues.insert(id, Vec::new());
                    });
                }
                Err(e) => {
                    eprintln!("wlift_window: window create failed: {}", e);
                }
            }
        }

        let to_close: Vec<u64> =
            APP.with(|cell| std::mem::take(&mut cell.borrow_mut().pending_close));
        for id in to_close {
            APP.with(|cell| {
                let mut app = cell.borrow_mut();
                if let Some(entry) = app.windows.remove(&id) {
                    let winit_id = entry.window.id();
                    app.by_winit.remove(&winit_id);
                    app.queues.remove(&id);
                    drop(entry);
                }
            });
        }
    }
}

impl ApplicationHandler for PumpHandler {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: StartCause) {
        self.drain_pending(event_loop);
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Resumed fires once on init plus on suspend/resume cycles
        // (mobile). On the desktop case it gives us our first
        // chance to drain the pre-init create queue.
        self.drain_pending(event_loop);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        winit_id: WindowId,
        event: WindowEvent,
    ) {
        // Translate the few events we care about into our enum; the
        // long tail (touchpad, IME, window-focus, etc.) lands later.
        let id = APP.with(|cell| cell.borrow().by_winit.get(&winit_id).copied());
        let Some(id) = id else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                APP.with(|cell| {
                    let mut app = cell.borrow_mut();
                    if let Some(w) = app.windows.get_mut(&id) {
                        w.close_requested = true;
                    }
                    if let Some(q) = app.queues.get_mut(&id) {
                        q.push(EventRecord::CloseRequested);
                    }
                });
            }
            WindowEvent::Resized(size) => {
                APP.with(|cell| {
                    let mut app = cell.borrow_mut();
                    if let Some(w) = app.windows.get_mut(&id) {
                        w.width = size.width;
                        w.height = size.height;
                    }
                    if let Some(q) = app.queues.get_mut(&id) {
                        q.push(EventRecord::Resized {
                            width: size.width,
                            height: size.height,
                        });
                    }
                });
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let code = format!("{:?}", event.physical_key);
                let record = if event.state == winit::event::ElementState::Pressed {
                    EventRecord::KeyDown { code }
                } else {
                    EventRecord::KeyUp { code }
                };
                APP.with(|cell| {
                    if let Some(q) = cell.borrow_mut().queues.get_mut(&id) {
                        q.push(record);
                    }
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                APP.with(|cell| {
                    if let Some(q) = cell.borrow_mut().queues.get_mut(&id) {
                        q.push(EventRecord::MouseMoved {
                            x: position.x,
                            y: position.y,
                        });
                    }
                });
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let name = match button {
                    winit::event::MouseButton::Left => "left",
                    winit::event::MouseButton::Right => "right",
                    winit::event::MouseButton::Middle => "middle",
                    _ => "other",
                }
                .to_string();
                let rec = if state == winit::event::ElementState::Pressed {
                    EventRecord::MouseDown { button: name }
                } else {
                    EventRecord::MouseUp { button: name }
                };
                APP.with(|cell| {
                    if let Some(q) = cell.borrow_mut().queues.get_mut(&id) {
                        q.push(rec);
                    }
                });
            }
            _ => {}
        }
    }
}

// One PumpHandler shared for the lifetime of the process. Mutating
// `new_windows` happens on the same thread that calls pump_app_events,
// so a Mutex is sufficient.
fn pump_handler() -> &'static Mutex<PumpHandler> {
    static H: OnceLock<Mutex<PumpHandler>> = OnceLock::new();
    H.get_or_init(|| Mutex::new(PumpHandler::default()))
}

fn ensure_event_loop() {
    EVENT_LOOP.with(|cell| {
        if cell.borrow().is_none() {
            let el = EventLoop::builder()
                .build()
                .expect("wlift_window: failed to build event loop");
            el.set_control_flow(ControlFlow::Poll);
            *cell.borrow_mut() = Some(el);
        }
    });
}

/// One pump cycle. Drains pending events into per-window queues
/// and processes any deferred create / close requests. The
/// timeout is zero so the call returns as soon as winit has
/// nothing more to do — Wren drives its own pacing.
fn pump_once() {
    ensure_event_loop();
    EVENT_LOOP.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let el = borrow.as_mut().expect("event loop missing");
        let mut handler = pump_handler().lock().unwrap();
        let _status = el.pump_app_events(Some(Duration::ZERO), &mut *handler);
    });
}

// ---------------------------------------------------------------------------
// Foreign entry points
// ---------------------------------------------------------------------------

/// `Window.create_(descriptor)` — descriptor keys:
///   "title":     String  (default "wlift")
///   "width":     Num     (default 1280)
///   "height":    Num     (default 720)
///   "resizable": Bool    (default true)
///
/// Returns the window id (Num).
#[no_mangle]
pub unsafe extern "C" fn wlift_window_create(vm: *mut VM) {
    unsafe {
        let desc = slot(vm, 1);
        let title = map_get(desc, "title")
            .and_then(|v| string_of(v))
            .unwrap_or_else(|| "wlift".to_string());
        let width = map_get(desc, "width")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(1280);
        let height = map_get(desc, "height")
            .and_then(|v| v.as_num())
            .map(|n| n as u32)
            .unwrap_or(720);
        let resizable = map_get(desc, "resizable")
            .map(|v| !v.is_falsy())
            .unwrap_or(true);

        let id = next_id();
        pump_handler().lock().unwrap().new_windows.push((
            id,
            NewWindowRequest {
                title,
                width,
                height,
                resizable,
            },
        ));
        // Run a pump so the window actually materialises before
        // the caller reads back its handle. winit needs to see at
        // least one resumed/window_event cycle to allocate the
        // OS-level window.
        pump_once();
        set_return(vm, Value::num(id as f64));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_destroy(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => {
                ctx(vm)
                    .runtime_error("Window.destroy: id must be a non-negative number.".to_string());
                return;
            }
        };
        APP.with(|cell| cell.borrow_mut().pending_close.push(id));
        pump_once();
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_pump(vm: *mut VM) {
    unsafe {
        pump_once();
        set_return(vm, Value::null());
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_close_requested(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => {
                set_return(vm, Value::bool(true));
                return;
            }
        };
        let flag = APP.with(|cell| {
            cell.borrow()
                .windows
                .get(&id)
                .map(|w| w.close_requested)
                .unwrap_or(true)
        });
        set_return(vm, Value::bool(flag));
    }
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_size(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => {
                ctx(vm).runtime_error("Window.size: id must be a non-negative number.".to_string());
                return;
            }
        };
        let (w, h) = APP.with(|cell| {
            cell.borrow()
                .windows
                .get(&id)
                .map(|wnd| (wnd.width, wnd.height))
                .unwrap_or((0, 0))
        });
        let context = ctx(vm);
        let map = context.alloc_map();
        let kw = context.alloc_string("width".to_string());
        let kh = context.alloc_string("height".to_string());
        let _ = context.call_method_on(map, "[_]=(_)", &[kw, Value::num(w as f64)]);
        let _ = context.call_method_on(map, "[_]=(_)", &[kh, Value::num(h as f64)]);
        set_return(vm, map);
    }
}

/// Drain pending events for a window. Returns a `List` of Maps:
///
///   {"type": "close"}
///   {"type": "resize", "width": Num, "height": Num}
///   {"type": "keyDown" | "keyUp", "code": String}
///   {"type": "mouseMoved", "x": Num, "y": Num}
///   {"type": "mouseDown" | "mouseUp", "button": String}
#[no_mangle]
pub unsafe extern "C" fn wlift_window_drain_events(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => {
                ctx(vm).runtime_error(
                    "Window.pollEvents: id must be a non-negative number.".to_string(),
                );
                return;
            }
        };
        // Pump first so any newly-arrived OS events are part of
        // the drain.
        pump_once();
        let events = APP.with(|cell| {
            cell.borrow_mut()
                .queues
                .get_mut(&id)
                .map(std::mem::take)
                .unwrap_or_default()
        });

        let context = ctx(vm);
        let mut entries: Vec<Value> = Vec::with_capacity(events.len());
        for ev in events {
            let m = context.alloc_map();
            let key_type = context.alloc_string("type".to_string());
            match ev {
                EventRecord::CloseRequested => {
                    let v = context.alloc_string("close".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                }
                EventRecord::Resized { width, height } => {
                    let v = context.alloc_string("resize".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                    let kw = context.alloc_string("width".to_string());
                    let kh = context.alloc_string("height".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[kw, Value::num(width as f64)]);
                    let _ = context.call_method_on(m, "[_]=(_)", &[kh, Value::num(height as f64)]);
                }
                EventRecord::KeyDown { code } => {
                    let v = context.alloc_string("keyDown".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                    let kc = context.alloc_string("code".to_string());
                    let cv = context.alloc_string(code);
                    let _ = context.call_method_on(m, "[_]=(_)", &[kc, cv]);
                }
                EventRecord::KeyUp { code } => {
                    let v = context.alloc_string("keyUp".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                    let kc = context.alloc_string("code".to_string());
                    let cv = context.alloc_string(code);
                    let _ = context.call_method_on(m, "[_]=(_)", &[kc, cv]);
                }
                EventRecord::MouseMoved { x, y } => {
                    let v = context.alloc_string("mouseMoved".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                    let kx = context.alloc_string("x".to_string());
                    let ky = context.alloc_string("y".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[kx, Value::num(x)]);
                    let _ = context.call_method_on(m, "[_]=(_)", &[ky, Value::num(y)]);
                }
                EventRecord::MouseDown { button } => {
                    let v = context.alloc_string("mouseDown".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                    let kb = context.alloc_string("button".to_string());
                    let bv = context.alloc_string(button);
                    let _ = context.call_method_on(m, "[_]=(_)", &[kb, bv]);
                }
                EventRecord::MouseUp { button } => {
                    let v = context.alloc_string("mouseUp".to_string());
                    let _ = context.call_method_on(m, "[_]=(_)", &[key_type, v]);
                    let kb = context.alloc_string("button".to_string());
                    let bv = context.alloc_string(button);
                    let _ = context.call_method_on(m, "[_]=(_)", &[kb, bv]);
                }
            }
            entries.push(m);
        }

        let list = context.alloc_list(entries);
        set_return(vm, list);
    }
}

/// Build the platform-tagged raw-window-handle Map @hatch:gpu's
/// `Device.createSurface` accepts.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_handle(vm: *mut VM) {
    unsafe {
        let id = match slot(vm, 1).as_num() {
            Some(n) if n >= 0.0 => n as u64,
            _ => {
                ctx(vm)
                    .runtime_error("Window.handle: id must be a non-negative number.".to_string());
                return;
            }
        };
        // Snapshot the raw handles inside the thread-local so we
        // can release the borrow before the FFI re-entrance into
        // alloc_map.
        let handles = APP.with(|cell| {
            let app = cell.borrow();
            let entry = app.windows.get(&id)?;
            let win_handle = entry.window.window_handle().ok().map(|h| h.as_raw());
            let disp_handle = entry.window.display_handle().ok().map(|h| h.as_raw());
            match (win_handle, disp_handle) {
                (Some(w), Some(d)) => Some((w, d)),
                _ => None,
            }
        });
        let Some((win, _disp)) = handles else {
            ctx(vm).runtime_error(
                "Window.handle: unknown window id or handle unavailable.".to_string(),
            );
            return;
        };

        use raw_window_handle::RawWindowHandle;
        let context = ctx(vm);
        let map = context.alloc_map();
        let key_platform = context.alloc_string("platform".to_string());

        match win {
            RawWindowHandle::AppKit(h) => {
                let v = context.alloc_string("appkit".to_string());
                let _ = context.call_method_on(map, "[_]=(_)", &[key_platform, v]);
                let kv = context.alloc_string("ns_view".to_string());
                let val = Value::num(h.ns_view.as_ptr() as usize as f64);
                let _ = context.call_method_on(map, "[_]=(_)", &[kv, val]);
            }
            RawWindowHandle::UiKit(h) => {
                let v = context.alloc_string("uikit".to_string());
                let _ = context.call_method_on(map, "[_]=(_)", &[key_platform, v]);
                let kv = context.alloc_string("ui_view".to_string());
                let val = Value::num(h.ui_view.as_ptr() as usize as f64);
                let _ = context.call_method_on(map, "[_]=(_)", &[kv, val]);
            }
            RawWindowHandle::Win32(h) => {
                let v = context.alloc_string("win32".to_string());
                let _ = context.call_method_on(map, "[_]=(_)", &[key_platform, v]);
                let kh = context.alloc_string("hwnd".to_string());
                let _ =
                    context.call_method_on(map, "[_]=(_)", &[kh, Value::num(h.hwnd.get() as f64)]);
                if let Some(hi) = h.hinstance {
                    let key_hinstance = context.alloc_string("hinstance".to_string());
                    let _ = context.call_method_on(
                        map,
                        "[_]=(_)",
                        &[key_hinstance, Value::num(hi.get() as f64)],
                    );
                }
            }
            RawWindowHandle::Xlib(h) => {
                let v = context.alloc_string("xlib".to_string());
                let _ = context.call_method_on(map, "[_]=(_)", &[key_platform, v]);
                let kw = context.alloc_string("window".to_string());
                let _ = context.call_method_on(map, "[_]=(_)", &[kw, Value::num(h.window as f64)]);
                if h.visual_id != 0 {
                    let kvi = context.alloc_string("visual_id".to_string());
                    let _ = context.call_method_on(
                        map,
                        "[_]=(_)",
                        &[kvi, Value::num(h.visual_id as f64)],
                    );
                }
            }
            RawWindowHandle::Wayland(h) => {
                let v = context.alloc_string("wayland".to_string());
                let _ = context.call_method_on(map, "[_]=(_)", &[key_platform, v]);
                let ks = context.alloc_string("surface".to_string());
                let val = Value::num(h.surface.as_ptr() as usize as f64);
                let _ = context.call_method_on(map, "[_]=(_)", &[ks, val]);
            }
            other => {
                let v = context.alloc_string(format!("{:?}", other).to_lowercase());
                let _ = context.call_method_on(map, "[_]=(_)", &[key_platform, v]);
            }
        }

        // The display half — most embedders only need the platform
        // tag plus `ns_view` / `hwnd` / etc., but X11 + Wayland
        // require the display pointer as a separate key.
        APP.with(|cell| {
            let app = cell.borrow();
            if let Some(entry) = app.windows.get(&id) {
                if let Ok(disp) = entry.window.display_handle() {
                    use raw_window_handle::RawDisplayHandle;
                    match disp.as_raw() {
                        RawDisplayHandle::Xlib(d) => {
                            if let Some(p) = d.display {
                                let key = context.alloc_string("display".to_string());
                                let val = Value::num(p.as_ptr() as usize as f64);
                                let _ = context.call_method_on(map, "[_]=(_)", &[key, val]);
                            }
                        }
                        RawDisplayHandle::Xcb(d) => {
                            if let Some(p) = d.connection {
                                let key = context.alloc_string("connection".to_string());
                                let val = Value::num(p.as_ptr() as usize as f64);
                                let _ = context.call_method_on(map, "[_]=(_)", &[key, val]);
                            }
                        }
                        RawDisplayHandle::Wayland(d) => {
                            let key = context.alloc_string("display".to_string());
                            let val = Value::num(d.display.as_ptr() as usize as f64);
                            let _ = context.call_method_on(map, "[_]=(_)", &[key, val]);
                        }
                        _ => {}
                    }
                }
            }
        });

        set_return(vm, map);
    }
}
