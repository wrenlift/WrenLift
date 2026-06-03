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

use wlift_abi::{
    alloc_list, alloc_map, alloc_string, list_add, map_iter, map_set, obj_type, push_root,
    reload_root, roots_restore, roots_snapshot, runtime_error, set_return, slot, string_str,
    ObjType, Value, WrenVm,
};

/// Plugin ABI handshake — see wlift_gpu::wlift_plugin_abi_version.
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

fn map_get(v: Value, key: &str) -> Option<Value> {
    if obj_type(v) != Some(ObjType::Map) {
        return None;
    }
    for (k, val) in map_iter(v) {
        if let Some(s) = string_str(k) {
            if s == key {
                return Some(val);
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
    MouseWheel { dx: f64, dy: f64 },
    /// Gamepad button / axis events. `code` is the canonical
    /// binding name (`"GamepadButtonA"`, `"GamepadAxisLX"`, etc.)
    /// the action layer maps against. Gamepad events broadcast
    /// across all windows — gamepads aren't owned by any one
    /// window — so they land in every active window's queue.
    GamepadButtonDown { code: String, gamepad: u32 },
    GamepadButtonUp { code: String, gamepad: u32 },
    GamepadAxis { code: String, gamepad: u32, value: f64 },
}

// ---------------------------------------------------------------------------
// Gamepad polling (gilrs)
// ---------------------------------------------------------------------------
//
// gilrs runs polled too — its `next_event()` returns an event from
// whichever connected pad has one buffered, or `None`. We poll
// every `pump_once` call and feed events into every window's queue
// so any window listening to `pollEvents` sees them. Action-mapping
// layers ride on top of these names (see actions.wren).

thread_local! {
    static GILRS: RefCell<Option<gilrs::Gilrs>> = const { RefCell::new(None) };
}

fn ensure_gilrs() {
    GILRS.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            // Construction can fail on Linux when no input device
            // is exposed (containerised CI). Log and keep going —
            // games still want keyboard / mouse.
            match gilrs::Gilrs::new() {
                Ok(g) => *slot = Some(g),
                Err(e) => eprintln!("wlift_window: gilrs init failed: {}; gamepad input disabled", e),
            }
        }
    });
}

/// Canonical button-binding name. Mirrors the SDL game-controller
/// labelling convention so Wren-side code reads `"GamepadButtonA"`,
/// `"GamepadDPadUp"`, etc., regardless of platform.
fn gilrs_button_name(b: gilrs::Button) -> Option<&'static str> {
    use gilrs::Button::*;
    Some(match b {
        South => "GamepadButtonA",
        East => "GamepadButtonB",
        West => "GamepadButtonX",
        North => "GamepadButtonY",
        LeftTrigger => "GamepadLeftBumper",
        RightTrigger => "GamepadRightBumper",
        LeftTrigger2 => "GamepadLeftTrigger",
        RightTrigger2 => "GamepadRightTrigger",
        Select => "GamepadBack",
        Start => "GamepadStart",
        Mode => "GamepadGuide",
        LeftThumb => "GamepadLeftStick",
        RightThumb => "GamepadRightStick",
        DPadUp => "GamepadDPadUp",
        DPadDown => "GamepadDPadDown",
        DPadLeft => "GamepadDPadLeft",
        DPadRight => "GamepadDPadRight",
        _ => return None,
    })
}

/// Canonical axis-binding name. Sticks are reported as four signed
/// axes (-1..1); the action layer is free to threshold them into
/// digital up/down/left/right or use the magnitude directly.
fn gilrs_axis_name(a: gilrs::Axis) -> Option<&'static str> {
    use gilrs::Axis::*;
    Some(match a {
        LeftStickX => "GamepadAxisLX",
        LeftStickY => "GamepadAxisLY",
        RightStickX => "GamepadAxisRX",
        RightStickY => "GamepadAxisRY",
        LeftZ => "GamepadAxisLZ",
        RightZ => "GamepadAxisRZ",
        DPadX => "GamepadAxisDX",
        DPadY => "GamepadAxisDY",
        _ => return None,
    })
}

/// Drain whatever gilrs has buffered into every window's queue.
fn drain_gilrs() {
    GILRS.with(|cell| {
        let mut slot = cell.borrow_mut();
        let Some(g) = slot.as_mut() else { return };
        while let Some(gilrs::Event { id, event, .. }) = g.next_event() {
            // `id` is a `GamepadId` newtype wrapping a usize. Cast
            // to u32 so the Wren side gets a plain Num.
            let gamepad: u32 = Into::<usize>::into(id) as u32;
            let record = match event {
                gilrs::EventType::ButtonPressed(b, _) => gilrs_button_name(b)
                    .map(|c| EventRecord::GamepadButtonDown {
                        code: c.to_string(),
                        gamepad,
                    }),
                gilrs::EventType::ButtonReleased(b, _) => gilrs_button_name(b)
                    .map(|c| EventRecord::GamepadButtonUp {
                        code: c.to_string(),
                        gamepad,
                    }),
                gilrs::EventType::AxisChanged(a, v, _) => gilrs_axis_name(a).map(|c| {
                    EventRecord::GamepadAxis {
                        code: c.to_string(),
                        gamepad,
                        value: v as f64,
                    }
                }),
                // Connected/Disconnected/etc. are surfaced as
                // dedicated event types in a follow-up — for now
                // only button + axis are wired.
                _ => None,
            };
            let Some(record) = record else { continue };
            APP.with(|cell| {
                let mut app = cell.borrow_mut();
                let ids: Vec<u64> = app.queues.keys().copied().collect();
                for wid in ids {
                    // Clone per-window. Records are small
                    // (one short String + Nums); the extra clone
                    // is negligible against the syscalls gilrs
                    // already did to read the event.
                    let clone = match &record {
                        EventRecord::GamepadButtonDown { code, gamepad } => {
                            EventRecord::GamepadButtonDown {
                                code: code.clone(),
                                gamepad: *gamepad,
                            }
                        }
                        EventRecord::GamepadButtonUp { code, gamepad } => {
                            EventRecord::GamepadButtonUp {
                                code: code.clone(),
                                gamepad: *gamepad,
                            }
                        }
                        EventRecord::GamepadAxis {
                            code,
                            gamepad,
                            value,
                        } => EventRecord::GamepadAxis {
                            code: code.clone(),
                            gamepad: *gamepad,
                            value: *value,
                        },
                        _ => continue,
                    };
                    if let Some(q) = app.queues.get_mut(&wid) {
                        q.push(clone);
                    }
                }
            });
        }
    });
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
    visible: bool,
}

impl PumpHandler {
    /// Drain pending create / close requests against the active
    /// event loop. Called from `new_events` on every pump cycle —
    /// `resumed` only fires once at init, so we can't rely on it
    /// for subsequent window allocations.
    fn drain_pending(&mut self, event_loop: &ActiveEventLoop) {
        let pending = std::mem::take(&mut self.new_windows);
        for (id, req) in pending {
            // LogicalSize so the request honours the monitor's
            // scale factor — on a Retina display PhysicalSize(1280,
            // 720) lands as 1280 PHYSICAL pixels = 640 logical
            // points = a tiny window. LogicalSize(1280, 720)
            // multiplies through to 2560 × 1440 physical pixels on
            // a 2× display, matching the size users actually
            // expect when they ask for "1280 × 720". The Resized
            // event still surfaces physical pixels (winit's
            // convention), which is what wgpu wants for surface
            // configure.
            let attrs = winit::window::Window::default_attributes()
                .with_title(req.title)
                .with_inner_size(winit::dpi::LogicalSize::new(
                    req.width as f64,
                    req.height as f64,
                ))
                .with_resizable(req.resizable)
                // Open hidden so Game.run can finish GPU init + paint
                // a real clear frame BEFORE the user sees anything.
                // Without this the OS shows the default background
                // (white on macOS) for the few hundred ms between
                // Window.create returning and the first present
                // landing — flashing the window is a worse first
                // impression than waiting a beat longer.
                // `Window.show()` flips this once init is done.
                .with_visible(req.visible);
            match event_loop.create_window(attrs) {
                Ok(window) => {
                    let winit_id = window.id();
                    // PHYSICAL pixels — what wgpu's surface needs and
                    // what the Resized event later reports. The
                    // request size was LOGICAL (so winit could honour
                    // the monitor's scale factor when opening); on a
                    // 2× retina display that means the actual surface
                    // backing is 2× larger than what we asked for.
                    // Storing the request value here would size the
                    // swap chain to a quarter of the visible window,
                    // surfacing as a small clear-coloured rectangle
                    // in the bottom-left corner with the rest of the
                    // window painted by the OS default background.
                    let inner = window.inner_size();
                    let entry = WindowEntry {
                        window,
                        width:  inner.width,
                        height: inner.height,
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
            WindowEvent::MouseWheel { delta, .. } => {
                // Normalise both `LineDelta` (discrete notches from a
                // wheel-mouse) and `PixelDelta` (continuous gestures
                // from a trackpad) into the same (dx, dy) scalar so
                // consumers don't branch on the source. Line deltas
                // come through as small integers, so the consumer
                // typically multiplies by a step size to compute the
                // effective scroll. Pixel deltas can be hundreds per
                // event; divide them down to roughly the same scale.
                let (dx, dy) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x as f64, y as f64),
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.x / 30.0, p.y / 30.0),
                };
                APP.with(|cell| {
                    if let Some(q) = cell.borrow_mut().queues.get_mut(&id) {
                        q.push(EventRecord::MouseWheel { dx, dy });
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
    ensure_gilrs();
    EVENT_LOOP.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let el = borrow.as_mut().expect("event loop missing");
        let mut handler = pump_handler().lock().unwrap();
        let _status = el.pump_app_events(Some(Duration::ZERO), &mut *handler);
    });
    drain_gilrs();
}

// ---------------------------------------------------------------------------
// Foreign entry points
// ---------------------------------------------------------------------------

/// `Window.create_(descriptor)` — descriptor keys:
///   "title":     String  (default "wlift")
///   "width":     Num     (default 1280)
///   "height":    Num     (default 720)
///   "resizable": Bool    (default true)
///   "visible":   Bool    (default true) — pass `false` to open
///                hidden; the caller then calls `Window.show(id)`
///                once it has painted the first frame, avoiding
///                the OS-default-background flash during GPU init.
///
/// Returns the window id (Num).
#[no_mangle]
pub unsafe extern "C" fn wlift_window_create(vm: *mut WrenVm) {
    let desc = slot(vm, 1);
    let title = map_get(desc, "title")
        .and_then(string_of)
        .unwrap_or_else(|| "wlift".to_string());
    let width = map_get(desc, "width")
        .and_then(Value::as_num)
        .map(|n| n as u32)
        .unwrap_or(1280);
    let height = map_get(desc, "height")
        .and_then(Value::as_num)
        .map(|n| n as u32)
        .unwrap_or(720);
    let resizable = map_get(desc, "resizable")
        .map(|v| !(v.is_null() || v == Value::FALSE))
        .unwrap_or(true);
    let visible = map_get(desc, "visible")
        .map(|v| !(v.is_null() || v == Value::FALSE))
        .unwrap_or(true);

    let id = next_id();
    pump_handler().lock().unwrap().new_windows.push((
        id,
        NewWindowRequest {
            title,
            width,
            height,
            resizable,
            visible,
        },
    ));
    // Run a pump so the window actually materialises before
    // the caller reads back its handle. winit needs to see at
    // least one resumed/window_event cycle to allocate the
    // OS-level window.
    pump_once();
    set_return(vm, Value::num(id as f64));
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_destroy(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.destroy: id must be a non-negative number.");
            return;
        }
    };
    APP.with(|cell| cell.borrow_mut().pending_close.push(id));
    pump_once();
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_pump(vm: *mut WrenVm) {
    pump_once();
    set_return(vm, Value::NULL);
}

/// `Window.setVisible(id, visible)` — flip the OS-window's
/// visibility. The typical use is opening hidden (`Window.create`
/// with `"visible": false`), running GPU device + surface + first
/// paint, then `Window.setVisible(id, true)`. Doing it that way
/// instead of paint-while-OS-shows-default-background avoids the
/// brief white flash macOS otherwise shows for the few hundred
/// milliseconds between window create and first present.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_set_visible(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.setVisible: id must be a non-negative number.");
            return;
        }
    };
    let v = slot(vm, 2);
    let visible = !(v.is_null() || v == Value::FALSE);
    APP.with(|cell| {
        let app = cell.borrow();
        if let Some(entry) = app.windows.get(&id) {
            entry.window.set_visible(visible);
        }
    });
    set_return(vm, Value::NULL);
}

/// `Window.lockCursor(id, lock)` — toggle cursor grab. When
/// locked the cursor stays inside the window; mouse-moved events
/// keep flowing so an FPS controller can integrate them. Falls
/// back transparently from Confined (preferred) to Locked
/// (macOS Cocoa restriction) since both achieve the FPS behaviour
/// the caller wants.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_set_cursor_lock(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.lockCursor: id must be a non-negative number.");
            return;
        }
    };
    let lock = slot(vm, 2).as_bool().unwrap_or(false);
    let result = APP.with(|cell| {
        let app = cell.borrow();
        let Some(entry) = app.windows.get(&id) else {
            return Err("unknown window id".to_string());
        };
        use winit::window::CursorGrabMode;
        let mode = if lock {
            CursorGrabMode::Confined
        } else {
            CursorGrabMode::None
        };
        match entry.window.set_cursor_grab(mode) {
            Ok(_) => Ok(()),
            Err(_) if lock => {
                // Confined isn't supported (macOS Cocoa) — fall
                // back to Locked, which warps the cursor to the
                // window centre every frame. Same end result for
                // mouselook.
                entry
                    .window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .map_err(|e| e.to_string())
            }
            Err(e) => Err(e.to_string()),
        }
    });
    if let Err(e) = result {
        runtime_error(vm, &format!("Window.lockCursor: {}", e));
        return;
    }
    set_return(vm, Value::NULL);
}

/// `Window.hideCursor(id, hide)` — show or hide the OS cursor
/// over this window. Independent of `lockCursor`; an FPS title
/// typically calls both.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_set_cursor_visible(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.hideCursor: id must be a non-negative number.");
            return;
        }
    };
    let hide = slot(vm, 2).as_bool().unwrap_or(false);
    APP.with(|cell| {
        if let Some(entry) = cell.borrow().windows.get(&id) {
            entry.window.set_cursor_visible(!hide);
        }
    });
    set_return(vm, Value::NULL);
}

#[no_mangle]
pub unsafe extern "C" fn wlift_window_close_requested(vm: *mut WrenVm) {
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

#[no_mangle]
pub unsafe extern "C" fn wlift_window_size(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.size: id must be a non-negative number.");
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
    // GC rooting: see `wlift_image_decode` for the pattern. The
    // map survives across each alloc_string, so reload it through
    // the JIT-roots stack on every map_set instead of relying on
    // the stale Rust local.
    let snap = roots_snapshot(vm);
    let map = alloc_map(vm);
    let map_r = push_root(vm, map);
    set_return(vm, map);
    let kw = alloc_string(vm, "width");
    map_set(vm, reload_root(vm, map_r), kw, Value::num(w as f64));
    let kh = alloc_string(vm, "height");
    map_set(vm, reload_root(vm, map_r), kh, Value::num(h as f64));
    roots_restore(vm, snap);
}

/// Drain pending events for a window. Returns a `List` of Maps:
///
///   {"type": "close"}
///   {"type": "resize", "width": Num, "height": Num}
///   {"type": "keyDown" | "keyUp", "code": String}
///   {"type": "mouseMoved", "x": Num, "y": Num}
///   {"type": "mouseDown" | "mouseUp", "button": String}
#[no_mangle]
pub unsafe extern "C" fn wlift_window_drain_events(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.pollEvents: id must be a non-negative number.");
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

    // GC-rooted result list build. Same shape as
    // `wlift_sqlite_query`'s rebuild — accumulating maps in a
    // plain `Vec<Value>` was leaving them unrooted across the
    // inner `alloc_string` calls, so a collection mid-loop
    // would free partially-built event maps. Allocate the
    // result list first, set it as slot 0 (GC-rooted via
    // `api_stack`), append each fresh map to the list before
    // populating fields. Switched per-field assignment from
    // `call_method_on(_, "[_]=(_)", ...)` to direct
    // `(*map_ptr).set` so key + value commit happens
    // immediately after the key alloc, with no intervening
    // method-dispatch path that could allocate.

    // Rooting protocol: every receiver carried across an allocator
    // call has to be a GC root, or a nursery collection during the
    // next alloc forwards the object and the Rust-local Value bits
    // decode to the stale from-space pointer. The next `list_add`
    // or `map_set` then writes through that zombie cell, smearing
    // NaN-tagged bits across the successor object's header — which
    // surfaces as a SIGSEGV in `trace_object` on the next GC.
    //
    // `result` is rooted via `set_return` (api_stack[0] is GC-walked);
    // every per-iteration handle (`map`, `key_type`, "code"/"button"
    // key strings) goes through `push_root` and is read back via
    // `reload_root` on the receiving side of each `map_set` /
    // `list_add`. The whole block pops back to `snap` so successive
    // calls don't leak roots.
    let snap = roots_snapshot(vm);
    let result = alloc_list(vm, 0);
    set_return(vm, result);

    for ev in events {
        let map = alloc_map(vm);
        let map_r = push_root(vm, map);
        list_add(vm, slot(vm, 0), reload_root(vm, map_r));

        let key_type = alloc_string(vm, "type");
        let kt_r = push_root(vm, key_type);
        match ev {
            EventRecord::CloseRequested => {
                let v = alloc_string(vm, "close");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
            }
            EventRecord::Resized { width, height } => {
                let v = alloc_string(vm, "resize");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kw = alloc_string(vm, "width");
                map_set(vm, reload_root(vm, map_r), kw, Value::num(width as f64));
                let kh = alloc_string(vm, "height");
                map_set(vm, reload_root(vm, map_r), kh, Value::num(height as f64));
            }
            EventRecord::KeyDown { code } => {
                let v = alloc_string(vm, "keyDown");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kc = alloc_string(vm, "code");
                let kc_r = push_root(vm, kc);
                let cv = alloc_string(vm, &code);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kc_r), cv);
            }
            EventRecord::KeyUp { code } => {
                let v = alloc_string(vm, "keyUp");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kc = alloc_string(vm, "code");
                let kc_r = push_root(vm, kc);
                let cv = alloc_string(vm, &code);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kc_r), cv);
            }
            EventRecord::MouseMoved { x, y } => {
                let v = alloc_string(vm, "mouseMoved");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kx = alloc_string(vm, "x");
                map_set(vm, reload_root(vm, map_r), kx, Value::num(x));
                let ky = alloc_string(vm, "y");
                map_set(vm, reload_root(vm, map_r), ky, Value::num(y));
            }
            EventRecord::MouseDown { button } => {
                let v = alloc_string(vm, "mouseDown");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kb = alloc_string(vm, "button");
                let kb_r = push_root(vm, kb);
                let bv = alloc_string(vm, &button);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kb_r), bv);
            }
            EventRecord::MouseUp { button } => {
                let v = alloc_string(vm, "mouseUp");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kb = alloc_string(vm, "button");
                let kb_r = push_root(vm, kb);
                let bv = alloc_string(vm, &button);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kb_r), bv);
            }
            EventRecord::MouseWheel { dx, dy } => {
                let v = alloc_string(vm, "mouseWheel");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kdx = alloc_string(vm, "dx");
                map_set(vm, reload_root(vm, map_r), kdx, Value::num(dx));
                let kdy = alloc_string(vm, "dy");
                map_set(vm, reload_root(vm, map_r), kdy, Value::num(dy));
            }
            EventRecord::GamepadButtonDown { code, gamepad } => {
                let v = alloc_string(vm, "gamepadButtonDown");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kc = alloc_string(vm, "code");
                let kc_r = push_root(vm, kc);
                let cv = alloc_string(vm, &code);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kc_r), cv);
                let kg = alloc_string(vm, "gamepad");
                map_set(vm, reload_root(vm, map_r), kg, Value::num(gamepad as f64));
            }
            EventRecord::GamepadButtonUp { code, gamepad } => {
                let v = alloc_string(vm, "gamepadButtonUp");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kc = alloc_string(vm, "code");
                let kc_r = push_root(vm, kc);
                let cv = alloc_string(vm, &code);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kc_r), cv);
                let kg = alloc_string(vm, "gamepad");
                map_set(vm, reload_root(vm, map_r), kg, Value::num(gamepad as f64));
            }
            EventRecord::GamepadAxis {
                code,
                gamepad,
                value,
            } => {
                let v = alloc_string(vm, "gamepadAxis");
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kt_r), v);
                let kc = alloc_string(vm, "code");
                let kc_r = push_root(vm, kc);
                let cv = alloc_string(vm, &code);
                map_set(vm, reload_root(vm, map_r), reload_root(vm, kc_r), cv);
                let kg = alloc_string(vm, "gamepad");
                map_set(vm, reload_root(vm, map_r), kg, Value::num(gamepad as f64));
                let kv = alloc_string(vm, "value");
                map_set(vm, reload_root(vm, map_r), kv, Value::num(value));
            }
        }
        // Drop the per-iteration roots; the next iteration's `map`
        // gets a fresh slot, and `result` stays anchored via slot(0).
        roots_restore(vm, snap);
    }
    roots_restore(vm, snap);
}

/// Build the platform-tagged raw-window-handle Map @hatch:gpu's
/// `Device.createSurface` accepts.
#[no_mangle]
pub unsafe extern "C" fn wlift_window_handle(vm: *mut WrenVm) {
    let id = match slot(vm, 1).as_num() {
        Some(n) if n >= 0.0 => n as u64,
        _ => {
            runtime_error(vm, "Window.handle: id must be a non-negative number.");
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
        runtime_error(
            vm,
            "Window.handle: unknown window id or handle unavailable.",
        );
        return;
    };

    // GC rooting: same JIT-roots-stack pattern as events_list /
    // wlift_image_decode. Both `map` and `key_platform` survive
    // across every subsequent alloc_string in the match arms and
    // the display-handle block, so reload them through the
    // roots-stack rather than holding stale Rust locals.
    use raw_window_handle::RawWindowHandle;

    let snap = roots_snapshot(vm);
    let map = alloc_map(vm);
    let map_r = push_root(vm, map);
    set_return(vm, map);
    let key_platform_r = push_root(vm, alloc_string(vm, "platform"));
    map_set(
        vm,
        reload_root(vm, map_r),
        reload_root(vm, key_platform_r),
        Value::NULL,
    );

    match win {
        RawWindowHandle::AppKit(h) => {
            let v = alloc_string(vm, "appkit");
            map_set(vm, reload_root(vm, map_r), reload_root(vm, key_platform_r), v);
            let kv = alloc_string(vm, "ns_view");
            map_set(
                vm,
                reload_root(vm, map_r),
                kv,
                Value::num(h.ns_view.as_ptr() as usize as f64),
            );
        }
        RawWindowHandle::UiKit(h) => {
            let v = alloc_string(vm, "uikit");
            map_set(vm, reload_root(vm, map_r), reload_root(vm, key_platform_r), v);
            let kv = alloc_string(vm, "ui_view");
            map_set(
                vm,
                reload_root(vm, map_r),
                kv,
                Value::num(h.ui_view.as_ptr() as usize as f64),
            );
        }
        RawWindowHandle::Win32(h) => {
            let v = alloc_string(vm, "win32");
            map_set(vm, reload_root(vm, map_r), reload_root(vm, key_platform_r), v);
            let kh = alloc_string(vm, "hwnd");
            map_set(
                vm,
                reload_root(vm, map_r),
                kh,
                Value::num(h.hwnd.get() as f64),
            );
            if let Some(hi) = h.hinstance {
                let key_hinstance = alloc_string(vm, "hinstance");
                map_set(
                    vm,
                    reload_root(vm, map_r),
                    key_hinstance,
                    Value::num(hi.get() as f64),
                );
            }
        }
        RawWindowHandle::Xlib(h) => {
            let v = alloc_string(vm, "xlib");
            map_set(vm, reload_root(vm, map_r), reload_root(vm, key_platform_r), v);
            let kw = alloc_string(vm, "window");
            map_set(
                vm,
                reload_root(vm, map_r),
                kw,
                Value::num(h.window as f64),
            );
            if h.visual_id != 0 {
                let kvi = alloc_string(vm, "visual_id");
                map_set(
                    vm,
                    reload_root(vm, map_r),
                    kvi,
                    Value::num(h.visual_id as f64),
                );
            }
        }
        RawWindowHandle::Wayland(h) => {
            let v = alloc_string(vm, "wayland");
            map_set(vm, reload_root(vm, map_r), reload_root(vm, key_platform_r), v);
            let ks = alloc_string(vm, "surface");
            map_set(
                vm,
                reload_root(vm, map_r),
                ks,
                Value::num(h.surface.as_ptr() as usize as f64),
            );
        }
        other => {
            let v = alloc_string(vm, &format!("{:?}", other).to_lowercase());
            map_set(vm, reload_root(vm, map_r), reload_root(vm, key_platform_r), v);
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
                            let key = alloc_string(vm, "display");
                            map_set(
                                vm,
                                reload_root(vm, map_r),
                                key,
                                Value::num(p.as_ptr() as usize as f64),
                            );
                        }
                    }
                    RawDisplayHandle::Xcb(d) => {
                        if let Some(p) = d.connection {
                            let key = alloc_string(vm, "connection");
                            map_set(
                                vm,
                                reload_root(vm, map_r),
                                key,
                                Value::num(p.as_ptr() as usize as f64),
                            );
                        }
                    }
                    RawDisplayHandle::Wayland(d) => {
                        let key = alloc_string(vm, "display");
                        map_set(
                            vm,
                            reload_root(vm, map_r),
                            key,
                            Value::num(d.display.as_ptr() as usize as f64),
                        );
                    }
                    _ => {}
                }
            }
        }
    });

    roots_restore(vm, snap);
}
