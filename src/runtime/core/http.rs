//! Optional `http` module — synchronous HTTP client.
//!
//! Backed by `ureq` with built-in TLS. All methods are blocking;
//! callers that want concurrency spawn a Fiber.
//!
//! Narrow surface — one `request(method, url, headers, body,
//! timeout)` call. @hatch:http layers the idiomatic API
//! (verb helpers, bearer/basicAuth shortcuts, form bodies, JSON
//! parsing) on top.
//!
//! Headers are passed in as a `Map<String, Value>` where each
//! value is either a String or a `List<String>` (for headers
//! that legitimately carry multiple values like `Accept`). We
//! validate names + values up-front because ureq panics on
//! malformed input — a library crash shouldn't be the way a
//! Wren script learns it sent a bad header.
//!
//! The returned `headers` map is always `Map<String, List<String>>`
//! with lowercased keys, preserving multi-value responses like
//! `Set-Cookie`. Callers who want the "just give me one value"
//! behaviour pick the first element.

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Streaming registry ---------------------------------------
//
// `Http.stream` starts a background drain thread that pumps the
// response body into a shared VecDeque. Both blocking
// (`streamReadBytes`) and non-blocking (`tryStreamReadBytes`)
// readers pull from the queue under the same mutex. A condvar
// signals blocking readers when new bytes arrive or EOF is hit.
//
// The drain thread owns the boxed `Read` (returned by
// `ureq::Response::into_reader`) and is joined either when the
// caller EOF-drains the reader or explicitly via `streamClose`.

struct StreamState {
    buf: VecDeque<u8>,
    eof: bool,
    err: Option<String>,
}

type SharedHttpStream = Arc<(Mutex<StreamState>, Condvar)>;

struct StreamEntry {
    state: SharedHttpStream,
    /// Drain thread handle. `None` once joined via close/EOF.
    drain: Option<JoinHandle<()>>,
}

fn stream_registry() -> &'static Mutex<HashMap<u64, StreamEntry>> {
    static REG: OnceLock<Mutex<HashMap<u64, StreamEntry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_stream_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

fn spawn_http_drain(
    mut reader: Box<dyn Read + Send>,
    state: SharedHttpStream,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => {
                    let (lock, cvar) = &*state;
                    let mut s = lock.lock().unwrap();
                    s.eof = true;
                    cvar.notify_all();
                    return;
                }
                Ok(n) => {
                    let (lock, cvar) = &*state;
                    let mut s = lock.lock().unwrap();
                    s.buf.extend(&buf[..n]);
                    cvar.notify_all();
                }
                Err(e) => {
                    let (lock, cvar) = &*state;
                    let mut s = lock.lock().unwrap();
                    s.err = Some(e.to_string());
                    s.eof = true;
                    cvar.notify_all();
                    return;
                }
            }
        }
    })
}

// --- Helpers --------------------------------------------------

fn string_of(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<String> {
    super::validate_string(ctx, value, label)
}

unsafe fn string_value(v: Value) -> Option<String> {
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

/// Pull a list's `String` elements, stopping at the first
/// non-string entry with an error.
fn list_of_strings(
    ctx: &mut dyn NativeContext,
    value: Value,
    label: &str,
) -> Option<Vec<String>> {
    let ptr = value.as_object()? as *const ObjList;
    let (count, data) = unsafe { ((*ptr).count as usize, (*ptr).elements) };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let v = unsafe { *data.add(i) };
        match unsafe { string_value(v) } {
            Some(s) => out.push(s),
            None => {
                ctx.runtime_error(format!(
                    "{}: every list entry must be a string.",
                    label
                ));
                return None;
            }
        }
    }
    Some(out)
}

fn put_str(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: String) {
    let k = ctx.alloc_string(key.to_string());
    let v = ctx.alloc_string(value);
    unsafe { (*map).set(k, v) };
}

fn put_num(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: f64) {
    let k = ctx.alloc_string(key.to_string());
    unsafe { (*map).set(k, Value::num(value)) };
}

// --- Header validation ---------------------------------------

/// Pre-flight reject anything ureq would panic on. Message
/// returned to the fiber includes the offending header so the
/// caller can fix it at the source rather than bisecting.
fn validate_header(name: &str, value: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Http.request: empty header name.".into());
    }
    for (i, c) in name.char_indices() {
        // RFC 7230 §3.2.6 token chars: ALPHA / DIGIT and these
        // specials. Excluded explicitly: whitespace and most
        // delimiters that would confuse the framing.
        let ok = c.is_ascii_alphanumeric()
            || matches!(
                c,
                '!' | '#'
                    | '$'
                    | '%'
                    | '&'
                    | '\''
                    | '*'
                    | '+'
                    | '-'
                    | '.'
                    | '^'
                    | '_'
                    | '`'
                    | '|'
                    | '~'
            );
        if !ok {
            return Err(format!(
                "Http.request: invalid header name '{}' (bad char {:?} at offset {}).",
                name, c, i
            ));
        }
    }
    // Values: CR / LF / NUL are the main framing hazards.
    for (i, b) in value.bytes().enumerate() {
        if b == b'\r' || b == b'\n' || b == 0 {
            return Err(format!(
                "Http.request: invalid header value for '{}' (control byte {:#x} at offset {}).",
                name, b, i
            ));
        }
    }
    Ok(())
}

/// Read the caller's headers map into a flat list of (name,
/// value) pairs. A single value → one pair. A List<String> →
/// comma-joined into one pair, which is the RFC-7230 §3.2.2
/// equivalent for everything except `Set-Cookie` (a server
/// concern, not relevant on request).
///
/// ureq's `.set(name, value)` replaces same-name headers, so we
/// have to do the join here rather than emitting N pairs.
fn flatten_request_headers(
    ctx: &mut dyn NativeContext,
    value: Value,
) -> Option<Vec<(String, String)>> {
    let mut out = Vec::new();
    if value.is_null() {
        return Some(out);
    }
    let ptr = match value.as_object() {
        Some(p) => p as *const ObjMap,
        None => {
            ctx.runtime_error("Http.request: headers must be a Map.".to_string());
            return None;
        }
    };
    let entries: Vec<(Value, Value)> = unsafe {
        (*ptr)
            .entries
            .iter()
            .map(|(k, v)| (k.0, *v))
            .collect()
    };
    for (k, v) in entries {
        let Some(name) = (unsafe { string_value(k) }) else {
            ctx.runtime_error("Http.request: header names must be strings.".to_string());
            return None;
        };
        let joined = if let Some(s) = unsafe { string_value(v) } {
            s
        } else if let Some(list) = list_of_strings(ctx, v, "Http.request: header value") {
            list.join(", ")
        } else {
            ctx.runtime_error(format!(
                "Http.request: header '{}' value must be a string or list of strings.",
                name
            ));
            return None;
        };
        if let Err(e) = validate_header(&name, &joined) {
            ctx.runtime_error(e);
            return None;
        }
        out.push((name, joined));
    }
    Some(out)
}

// --- Request -------------------------------------------------

fn http_request(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // args[0] = HttpCore class (receiver)
    // args[1] = method (String)
    // args[2] = url (String)
    // args[3] = headers (Map or null); values are String or List<String>
    // args[4] = body (String or null)
    // args[5] = timeout (Num seconds or null)
    let Some(method) = string_of(ctx, args[1], "Method") else {
        return Value::null();
    };
    let Some(url) = string_of(ctx, args[2], "Url") else {
        return Value::null();
    };
    let Some(headers) = flatten_request_headers(ctx, args[3]) else {
        return Value::null();
    };

    let body = if args[4].is_null() {
        None
    } else {
        match unsafe { string_value(args[4]) } {
            Some(s) => Some(s),
            None => {
                ctx.runtime_error("Http.request: body must be a string or null.".to_string());
                return Value::null();
            }
        }
    };

    let timeout = if args[5].is_null() {
        Some(Duration::from_secs(30))
    } else {
        match args[5].as_num() {
            Some(n) if n > 0.0 && n.is_finite() => Some(Duration::from_secs_f64(n)),
            _ => {
                ctx.runtime_error(
                    "Http.request: timeout must be a positive finite number.".to_string(),
                );
                return Value::null();
            }
        }
    };

    // Panic safety is handled centrally in
    // `call_native_with_frame_sync` — we just do the call here.
    // Upstream catches any crash in ureq / its transitive deps
    // and surfaces it as a fiber-catchable runtime error.
    let agent = ureq::AgentBuilder::new().timeout(timeout.unwrap()).build();
    let mut req = agent.request(&method, &url);
    for (k, v) in &headers {
        req = req.set(k, v);
    }
    let result = if let Some(b) = body {
        req.send_string(&b)
    } else {
        req.call()
    };

    let response = match result {
        Ok(resp) => resp,
        Err(ureq::Error::Status(_, resp)) => resp, // non-2xx still returns a response
        Err(ureq::Error::Transport(t)) => {
            ctx.runtime_error(format!("Http.request: {}: {}", url, t));
            return Value::null();
        }
    };

    let status = response.status();
    // Gather unique header names in order, then pull *all* values
    // for each — preserves `Set-Cookie: a` / `Set-Cookie: b` which
    // a Map<String, String> would collapse.
    let names: Vec<String> = response
        .headers_names()
        .into_iter()
        .map(|n| n.to_string())
        .collect();
    let mut resp_headers: Vec<(String, Vec<String>)> = Vec::with_capacity(names.len());
    for name in &names {
        let values: Vec<String> = response
            .all(name)
            .into_iter()
            .map(|v| v.to_string())
            .collect();
        resp_headers.push((name.to_ascii_lowercase(), values));
    }
    let text = response
        .into_string()
        .unwrap_or_else(|e| format!("<body read failed: {}>", e));

    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;

    put_num(ctx, out_ptr, "status", status as f64);
    put_str(ctx, out_ptr, "body", text);

    let hdr_val = ctx.alloc_map();
    let hdr_ptr = hdr_val.as_object().unwrap() as *mut ObjMap;
    for (k, values) in resp_headers {
        let list_elems: Vec<Value> = values.into_iter().map(|s| ctx.alloc_string(s)).collect();
        let list = ctx.alloc_list(list_elems);
        let key = ctx.alloc_string(k);
        unsafe { (*hdr_ptr).set(key, list) };
    }
    let hdr_key = ctx.alloc_string("headers".to_string());
    unsafe { (*out_ptr).set(hdr_key, hdr_val) };

    out
}

// --- Streaming ------------------------------------------------
//
// `stream(method, url, headers, body, timeout)` — issues the
// request synchronously but doesn't read the body. Returns a Map
// {id, status, headers} where `id` is a registry handle. The
// caller drains the body via `streamReadBytes(id, max)` which
// returns `List<Num>` chunks or null at EOF. `streamClose(id)`
// drops the entry if the caller bails early.
//
// Large responses (downloads, SSE, chunked endpoints) no longer
// need to fit in memory: the response body is pulled lazily from
// the socket as the Wren side consumes it.

fn http_stream(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(method) = string_of(ctx, args[1], "Method") else {
        return Value::null();
    };
    let Some(url) = string_of(ctx, args[2], "Url") else {
        return Value::null();
    };
    let Some(headers) = flatten_request_headers(ctx, args[3]) else {
        return Value::null();
    };
    let body = if args[4].is_null() {
        None
    } else {
        match unsafe { string_value(args[4]) } {
            Some(s) => Some(s),
            None => {
                ctx.runtime_error("Http.stream: body must be a string or null.".to_string());
                return Value::null();
            }
        }
    };
    let timeout = if args[5].is_null() {
        Some(Duration::from_secs(30))
    } else {
        match args[5].as_num() {
            Some(n) if n > 0.0 && n.is_finite() => Some(Duration::from_secs_f64(n)),
            _ => {
                ctx.runtime_error(
                    "Http.stream: timeout must be a positive finite number.".to_string(),
                );
                return Value::null();
            }
        }
    };

    let agent = ureq::AgentBuilder::new().timeout(timeout.unwrap()).build();
    let mut req = agent.request(&method, &url);
    for (k, v) in &headers {
        req = req.set(k, v);
    }
    let result = if let Some(b) = body {
        req.send_string(&b)
    } else {
        req.call()
    };

    let response = match result {
        Ok(resp) => resp,
        Err(ureq::Error::Status(_, resp)) => resp,
        Err(ureq::Error::Transport(t)) => {
            ctx.runtime_error(format!("Http.stream: {}: {}", url, t));
            return Value::null();
        }
    };

    let status = response.status();
    // Gather headers the same way `request` does, before consuming
    // the response into its streaming reader.
    let names: Vec<String> = response
        .headers_names()
        .into_iter()
        .map(|n| n.to_string())
        .collect();
    let mut resp_headers: Vec<(String, Vec<String>)> = Vec::with_capacity(names.len());
    for name in &names {
        let values: Vec<String> = response
            .all(name)
            .into_iter()
            .map(|v| v.to_string())
            .collect();
        resp_headers.push((name.to_ascii_lowercase(), values));
    }

    // Hand the body reader to a background drain thread so Wren
    // can pull bytes (blocking or non-blocking) from a shared
    // queue. The thread owns the `Read` until EOF; the registry
    // holds only the shared state + join handle.
    let reader = response.into_reader();
    let id = next_stream_id();
    let state: SharedHttpStream = Arc::new((
        Mutex::new(StreamState {
            buf: VecDeque::new(),
            eof: false,
            err: None,
        }),
        Condvar::new(),
    ));
    let drain = spawn_http_drain(reader, Arc::clone(&state));
    stream_registry().lock().unwrap().insert(
        id,
        StreamEntry {
            state,
            drain: Some(drain),
        },
    );

    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;
    put_num(ctx, out_ptr, "id", id as f64);
    put_num(ctx, out_ptr, "status", status as f64);
    let hdr_val = ctx.alloc_map();
    let hdr_ptr = hdr_val.as_object().unwrap() as *mut ObjMap;
    for (k, values) in resp_headers {
        let list_elems: Vec<Value> = values.into_iter().map(|s| ctx.alloc_string(s)).collect();
        let list = ctx.alloc_list(list_elems);
        let key = ctx.alloc_string(k);
        unsafe { (*hdr_ptr).set(key, list) };
    }
    let hdr_key = ctx.alloc_string("headers".to_string());
    unsafe { (*out_ptr).set(hdr_key, hdr_val) };
    out
}

fn http_stream_read_impl(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    nonblocking: bool,
) -> Value {
    let label = if nonblocking {
        "Http.tryStreamReadBytes"
    } else {
        "Http.streamReadBytes"
    };
    let id = match args[1].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as u64,
        _ => {
            ctx.runtime_error(format!(
                "{}: id must be a non-negative integer.",
                label
            ));
            return Value::null();
        }
    };
    let max = match args[2].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as usize,
        _ => {
            ctx.runtime_error(format!(
                "{}: max must be a non-negative integer.",
                label
            ));
            return Value::null();
        }
    };
    if max == 0 {
        return ctx.alloc_list(Vec::new());
    }

    // Snapshot the shared state so we can drop the registry lock
    // before blocking on the condvar.
    let state = {
        let reg = stream_registry().lock().unwrap();
        let Some(entry) = reg.get(&id) else {
            ctx.runtime_error(format!("{}: unknown stream id {}.", label, id));
            return Value::null();
        };
        Arc::clone(&entry.state)
    };

    let chunk_opt: Option<Vec<u8>> = {
        let (lock, cvar) = &*state;
        let mut s = lock.lock().unwrap();
        if nonblocking {
            if s.buf.is_empty() {
                if s.eof {
                    None
                } else {
                    // "Try again" — caller yields between polls.
                    Some(Vec::new())
                }
            } else {
                let take = max.min(s.buf.len());
                Some(s.buf.drain(..take).collect())
            }
        } else {
            while s.buf.is_empty() && !s.eof {
                s = cvar.wait(s).unwrap();
            }
            if s.buf.is_empty() {
                None
            } else {
                let take = max.min(s.buf.len());
                Some(s.buf.drain(..take).collect())
            }
        }
    };

    match chunk_opt {
        None => {
            // We're about to report EOF to the caller — tear down
            // the registry entry so the underlying connection is
            // released. We deliberately wait for THIS call (not a
            // chunk-ending-on-EOF one) so repeat reads after the
            // last bytes still find the entry and see null.
            let mut reg = stream_registry().lock().unwrap();
            if let Some(mut entry) = reg.remove(&id) {
                if let Some(th) = entry.drain.take() {
                    drop(reg);
                    let _ = th.join();
                }
            }
            Value::null()
        }
        Some(chunk) => {
            let elements: Vec<Value> =
                chunk.iter().map(|&b| Value::num(b as f64)).collect();
            ctx.alloc_list(elements)
        }
    }
}

fn http_stream_read_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    http_stream_read_impl(ctx, args, false)
}

fn http_try_stream_read_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    http_stream_read_impl(ctx, args, true)
}

fn http_stream_close(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let id = match args[1].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as u64,
        _ => {
            ctx.runtime_error(
                "Http.streamClose: id must be a non-negative integer.".to_string(),
            );
            return Value::null();
        }
    };
    // Remove from the registry first so the drain thread loses
    // its registry reference — its Arc is held separately, so the
    // thread keeps running (it will exit on its own at EOF).
    if let Some(mut entry) = stream_registry().lock().unwrap().remove(&id) {
        // Detach the drain thread: joining here would wait for
        // the server to finish sending, which is the opposite of
        // "close early". Closing the underlying `Read` would
        // force the drain to exit, but `Response::into_reader`
        // returns an opaque Box<dyn Read> without a close method.
        // Letting the thread finish in the background is the
        // least-surprising behavior for now.
        entry.drain.take();
    }
    Value::null()
}

// --- Registration --------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("HttpCore", vm.object_class);
    vm.primitive_static(class, "request(_,_,_,_,_)", http_request);
    vm.primitive_static(class, "stream(_,_,_,_,_)", http_stream);
    vm.primitive_static(class, "streamReadBytes(_,_)", http_stream_read_bytes);
    vm.primitive_static(class, "tryStreamReadBytes(_,_)", http_try_stream_read_bytes);
    vm.primitive_static(class, "streamClose(_)", http_stream_close);
    class
}
