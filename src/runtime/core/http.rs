//! Optional `http` module — synchronous HTTP client.
//!
//! Backed by `ureq` with rustls TLS. All methods are blocking;
//! callers that want concurrency spawn a Fiber (or a real thread
//! once @hatch:thread lands).
//!
//! This module exposes a narrow surface — everything a consumer
//! needs to make one request and read the response back.
//! @hatch:http layers the convenience API (method-specific
//! helpers, JSON body, response.json, etc.) on top.
//!
//! Each `request(...)` call takes:
//!   method  — "GET" / "POST" / …
//!   url     — String
//!   headers — Map<String, String>  (or null)
//!   body    — String  (or null)
//!   timeout — Num seconds  (or null for the default)
//!
//! Returns a Map with:
//!   status  — Num (100..=599)
//!   headers — Map<String, String> (response headers, lowercase keys)
//!   body    — String (response text, UTF-8)
//!
//! Transport errors (DNS, connect, TLS, timeout) abort the fiber
//! with the error message. Non-2xx responses are not errors —
//! the caller inspects `status`.

use std::time::Duration;

use crate::runtime::object::{NativeContext, ObjMap, ObjString};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Helpers --------------------------------------------------

fn string_of(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<String> {
    super::validate_string(ctx, value, label)
}

/// Best-effort extraction of a Map<String, String> into a Vec of
/// (key, value) pairs. Skips entries with non-string keys or
/// values rather than aborting, so callers can mix simple shapes.
fn pairs_from_map(value: Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if value.is_null() {
        return out;
    }
    let ptr = match value.as_object() {
        Some(p) => p as *const ObjMap,
        None => return out,
    };
    unsafe {
        for (k, v) in (*ptr).entries.iter() {
            let Some(key) = string_value(k.0) else { continue };
            let Some(val) = string_value(*v) else { continue };
            out.push((key, val));
        }
    }
    out
}

unsafe fn string_value(v: Value) -> Option<String> {
    if !v.is_object() {
        return None;
    }
    let ptr = v.as_object()?;
    let header = ptr as *const crate::runtime::object::ObjHeader;
    if unsafe { (*header).obj_type } != crate::runtime::object::ObjType::String {
        return None;
    }
    let s = ptr as *const ObjString;
    Some(unsafe { (*s).as_str().to_string() })
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

// --- Request -------------------------------------------------

fn http_request(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // args[0] = HTTP class itself (receiver)
    // args[1] = method (String)
    // args[2] = url (String)
    // args[3] = headers (Map or null)
    // args[4] = body (String or null)
    // args[5] = timeout (Num seconds or null)
    let Some(method) = string_of(ctx, args[1], "Method") else {
        return Value::null();
    };
    let Some(url) = string_of(ctx, args[2], "Url") else {
        return Value::null();
    };
    let headers = pairs_from_map(args[3]);

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
        None
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

    // Build and fire the request. ureq's agent is cheap to build
    // per-call; pooling happens via the OS keep-alive. For the
    // v0.1 API we don't expose a persistent client — the HTTP
    // class is a facade, not a stateful connection.
    let agent = ureq::AgentBuilder::new()
        .timeout(timeout.unwrap_or_else(|| Duration::from_secs(30)))
        .build();

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
    let resp_headers: Vec<(String, String)> = response
        .headers_names()
        .iter()
        .filter_map(|name| {
            response
                .header(name)
                .map(|v| (name.to_ascii_lowercase(), v.to_string()))
        })
        .collect();
    let text = response
        .into_string()
        .unwrap_or_else(|e| format!("<body read failed: {}>", e));

    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;

    put_num(ctx, out_ptr, "status", status as f64);
    put_str(ctx, out_ptr, "body", text);

    let hdr_val = ctx.alloc_map();
    let hdr_ptr = hdr_val.as_object().unwrap() as *mut ObjMap;
    for (k, v) in resp_headers {
        put_str(ctx, hdr_ptr, &k, v);
    }
    let hdr_key = ctx.alloc_string("headers".to_string());
    unsafe { (*out_ptr).set(hdr_key, hdr_val) };

    out
}

// --- Registration --------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("HttpCore", vm.object_class);
    vm.primitive_static(class, "request(_,_,_,_,_)", http_request);
    class
}
