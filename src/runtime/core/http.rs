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

use std::time::Duration;

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

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

// --- Registration --------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("HttpCore", vm.object_class);
    vm.primitive_static(class, "request(_,_,_,_,_)", http_request);
    class
}
