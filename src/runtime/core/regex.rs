//! Optional `regex` module — compiled-pattern handle API. Patterns
//! live in a global registry indexed by monotonic u64 ids; Wren
//! holds the id in a `Regex` wrapper and drops it via `free`.
//!
//! Shape mirrors `proc`: each API takes a leading id and reaches
//! into the registry under a mutex. Single-threaded VM, so the
//! mutex is uncontended in practice.
//!
//! Flags string (second compile arg): any combination of
//!   i — case-insensitive
//!   m — multi-line (^/$ match line boundaries)
//!   s — dot matches newline
//!   U — swap greediness (lazy by default, `?` makes greedy)
//!   x — ignore whitespace + allow `#` comments
//!
//! Returned match maps look like
//!   {
//!     "text":   matched substring,
//!     "start":  byte offset of match start,
//!     "end":    byte offset of match end (exclusive),
//!     "groups": [full, g1, g2, ...],   // Strings, null for non-participating
//!     "named":  {"name": "value", ...}
//!   }

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use regex::{Regex, RegexBuilder};

use crate::runtime::object::{NativeContext, ObjHeader, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Registry --------------------------------------------------

fn registry() -> &'static Mutex<HashMap<u64, Regex>> {
    static REG: OnceLock<Mutex<HashMap<u64, Regex>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

// --- Helpers ---------------------------------------------------

unsafe fn string_of_value(v: Value) -> Option<String> {
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

fn id_of(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<u64> {
    match v.as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Some(n as u64),
        _ => {
            ctx.runtime_error(format!("{}: id must be a non-negative integer.", label));
            None
        }
    }
}

fn string_arg(ctx: &mut dyn NativeContext, v: Value, label: &str, field: &str) -> Option<String> {
    match unsafe { string_of_value(v) } {
        Some(s) => Some(s),
        None => {
            ctx.runtime_error(format!("{}: {} must be a string.", label, field));
            None
        }
    }
}

fn with_regex<F, R>(
    ctx: &mut dyn NativeContext,
    id: u64,
    label: &str,
    f: F,
) -> Option<R>
where
    F: FnOnce(&Regex) -> R,
{
    let reg = registry().lock().unwrap();
    match reg.get(&id) {
        Some(re) => Some(f(re)),
        None => {
            ctx.runtime_error(format!("{}: unknown regex id {}.", label, id));
            None
        }
    }
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

fn put_val(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: Value) {
    let k = ctx.alloc_string(key.to_string());
    unsafe { (*map).set(k, value) };
}

/// Convert a regex::Captures into a Wren Map with text/start/end/groups/named.
fn captures_to_map(ctx: &mut dyn NativeContext, caps: &regex::Captures<'_>, re: &Regex) -> Value {
    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;

    let m0 = caps.get(0).unwrap();
    put_str(ctx, out_ptr, "text", m0.as_str().to_string());
    put_num(ctx, out_ptr, "start", m0.start() as f64);
    put_num(ctx, out_ptr, "end", m0.end() as f64);

    // groups: [full, g1, g2, ...] — non-participating groups are null.
    let mut group_vals: Vec<Value> = Vec::with_capacity(caps.len());
    for i in 0..caps.len() {
        match caps.get(i) {
            Some(m) => group_vals.push(ctx.alloc_string(m.as_str().to_string())),
            None => group_vals.push(Value::null()),
        }
    }
    let groups = ctx.alloc_list(group_vals);
    put_val(ctx, out_ptr, "groups", groups);

    // named: only present groups. Absent-but-declared groups omitted.
    let named = ctx.alloc_map();
    let named_ptr = named.as_object().unwrap() as *mut ObjMap;
    for name in re.capture_names().flatten() {
        if let Some(m) = caps.name(name) {
            put_str(ctx, named_ptr, name, m.as_str().to_string());
        }
    }
    put_val(ctx, out_ptr, "named", named);

    out
}

// --- compile / free --------------------------------------------

fn parse_flags(flags: &str) -> Result<(bool, bool, bool, bool, bool), String> {
    let mut case_insensitive = false;
    let mut multi_line = false;
    let mut dot_all = false;
    let mut swap_greed = false;
    let mut ignore_ws = false;
    for c in flags.chars() {
        match c {
            'i' => case_insensitive = true,
            'm' => multi_line = true,
            's' => dot_all = true,
            'U' => swap_greed = true,
            'x' => ignore_ws = true,
            other => return Err(format!("unknown flag '{}'", other)),
        }
    }
    Ok((case_insensitive, multi_line, dot_all, swap_greed, ignore_ws))
}

fn regex_compile(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(pattern) = string_arg(ctx, args[1], "Regex.compile", "pattern") else {
        return Value::null();
    };
    let flags = if args[2].is_null() {
        String::new()
    } else {
        match unsafe { string_of_value(args[2]) } {
            Some(s) => s,
            None => {
                ctx.runtime_error("Regex.compile: flags must be a string or null.".to_string());
                return Value::null();
            }
        }
    };
    let (ci, ml, da, sg, xws) = match parse_flags(&flags) {
        Ok(v) => v,
        Err(e) => {
            ctx.runtime_error(format!("Regex.compile: {}", e));
            return Value::null();
        }
    };
    let re = match RegexBuilder::new(&pattern)
        .case_insensitive(ci)
        .multi_line(ml)
        .dot_matches_new_line(da)
        .swap_greed(sg)
        .ignore_whitespace(xws)
        .build()
    {
        Ok(re) => re,
        Err(e) => {
            ctx.runtime_error(format!("Regex.compile: {}", e));
            return Value::null();
        }
    };
    let id = next_id();
    registry().lock().unwrap().insert(id, re);
    Value::num(id as f64)
}

fn regex_free(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.free") else {
        return Value::null();
    };
    registry().lock().unwrap().remove(&id);
    Value::null()
}

// --- match queries ---------------------------------------------

fn regex_is_match(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.isMatch") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.isMatch", "haystack") else {
        return Value::null();
    };
    let Some(v) = with_regex(ctx, id, "Regex.isMatch", |re| re.is_match(&hay)) else {
        return Value::null();
    };
    Value::bool(v)
}

fn regex_find(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.find") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.find", "haystack") else {
        return Value::null();
    };
    // We need &Regex inside to produce Captures tied to `hay`'s lifetime.
    // Clone the Regex out of the registry so we can drop the lock before
    // allocating into the VM (alloc can trigger GC, which must not run
    // while we hold a global mutex the GC might transitively touch).
    let re_opt: Option<Regex> = {
        let reg = registry().lock().unwrap();
        reg.get(&id).cloned()
    };
    let Some(re) = re_opt else {
        ctx.runtime_error(format!("Regex.find: unknown regex id {}.", id));
        return Value::null();
    };
    match re.captures(&hay) {
        Some(caps) => captures_to_map(ctx, &caps, &re),
        None => Value::null(),
    }
}

fn regex_find_all(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.findAll") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.findAll", "haystack") else {
        return Value::null();
    };
    let re_opt: Option<Regex> = {
        let reg = registry().lock().unwrap();
        reg.get(&id).cloned()
    };
    let Some(re) = re_opt else {
        ctx.runtime_error(format!("Regex.findAll: unknown regex id {}.", id));
        return Value::null();
    };
    let mut out: Vec<Value> = Vec::new();
    for caps in re.captures_iter(&hay) {
        out.push(captures_to_map(ctx, &caps, &re));
    }
    ctx.alloc_list(out)
}

// --- replace ---------------------------------------------------

fn regex_replace(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.replace") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.replace", "haystack") else {
        return Value::null();
    };
    let Some(rep) = string_arg(ctx, args[3], "Regex.replace", "replacement") else {
        return Value::null();
    };
    let out_opt = with_regex(ctx, id, "Regex.replace", |re| {
        re.replace(&hay, rep.as_str()).into_owned()
    });
    let Some(out) = out_opt else {
        return Value::null();
    };
    ctx.alloc_string(out)
}

fn regex_replace_all(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.replaceAll") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.replaceAll", "haystack") else {
        return Value::null();
    };
    let Some(rep) = string_arg(ctx, args[3], "Regex.replaceAll", "replacement") else {
        return Value::null();
    };
    let out_opt = with_regex(ctx, id, "Regex.replaceAll", |re| {
        re.replace_all(&hay, rep.as_str()).into_owned()
    });
    let Some(out) = out_opt else {
        return Value::null();
    };
    ctx.alloc_string(out)
}

// --- split -----------------------------------------------------

fn regex_split(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.split") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.split", "haystack") else {
        return Value::null();
    };
    let re_opt: Option<Regex> = {
        let reg = registry().lock().unwrap();
        reg.get(&id).cloned()
    };
    let Some(re) = re_opt else {
        ctx.runtime_error(format!("Regex.split: unknown regex id {}.", id));
        return Value::null();
    };
    let parts: Vec<Value> = re
        .split(&hay)
        .map(|s| ctx.alloc_string(s.to_string()))
        .collect();
    ctx.alloc_list(parts)
}

fn regex_splitn(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.splitN") else {
        return Value::null();
    };
    let Some(hay) = string_arg(ctx, args[2], "Regex.splitN", "haystack") else {
        return Value::null();
    };
    let n = match args[3].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as usize,
        _ => {
            ctx.runtime_error("Regex.splitN: n must be a non-negative integer.".to_string());
            return Value::null();
        }
    };
    let re_opt: Option<Regex> = {
        let reg = registry().lock().unwrap();
        reg.get(&id).cloned()
    };
    let Some(re) = re_opt else {
        ctx.runtime_error(format!("Regex.splitN: unknown regex id {}.", id));
        return Value::null();
    };
    let parts: Vec<Value> = re
        .splitn(&hay, n)
        .map(|s| ctx.alloc_string(s.to_string()))
        .collect();
    ctx.alloc_list(parts)
}

// --- metadata --------------------------------------------------

fn regex_pattern(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = id_of(ctx, args[1], "Regex.pattern") else {
        return Value::null();
    };
    let s_opt = with_regex(ctx, id, "Regex.pattern", |re| re.as_str().to_string());
    let Some(s) = s_opt else {
        return Value::null();
    };
    ctx.alloc_string(s)
}

// --- static helper: escape -------------------------------------

fn regex_escape(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(s) = string_arg(ctx, args[1], "Regex.escape", "text") else {
        return Value::null();
    };
    ctx.alloc_string(regex::escape(&s))
}

// --- Registration ----------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("RegexCore", vm.object_class);

    vm.primitive_static(class, "compile(_,_)", regex_compile);
    vm.primitive_static(class, "free(_)", regex_free);

    vm.primitive_static(class, "isMatch(_,_)", regex_is_match);
    vm.primitive_static(class, "find(_,_)", regex_find);
    vm.primitive_static(class, "findAll(_,_)", regex_find_all);

    vm.primitive_static(class, "replace(_,_,_)", regex_replace);
    vm.primitive_static(class, "replaceAll(_,_,_)", regex_replace_all);

    vm.primitive_static(class, "split(_,_)", regex_split);
    vm.primitive_static(class, "splitN(_,_,_)", regex_splitn);

    vm.primitive_static(class, "pattern(_)", regex_pattern);
    vm.primitive_static(class, "escape(_)", regex_escape);

    class
}
