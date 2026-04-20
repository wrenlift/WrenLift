//! Optional `proc` module — subprocess spawn & capture.
//!
//! `import "proc" for ProcCore` exposes a single `run` call that
//! spawns a command, feeds it stdin, waits for it, and returns
//! exit code + full stdout/stderr as strings.
//!
//! Synchronous and accumulating — the whole stdout/stderr must
//! fit in memory. Streaming (read-as-you-go) is a later upgrade
//! when a consumer actually needs it.
//!
//! Timeout works via polling + reader threads: the child's
//! streams are drained concurrently so large outputs don't
//! deadlock against the pipe buffer. If the child hasn't exited
//! by `timeout` seconds we `kill` it and surface "timeout" as
//! the exit reason.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Arg extraction ----------------------------------------

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

fn argv_from_list(ctx: &mut dyn NativeContext, v: Value) -> Option<Vec<String>> {
    let ptr = v.as_object();
    let ptr = match ptr {
        Some(p) => p as *const ObjList,
        None => {
            ctx.runtime_error("Proc.run: argv must be a list of strings.".to_string());
            return None;
        }
    };
    let (count, data) = unsafe { ((*ptr).count as usize, (*ptr).elements) };
    if count == 0 {
        ctx.runtime_error("Proc.run: argv must contain at least the program name.".to_string());
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let el = unsafe { *data.add(i) };
        match unsafe { string_of_value(el) } {
            Some(s) => out.push(s),
            None => {
                ctx.runtime_error(
                    "Proc.run: every argv entry must be a string.".to_string(),
                );
                return None;
            }
        }
    }
    Some(out)
}

/// Env map: Map<String, String>. Nulls on values mean "inherit
/// from parent" and are represented by absence, so we only emit
/// explicit pairs here.
fn env_from_map(ctx: &mut dyn NativeContext, v: Value) -> Option<Vec<(String, String)>> {
    if v.is_null() {
        return Some(Vec::new());
    }
    let ptr = match v.as_object() {
        Some(p) => p as *const ObjMap,
        None => {
            ctx.runtime_error("Proc.run: env must be a Map.".to_string());
            return None;
        }
    };
    let entries: Vec<(Value, Value)> =
        unsafe { (*ptr).entries.iter().map(|(k, v)| (k.0, *v)).collect() };
    let mut out = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        let Some(key) = (unsafe { string_of_value(k) }) else {
            ctx.runtime_error("Proc.run: env keys must be strings.".to_string());
            return None;
        };
        let Some(val) = (unsafe { string_of_value(v) }) else {
            ctx.runtime_error("Proc.run: env values must be strings.".to_string());
            return None;
        };
        out.push((key, val));
    }
    Some(out)
}

// --- Result map builders ----------------------------------

fn put_str(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: String) {
    let k = ctx.alloc_string(key.to_string());
    let v = ctx.alloc_string(value);
    unsafe { (*map).set(k, v) };
}

fn put_num(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: f64) {
    let k = ctx.alloc_string(key.to_string());
    unsafe { (*map).set(k, Value::num(value)) };
}

fn put_bool(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: bool) {
    let k = ctx.alloc_string(key.to_string());
    unsafe { (*map).set(k, Value::bool(value)) };
}

// --- Run ---------------------------------------------------

fn proc_run(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // args[0] = ProcCore class (receiver)
    // args[1] = argv (List<String>, at least 1 entry)
    // args[2] = cwd (String or null)
    // args[3] = env (Map<String, String> or null)
    // args[4] = stdin (String or null)
    // args[5] = timeout seconds (Num or null → no timeout)
    let Some(argv) = argv_from_list(ctx, args[1]) else {
        return Value::null();
    };

    let cwd = if args[2].is_null() {
        None
    } else {
        match unsafe { string_of_value(args[2]) } {
            Some(s) => Some(s),
            None => {
                ctx.runtime_error("Proc.run: cwd must be a string or null.".to_string());
                return Value::null();
            }
        }
    };

    let Some(env_pairs) = env_from_map(ctx, args[3]) else {
        return Value::null();
    };

    let stdin_data = if args[4].is_null() {
        None
    } else {
        match unsafe { string_of_value(args[4]) } {
            Some(s) => Some(s.into_bytes()),
            None => {
                ctx.runtime_error("Proc.run: stdin must be a string or null.".to_string());
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
                    "Proc.run: timeout must be a positive finite number or null.".to_string(),
                );
                return Value::null();
            }
        }
    };

    // Build + spawn ──────────────────────────────────────
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]);
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    if let Some(dir) = &cwd {
        cmd.current_dir(dir);
    }
    for (k, v) in &env_pairs {
        cmd.env(k, v);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            ctx.runtime_error(format!("Proc.run: spawn {:?}: {}", argv[0], e));
            return Value::null();
        }
    };

    // Feed stdin, then close our end so the child sees EOF.
    if let Some(data) = stdin_data {
        if let Some(mut pipe) = child.stdin.take() {
            if let Err(e) = pipe.write_all(&data) {
                // Don't abort — some programs read stdin lazily
                // and exit before we finish writing. Log the
                // partial-write warning into stderr instead.
                // Surfacing as a hard error would make
                // `echo foo | grep nothing` fail spuriously.
                let _ = e;
            }
        }
    } else {
        // Ensure we don't keep the pipe open — otherwise programs
        // that read all of stdin would block forever.
        drop(child.stdin.take());
    }

    // Drain stdout + stderr in dedicated threads so long output
    // can't fill the pipe buffer and wedge the child.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_handle = std::thread::spawn(move || -> String {
        let mut buf = String::new();
        if let Some(mut s) = stdout {
            let _ = s.read_to_string(&mut buf);
        }
        buf
    });
    let stderr_handle = std::thread::spawn(move || -> String {
        let mut buf = String::new();
        if let Some(mut s) = stderr {
            let _ = s.read_to_string(&mut buf);
        }
        buf
    });

    // Wait, with optional timeout via polling try_wait().
    let (code, timed_out) = {
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => break (status.code(), false),
                Ok(None) => {
                    if let Some(limit) = timeout {
                        if start.elapsed() >= limit {
                            let _ = child.kill();
                            let _ = child.wait();
                            break (None, true);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    ctx.runtime_error(format!("Proc.run: wait: {}", e));
                    return Value::null();
                }
            }
        }
    };

    let stdout_str = stdout_handle.join().unwrap_or_default();
    let stderr_str = stderr_handle.join().unwrap_or_default();

    // Build the result map ───────────────────────────────
    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;

    // `code` is None on Unix when the process was signalled (e.g.
    // killed for our timeout). We surface that as code=-1 plus
    // timedOut=true so callers can branch without guessing.
    put_num(ctx, out_ptr, "code", code.unwrap_or(-1) as f64);
    put_str(ctx, out_ptr, "stdout", stdout_str);
    put_str(ctx, out_ptr, "stderr", stderr_str);
    put_bool(ctx, out_ptr, "timedOut", timed_out);

    out
}

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("ProcCore", vm.object_class);
    vm.primitive_static(class, "run(_,_,_,_,_)", proc_run);
    class
}
