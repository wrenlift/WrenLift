//! Optional `proc` module — subprocess spawn, capture, IPC, and
//! chaining.
//!
//! Two API shapes:
//!
//! - `run(argv, ...)` — spawn + wait in one blocking call,
//!   returns the full result map. Kept for the one-liner case.
//!
//! - `spawn(argv, ...)` → numeric process **id** — non-blocking.
//!   Callers drive lifecycle with `tryWait` / `wait` / `kill`,
//!   stream stdin with `writeStdin` / `closeStdin`, and chain
//!   pipelines by passing another process's id as `stdin_from_id`
//!   (the runtime transfers the pipe handle).
//!
//! @hatch:proc layers the `Process` class over the id so Wren
//! callers don't deal in raw integers.
//!
//! Design notes:
//! * Stdout/stderr reader threads run for every spawned process
//!   unless its stdout is consumed as another process's stdin
//!   (chained). That prevents pipe-buffer deadlocks for large
//!   output.
//! * Once a process has been waited on, its registry entry is
//!   kept (so repeat queries of stdout / code work), but the
//!   Child and pipe handles are dropped. Callers that want to
//!   free memory eagerly call `forget(id)`.
//! * All access goes through a global Mutex — this module is
//!   designed for a single-threaded Wren VM. Concurrent Wren
//!   threads don't exist yet.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::runtime::object::{NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType};
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Stream state (shared between drain thread and readers) ---

/// Bytes pulled off a subprocess pipe. A background drain thread
/// owns the `ChildStdout` / `ChildStderr` handle, reads chunks,
/// and pushes them into this queue. Readers — both blocking
/// (`readStdoutBytes`) and non-blocking (`tryReadStdoutBytes`) —
/// pull from the queue under the same mutex.
///
/// A condvar wakes blocking readers when new bytes arrive or EOF
/// is reached, so they don't spin-poll.
struct StreamState {
    buf: VecDeque<u8>,
    /// Set when the drain thread has hit EOF or a read error.
    eof: bool,
    /// Populated on read errors so the surface can report them.
    err: Option<String>,
    /// Total bytes ever pushed — used by `build_result` to size
    /// the final String allocation.
    total_pushed: usize,
}

type SharedStream = Arc<(Mutex<StreamState>, Condvar)>;

fn new_stream() -> SharedStream {
    Arc::new((
        Mutex::new(StreamState {
            buf: VecDeque::new(),
            eof: false,
            err: None,
            total_pushed: 0,
        }),
        Condvar::new(),
    ))
}

/// Spawn a drain thread that pumps `reader` into `state` until
/// EOF or error. Used for both stdout and stderr — generic over
/// the concrete pipe type via a boxed reader.
fn spawn_drain_thread<R: Read + Send + 'static>(
    mut reader: R,
    state: SharedStream,
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
                    s.total_pushed += n;
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

// --- Registry --------------------------------------------------

struct ProcEntry {
    /// Handle to the underlying child. Consumed on final wait /
    /// kill so repeat calls are cheap.
    child: Option<Child>,
    /// Writable stdin end. `None` once `closeStdin` was called
    /// or once the child exited.
    stdin: Option<ChildStdin>,
    /// Child's stdout, held *unread* until one of:
    ///   * Another process asks to chain from us — we take() it
    ///     out and hand it over as that child's stdin.
    ///   * A read (blocking or non-blocking) starts the drain
    ///     thread that owns the handle from then on.
    ///   * On `wait` / `tryWait` completion we start the drain
    ///     thread if it hasn't started yet.
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    /// Drain threads — spawned lazily by `ensure_*_reader`. They
    /// own the corresponding pipe handle and push bytes into
    /// `stdout_state` / `stderr_state`. Joined at reap.
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    /// Shared state between drain thread and readers. Holds the
    /// pending byte queue, EOF flag, and any drain error.
    stdout_state: SharedStream,
    stderr_state: SharedStream,
    /// Everything the drain thread has pushed so far, kept
    /// separately so the final `Result.stdout` string survives the
    /// VecDeque being drained by streaming reads.
    stdout_captured: Vec<u8>,
    stderr_captured: Vec<u8>,
    /// True if the stdout handle was transferred to another
    /// process — the drain thread never starts, and
    /// `stdout_captured` stays empty.
    stdout_chained: bool,
    exit_code: Option<i32>,
    /// Set by our own `kill` when it was a timeout-driven kill.
    timed_out: bool,
    /// Original pid for debugging / signalling outside the registry.
    pid: u32,
}

fn registry() -> &'static Mutex<HashMap<u64, ProcEntry>> {
    static REG: OnceLock<Mutex<HashMap<u64, ProcEntry>>> = OnceLock::new();
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

fn argv_from_list(ctx: &mut dyn NativeContext, v: Value) -> Option<Vec<String>> {
    let ptr = match v.as_object() {
        Some(p) => p as *const ObjList,
        None => {
            ctx.runtime_error("Proc: argv must be a list of strings.".to_string());
            return None;
        }
    };
    let (count, data) = unsafe { ((*ptr).count as usize, (*ptr).elements) };
    if count == 0 {
        ctx.runtime_error("Proc: argv must contain at least the program name.".to_string());
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let el = unsafe { *data.add(i) };
        match unsafe { string_of_value(el) } {
            Some(s) => out.push(s),
            None => {
                ctx.runtime_error("Proc: every argv entry must be a string.".to_string());
                return None;
            }
        }
    }
    Some(out)
}

fn env_from_map(ctx: &mut dyn NativeContext, v: Value) -> Option<Vec<(String, String)>> {
    if v.is_null() {
        return Some(Vec::new());
    }
    let ptr = match v.as_object() {
        Some(p) => p as *const ObjMap,
        None => {
            ctx.runtime_error("Proc: env must be a Map.".to_string());
            return None;
        }
    };
    let entries: Vec<(Value, Value)> =
        unsafe { (*ptr).entries.iter().map(|(k, v)| (k.0, *v)).collect() };
    let mut out = Vec::with_capacity(entries.len());
    for (k, v) in entries {
        let Some(key) = (unsafe { string_of_value(k) }) else {
            ctx.runtime_error("Proc: env keys must be strings.".to_string());
            return None;
        };
        let Some(val) = (unsafe { string_of_value(v) }) else {
            ctx.runtime_error("Proc: env values must be strings.".to_string());
            return None;
        };
        out.push((key, val));
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
fn put_bool(ctx: &mut dyn NativeContext, map: *mut ObjMap, key: &str, value: bool) {
    let k = ctx.alloc_string(key.to_string());
    unsafe { (*map).set(k, Value::bool(value)) };
}

// --- Core spawn ------------------------------------------------

/// Spawn arguments shared by `spawn` and `run`.
struct SpawnOpts<'a> {
    argv: Vec<String>,
    cwd: Option<String>,
    env: Vec<(String, String)>,
    /// Either raw bytes to pipe in, or a donor process to chain
    /// from. At most one should be set.
    stdin_data: Option<Vec<u8>>,
    stdin_from_id: Option<u64>,
    /// `run` wants stdin closed after the initial write so that
    /// programs reading from it (cat, grep, etc.) see EOF and
    /// can exit. `spawn` leaves it open so the handle-based
    /// caller can keep writing.
    close_stdin_after_write: bool,
    #[allow(dead_code)]
    label: &'a str,
}

fn build_command(opts: &SpawnOpts<'_>) -> Command {
    let mut cmd = Command::new(&opts.argv[0]);
    cmd.args(&opts.argv[1..]);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    // Stdin is piped unless the caller has explicitly wired it
    // from another process's stdout — that case sets Stdio
    // ourselves below.
    if opts.stdin_from_id.is_none() {
        cmd.stdin(Stdio::piped());
    }
    if let Some(dir) = &opts.cwd {
        cmd.current_dir(dir);
    }
    for (k, v) in &opts.env {
        cmd.env(k, v);
    }
    cmd
}

fn spawn_entry(ctx: &mut dyn NativeContext, mut opts: SpawnOpts<'_>) -> Option<(u64, ProcEntry)> {
    let mut cmd = build_command(&opts);

    // Chaining: pull the donor's (unread) stdout out of the
    // registry and hand it to the new Command as stdin. We take
    // it before the donor's reader thread is ever started — the
    // deferred-reader design is what makes this safe.
    let mut donor_chained = false;
    if let Some(donor_id) = opts.stdin_from_id {
        let donor_stdout: Option<ChildStdout> = {
            let mut reg = registry().lock().unwrap();
            let entry = match reg.get_mut(&donor_id) {
                Some(e) => e,
                None => {
                    ctx.runtime_error(format!(
                        "Proc: stdin donor process id {} not found.",
                        donor_id
                    ));
                    return None;
                }
            };
            if entry.stdout_chained {
                ctx.runtime_error(format!(
                    "Proc: stdin donor {} already piped to another process.",
                    donor_id
                ));
                return None;
            }
            let stdout = entry.stdout.take();
            entry.stdout_chained = true;
            stdout
        };
        if let Some(out) = donor_stdout {
            cmd.stdin(Stdio::from(out));
            donor_chained = true;
        } else {
            // Donor's stdout was already drained (e.g. wait was
            // called before chaining). Run with no input — shows
            // up as EOF on stdin, which is the least-surprising
            // outcome.
            cmd.stdin(Stdio::null());
        }
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            ctx.runtime_error(format!("Proc: spawn {:?}: {}", opts.argv[0], e));
            return None;
        }
    };

    // Pull stdin out of the child so we own it. If inline bytes
    // were supplied, push them now. By default we leave the pipe
    // open so later `writeStdin` calls can append; `run` sets
    // `close_stdin_after_write` so programs that read to EOF
    // (cat, grep, …) actually see it and exit.
    let mut stdin_handle: Option<ChildStdin> = if donor_chained {
        None
    } else {
        child.stdin.take()
    };
    if let (Some(bytes), Some(pipe)) = (opts.stdin_data.take(), stdin_handle.as_mut()) {
        let _ = pipe.write_all(&bytes);
    }
    if opts.close_stdin_after_write {
        stdin_handle = None;
    }

    // Keep both stdout and stderr idle. Reader threads start
    // lazily at reap unless streaming reads or chaining got there
    // first. Eager readers would race against streaming callers.
    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    let pid = child.id();
    let id = next_id();
    // `stdout_chained` here describes whether someone else has
    // taken OUR stdout — starts false, flips when a later spawn
    // cites us as `stdin_from_id`. `donor_chained` is about our
    // stdin, which is separate.
    let _ = donor_chained;
    let entry = ProcEntry {
        child: Some(child),
        stdin: stdin_handle,
        stdout: stdout_handle,
        stderr: stderr_handle,
        stdout_thread: None,
        stderr_thread: None,
        stdout_state: new_stream(),
        stderr_state: new_stream(),
        stdout_captured: Vec::new(),
        stderr_captured: Vec::new(),
        stdout_chained: false,
        exit_code: None,
        timed_out: false,
        pid,
    };
    Some((id, entry))
}

/// Kick off the stdout drain thread on demand. No-op if the
/// stdout was chained (the handle is gone) or if the drain is
/// already running.
fn ensure_stdout_reader(entry: &mut ProcEntry) {
    if entry.stdout_thread.is_some() {
        return;
    }
    let Some(s) = entry.stdout.take() else {
        return;
    };
    let state = Arc::clone(&entry.stdout_state);
    entry.stdout_thread = Some(spawn_drain_thread(s, state));
}

/// Symmetric helper for stderr. Both stdout and stderr are lazy
/// — the drain thread starts on first read or at reap, whichever
/// comes first.
fn ensure_stderr_reader(entry: &mut ProcEntry) {
    if entry.stderr_thread.is_some() {
        return;
    }
    let Some(s) = entry.stderr.take() else {
        return;
    };
    let state = Arc::clone(&entry.stderr_state);
    entry.stderr_thread = Some(spawn_drain_thread(s, state));
}

// --- Wait helpers ---------------------------------------------

/// Block on the child, collect reader threads, append their
/// output to the streaming buffers. Idempotent — repeat calls
/// after the first just echo the stored values.
fn reap(entry: &mut ProcEntry) {
    if entry.exit_code.is_some() || entry.timed_out {
        return;
    }
    // Kick off the reader threads now so they drain in parallel
    // with child.wait(). Without this, verbose programs could
    // deadlock on a full pipe buffer. If a streaming read has
    // already drained a handle, the corresponding ensure_* is a
    // no-op.
    ensure_stdout_reader(entry);
    ensure_stderr_reader(entry);
    let Some(mut child) = entry.child.take() else {
        return;
    };
    match child.wait() {
        Ok(status) => {
            entry.exit_code = Some(status.code().unwrap_or(-1));
        }
        Err(_) => {
            entry.exit_code = Some(-1);
        }
    }
    if let Some(th) = entry.stdout_thread.take() {
        let _ = th.join();
    }
    if let Some(th) = entry.stderr_thread.take() {
        let _ = th.join();
    }
    // Drain any bytes the reader thread pushed but no streaming
    // read consumed. Post-reap the VecDeque is whatever the drain
    // thread produced after the last `try_read` / `read`.
    drain_remaining(&entry.stdout_state, &mut entry.stdout_captured);
    drain_remaining(&entry.stderr_state, &mut entry.stderr_captured);
    // Drop stdin so writes after reap fail cleanly rather than
    // silently buffering into a dead pipe.
    entry.stdin = None;
}

/// Pop everything currently in `state.buf` into `captured`. Used
/// at reap to build the final Result, and by streaming reads to
/// record what they returned.
fn drain_remaining(state: &SharedStream, captured: &mut Vec<u8>) {
    let (lock, _) = &**state;
    let mut s = lock.lock().unwrap();
    captured.reserve(s.buf.len());
    captured.extend(s.buf.drain(..));
}

/// Return the result map, populating missing fields from the
/// entry. Does not consume the entry — the wrapper keeps the
/// slot alive so later queries (stdout, code) still work.
fn build_result(ctx: &mut dyn NativeContext, entry: &ProcEntry) -> Value {
    let out = ctx.alloc_map();
    let out_ptr = out.as_object().unwrap() as *mut ObjMap;
    put_num(ctx, out_ptr, "code", entry.exit_code.unwrap_or(-1) as f64);
    put_str(
        ctx,
        out_ptr,
        "stdout",
        String::from_utf8_lossy(&entry.stdout_captured).into_owned(),
    );
    put_str(
        ctx,
        out_ptr,
        "stderr",
        String::from_utf8_lossy(&entry.stderr_captured).into_owned(),
    );
    put_bool(ctx, out_ptr, "timedOut", entry.timed_out);
    out
}

// --- `run` (blocking, one-shot) -------------------------------

fn proc_run(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // argv, cwd, env, stdin_data, timeout
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
    let Some(env) = env_from_map(ctx, args[3]) else {
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

    let opts = SpawnOpts {
        argv,
        cwd,
        env,
        stdin_data,
        stdin_from_id: None,
        close_stdin_after_write: true,
        label: "Proc.run",
    };
    let Some((id, entry)) = spawn_entry(ctx, opts) else {
        return Value::null();
    };
    registry().lock().unwrap().insert(id, entry);

    // Start both reader threads before blocking so neither pipe
    // fills up while we poll. Stderr is lazy now (see the
    // spawn_entry comment); without this kick, a chatty program
    // would deadlock before we got to reap.
    {
        let mut reg = registry().lock().unwrap();
        let entry = reg.get_mut(&id).unwrap();
        ensure_stdout_reader(entry);
        ensure_stderr_reader(entry);
    }

    // Wait with optional timeout.
    let start = Instant::now();
    let (code, timed_out) = loop {
        let mut reg = registry().lock().unwrap();
        let entry = reg.get_mut(&id).expect("entry vanished");
        let child = entry.child.as_mut().expect("child taken");
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
            }
            Err(e) => {
                ctx.runtime_error(format!("Proc.run: wait: {}", e));
                reg.remove(&id);
                return Value::null();
            }
        }
        drop(reg);
        std::thread::sleep(Duration::from_millis(10));
    };
    // Seal the entry.
    {
        let mut reg = registry().lock().unwrap();
        if let Some(entry) = reg.get_mut(&id) {
            entry.exit_code = Some(code.unwrap_or(-1));
            entry.timed_out = timed_out;
            if let Some(th) = entry.stdout_thread.take() {
                let _ = th.join();
            }
            if let Some(th) = entry.stderr_thread.take() {
                let _ = th.join();
            }
            drain_remaining(&entry.stdout_state, &mut entry.stdout_captured);
            drain_remaining(&entry.stderr_state, &mut entry.stderr_captured);
            entry.stdin = None;
            // Drop the Child to release any remaining handles.
            let _ = entry.child.take();
        }
    }
    let reg = registry().lock().unwrap();
    let entry = reg.get(&id).unwrap();
    build_result(ctx, entry)
}

// --- Handle-based API -----------------------------------------

fn proc_spawn(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    // argv, cwd, env, stdin_data (String or null), stdin_from_id
    // (Num or null)
    let Some(argv) = argv_from_list(ctx, args[1]) else {
        return Value::null();
    };
    let cwd = if args[2].is_null() {
        None
    } else {
        match unsafe { string_of_value(args[2]) } {
            Some(s) => Some(s),
            None => {
                ctx.runtime_error("Proc.spawn: cwd must be a string or null.".to_string());
                return Value::null();
            }
        }
    };
    let Some(env) = env_from_map(ctx, args[3]) else {
        return Value::null();
    };
    let stdin_data = if args[4].is_null() {
        None
    } else {
        match unsafe { string_of_value(args[4]) } {
            Some(s) => Some(s.into_bytes()),
            None => {
                ctx.runtime_error("Proc.spawn: stdin must be a string or null.".to_string());
                return Value::null();
            }
        }
    };
    let stdin_from_id = if args[5].is_null() {
        None
    } else {
        match args[5].as_num() {
            Some(n) if n > 0.0 && n.fract() == 0.0 => Some(n as u64),
            _ => {
                ctx.runtime_error(
                    "Proc.spawn: stdinFromId must be a positive integer or null.".to_string(),
                );
                return Value::null();
            }
        }
    };
    if stdin_data.is_some() && stdin_from_id.is_some() {
        ctx.runtime_error(
            "Proc.spawn: pass either stdin bytes or stdinFromId, not both.".to_string(),
        );
        return Value::null();
    }

    let opts = SpawnOpts {
        argv,
        cwd,
        env,
        stdin_data,
        stdin_from_id,
        close_stdin_after_write: false,
        label: "Proc.spawn",
    };
    let Some((id, entry)) = spawn_entry(ctx, opts) else {
        return Value::null();
    };
    registry().lock().unwrap().insert(id, entry);
    Value::num(id as f64)
}

fn resolve_id(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<u64> {
    match v.as_num() {
        Some(n) if n > 0.0 && n.fract() == 0.0 => Some(n as u64),
        _ => {
            ctx.runtime_error(format!("{}: id must be a positive integer.", label));
            None
        }
    }
}

fn proc_write_stdin(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.writeStdin") else {
        return Value::null();
    };
    let Some(data) = (unsafe { string_of_value(args[2]) }) else {
        ctx.runtime_error("Proc.writeStdin: data must be a string.".to_string());
        return Value::null();
    };
    let mut reg = registry().lock().unwrap();
    let Some(entry) = reg.get_mut(&id) else {
        ctx.runtime_error(format!("Proc.writeStdin: process {} not found.", id));
        return Value::null();
    };
    let Some(pipe) = entry.stdin.as_mut() else {
        ctx.runtime_error(format!(
            "Proc.writeStdin: process {}'s stdin is closed.",
            id
        ));
        return Value::null();
    };
    if let Err(e) = pipe.write_all(data.as_bytes()) {
        ctx.runtime_error(format!("Proc.writeStdin: {}: {}", id, e));
    }
    Value::null()
}

fn proc_close_stdin(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.closeStdin") else {
        return Value::null();
    };
    let mut reg = registry().lock().unwrap();
    if let Some(entry) = reg.get_mut(&id) {
        entry.stdin = None;
    }
    Value::null()
}

fn proc_try_wait(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.tryWait") else {
        return Value::null();
    };
    let mut reg = registry().lock().unwrap();
    let Some(entry) = reg.get_mut(&id) else {
        ctx.runtime_error(format!("Proc.tryWait: process {} not found.", id));
        return Value::null();
    };
    // Already reaped → answer from the cache.
    if entry.exit_code.is_some() || entry.timed_out {
        return build_result(ctx, entry);
    }
    let Some(child) = entry.child.as_mut() else {
        return build_result(ctx, entry);
    };
    match child.try_wait() {
        Ok(Some(status)) => {
            entry.exit_code = Some(status.code().unwrap_or(-1));
            // Start the readers if they're not running yet, then
            // join them. Child already exited so joins return
            // quickly.
            ensure_stdout_reader(entry);
            ensure_stderr_reader(entry);
            if let Some(th) = entry.stdout_thread.take() {
                let _ = th.join();
            }
            if let Some(th) = entry.stderr_thread.take() {
                let _ = th.join();
            }
            drain_remaining(&entry.stdout_state, &mut entry.stdout_captured);
            drain_remaining(&entry.stderr_state, &mut entry.stderr_captured);
            entry.stdin = None;
            let _ = entry.child.take();
            build_result(ctx, entry)
        }
        Ok(None) => Value::null(),
        Err(e) => {
            ctx.runtime_error(format!("Proc.tryWait: {}: {}", id, e));
            Value::null()
        }
    }
}

fn proc_wait(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.wait") else {
        return Value::null();
    };
    {
        let mut reg = registry().lock().unwrap();
        let Some(entry) = reg.get_mut(&id) else {
            ctx.runtime_error(format!("Proc.wait: process {} not found.", id));
            return Value::null();
        };
        reap(entry);
    }
    let reg = registry().lock().unwrap();
    let entry = reg.get(&id).unwrap();
    build_result(ctx, entry)
}

fn proc_kill(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.kill") else {
        return Value::null();
    };
    let mut reg = registry().lock().unwrap();
    let Some(entry) = reg.get_mut(&id) else {
        return Value::null();
    };
    if let Some(child) = entry.child.as_mut() {
        let _ = child.kill();
    }
    entry.timed_out = false; // Explicit kill, not a timeout.
    Value::null()
}

fn proc_alive(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.alive") else {
        return Value::bool(false);
    };
    let mut reg = registry().lock().unwrap();
    let Some(entry) = reg.get_mut(&id) else {
        return Value::bool(false);
    };
    if entry.exit_code.is_some() || entry.timed_out {
        return Value::bool(false);
    }
    let Some(child) = entry.child.as_mut() else {
        return Value::bool(false);
    };
    match child.try_wait() {
        Ok(Some(_)) => Value::bool(false),
        Ok(None) => Value::bool(true),
        Err(_) => Value::bool(false),
    }
}

fn proc_pid(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.pid") else {
        return Value::null();
    };
    let reg = registry().lock().unwrap();
    match reg.get(&id) {
        Some(e) => Value::num(e.pid as f64),
        None => Value::null(),
    }
}

/// Drop the registry entry. Callers who want to free memory
/// after consuming a result. Otherwise entries live until the
/// VM exits — fine for short-lived scripts, a liability for
/// long-running servers.
fn proc_forget(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.forget") else {
        return Value::null();
    };
    registry().lock().unwrap().remove(&id);
    Value::null()
}

// --- Streaming I/O --------------------------------------------

/// Convert a `&[u8]` slice to a Wren `List<Num>`. Allocates
/// through the ctx so the caller doesn't reach into GC details.
fn bytes_slice_to_list(ctx: &mut dyn NativeContext, bytes: &[u8]) -> Value {
    let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
    ctx.alloc_list(elements)
}

/// Pull a `Vec<u8>` out of a Wren `List<Num>` argument. Rejects
/// non-byte-shaped entries; used by `write_stdin_bytes`.
fn bytes_from_list_arg(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<Vec<u8>> {
    let ptr = match v.as_object() {
        Some(p) => p as *const ObjList,
        None => {
            ctx.runtime_error(format!("{}: expected a list of bytes.", label));
            return None;
        }
    };
    let (count, data) = unsafe { ((*ptr).count as usize, (*ptr).elements) };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let v = unsafe { *data.add(i) };
        let n = match v.as_num() {
            Some(n) => n,
            None => {
                ctx.runtime_error(format!("{}: bytes must be numbers.", label));
                return None;
            }
        };
        if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
            ctx.runtime_error(format!("{}: bytes must be integers in 0..=255.", label));
            return None;
        }
        out.push(n as u8);
    }
    Some(out)
}

/// Shared body for `readStdoutBytes` / `readStderrBytes`. Returns
/// either a List<Num> of up to `max` bytes or null at EOF.
///
/// Contract:
///   * `null`      — no more bytes will ever arrive (EOF).
///   * empty list  — only from `try_read`; "nothing right now, yield and retry".
///   * non-empty   — up to `max` bytes.
///
/// Implementation: the drain thread (started here if not already
/// running) pushes bytes into `stdout_state` / `stderr_state`;
/// the blocking variant waits on the condvar until bytes arrive
/// or EOF is reached.
fn read_stream_bytes(
    ctx: &mut dyn NativeContext,
    args: &[Value],
    which: Stream,
    nonblocking: bool,
) -> Value {
    let label = match (which, nonblocking) {
        (Stream::Stdout, false) => "Proc.readStdoutBytes",
        (Stream::Stderr, false) => "Proc.readStderrBytes",
        (Stream::Stdout, true) => "Proc.tryReadStdoutBytes",
        (Stream::Stderr, true) => "Proc.tryReadStderrBytes",
    };
    let Some(id) = resolve_id(ctx, args[1], label) else {
        return Value::null();
    };
    let max = match args[2].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as usize,
        _ => {
            ctx.runtime_error(format!("{}: max must be a non-negative integer.", label));
            return Value::null();
        }
    };
    if max == 0 {
        return ctx.alloc_list(Vec::new());
    }

    // Ensure the drain thread has started. If the handle has been
    // chained, this is a no-op — the queue will stay empty and we
    // fall through to the EOF path once the downstream process
    // closes the pipe.
    let state = {
        let mut reg = registry().lock().unwrap();
        let Some(entry) = reg.get_mut(&id) else {
            ctx.runtime_error(format!("{}: unknown process id {}.", label, id));
            return Value::null();
        };
        match which {
            Stream::Stdout => {
                ensure_stdout_reader(entry);
                Arc::clone(&entry.stdout_state)
            }
            Stream::Stderr => {
                ensure_stderr_reader(entry);
                Arc::clone(&entry.stderr_state)
            }
        }
    };

    // Pull bytes out of the shared queue. Blocking variants wait
    // on the condvar; non-blocking variants return whatever's
    // ready (possibly nothing).
    let chunk_opt: Option<Vec<u8>> = {
        let (lock, cvar) = &*state;
        let mut s = lock.lock().unwrap();
        if nonblocking {
            if s.buf.is_empty() {
                if s.eof {
                    None
                } else {
                    // WouldBlock — caller retries after
                    // `Fiber.yield()`. Signal via empty list.
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
        None => Value::null(),
        Some(chunk) => {
            // Record what we just handed out so build_result can
            // reconstruct the full stdout/stderr string later.
            if !chunk.is_empty() {
                let mut reg = registry().lock().unwrap();
                if let Some(entry) = reg.get_mut(&id) {
                    let captured = match which {
                        Stream::Stdout => &mut entry.stdout_captured,
                        Stream::Stderr => &mut entry.stderr_captured,
                    };
                    captured.extend_from_slice(&chunk);
                }
            }
            bytes_slice_to_list(ctx, &chunk)
        }
    }
}

#[derive(Clone, Copy)]
enum Stream {
    Stdout,
    Stderr,
}

fn proc_read_stdout_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    read_stream_bytes(ctx, args, Stream::Stdout, false)
}

fn proc_read_stderr_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    read_stream_bytes(ctx, args, Stream::Stderr, false)
}

fn proc_try_read_stdout_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    read_stream_bytes(ctx, args, Stream::Stdout, true)
}

fn proc_try_read_stderr_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    read_stream_bytes(ctx, args, Stream::Stderr, true)
}

fn proc_write_stdin_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Proc.writeStdinBytes") else {
        return Value::null();
    };
    let Some(bytes) = bytes_from_list_arg(ctx, args[2], "Proc.writeStdinBytes") else {
        return Value::null();
    };
    let mut reg = registry().lock().unwrap();
    let Some(entry) = reg.get_mut(&id) else {
        ctx.runtime_error(format!("Proc.writeStdinBytes: unknown process id {}.", id));
        return Value::null();
    };
    let Some(pipe) = entry.stdin.as_mut() else {
        ctx.runtime_error("Proc.writeStdinBytes: stdin is closed.".to_string());
        return Value::null();
    };
    if let Err(e) = pipe.write_all(&bytes) {
        ctx.runtime_error(format!("Proc.writeStdinBytes: {}", e));
    }
    Value::null()
}

// --- Registration ---------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("ProcCore", vm.object_class);
    vm.primitive_static(class, "run(_,_,_,_,_)", proc_run);
    vm.primitive_static(class, "spawn(_,_,_,_,_)", proc_spawn);
    vm.primitive_static(class, "writeStdin(_,_)", proc_write_stdin);
    vm.primitive_static(class, "writeStdinBytes(_,_)", proc_write_stdin_bytes);
    vm.primitive_static(class, "closeStdin(_)", proc_close_stdin);
    vm.primitive_static(class, "readStdoutBytes(_,_)", proc_read_stdout_bytes);
    vm.primitive_static(class, "readStderrBytes(_,_)", proc_read_stderr_bytes);
    vm.primitive_static(class, "tryReadStdoutBytes(_,_)", proc_try_read_stdout_bytes);
    vm.primitive_static(class, "tryReadStderrBytes(_,_)", proc_try_read_stderr_bytes);
    vm.primitive_static(class, "tryWait(_)", proc_try_wait);
    vm.primitive_static(class, "wait(_)", proc_wait);
    vm.primitive_static(class, "kill(_)", proc_kill);
    vm.primitive_static(class, "alive(_)", proc_alive);
    vm.primitive_static(class, "pid(_)", proc_pid);
    vm.primitive_static(class, "forget(_)", proc_forget);
    class
}
