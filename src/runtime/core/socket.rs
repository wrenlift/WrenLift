//! Optional `socket` module — TCP listeners, connections, and UDP
//! datagram sockets. Paired with @hatch:socket on the Wren side.
//!
//! Design follows @hatch:proc: every long-lived handle lives in a
//! global `Mutex<HashMap<u64, _>>` keyed by a monotonically
//! increasing id. Wren callers hold the id as a `Num`, and every
//! operation looks up the entry by id. Each handle type has its
//! own registry so id spaces don't collide.
//!
//! Each call exists in two forms where it makes sense:
//!
//!   * a blocking variant (`read`, `accept`, ...) that parks the
//!     calling thread on the socket,
//!   * a non-blocking variant (`tryRead`, `tryAccept`, ...) that
//!     returns `null` immediately when nothing is ready.
//!
//! The non-blocking form lets Wren drive cooperative schedulers
//! (pair with `Fiber.yield()` in @hatch:events) without needing
//! true async primitives in the runtime.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Registries -------------------------------------------------

fn tcp_listeners() -> &'static Mutex<HashMap<u64, TcpListener>> {
    static R: OnceLock<Mutex<HashMap<u64, TcpListener>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn tcp_streams() -> &'static Mutex<HashMap<u64, TcpStream>> {
    static R: OnceLock<Mutex<HashMap<u64, TcpStream>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn udp_sockets() -> &'static Mutex<HashMap<u64, UdpSocket>> {
    static R: OnceLock<Mutex<HashMap<u64, UdpSocket>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, Ordering::SeqCst)
}

use super::bytes_from_value;

fn bytes_to_list(ctx: &mut dyn NativeContext, bytes: &[u8]) -> Value {
    let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
    ctx.alloc_list(elements)
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

fn resolve_count(ctx: &mut dyn NativeContext, v: Value, label: &str) -> Option<usize> {
    match v.as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Some(n as usize),
        _ => {
            ctx.runtime_error(format!("{}: count must be a non-negative integer.", label));
            None
        }
    }
}

// --- TCP listener ----------------------------------------------

fn tcp_listen(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(addr) = super::validate_string(ctx, args[1], "Tcp.listen") else {
        return Value::null();
    };
    match TcpListener::bind(&addr) {
        Ok(listener) => {
            // Accept is blocking by default. The non-blocking
            // switch only flips for `tryAccept`; we keep listeners
            // in blocking mode here so the first `accept` after
            // `listen` doesn't race on the not-yet-attached client.
            let id = next_id();
            tcp_listeners().lock().unwrap().insert(id, listener);
            Value::num(id as f64)
        }
        Err(e) => {
            ctx.runtime_error(format!("Tcp.listen: {}: {}", addr, e));
            Value::null()
        }
    }
}

fn tcp_accept(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.accept") else {
        return Value::null();
    };
    // Clone the listener out of the registry so we can release the
    // mutex before blocking on accept(). Otherwise every parallel
    // accept on unrelated listeners would serialize.
    let listener = {
        let reg = tcp_listeners().lock().unwrap();
        match reg.get(&id) {
            Some(l) => match l.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Tcp.accept: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Tcp.accept: listener {} not found.", id));
                return Value::null();
            }
        }
    };
    // Ensure blocking mode; a previous tryAccept may have left it
    // non-blocking on this listener's underlying fd.
    let _ = listener.set_nonblocking(false);
    match listener.accept() {
        Ok((stream, _peer)) => {
            let sid = next_id();
            tcp_streams().lock().unwrap().insert(sid, stream);
            Value::num(sid as f64)
        }
        Err(e) => {
            ctx.runtime_error(format!("Tcp.accept: {}: {}", id, e));
            Value::null()
        }
    }
}

fn tcp_try_accept(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.tryAccept") else {
        return Value::null();
    };
    let listener = {
        let reg = tcp_listeners().lock().unwrap();
        match reg.get(&id) {
            Some(l) => match l.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Tcp.tryAccept: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Tcp.tryAccept: listener {} not found.", id));
                return Value::null();
            }
        }
    };
    if let Err(e) = listener.set_nonblocking(true) {
        ctx.runtime_error(format!("Tcp.tryAccept: {}: {}", id, e));
        return Value::null();
    }
    match listener.accept() {
        Ok((stream, _peer)) => {
            // Hand the stream back in blocking mode regardless of
            // the listener's flag — callers expect `read`/`write`
            // on the returned id to be synchronous by default.
            let _ = stream.set_nonblocking(false);
            let sid = next_id();
            tcp_streams().lock().unwrap().insert(sid, stream);
            Value::num(sid as f64)
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Value::null(),
        Err(e) => {
            ctx.runtime_error(format!("Tcp.tryAccept: {}: {}", id, e));
            Value::null()
        }
    }
}

fn tcp_listener_local_addr(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.listenerLocalAddr") else {
        return Value::null();
    };
    let reg = tcp_listeners().lock().unwrap();
    let Some(l) = reg.get(&id) else {
        ctx.runtime_error(format!("Tcp.listenerLocalAddr: listener {} not found.", id));
        return Value::null();
    };
    match l.local_addr() {
        Ok(a) => ctx.alloc_string(a.to_string()),
        Err(e) => {
            ctx.runtime_error(format!("Tcp.listenerLocalAddr: {}", e));
            Value::null()
        }
    }
}

fn tcp_close_listener(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.closeListener") else {
        return Value::null();
    };
    tcp_listeners().lock().unwrap().remove(&id);
    Value::null()
}

// --- TCP stream ------------------------------------------------

fn tcp_connect(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(addr) = super::validate_string(ctx, args[1], "Tcp.connect") else {
        return Value::null();
    };
    // args[2] is either null (block with OS default) or a Num of
    // milliseconds for an explicit connect timeout.
    let timeout_ms = if args[2].is_null() {
        None
    } else {
        match args[2].as_num() {
            Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Some(n as u64),
            _ => {
                ctx.runtime_error(
                    "Tcp.connect: timeoutMs must be a non-negative integer or null.".to_string(),
                );
                return Value::null();
            }
        }
    };
    let stream_result = match timeout_ms {
        None => TcpStream::connect(&addr),
        Some(ms) => {
            // `connect_timeout` needs a SocketAddr, not a String.
            // Resolve first so callers can still use "host:port"
            // notation.
            let resolved = match std::net::ToSocketAddrs::to_socket_addrs(&addr) {
                Ok(mut it) => match it.next() {
                    Some(a) => a,
                    None => {
                        ctx.runtime_error(format!("Tcp.connect: {}: no addresses resolved.", addr));
                        return Value::null();
                    }
                },
                Err(e) => {
                    ctx.runtime_error(format!("Tcp.connect: {}: {}", addr, e));
                    return Value::null();
                }
            };
            TcpStream::connect_timeout(&resolved, Duration::from_millis(ms))
        }
    };
    match stream_result {
        Ok(stream) => {
            let id = next_id();
            tcp_streams().lock().unwrap().insert(id, stream);
            Value::num(id as f64)
        }
        Err(e) => {
            ctx.runtime_error(format!("Tcp.connect: {}: {}", addr, e));
            Value::null()
        }
    }
}

fn tcp_read(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.read") else {
        return Value::null();
    };
    let Some(max) = resolve_count(ctx, args[2], "Tcp.read") else {
        return Value::null();
    };
    let mut stream = {
        let reg = tcp_streams().lock().unwrap();
        match reg.get(&id) {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Tcp.read: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Tcp.read: stream {} not found.", id));
                return Value::null();
            }
        }
    };
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(None);
    let mut buf = vec![0u8; max];
    match stream.read(&mut buf) {
        // EOF → empty list (NOT null; null is reserved for the
        // non-blocking would-block case).
        Ok(0) => bytes_to_list(ctx, &[]),
        Ok(n) => {
            buf.truncate(n);
            bytes_to_list(ctx, &buf)
        }
        Err(e) => {
            ctx.runtime_error(format!("Tcp.read: {}: {}", id, e));
            Value::null()
        }
    }
}

fn tcp_try_read(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.tryRead") else {
        return Value::null();
    };
    let Some(max) = resolve_count(ctx, args[2], "Tcp.tryRead") else {
        return Value::null();
    };
    let mut stream = {
        let reg = tcp_streams().lock().unwrap();
        match reg.get(&id) {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Tcp.tryRead: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Tcp.tryRead: stream {} not found.", id));
                return Value::null();
            }
        }
    };
    if let Err(e) = stream.set_nonblocking(true) {
        ctx.runtime_error(format!("Tcp.tryRead: {}: {}", id, e));
        return Value::null();
    }
    let mut buf = vec![0u8; max];
    let result = stream.read(&mut buf);
    // Always flip back so a subsequent blocking read does what it
    // says on the tin.
    let _ = stream.set_nonblocking(false);
    match result {
        Ok(0) => bytes_to_list(ctx, &[]), // EOF
        Ok(n) => {
            buf.truncate(n);
            bytes_to_list(ctx, &buf)
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Value::null(),
        Err(e) => {
            ctx.runtime_error(format!("Tcp.tryRead: {}: {}", id, e));
            Value::null()
        }
    }
}

fn tcp_write(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.write") else {
        return Value::null();
    };
    let Some(data) = bytes_from_value(ctx, args[2], "Tcp.write") else {
        return Value::null();
    };
    let mut stream = {
        let reg = tcp_streams().lock().unwrap();
        match reg.get(&id) {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Tcp.write: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Tcp.write: stream {} not found.", id));
                return Value::null();
            }
        }
    };
    let _ = stream.set_nonblocking(false);
    match stream.write_all(&data) {
        Ok(()) => Value::num(data.len() as f64),
        Err(e) => {
            ctx.runtime_error(format!("Tcp.write: {}: {}", id, e));
            Value::null()
        }
    }
}

fn tcp_set_timeout(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.setReadTimeout") else {
        return Value::null();
    };
    // null → clears the timeout (block indefinitely).
    let dur = if args[2].is_null() {
        None
    } else {
        match args[2].as_num() {
            Some(n) if n.is_finite() && n > 0.0 && n.fract() == 0.0 => {
                Some(Duration::from_millis(n as u64))
            }
            _ => {
                ctx.runtime_error(
                    "Tcp.setReadTimeout: ms must be a positive integer or null.".to_string(),
                );
                return Value::null();
            }
        }
    };
    let reg = tcp_streams().lock().unwrap();
    let Some(stream) = reg.get(&id) else {
        ctx.runtime_error(format!("Tcp.setReadTimeout: stream {} not found.", id));
        return Value::null();
    };
    if let Err(e) = stream.set_read_timeout(dur) {
        ctx.runtime_error(format!("Tcp.setReadTimeout: {}", e));
    }
    Value::null()
}

fn tcp_peer_addr(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.peerAddr") else {
        return Value::null();
    };
    let reg = tcp_streams().lock().unwrap();
    let Some(s) = reg.get(&id) else {
        ctx.runtime_error(format!("Tcp.peerAddr: stream {} not found.", id));
        return Value::null();
    };
    match s.peer_addr() {
        Ok(a) => ctx.alloc_string(a.to_string()),
        Err(e) => {
            ctx.runtime_error(format!("Tcp.peerAddr: {}", e));
            Value::null()
        }
    }
}

fn tcp_stream_local_addr(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.localAddr") else {
        return Value::null();
    };
    let reg = tcp_streams().lock().unwrap();
    let Some(s) = reg.get(&id) else {
        ctx.runtime_error(format!("Tcp.localAddr: stream {} not found.", id));
        return Value::null();
    };
    match s.local_addr() {
        Ok(a) => ctx.alloc_string(a.to_string()),
        Err(e) => {
            ctx.runtime_error(format!("Tcp.localAddr: {}", e));
            Value::null()
        }
    }
}

fn tcp_close_stream(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Tcp.close") else {
        return Value::null();
    };
    // Dropping the TcpStream closes both halves. `shutdown(Both)`
    // first gives a deterministic close signal to the peer even
    // if another clone of this stream is still alive somewhere.
    if let Some(stream) = tcp_streams().lock().unwrap().remove(&id) {
        let _ = stream.shutdown(std::net::Shutdown::Both);
    }
    Value::null()
}

// --- UDP -------------------------------------------------------

fn udp_bind(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(addr) = super::validate_string(ctx, args[1], "Udp.bind") else {
        return Value::null();
    };
    match UdpSocket::bind(&addr) {
        Ok(sock) => {
            let id = next_id();
            udp_sockets().lock().unwrap().insert(id, sock);
            Value::num(id as f64)
        }
        Err(e) => {
            ctx.runtime_error(format!("Udp.bind: {}: {}", addr, e));
            Value::null()
        }
    }
}

fn udp_send_to(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Udp.sendTo") else {
        return Value::null();
    };
    let Some(data) = bytes_from_value(ctx, args[2], "Udp.sendTo") else {
        return Value::null();
    };
    let Some(dest) = super::validate_string(ctx, args[3], "Udp.sendTo (dest)") else {
        return Value::null();
    };
    let sock = {
        let reg = udp_sockets().lock().unwrap();
        match reg.get(&id) {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Udp.sendTo: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Udp.sendTo: socket {} not found.", id));
                return Value::null();
            }
        }
    };
    match sock.send_to(&data, &dest) {
        Ok(n) => Value::num(n as f64),
        Err(e) => {
            ctx.runtime_error(format!("Udp.sendTo: {}: {}", id, e));
            Value::null()
        }
    }
}

fn udp_recv_from(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Udp.recvFrom") else {
        return Value::null();
    };
    let Some(max) = resolve_count(ctx, args[2], "Udp.recvFrom") else {
        return Value::null();
    };
    let sock = {
        let reg = udp_sockets().lock().unwrap();
        match reg.get(&id) {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Udp.recvFrom: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Udp.recvFrom: socket {} not found.", id));
                return Value::null();
            }
        }
    };
    let _ = sock.set_nonblocking(false);
    let mut buf = vec![0u8; max];
    match sock.recv_from(&mut buf) {
        Ok((n, peer)) => {
            buf.truncate(n);
            let bytes = bytes_to_list(ctx, &buf);
            let addr = ctx.alloc_string(peer.to_string());
            ctx.alloc_list(vec![bytes, addr])
        }
        Err(e) => {
            ctx.runtime_error(format!("Udp.recvFrom: {}: {}", id, e));
            Value::null()
        }
    }
}

fn udp_try_recv_from(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Udp.tryRecvFrom") else {
        return Value::null();
    };
    let Some(max) = resolve_count(ctx, args[2], "Udp.tryRecvFrom") else {
        return Value::null();
    };
    let sock = {
        let reg = udp_sockets().lock().unwrap();
        match reg.get(&id) {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => {
                    ctx.runtime_error(format!("Udp.tryRecvFrom: {}: {}", id, e));
                    return Value::null();
                }
            },
            None => {
                ctx.runtime_error(format!("Udp.tryRecvFrom: socket {} not found.", id));
                return Value::null();
            }
        }
    };
    if let Err(e) = sock.set_nonblocking(true) {
        ctx.runtime_error(format!("Udp.tryRecvFrom: {}: {}", id, e));
        return Value::null();
    }
    let mut buf = vec![0u8; max];
    let result = sock.recv_from(&mut buf);
    let _ = sock.set_nonblocking(false);
    match result {
        Ok((n, peer)) => {
            buf.truncate(n);
            let bytes = bytes_to_list(ctx, &buf);
            let addr = ctx.alloc_string(peer.to_string());
            ctx.alloc_list(vec![bytes, addr])
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Value::null(),
        Err(e) => {
            ctx.runtime_error(format!("Udp.tryRecvFrom: {}: {}", id, e));
            Value::null()
        }
    }
}

fn udp_local_addr(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Udp.localAddr") else {
        return Value::null();
    };
    let reg = udp_sockets().lock().unwrap();
    let Some(s) = reg.get(&id) else {
        ctx.runtime_error(format!("Udp.localAddr: socket {} not found.", id));
        return Value::null();
    };
    match s.local_addr() {
        Ok(a) => ctx.alloc_string(a.to_string()),
        Err(e) => {
            ctx.runtime_error(format!("Udp.localAddr: {}", e));
            Value::null()
        }
    }
}

fn udp_close(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(id) = resolve_id(ctx, args[1], "Udp.close") else {
        return Value::null();
    };
    udp_sockets().lock().unwrap().remove(&id);
    Value::null()
}

// --- Registration ----------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("SocketCore", vm.object_class);

    // TCP listener
    vm.primitive_static(class, "tcpListen(_)", tcp_listen);
    vm.primitive_static(class, "tcpAccept(_)", tcp_accept);
    vm.primitive_static(class, "tcpTryAccept(_)", tcp_try_accept);
    vm.primitive_static(class, "tcpListenerLocalAddr(_)", tcp_listener_local_addr);
    vm.primitive_static(class, "tcpCloseListener(_)", tcp_close_listener);

    // TCP stream
    vm.primitive_static(class, "tcpConnect(_,_)", tcp_connect);
    vm.primitive_static(class, "tcpRead(_,_)", tcp_read);
    vm.primitive_static(class, "tcpTryRead(_,_)", tcp_try_read);
    vm.primitive_static(class, "tcpWrite(_,_)", tcp_write);
    vm.primitive_static(class, "tcpSetReadTimeout(_,_)", tcp_set_timeout);
    vm.primitive_static(class, "tcpPeerAddr(_)", tcp_peer_addr);
    vm.primitive_static(class, "tcpLocalAddr(_)", tcp_stream_local_addr);
    vm.primitive_static(class, "tcpClose(_)", tcp_close_stream);

    // UDP
    vm.primitive_static(class, "udpBind(_)", udp_bind);
    vm.primitive_static(class, "udpSendTo(_,_,_)", udp_send_to);
    vm.primitive_static(class, "udpRecvFrom(_,_)", udp_recv_from);
    vm.primitive_static(class, "udpTryRecvFrom(_,_)", udp_try_recv_from);
    vm.primitive_static(class, "udpLocalAddr(_)", udp_local_addr);
    vm.primitive_static(class, "udpClose(_)", udp_close);

    class
}
