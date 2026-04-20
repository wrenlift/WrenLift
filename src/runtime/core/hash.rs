//! Optional `hash` module — cryptographic primitives Wren can't
//! implement in pure bytecode efficiently. Small surface by
//! design: hashers (MD5, SHA-1, SHA-256, SHA-512), HMAC for each,
//! and base64 encode/decode. @hatch:hash layers nicer names +
//! one-shot helpers on top.
//!
//! Everything takes `List<Num>` bytes in / out, where each
//! element is an integer in 0..=255. String-returning helpers
//! emit lower-case hex.

use base64::Engine;
use hmac::{Hmac, Mac};
use md5::Digest as _;

use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Hex + byte helpers ---------------------------------------

fn hex_of(bytes: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(*b as usize >> 4) & 0xf] as char);
        out.push(HEX[*b as usize & 0xf] as char);
    }
    out
}

/// Extract a byte slice from a Wren List<Num> argument. Each
/// element must be an integer in 0..=255; anything else is a
/// runtime error.
fn bytes_from_list(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<Vec<u8>> {
    let ptr = value.as_object();
    let ptr = match ptr {
        Some(p) => p as *const crate::runtime::object::ObjList,
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

/// Extract a byte slice from either a Wren String (UTF-8 bytes)
/// or a List<Num>. HMAC APIs often pass either depending on the
/// protocol.
fn bytes_from_value(ctx: &mut dyn NativeContext, value: Value, label: &str) -> Option<Vec<u8>> {
    if !value.is_object() {
        ctx.runtime_error(format!("{}: expected a string or list of bytes.", label));
        return None;
    }
    let ptr = value.as_object().unwrap();
    let header = ptr as *const crate::runtime::object::ObjHeader;
    unsafe {
        match (*header).obj_type {
            crate::runtime::object::ObjType::String => {
                let s = ptr as *const crate::runtime::object::ObjString;
                Some((*s).as_str().as_bytes().to_vec())
            }
            crate::runtime::object::ObjType::List => bytes_from_list(ctx, value, label),
            _ => {
                ctx.runtime_error(format!(
                    "{}: expected a string or list of bytes.",
                    label
                ));
                None
            }
        }
    }
}

fn bytes_to_list(ctx: &mut dyn NativeContext, bytes: &[u8]) -> Value {
    let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
    ctx.alloc_list(elements)
}

// --- Hashers --------------------------------------------------

macro_rules! hex_digest_fn {
    ($name:ident, $hasher:path, $label:literal) => {
        fn $name(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
            let Some(bytes) = bytes_from_value(ctx, args[1], $label) else {
                return Value::null();
            };
            let mut h = <$hasher>::new();
            h.update(&bytes);
            ctx.alloc_string(hex_of(&h.finalize()))
        }
    };
}

macro_rules! bytes_digest_fn {
    ($name:ident, $hasher:path, $label:literal) => {
        fn $name(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
            let Some(bytes) = bytes_from_value(ctx, args[1], $label) else {
                return Value::null();
            };
            let mut h = <$hasher>::new();
            h.update(&bytes);
            bytes_to_list(ctx, &h.finalize())
        }
    };
}

hex_digest_fn!(hash_md5_hex,    md5::Md5,       "Hash.md5Hex");
hex_digest_fn!(hash_sha1_hex,   sha1::Sha1,     "Hash.sha1Hex");
hex_digest_fn!(hash_sha256_hex, sha2::Sha256,   "Hash.sha256Hex");
hex_digest_fn!(hash_sha512_hex, sha2::Sha512,   "Hash.sha512Hex");

bytes_digest_fn!(hash_md5_bytes,    md5::Md5,     "Hash.md5Bytes");
bytes_digest_fn!(hash_sha1_bytes,   sha1::Sha1,   "Hash.sha1Bytes");
bytes_digest_fn!(hash_sha256_bytes, sha2::Sha256, "Hash.sha256Bytes");
bytes_digest_fn!(hash_sha512_bytes, sha2::Sha512, "Hash.sha512Bytes");

// --- HMAC -----------------------------------------------------

macro_rules! hmac_hex_fn {
    ($name:ident, $hasher:path, $label:literal) => {
        fn $name(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
            let Some(key) = bytes_from_value(ctx, args[1], concat!($label, " (key)")) else {
                return Value::null();
            };
            let Some(msg) = bytes_from_value(ctx, args[2], concat!($label, " (message)")) else {
                return Value::null();
            };
            let mut mac = <Hmac<$hasher>>::new_from_slice(&key)
                .expect("HMAC can take a key of any size");
            mac.update(&msg);
            ctx.alloc_string(hex_of(&mac.finalize().into_bytes()))
        }
    };
}

hmac_hex_fn!(hash_hmac_sha256_hex, sha2::Sha256, "Hash.hmacSha256Hex");
hmac_hex_fn!(hash_hmac_sha512_hex, sha2::Sha512, "Hash.hmacSha512Hex");
hmac_hex_fn!(hash_hmac_sha1_hex,   sha1::Sha1,   "Hash.hmacSha1Hex");

// --- Base64 ---------------------------------------------------

fn hash_base64_encode(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(bytes) = bytes_from_value(ctx, args[1], "Hash.base64Encode") else {
        return Value::null();
    };
    ctx.alloc_string(base64::engine::general_purpose::STANDARD.encode(&bytes))
}

fn hash_base64_decode(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(text) = super::validate_string(ctx, args[1], "Input") else {
        return Value::null();
    };
    match base64::engine::general_purpose::STANDARD.decode(text.as_bytes()) {
        Ok(bytes) => bytes_to_list(ctx, &bytes),
        Err(e) => {
            ctx.runtime_error(format!("Hash.base64Decode: {}", e));
            Value::null()
        }
    }
}

/// URL-safe base64 without padding, matching JWT's alphabet.
fn hash_base64_url_encode(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(bytes) = bytes_from_value(ctx, args[1], "Hash.base64UrlEncode") else {
        return Value::null();
    };
    ctx.alloc_string(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes))
}

fn hash_base64_url_decode(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(text) = super::validate_string(ctx, args[1], "Input") else {
        return Value::null();
    };
    match base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(text.as_bytes()) {
        Ok(bytes) => bytes_to_list(ctx, &bytes),
        Err(e) => {
            ctx.runtime_error(format!("Hash.base64UrlDecode: {}", e));
            Value::null()
        }
    }
}

// --- Registration ---------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    // Registered under `HashCore` so @hatch:hash can re-export a
    // `Hash` class with nicer names without the not-yet-wired
    // `import … as` rename.
    let class = vm.make_class("HashCore", vm.object_class);

    vm.primitive_static(class, "md5Hex(_)",       hash_md5_hex);
    vm.primitive_static(class, "sha1Hex(_)",      hash_sha1_hex);
    vm.primitive_static(class, "sha256Hex(_)",    hash_sha256_hex);
    vm.primitive_static(class, "sha512Hex(_)",    hash_sha512_hex);

    vm.primitive_static(class, "md5Bytes(_)",     hash_md5_bytes);
    vm.primitive_static(class, "sha1Bytes(_)",    hash_sha1_bytes);
    vm.primitive_static(class, "sha256Bytes(_)",  hash_sha256_bytes);
    vm.primitive_static(class, "sha512Bytes(_)",  hash_sha512_bytes);

    vm.primitive_static(class, "hmacSha1Hex(_,_)",   hash_hmac_sha1_hex);
    vm.primitive_static(class, "hmacSha256Hex(_,_)", hash_hmac_sha256_hex);
    vm.primitive_static(class, "hmacSha512Hex(_,_)", hash_hmac_sha512_hex);

    vm.primitive_static(class, "base64Encode(_)",      hash_base64_encode);
    vm.primitive_static(class, "base64Decode(_)",      hash_base64_decode);
    vm.primitive_static(class, "base64UrlEncode(_)",   hash_base64_url_encode);
    vm.primitive_static(class, "base64UrlDecode(_)",   hash_base64_url_decode);

    class
}
