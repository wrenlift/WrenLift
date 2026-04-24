//! Optional `crypto` module — authenticated symmetric encryption
//! (AES-256-GCM), public-key signatures (Ed25519), and a CSPRNG.
//! Sits alongside `hash` — that one's digests + HMAC + base64,
//! this one's the encryption / signing / random side.
//!
//! All byte inputs and outputs are `List<Num>` (each entry an
//! integer in 0..=255), matching the convention @hatch:hash
//! already uses. Wren callers never touch Rust byte slices
//! directly.
//!
//! @hatch:crypto layers ergonomic class names on top.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand_core::{OsRng, RngCore};

use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Byte-list helpers ------------------------------------------
//
// Shared by every function below — every input is either a String
// or a List<Num in 0..=255>. Outputs are always List<Num>.

use super::bytes_from_byte_list as bytes_from_list;
use super::bytes_from_value;

fn bytes_to_list(ctx: &mut dyn NativeContext, bytes: &[u8]) -> Value {
    let elements: Vec<Value> = bytes.iter().map(|&b| Value::num(b as f64)).collect();
    ctx.alloc_list(elements)
}

// --- Random -----------------------------------------------------

fn crypto_random_bytes(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let n = match args[1].as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => n as usize,
        _ => {
            ctx.runtime_error(
                "Crypto.randomBytes: count must be a non-negative integer.".to_string(),
            );
            return Value::null();
        }
    };
    let mut out = vec![0u8; n];
    OsRng.fill_bytes(&mut out);
    bytes_to_list(ctx, &out)
}

// --- AES-256-GCM ------------------------------------------------
//
// 256-bit key (32 bytes) + 96-bit nonce (12 bytes) → ciphertext
// that includes a 16-byte authentication tag at the end.
// Optional Additional Authenticated Data (AAD) is covered by
// the tag but not encrypted.

fn build_cipher(ctx: &mut dyn NativeContext, key_bytes: &[u8], label: &str) -> Option<Aes256Gcm> {
    if key_bytes.len() != 32 {
        ctx.runtime_error(format!(
            "{}: key must be 32 bytes (AES-256), got {}.",
            label,
            key_bytes.len()
        ));
        return None;
    }
    let key = Key::<Aes256Gcm>::from_slice(key_bytes);
    Some(Aes256Gcm::new(key))
}

fn check_nonce(ctx: &mut dyn NativeContext, nonce_bytes: &[u8], label: &str) -> bool {
    if nonce_bytes.len() != 12 {
        ctx.runtime_error(format!(
            "{}: nonce must be 12 bytes, got {}.",
            label,
            nonce_bytes.len()
        ));
        return false;
    }
    true
}

fn crypto_aes_gcm_key(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let key = Aes256Gcm::generate_key(OsRng);
    bytes_to_list(ctx, key.as_slice())
}

fn crypto_aes_gcm_nonce(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let nonce = Aes256Gcm::generate_nonce(OsRng);
    bytes_to_list(ctx, nonce.as_slice())
}

fn crypto_aes_gcm_encrypt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(key_bytes) = bytes_from_list(ctx, args[1], "Aes.encrypt (key)") else {
        return Value::null();
    };
    let Some(nonce_bytes) = bytes_from_list(ctx, args[2], "Aes.encrypt (nonce)") else {
        return Value::null();
    };
    let Some(plaintext) = bytes_from_value(ctx, args[3], "Aes.encrypt (plaintext)") else {
        return Value::null();
    };
    // AAD: null → none, anything else must be a byte list / string.
    let aad = if args[4].is_null() {
        Vec::new()
    } else {
        let Some(a) = bytes_from_value(ctx, args[4], "Aes.encrypt (aad)") else {
            return Value::null();
        };
        a
    };
    let Some(cipher) = build_cipher(ctx, &key_bytes, "Aes.encrypt") else {
        return Value::null();
    };
    if !check_nonce(ctx, &nonce_bytes, "Aes.encrypt") {
        return Value::null();
    }
    let nonce = Nonce::from_slice(&nonce_bytes);
    let result = cipher.encrypt(
        nonce,
        Payload {
            msg: &plaintext,
            aad: &aad,
        },
    );
    match result {
        Ok(ciphertext) => bytes_to_list(ctx, &ciphertext),
        Err(e) => {
            ctx.runtime_error(format!("Aes.encrypt: {}", e));
            Value::null()
        }
    }
}

fn crypto_aes_gcm_decrypt(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(key_bytes) = bytes_from_list(ctx, args[1], "Aes.decrypt (key)") else {
        return Value::null();
    };
    let Some(nonce_bytes) = bytes_from_list(ctx, args[2], "Aes.decrypt (nonce)") else {
        return Value::null();
    };
    let Some(ciphertext) = bytes_from_list(ctx, args[3], "Aes.decrypt (ciphertext)") else {
        return Value::null();
    };
    let aad = if args[4].is_null() {
        Vec::new()
    } else {
        let Some(a) = bytes_from_value(ctx, args[4], "Aes.decrypt (aad)") else {
            return Value::null();
        };
        a
    };
    let Some(cipher) = build_cipher(ctx, &key_bytes, "Aes.decrypt") else {
        return Value::null();
    };
    if !check_nonce(ctx, &nonce_bytes, "Aes.decrypt") {
        return Value::null();
    }
    let nonce = Nonce::from_slice(&nonce_bytes);
    match cipher.decrypt(
        nonce,
        Payload {
            msg: &ciphertext,
            aad: &aad,
        },
    ) {
        Ok(plaintext) => bytes_to_list(ctx, &plaintext),
        Err(_) => {
            // Auth failure returns null so callers can `?.ok` check.
            // All decrypt failures (wrong key, nonce, tag, AAD) look
            // the same by design — a real attacker learns nothing.
            Value::null()
        }
    }
}

// --- Ed25519 ----------------------------------------------------

fn crypto_ed25519_keypair(ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Generate from OS RNG. Returns a two-element list:
    //   [secret (32 bytes), public (32 bytes)]
    let signing_key = SigningKey::generate(&mut OsRng);
    let secret_bytes = signing_key.to_bytes();
    let public_bytes = signing_key.verifying_key().to_bytes();
    let secret = bytes_to_list(ctx, &secret_bytes);
    let public = bytes_to_list(ctx, &public_bytes);
    ctx.alloc_list(vec![secret, public])
}

fn crypto_ed25519_public_from_secret(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(secret_bytes) = bytes_from_list(ctx, args[1], "Ed25519.publicFromSecret (secret)")
    else {
        return Value::null();
    };
    if secret_bytes.len() != 32 {
        ctx.runtime_error(format!(
            "Ed25519.publicFromSecret: secret must be 32 bytes, got {}.",
            secret_bytes.len()
        ));
        return Value::null();
    }
    let arr: [u8; 32] = secret_bytes.try_into().unwrap();
    let signing_key = SigningKey::from_bytes(&arr);
    bytes_to_list(ctx, signing_key.verifying_key().as_bytes())
}

fn crypto_ed25519_sign(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(secret_bytes) = bytes_from_list(ctx, args[1], "Ed25519.sign (secret)") else {
        return Value::null();
    };
    let Some(message) = bytes_from_value(ctx, args[2], "Ed25519.sign (message)") else {
        return Value::null();
    };
    if secret_bytes.len() != 32 {
        ctx.runtime_error(format!(
            "Ed25519.sign: secret must be 32 bytes, got {}.",
            secret_bytes.len()
        ));
        return Value::null();
    }
    let arr: [u8; 32] = secret_bytes.try_into().unwrap();
    let signing_key = SigningKey::from_bytes(&arr);
    let signature = signing_key.sign(&message);
    bytes_to_list(ctx, &signature.to_bytes())
}

fn crypto_ed25519_verify(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let Some(public_bytes) = bytes_from_list(ctx, args[1], "Ed25519.verify (public)") else {
        return Value::null();
    };
    let Some(message) = bytes_from_value(ctx, args[2], "Ed25519.verify (message)") else {
        return Value::null();
    };
    let Some(sig_bytes) = bytes_from_list(ctx, args[3], "Ed25519.verify (signature)") else {
        return Value::null();
    };
    if public_bytes.len() != 32 {
        ctx.runtime_error(format!(
            "Ed25519.verify: public must be 32 bytes, got {}.",
            public_bytes.len()
        ));
        return Value::null();
    }
    if sig_bytes.len() != 64 {
        ctx.runtime_error(format!(
            "Ed25519.verify: signature must be 64 bytes, got {}.",
            sig_bytes.len()
        ));
        return Value::null();
    }
    let pub_arr: [u8; 32] = public_bytes.try_into().unwrap();
    let sig_arr: [u8; 64] = sig_bytes.try_into().unwrap();
    let verifying_key = match VerifyingKey::from_bytes(&pub_arr) {
        Ok(k) => k,
        Err(_) => return Value::bool(false),
    };
    let signature = Signature::from_bytes(&sig_arr);
    Value::bool(verifying_key.verify(&message, &signature).is_ok())
}

// --- Registration -----------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    let class = vm.make_class("CryptoCore", vm.object_class);

    vm.primitive_static(class, "randomBytes(_)", crypto_random_bytes);

    vm.primitive_static(class, "aesGcmKey()", crypto_aes_gcm_key);
    vm.primitive_static(class, "aesGcmNonce()", crypto_aes_gcm_nonce);
    vm.primitive_static(class, "aesGcmEncrypt(_,_,_,_)", crypto_aes_gcm_encrypt);
    vm.primitive_static(class, "aesGcmDecrypt(_,_,_,_)", crypto_aes_gcm_decrypt);

    vm.primitive_static(class, "ed25519Keypair()", crypto_ed25519_keypair);
    vm.primitive_static(
        class,
        "ed25519PublicFromSecret(_)",
        crypto_ed25519_public_from_secret,
    );
    vm.primitive_static(class, "ed25519Sign(_,_)", crypto_ed25519_sign);
    vm.primitive_static(class, "ed25519Verify(_,_,_)", crypto_ed25519_verify);

    class
}
