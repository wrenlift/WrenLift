//! Portable bytecode-cache serializer (`*.wlbc`).
//!
//! Skips the parse → sema → MIR-build → optimize pipeline on subsequent
//! launches. The serializer captures `ModuleMir` + the interner that
//! produced its `SymbolId`s, plus a small magic / version envelope so we
//! can reject old blobs when the format evolves.
//!
//! The serialized form is MIR-level (not bytecode-level) — bytecode gets
//! re-lowered at load time, which is cheap (~100µs per function) and
//! keeps the snapshot JIT-compatible: Cranelift compiles from MIR, so a
//! bytecode-only cache would be stuck in the interpreter. Class method
//! tables, module var layouts, and closure bodies all ride inside
//! `ModuleMir`, so the format is complete for a single module.
//!
//! # Wire format (all little-endian)
//!
//! ```text
//! magic       b"WLBC"         4 bytes
//! version     u32             (bump on incompatible change)
//! payload_len u32             (bincode-encoded bytes that follow)
//! payload     bincode(ModuleBlob)
//! ```
//!
//! `ModuleBlob` bundles the interner + `ModuleMir` into a single
//! bincode record so the on-disk layout is one `length | bytes` frame
//! regardless of how many top-level / closure / method functions the
//! module contains.

use crate::intern::Interner;
use crate::mir::ModuleMir;

/// Magic header at the start of every serialized module. "WLBC" =
/// wren_lift bytecode cache.
pub const MAGIC: [u8; 4] = *b"WLBC";

/// Current format revision. Bump when `ModuleBlob`'s shape changes in a
/// way a previous-version loader can't handle (new Instruction variants
/// are additive from bincode's perspective as long as we never reorder
/// existing variants, so most upgrades won't need a version bump — but
/// reshuffling a struct field or swapping enum variant order will).
pub const VERSION: u32 = 1;

/// Combined payload: everything a fresh `VM` needs to materialise the
/// module without touching the parser, resolver, MIR builder, or the
/// optimizer.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModuleBlob {
    pub interner: Interner,
    pub module: ModuleMir,
}

/// Errors surfaced by `emit` / `load`.
#[derive(Debug)]
pub enum SerializeError {
    /// `load` received bytes that don't start with the `WLBC` magic.
    BadMagic,
    /// The blob's version number doesn't match what this binary speaks.
    /// The loader could later keep a compat table, but today we reject.
    VersionMismatch { expected: u32, found: u32 },
    /// Payload length header claims more (or fewer) bytes than the
    /// buffer actually contains.
    TruncatedPayload { declared: u32, available: usize },
    /// bincode encode failed — almost always indicates a programmer
    /// error (a type deep inside `ModuleMir` lost its `Serialize`
    /// derive).
    Encode(String),
    /// bincode decode failed — typically a genuinely corrupt blob or a
    /// cross-version format drift we didn't version-bump for.
    Decode(String),
}

impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializeError::BadMagic => write!(f, "not a wlift bytecode cache (missing WLBC magic)"),
            SerializeError::VersionMismatch { expected, found } => write!(
                f,
                "wlift bytecode cache version mismatch: expected {expected}, found {found}"
            ),
            SerializeError::TruncatedPayload { declared, available } => write!(
                f,
                "wlift bytecode cache payload is truncated: header says {declared} bytes, only {available} available"
            ),
            SerializeError::Encode(e) => write!(f, "encode: {e}"),
            SerializeError::Decode(e) => write!(f, "decode: {e}"),
        }
    }
}

impl std::error::Error for SerializeError {}

/// Serialize a compiled module + its interner into a self-describing
/// blob suitable for `.wlbc` files.
pub fn emit(interner: &Interner, module: &ModuleMir) -> Result<Vec<u8>, SerializeError> {
    let blob = ModuleBlob {
        interner: interner.clone(),
        module: module.clone(),
    };

    let payload = bincode::serde::encode_to_vec(&blob, bincode::config::standard())
        .map_err(|e| SerializeError::Encode(e.to_string()))?;

    let mut out = Vec::with_capacity(4 + 4 + 4 + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Parse a blob produced by `emit`. Validates magic + version + declared
/// payload length before handing bytes to bincode.
pub fn load(bytes: &[u8]) -> Result<(Interner, ModuleMir), SerializeError> {
    if bytes.len() < 12 || bytes[..4] != MAGIC {
        return Err(SerializeError::BadMagic);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(SerializeError::VersionMismatch {
            expected: VERSION,
            found: version,
        });
    }
    let declared = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let payload = &bytes[12..];
    if payload.len() != declared as usize {
        return Err(SerializeError::TruncatedPayload {
            declared,
            available: payload.len(),
        });
    }
    let (blob, _consumed) =
        bincode::serde::decode_from_slice::<ModuleBlob, _>(payload, bincode::config::standard())
            .map_err(|e| SerializeError::Decode(e.to_string()))?;
    Ok((blob.interner, blob.module))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MirFunction, ModuleMir, Terminator};

    fn empty_module(interner: &mut Interner) -> ModuleMir {
        let top_sym = interner.intern("<module>");
        let mut top_level = MirFunction::new(top_sym, 0);
        let bb = top_level.new_block();
        top_level.block_mut(bb).terminator = Terminator::ReturnNull;
        ModuleMir {
            top_level,
            classes: Vec::new(),
            closures: Vec::new(),
        }
    }

    #[test]
    fn emit_load_round_trips_empty_module() {
        let mut interner = Interner::new();
        let module = empty_module(&mut interner);

        let blob = emit(&interner, &module).expect("emit");
        let (interner_back, module_back) = load(&blob).expect("load");

        assert_eq!(
            interner.resolve(module.top_level.name),
            interner_back.resolve(module_back.top_level.name)
        );
        assert_eq!(module.top_level.arity, module_back.top_level.arity);
        assert_eq!(
            module.top_level.blocks.len(),
            module_back.top_level.blocks.len()
        );
    }

    #[test]
    fn load_rejects_bad_magic() {
        let junk = vec![0u8; 32];
        assert!(matches!(load(&junk), Err(SerializeError::BadMagic)));
    }

    #[test]
    fn load_rejects_version_skew() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&(VERSION + 1).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            load(&buf),
            Err(SerializeError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn load_rejects_truncated_payload() {
        let mut interner = Interner::new();
        let module = empty_module(&mut interner);
        let blob = emit(&interner, &module).expect("emit");
        let truncated = &blob[..blob.len() - 1];
        assert!(matches!(
            load(truncated),
            Err(SerializeError::TruncatedPayload { .. })
        ));
    }
}
