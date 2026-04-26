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

/// Current format revision. **Bump this whenever any
/// wlbc-serialized type changes shape**: a new field on a struct
/// variant of an enum, a new variant in the middle of an enum,
/// reordered fields — anything that changes the bincode layout.
/// A missed bump produces silent decoder confusion ("decode:
/// InvalidBooleanValue(5)" or similar) on every stale `.hatch`
/// artifact a developer hasn't rebuilt yet, instead of a clean
/// `VersionMismatch` error pointing at the exact remediation.
///
/// Adding a new (last) variant to an enum is the only safe edit
/// without bumping — bincode encodes variants by index, and a
/// new tail variant is encode-only until the first program
/// produces one in serialized output.
///
/// History:
/// - v5 (2026-04-26): `Instruction::Call::pure_call` field added
///   for the Phase 6 effect-summary seed.
/// - v4: prior; first version this constant gained a written-down
///   bump policy.
pub const VERSION: u32 = 5;

/// Combined payload: everything a fresh `VM` needs to materialise the
/// module without touching the parser, resolver, MIR builder, or the
/// optimizer.
///
/// `var_names` is the declared module-var layout (one entry per slot,
/// in the order the resolver assigned them). On load, the VM looks up
/// each name against its own core classes / imported modules and fills
/// the corresponding slot, falling back to `null`. This replaces the
/// `resolve_result.module_vars` list that the source path produces.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModuleBlob {
    pub interner: Interner,
    pub module: ModuleMir,
    pub var_names: Vec<String>,
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
                "wlift bytecode cache version mismatch: expected v{expected}, found v{found}; \
                 rebuild the artifact with `hatch build` against the current wren_lift sources"
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
///
/// `var_names` is the module's declared top-level variable order. The
/// loader uses it to reconstruct the module's var slot layout.
pub fn emit(
    interner: &Interner,
    module: &ModuleMir,
    var_names: &[String],
) -> Result<Vec<u8>, SerializeError> {
    let blob = ModuleBlob {
        interner: interner.clone(),
        module: module.clone(),
        var_names: var_names.to_vec(),
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
pub fn load(bytes: &[u8]) -> Result<ModuleBlob, SerializeError> {
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
    Ok(blob)
}

/// Cheap magic-bytes probe so the CLI can pick the .wlbc path without
/// committing to a full `load()` up front (and its bincode dependency).
pub fn looks_like_wlbc(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[..4] == MAGIC
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
        let var_names = vec!["System".to_string(), "greeting".to_string()];

        let blob_bytes = emit(&interner, &module, &var_names).expect("emit");
        let blob = load(&blob_bytes).expect("load");

        assert_eq!(
            interner.resolve(module.top_level.name),
            blob.interner.resolve(blob.module.top_level.name)
        );
        assert_eq!(module.top_level.arity, blob.module.top_level.arity);
        assert_eq!(
            module.top_level.blocks.len(),
            blob.module.top_level.blocks.len()
        );
        assert_eq!(blob.var_names, var_names);
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

    /// Stale artifacts carrying a *lower* version number should
    /// also fail at the version check rather than tumble down into
    /// bincode and surface as misleading "InvalidBooleanValue"-style
    /// decoder errors. Locks the rebuild-guidance message format too
    /// — every developer pulling a wlbc-format change benefits from
    /// the user-visible "rebuild with hatch build" hint instead of
    /// having to grep the error.
    #[test]
    fn load_rejects_older_version_with_rebuild_hint() {
        if VERSION == 0 {
            return; // can't synthesise a lower version
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&(VERSION - 1).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let err = match load(&buf) {
            Err(e) => e,
            Ok(_) => panic!("stale version should reject"),
        };
        assert!(matches!(err, SerializeError::VersionMismatch { .. }));
        let msg = format!("{}", err);
        assert!(msg.contains(&format!("v{}", VERSION)));
        assert!(msg.contains(&format!("v{}", VERSION - 1)));
        assert!(
            msg.contains("hatch build"),
            "expected rebuild guidance, got: {}",
            msg
        );
    }

    #[test]
    fn load_rejects_truncated_payload() {
        let mut interner = Interner::new();
        let module = empty_module(&mut interner);
        let blob = emit(&interner, &module, &[]).expect("emit");
        let truncated = &blob[..blob.len() - 1];
        assert!(matches!(
            load(truncated),
            Err(SerializeError::TruncatedPayload { .. })
        ));
    }

    #[test]
    fn looks_like_wlbc_matches_magic() {
        assert!(!looks_like_wlbc(&[]));
        assert!(!looks_like_wlbc(b"WLB"));
        assert!(looks_like_wlbc(b"WLBC"));
        assert!(looks_like_wlbc(b"WLBCextra"));
        assert!(!looks_like_wlbc(b"ABCD"));
    }
}
