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
/// - v7 (2026-06-14): `class_field_names` field on `ModuleBlob` —
///   ordered (inherited + own) field-name list per class declared
///   in the module. Installed at module-install time into
///   `vm.field_layouts` so downstream source compiles (e.g. the
///   wasm playground compiling user code that subclasses `Game`
///   from a pre-built `@hatch:game` bundle) can resolve the parent
///   layout without re-running the parent's source compile.
/// - v6 (2026-04-29): `var_sources` field on `ModuleBlob` so the
///   install path can honour `import "<module>" for <name>` source
///   pins (was lost on the wlbc round-trip, breaking dispatcher
///   re-export modules where two siblings export the same class).
/// - v5 (2026-04-26): `Instruction::Call::pure_call` field added
///   to seed the effect-summary pass.
/// - v4: prior; first version this constant gained a written-down
///   bump policy.
pub const VERSION: u32 = 7;

/// Combined payload: everything a fresh `VM` needs to materialise the
/// module without touching the parser, resolver, MIR builder, or the
/// optimizer.
///
/// `var_names` is the declared module-var layout (one entry per slot,
/// in the order the resolver assigned them). `var_sources` parallels
/// it: `Some("<module>")` when the slot came from `import "<module>"
/// for <name>`, `None` for local declarations or prelude classes. On
/// load, the VM resolves each slot against the named source first
/// (so two modules exporting the same class name don't collide on
/// HashMap iteration order) and falls back to a name-only scan when
/// the source isn't recorded.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModuleBlob {
    pub interner: Interner,
    pub module: ModuleMir,
    pub var_names: Vec<String>,
    pub var_sources: Vec<Option<String>>,
    /// Ordered (inherited + own) field-name list per class declared
    /// in this module. Harvested into `vm.field_layouts` at install
    /// time so a later source compile (e.g. user code subclassing a
    /// class defined in this module) can resolve the parent layout
    /// without re-running the parent's source compile.
    pub class_field_names: std::collections::HashMap<String, Vec<String>>,
}

/// v6 layout — kept solely for the back-compat read path in
/// `load()`. v6 artifacts published to the hatch registry before
/// the v7 bump still install (with empty class_field_names — a
/// downstream source compile that subclasses a class defined in
/// a v6 module will still hit the "field layout is not yet
/// registered" panic until the module is republished).
#[derive(serde::Deserialize)]
struct ModuleBlobV6 {
    interner: Interner,
    module: ModuleMir,
    var_names: Vec<String>,
    var_sources: Vec<Option<String>>,
}

/// v5 layout — kept solely for the back-compat read path in
/// `load()`. v5 artifacts published to the hatch registry before
/// the v6 bump still install (with empty source pins), so we don't
/// have to republish every existing package.
#[derive(serde::Deserialize)]
struct ModuleBlobV5 {
    interner: Interner,
    module: ModuleMir,
    var_names: Vec<String>,
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
/// `var_names` is the module's declared top-level variable order;
/// `var_sources` parallels it with the import source for slots that
/// came from `import "<module>" for <name>`, or `None` for local
/// declarations.
pub fn emit(
    interner: &Interner,
    module: &ModuleMir,
    var_names: &[String],
    var_sources: &[Option<String>],
    class_field_names: &std::collections::HashMap<String, Vec<String>>,
) -> Result<Vec<u8>, SerializeError> {
    let blob = ModuleBlob {
        interner: interner.clone(),
        module: module.clone(),
        var_names: var_names.to_vec(),
        var_sources: var_sources.to_vec(),
        class_field_names: class_field_names.clone(),
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
    // v5 and v6 are the previous wlbc layouts kept for back-compat
    // so already-published registry artifacts still install. v5/v6
    // lack `class_field_names`, so downstream source compiles that
    // subclass a class defined in a legacy bundle will hit the
    // "field layout is not yet registered" panic — republish the
    // affected package against the current wren_lift sources.
    if version != VERSION && version != 5 && version != 6 {
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
    if version == 5 {
        let (v5, _consumed) = bincode::serde::decode_from_slice::<ModuleBlobV5, _>(
            payload,
            bincode::config::standard(),
        )
        .map_err(|e| SerializeError::Decode(e.to_string()))?;
        let var_sources = vec![None; v5.var_names.len()];
        return Ok(ModuleBlob {
            interner: v5.interner,
            module: v5.module,
            var_names: v5.var_names,
            var_sources,
            class_field_names: std::collections::HashMap::new(),
        });
    }
    if version == 6 {
        let (v6, _consumed) = bincode::serde::decode_from_slice::<ModuleBlobV6, _>(
            payload,
            bincode::config::standard(),
        )
        .map_err(|e| SerializeError::Decode(e.to_string()))?;
        return Ok(ModuleBlob {
            interner: v6.interner,
            module: v6.module,
            var_names: v6.var_names,
            var_sources: v6.var_sources,
            class_field_names: std::collections::HashMap::new(),
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

        let var_sources = vec![None; var_names.len()];
        let blob_bytes = emit(
            &interner,
            &module,
            &var_names,
            &var_sources,
            &std::collections::HashMap::new(),
        )
        .expect("emit");
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

    /// Stale artifacts carrying an *unsupported* lower version should
    /// fail at the version check rather than tumble down into bincode
    /// and surface as misleading "InvalidBooleanValue"-style decoder
    /// errors. Locks the rebuild-guidance message format too — every
    /// developer pulling a wlbc-format change benefits from the
    /// user-visible "rebuild with hatch build" hint instead of having
    /// to grep the error. (The current loader also accepts v5 for
    /// registry back-compat; we synthesise v4 here for the reject
    /// path.)
    #[test]
    fn load_rejects_older_version_with_rebuild_hint() {
        if VERSION <= 5 {
            return; // can't synthesise an unsupported lower version
        }
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let err = match load(&buf) {
            Err(e) => e,
            Ok(_) => panic!("stale version should reject"),
        };
        assert!(matches!(err, SerializeError::VersionMismatch { .. }));
        let msg = format!("{}", err);
        assert!(msg.contains(&format!("v{}", VERSION)));
        assert!(msg.contains("v4"));
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
        let blob = emit(
            &interner,
            &module,
            &[],
            &[],
            &std::collections::HashMap::new(),
        )
        .expect("emit");
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
