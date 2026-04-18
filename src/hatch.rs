//! `.hatch` package format — library / distribution container.
//!
//! Where `.wlbc` is a single-module executable cache, `.hatch` bundles
//! one or more compiled modules plus a `hatch.toml` manifest plus
//! arbitrary resource blobs into a single zstd-compressed file. Loaders
//! install the contained modules in dependency order and then run the
//! entry-point module.
//!
//! # Wire format
//!
//! ```text
//! header (16 bytes, uncompressed)
//!   magic       b"HATCH"              5 bytes
//!   version     u32 LE                4 bytes   — bump on break
//!   flags       u8                    1 byte    — bit 0 = zstd-compressed
//!   reserved    3 bytes               always 0, future-use
//!   payload_len u64 LE                — bytes in the payload that
//!                                       follows the header, on disk.
//!                                       zstd-compressed unless flag 0
//!                                       is clear.
//!
//! payload (optionally zstd-wrapped)
//!   section_count u32 LE
//!   for each section:
//!     kind        u8                  — 0 = manifest, 1 = wlbc,
//!                                        2 = resource, 3 = native lib
//!                                        (3 deferred to commit 3b)
//!     name_len    u16 LE
//!     name        utf-8 bytes         e.g. "hatch.toml", "util.wlbc"
//!     data_len    u32 LE
//!     data        bytes
//! ```
//!
//! The header is never compressed so `file(1)`-style magic probes and
//! `wlift --inspect` can read metadata without a zstd roundtrip. Future
//! variants that ship uncompressed (e.g. pre-compressed CDN delivery)
//! can clear flag-0 without changing the framing.

use std::collections::BTreeMap;
use std::path::Path;

/// Magic bytes at the start of every `.hatch` file.
pub const MAGIC: [u8; 5] = *b"HATCH";

/// Current wire-format revision. Bump on any incompatible shape change.
pub const VERSION: u32 = 1;

/// Flag bit 0: payload is zstd-compressed. New flag bits can be added
/// additively in future versions without a version bump as long as
/// older loaders see them as no-ops.
pub const FLAG_ZSTD: u8 = 1 << 0;

/// zstd compression level. Level 3 is the default for most tooling
/// (git, facebook zstd). Level 19+ trades a lot of build time for
/// marginal size wins; stick with the default.
const ZSTD_LEVEL: i32 = 3;

// ---------------------------------------------------------------------------
// Section kinds
// ---------------------------------------------------------------------------

/// Kind tag for each section inside a hatch payload.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// The `hatch.toml` manifest, utf-8 TOML text. Exactly one required.
    Manifest = 0,
    /// One compiled `.wlbc` blob for a single module. The section name
    /// is the module name (e.g. `"util"`), not the filename.
    Wlbc = 1,
    /// Pass-through resource bytes. Not installed into the VM; loaders
    /// can surface them to application code via a future
    /// `wlift.resource(...)` API.
    Resource = 2,
    /// Native dynamic library (per-target variant). Added in commit 3b;
    /// rejected by the loader today so forward-compat hatches don't
    /// silently run on a build that can't honor them.
    NativeLib = 3,
}

impl SectionKind {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(SectionKind::Manifest),
            1 => Some(SectionKind::Wlbc),
            2 => Some(SectionKind::Resource),
            3 => Some(SectionKind::NativeLib),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// In-memory representation
// ---------------------------------------------------------------------------

/// A parsed `hatch.toml` manifest. Fields the builder writes and the
/// loader reads are listed here with `#[serde(default)]` where they're
/// optional so older / newer hatch versions round-trip through unknown
/// producers without breaking.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    /// Package name. Required.
    pub name: String,
    /// Package version, SemVer-ish. Required.
    pub version: String,
    /// Module name to run as the entry point (e.g. `"main"` for a
    /// package whose `main.wren` is the program's top-level).
    pub entry: String,
    /// Ordered list of module names in this hatch. The loader installs
    /// them in this order so a module's imports resolve against
    /// already-loaded peers. Producers are expected to write this in
    /// dependency order; the loader does not topologically sort.
    #[serde(default)]
    pub modules: Vec<String>,
    /// `name → version` dependency list for future resolver work.
    /// Today the loader doesn't enforce anything here; kept so hatches
    /// can declare their deps for tooling.
    #[serde(default)]
    pub dependencies: BTreeMap<String, String>,
    /// Native library declarations. Keys match the string a Wren
    /// `#!native = "..."` attribute resolves; values describe how the
    /// loader should find the underlying `.dylib` / `.so` / `.dll` —
    /// either an explicit path (relative to the workspace, or
    /// absolute for system locations like `/usr/lib/libssl.dylib`)
    /// or a bare name that's looked up through the ambient OS search
    /// plus `native_search_paths`.
    #[serde(default)]
    pub native_libs: BTreeMap<String, NativeLibEntry>,
    /// Extra filesystem directories to search when resolving a
    /// `#!native` reference, tried ahead of the OS loader's ambient
    /// search. Absolute paths (`/opt/homebrew/lib`) and workspace-
    /// relative paths both work.
    #[serde(default)]
    pub native_search_paths: Vec<String>,
}

/// Shape of a `[native_libs.<name>]` entry in a hatchfile. Accepts
/// either a bare string (treated as `path = "..."`) or a table with
/// richer fields so future keys (e.g. `version`, `symbols`) can land
/// without breaking existing manifests.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum NativeLibEntry {
    /// `libssl = "/usr/lib/libssl.dylib"` — path-only shorthand.
    Path(String),
    /// `[native_libs.libssl]` table form with `path = "..."`.
    Detailed {
        /// Filesystem path (absolute for system locations, relative to
        /// the workspace otherwise). If omitted, the loader falls back
        /// to platform-specific bare-name resolution using the key.
        #[serde(default)]
        path: Option<String>,
    },
}

impl NativeLibEntry {
    /// Extract the explicit path, if any. `None` means "use the key as
    /// a bare name and let the OS loader find it".
    pub fn path(&self) -> Option<&str> {
        match self {
            NativeLibEntry::Path(p) => Some(p.as_str()),
            NativeLibEntry::Detailed { path } => path.as_deref(),
        }
    }
}

/// One named blob inside a hatch. Owns its payload; for very large
/// hatches a streaming variant could avoid materializing everything at
/// once, but packages are typically < a few MB.
#[derive(Debug, Clone)]
pub struct Section {
    pub kind: SectionKind,
    pub name: String,
    pub data: Vec<u8>,
}

/// The top-level in-memory hatch. `load()` produces this from bytes;
/// `emit()` consumes it back into bytes. Manifest is stored separately
/// from sections because every hatch has exactly one, and callers
/// usually want structured access.
#[derive(Debug, Clone)]
pub struct Hatch {
    pub manifest: Manifest,
    pub sections: Vec<Section>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum HatchError {
    BadMagic,
    VersionMismatch { expected: u32, found: u32 },
    /// Header advertises a payload length the buffer can't cover.
    TruncatedPayload { declared: u64, available: usize },
    /// zstd decompression failed (corrupt blob or wrong flag bit).
    Decompress(String),
    /// A section header's utf-8 name didn't parse or data_len went past
    /// the payload end.
    MalformedSection(String),
    /// `hatch.toml` missing or parse failed.
    ManifestMissing,
    ManifestParse(String),
    /// Loader refused to run the hatch (e.g. it carried a `NativeLib`
    /// section and this build didn't enable native-lib support yet).
    Unsupported(String),
    Io(std::io::Error),
    Encode(String),
}

impl std::fmt::Display for HatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HatchError::BadMagic => write!(f, "not a hatch package (missing HATCH magic)"),
            HatchError::VersionMismatch { expected, found } => write!(
                f,
                "hatch version mismatch: expected {expected}, found {found}"
            ),
            HatchError::TruncatedPayload { declared, available } => write!(
                f,
                "hatch payload truncated: header says {declared} bytes, only {available} available"
            ),
            HatchError::Decompress(e) => write!(f, "zstd decompression failed: {e}"),
            HatchError::MalformedSection(e) => write!(f, "malformed section: {e}"),
            HatchError::ManifestMissing => write!(f, "hatch has no manifest section"),
            HatchError::ManifestParse(e) => write!(f, "hatch.toml parse failed: {e}"),
            HatchError::Unsupported(s) => write!(f, "unsupported: {s}"),
            HatchError::Io(e) => write!(f, "io: {e}"),
            HatchError::Encode(e) => write!(f, "encode: {e}"),
        }
    }
}

impl std::error::Error for HatchError {}

impl From<std::io::Error> for HatchError {
    fn from(e: std::io::Error) -> Self {
        HatchError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Serialize a `Hatch` to bytes. Emits the manifest as the first
/// section (kind = Manifest) with name `"hatch.toml"`, followed by the
/// caller-supplied sections in the order given. The payload is always
/// zstd-compressed in this build; loaders that see the flag clear on a
/// future hatch get an uncompressed payload instead.
pub fn emit(hatch: &Hatch) -> Result<Vec<u8>, HatchError> {
    // --- Build payload in-memory first so we know its post-
    //     compression length for the header. ---
    let manifest_toml = toml::to_string_pretty(&hatch.manifest)
        .map_err(|e| HatchError::Encode(format!("serialize hatch.toml: {e}")))?;
    let manifest_section = Section {
        kind: SectionKind::Manifest,
        name: "hatch.toml".to_string(),
        data: manifest_toml.into_bytes(),
    };

    let mut payload = Vec::new();
    let total_sections = 1 + hatch.sections.len() as u32;
    payload.extend_from_slice(&total_sections.to_le_bytes());

    emit_section(&manifest_section, &mut payload)?;
    for section in &hatch.sections {
        emit_section(section, &mut payload)?;
    }

    // --- Wrap payload + header. ---
    let compressed = zstd::encode_all(std::io::Cursor::new(&payload), ZSTD_LEVEL)
        .map_err(|e| HatchError::Encode(format!("zstd encode: {e}")))?;

    let mut out = Vec::with_capacity(16 + compressed.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.push(FLAG_ZSTD); // flags
    out.extend_from_slice(&[0u8; 3]); // reserved
    out.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

fn emit_section(section: &Section, out: &mut Vec<u8>) -> Result<(), HatchError> {
    if section.name.len() > u16::MAX as usize {
        return Err(HatchError::Encode(format!(
            "section name too long: {}",
            section.name
        )));
    }
    if section.data.len() > u32::MAX as usize {
        return Err(HatchError::Encode(format!(
            "section data too large for u32 length: {} bytes in {}",
            section.data.len(),
            section.name
        )));
    }
    out.push(section.kind as u8);
    out.extend_from_slice(&(section.name.len() as u16).to_le_bytes());
    out.extend_from_slice(section.name.as_bytes());
    out.extend_from_slice(&(section.data.len() as u32).to_le_bytes());
    out.extend_from_slice(&section.data);
    Ok(())
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Parse a `.hatch` byte stream.
pub fn load(bytes: &[u8]) -> Result<Hatch, HatchError> {
    if !looks_like_hatch(bytes) {
        return Err(HatchError::BadMagic);
    }
    if bytes.len() < 16 {
        return Err(HatchError::BadMagic);
    }
    let version = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
    if version != VERSION {
        return Err(HatchError::VersionMismatch {
            expected: VERSION,
            found: version,
        });
    }
    let flags = bytes[9];
    // bytes[10..13] reserved
    let payload_len = u64::from_le_bytes(bytes[13..21].try_into().unwrap());
    let header_len = 21usize;
    if bytes.len() < header_len || (bytes.len() - header_len) < payload_len as usize {
        return Err(HatchError::TruncatedPayload {
            declared: payload_len,
            available: bytes.len().saturating_sub(header_len),
        });
    }
    let raw_payload = &bytes[header_len..header_len + payload_len as usize];

    let payload = if flags & FLAG_ZSTD != 0 {
        zstd::decode_all(std::io::Cursor::new(raw_payload))
            .map_err(|e| HatchError::Decompress(e.to_string()))?
    } else {
        raw_payload.to_vec()
    };

    let sections = parse_sections(&payload)?;

    let mut manifest: Option<Manifest> = None;
    let mut rest = Vec::with_capacity(sections.len().saturating_sub(1));
    for section in sections {
        match section.kind {
            SectionKind::Manifest => {
                let text = std::str::from_utf8(&section.data)
                    .map_err(|e| HatchError::ManifestParse(e.to_string()))?;
                manifest = Some(
                    toml::from_str(text).map_err(|e| HatchError::ManifestParse(e.to_string()))?,
                );
            }
            _ => rest.push(section),
        }
    }
    let manifest = manifest.ok_or(HatchError::ManifestMissing)?;
    Ok(Hatch {
        manifest,
        sections: rest,
    })
}

fn parse_sections(payload: &[u8]) -> Result<Vec<Section>, HatchError> {
    if payload.len() < 4 {
        return Err(HatchError::MalformedSection(
            "payload too short for section count".into(),
        ));
    }
    let count = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
    let mut cursor = 4usize;
    let mut sections = Vec::with_capacity(count);
    for _ in 0..count {
        if payload.len() < cursor + 1 + 2 {
            return Err(HatchError::MalformedSection(
                "truncated section header".into(),
            ));
        }
        let kind_byte = payload[cursor];
        cursor += 1;
        let kind = SectionKind::from_u8(kind_byte).ok_or_else(|| {
            HatchError::MalformedSection(format!("unknown section kind {kind_byte}"))
        })?;

        let name_len = u16::from_le_bytes(payload[cursor..cursor + 2].try_into().unwrap()) as usize;
        cursor += 2;
        if payload.len() < cursor + name_len + 4 {
            return Err(HatchError::MalformedSection(
                "truncated section name or data header".into(),
            ));
        }
        let name = std::str::from_utf8(&payload[cursor..cursor + name_len])
            .map_err(|e| HatchError::MalformedSection(format!("non-utf-8 section name: {e}")))?
            .to_string();
        cursor += name_len;

        let data_len =
            u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if payload.len() < cursor + data_len {
            return Err(HatchError::MalformedSection(format!(
                "section {name} declares {data_len} data bytes, only {} available",
                payload.len() - cursor
            )));
        }
        let data = payload[cursor..cursor + data_len].to_vec();
        cursor += data_len;

        sections.push(Section { kind, name, data });
    }
    Ok(sections)
}

/// Cheap magic-bytes probe.
pub fn looks_like_hatch(bytes: &[u8]) -> bool {
    bytes.len() >= MAGIC.len() && bytes[..MAGIC.len()] == MAGIC
}

// ---------------------------------------------------------------------------
// Builder: walk a source directory, compile each .wren to .wlbc
// ---------------------------------------------------------------------------

/// On-disk manifest filename at a project root. `hatch-cli` treats a
/// directory containing one of these as a wrenlift workspace.
pub const HATCHFILE: &str = "hatchfile";

/// Compile an on-disk source tree into a `.hatch` byte stream.
///
/// Walks `root` recursively for `*.wren` files, compiles each one to a
/// `.wlbc` blob, and writes a single hatch containing all of them plus
/// a manifest section. If `root/hatchfile` (the workspace manifest)
/// exists it is parsed as TOML and used as-is; otherwise a minimal one
/// is synthesized.
///
/// The module ordering in the emitted manifest is alphabetical by
/// module name. Callers that need a specific dependency order should
/// supply a `hatchfile` with `modules` set explicitly.
pub fn build_from_source_tree(root: &Path) -> Result<Vec<u8>, HatchError> {
    let mut wren_files: Vec<(String, std::path::PathBuf)> = Vec::new();
    collect_wren_files(root, root, &mut wren_files)?;
    wren_files.sort_by(|a, b| a.0.cmp(&b.0));

    // Compile each to a .wlbc section.
    let mut sections: Vec<Section> = Vec::with_capacity(wren_files.len());
    let mut module_names: Vec<String> = Vec::with_capacity(wren_files.len());

    for (module_name, path) in &wren_files {
        let source = std::fs::read_to_string(path)?;
        // Fresh VM per compile so interners don't leak across modules.
        // Compilation is cheap and stateless here; the VM's runtime
        // state (modules registered, classes allocated) is never used.
        let mut vm = crate::runtime::vm::VM::new_default();
        let blob = vm.compile_source_to_blob(&source).map_err(|_| {
            HatchError::Encode(format!(
                "compile of {} failed (see diagnostics on stderr)",
                path.display()
            ))
        })?;
        sections.push(Section {
            kind: SectionKind::Wlbc,
            name: module_name.clone(),
            data: blob,
        });
        module_names.push(module_name.clone());
    }

    // Manifest: `hatchfile` at project root, or synthesized.
    let manifest = match std::fs::read_to_string(root.join(HATCHFILE)) {
        Ok(text) => {
            let mut m: Manifest =
                toml::from_str(&text).map_err(|e| HatchError::ManifestParse(e.to_string()))?;
            if m.modules.is_empty() {
                m.modules = module_names.clone();
            }
            m
        }
        Err(_) => Manifest {
            name: root
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("hatch")
                .to_string(),
            version: "0.1.0".to_string(),
            entry: pick_default_entry(&module_names),
            modules: module_names.clone(),
            dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
        },
    };

    let hatch = Hatch { manifest, sections };
    emit(&hatch)
}

fn collect_wren_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, std::path::PathBuf)>,
) -> Result<(), HatchError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_wren_files(root, &path, out)?;
            continue;
        }
        if file_type.is_file() && path.extension().and_then(|e| e.to_str()) == Some("wren") {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let module_name = relative
                .with_extension("")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(".");
            out.push((module_name, path.clone()));
        }
    }
    Ok(())
}

fn pick_default_entry(modules: &[String]) -> String {
    // Prefer `main` if present; otherwise the first module alphabetically.
    if modules.iter().any(|m| m == "main") {
        "main".to_string()
    } else {
        modules
            .first()
            .cloned()
            .unwrap_or_else(|| "main".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Hatch {
        Hatch {
            manifest: Manifest {
                name: "sample".to_string(),
                version: "0.1.0".to_string(),
                entry: "main".to_string(),
                modules: vec!["main".to_string(), "util".to_string()],
                dependencies: BTreeMap::new(),
                native_libs: BTreeMap::new(),
                native_search_paths: Vec::new(),
            },
            sections: vec![
                Section {
                    kind: SectionKind::Wlbc,
                    name: "main".to_string(),
                    data: b"fake main wlbc".to_vec(),
                },
                Section {
                    kind: SectionKind::Wlbc,
                    name: "util".to_string(),
                    data: b"fake util wlbc".to_vec(),
                },
                Section {
                    kind: SectionKind::Resource,
                    name: "assets/banner.txt".to_string(),
                    data: b"hello".to_vec(),
                },
            ],
        }
    }

    #[test]
    fn emit_load_round_trip_preserves_manifest_and_sections() {
        let original = sample();
        let bytes = emit(&original).expect("emit");
        assert!(looks_like_hatch(&bytes));
        let back = load(&bytes).expect("load");

        assert_eq!(back.manifest.name, original.manifest.name);
        assert_eq!(back.manifest.version, original.manifest.version);
        assert_eq!(back.manifest.entry, original.manifest.entry);
        assert_eq!(back.manifest.modules, original.manifest.modules);

        assert_eq!(back.sections.len(), original.sections.len());
        for (a, b) in original.sections.iter().zip(back.sections.iter()) {
            assert_eq!(a.kind as u8, b.kind as u8);
            assert_eq!(a.name, b.name);
            assert_eq!(a.data, b.data);
        }
    }

    #[test]
    fn load_rejects_bad_magic() {
        let bytes = vec![0u8; 32];
        assert!(matches!(load(&bytes), Err(HatchError::BadMagic)));
    }

    #[test]
    fn load_rejects_version_skew() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&(VERSION + 1).to_le_bytes());
        bytes.push(FLAG_ZSTD);
        bytes.extend_from_slice(&[0u8; 3]);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            load(&bytes),
            Err(HatchError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn load_rejects_truncated_payload() {
        let original = sample();
        let bytes = emit(&original).expect("emit");
        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            load(truncated),
            Err(HatchError::TruncatedPayload { .. })
        ));
    }

    #[test]
    fn load_rejects_missing_manifest() {
        // Fabricate a hatch with one Wlbc section and no manifest.
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes()); // section_count
        payload.push(SectionKind::Wlbc as u8);
        payload.extend_from_slice(&4u16.to_le_bytes());
        payload.extend_from_slice(b"main");
        payload.extend_from_slice(&0u32.to_le_bytes());

        let compressed =
            zstd::encode_all(std::io::Cursor::new(&payload), ZSTD_LEVEL).expect("encode");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(FLAG_ZSTD);
        bytes.extend_from_slice(&[0u8; 3]);
        bytes.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&compressed);

        assert!(matches!(load(&bytes), Err(HatchError::ManifestMissing)));
    }

    #[test]
    fn looks_like_hatch_matches_magic() {
        assert!(!looks_like_hatch(&[]));
        assert!(!looks_like_hatch(b"HATC"));
        assert!(looks_like_hatch(b"HATCH"));
        assert!(looks_like_hatch(b"HATCHxtra"));
        assert!(!looks_like_hatch(b"ABCDE"));
    }

    #[test]
    fn manifest_native_libs_accepts_both_forms() {
        // Shorthand: key = "path" and table form must both deserialize.
        // Bare keys with no path become `Detailed { path: None }`,
        // meaning "use the key as a bare name".
        let text = r#"
name = "x"
version = "0.0.0"
entry = "main"

native_search_paths = ["/opt/custom/lib"]

[native_libs]
libssl = "/usr/lib/libssl.dylib"

[native_libs.sqlite3]
path = "vendor/libsqlite3.dylib"

[native_libs.curl]
# no path — fall back to bare-name resolution
"#;
        let m: Manifest = toml::from_str(text).expect("parse");
        assert_eq!(m.native_search_paths, vec!["/opt/custom/lib"]);
        assert_eq!(
            m.native_libs.get("libssl").and_then(|e| e.path()),
            Some("/usr/lib/libssl.dylib")
        );
        assert_eq!(
            m.native_libs.get("sqlite3").and_then(|e| e.path()),
            Some("vendor/libsqlite3.dylib")
        );
        assert!(m.native_libs.contains_key("curl"));
        assert_eq!(m.native_libs.get("curl").and_then(|e| e.path()), None);
    }
}
