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
    /// UTF-8 source text for a Wren module. Section name matches the
    /// module's compiled Wlbc name. Carried so runtime errors that
    /// originate inside a hatch-bundled module can render through the
    /// same ariadne label / span path the on-disk module loader uses;
    /// without source the loader falls back to bare prose.
    Source = 4,
}

impl SectionKind {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(SectionKind::Manifest),
            1 => Some(SectionKind::Wlbc),
            2 => Some(SectionKind::Resource),
            3 => Some(SectionKind::NativeLib),
            4 => Some(SectionKind::Source),
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
    /// One-line description surfaced in the catalog and `hatch find`
    /// output. Optional but strongly encouraged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Ordered list of module names in this hatch. The loader installs
    /// them in this order so a module's imports resolve against
    /// already-loaded peers. Producers are expected to write this in
    /// dependency order; the loader does not topologically sort.
    #[serde(default)]
    pub modules: Vec<String>,
    /// `name → dependency` declaration list. Each entry is either a
    /// version string (advisory for now — no registry yet) or an
    /// inline table with a `path` key pointing at another workspace
    /// directory. Path deps are recursively built and their sections
    /// merged into the enclosing hatch at build time.
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,
    /// Dependencies only needed while running `*.spec.wren` files —
    /// test runners, assertion libraries, fixtures. Not installed when
    /// a consumer builds against this package. Same shape as
    /// `[dependencies]`.
    #[serde(default, rename = "spec-dependencies")]
    pub spec_dependencies: BTreeMap<String, Dependency>,
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
    /// Optional declaration that this package's native library is
    /// built from an out-of-tree Rust `cdylib` crate living in
    /// another repo. Read by CI pipelines (not the runtime) to
    /// reproducibly build the dylib from a pinned source rev
    /// before bundling via `hatch publish`. Pinning lives here so
    /// each published package version ties itself to an exact
    /// upstream SHA — re-runs reproduce the same bytes.
    #[serde(default, rename = "plugin_source")]
    pub plugin_source: Option<PluginSource>,
    /// Target triple this hatch was built for. `None` (the default
    /// for older hatches) means "host target" — preserves the
    /// pre-target-aware behavior. When set, the loader checks the
    /// triple matches its own runtime; mismatched bundles refuse
    /// to load with `HatchError::WrongTarget`.
    ///
    /// Recognised values:
    ///   * Any concrete triple (`x86_64-apple-darwin`,
    ///     `wasm32-unknown-unknown`, `wasm32-wasip1`, etc.).
    ///   * `wasm32` — a family marker that matches any
    ///     `wasm32-*` runtime (so a single bundle works on both
    ///     `unknown-unknown` and `wasi`).
    ///
    /// For `wasm32-*` targets, `pack_bundled_native_libs` skips
    /// shipping `.dylib`/`.so` bytes — wasm runtimes use
    /// statically-linked plugins (cf. `wlift_wasm`'s
    /// `register_static_plugins`), and a wasm hatch's native_libs
    /// list is treated as a *required* set the runtime must
    /// already carry, not a build-time bundle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

/// Tells CI how to build this package's native library from an
/// out-of-tree Rust crate. Pinned by either `rev` (commit SHA,
/// preferred for reproducibility) or `tag` (friendly, mutable in
/// theory but stable in practice once you own the tag).
///
/// ```toml
/// [plugin_source]
/// repo  = "https://github.com/wrenlift/WrenLift.git"
/// rev   = "29ac35a"
/// crate = "wlift_sqlite"
/// ```
///
/// The runtime itself never reads this — it just loads whatever
/// dylib bytes the bundle carries. The hatch CI workflow in the
/// package's repo consumes it: clone the repo at `rev` / `tag`,
/// `cargo build -p <crate> --release` per platform matrix, copy
/// the output into the package's `libs/` dir, then publish.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PluginSource {
    /// Git URL — https, ssh, or file:// for local checkouts.
    pub repo: String,
    /// Pinned commit SHA (short or long form).
    #[serde(default)]
    pub rev: Option<String>,
    /// Alternative to `rev`: pin by git tag. One of `rev` / `tag`
    /// should be set; CI errors out if both are absent.
    #[serde(default)]
    pub tag: Option<String>,
    /// Cargo package name to build from within the repo (the one
    /// that produces the cdylib we bundle).
    #[serde(rename = "crate")]
    pub crate_name: String,
}

/// Shape of a `[dependencies.<name>]` entry. Four shapes, in rising
/// order of specificity:
///
/// ```toml
/// [dependencies]
/// json     = "1.0.0"                                        # official registry
/// counter  = { path = "../counter" }                        # workspace sibling
/// mylib    = { git = "https://github.com/alice/mylib.git",  # self-hosted git
///              tag = "v0.3.0" }
/// otherlib = { git = "https://github.com/bob/otherlib.git",
///              rev = "deadbeef..." }
/// ```
///
/// Published packages take two routes: contributors either open a PR
/// against the official registry monorepo (which builds + releases a
/// `.hatch` on tag push) and consumers pin by version, or they host
/// a package in their own git repo and consumers pin by tag / rev /
/// branch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    /// `name = "1.2.3"` — resolved through the registry release
    /// artifact cache populated by `hatch install`.
    Version(String),
    /// `name = { path = "../sibling" }` — workspace-relative path
    /// resolved at `hatch build`, recursively built, and its sections
    /// merged into the enclosing hatch.
    Path {
        path: String,
        #[serde(default)]
        version: Option<String>,
    },
    /// `name = { git = "...", tag/rev/branch = "..." }` — self-hosted
    /// package. `hatch install` shallow-clones the repo at the given
    /// ref into the git cache; `hatch build` reads the checkout like
    /// a path dep.
    Git {
        git: String,
        #[serde(default)]
        tag: Option<String>,
        #[serde(default)]
        rev: Option<String>,
        #[serde(default)]
        branch: Option<String>,
    },
}

impl Dependency {
    pub fn path(&self) -> Option<&str> {
        match self {
            Dependency::Path { path, .. } => Some(path.as_str()),
            _ => None,
        }
    }

    /// Pick the single git ref the user declared, enforcing the "only
    /// one of tag / rev / branch" invariant. Returns the stringified
    /// ref plus a label suitable for the cache-directory name.
    pub fn git_ref(&self) -> Option<GitRef<'_>> {
        match self {
            Dependency::Git {
                tag, rev, branch, ..
            } => {
                // Prefer the most specific form if multiple set (rev
                // pins harder than tag, tag harder than branch), but
                // publishers really shouldn't list more than one.
                if let Some(r) = rev.as_deref() {
                    Some(GitRef::Rev(r))
                } else if let Some(t) = tag.as_deref() {
                    Some(GitRef::Tag(t))
                } else {
                    branch.as_deref().map(GitRef::Branch)
                }
            }
            _ => None,
        }
    }
}

/// A checkout target inside a git-hosted dependency. Variants differ
/// in how much the cache dedupes: tags + revs are immutable, branches
/// aren't (could be refreshed with a future `hatch install --update`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitRef<'a> {
    Tag(&'a str),
    Rev(&'a str),
    Branch(&'a str),
}

impl GitRef<'_> {
    pub fn as_str(&self) -> &str {
        match self {
            GitRef::Tag(s) | GitRef::Rev(s) | GitRef::Branch(s) => s,
        }
    }

    /// Short label that goes into cache paths — keeps rev / tag /
    /// branch namespaces separate so a branch named `v1` can coexist
    /// with a tag named `v1`.
    pub fn kind(&self) -> &'static str {
        match self {
            GitRef::Tag(_) => "tag",
            GitRef::Rev(_) => "rev",
            GitRef::Branch(_) => "branch",
        }
    }
}

/// Shape of a `[native_libs.<name>]` entry in a hatchfile. Either a
/// bare string path, or an inline table whose keys select a path per
/// runtime platform:
///
/// ```toml
/// [native_libs]
/// libssl  = "/usr/lib/libssl.dylib"            # bare shorthand
/// sqlite3 = { macos = "libs/libsqlite3.dylib",
///             linux = "libs/libsqlite3.so" }
/// openssl = { any = "libssl",                  # catch-all fallback
///             macos = "libs/libssl.dylib" }
/// zlib    = { "macos-arm64"  = "libs/arm64/libz.dylib",
///             "macos-x86_64" = "libs/x86_64/libz.dylib" }
/// ```
///
/// Recognized table keys:
///
/// * `<os>-<arch>` — most specific; e.g. `macos-arm64`, `linux-x86_64`.
/// * `<os>` — OS bucket; e.g. `macos`, `linux`, `windows`, `freebsd`.
/// * `any` — fallback for any platform with no more specific match.
/// * `path` — legacy alias for `any`, kept for backward compat.
///
/// Architecture names use the short Wren-user vocabulary — `arm64`
/// (not `aarch64`), `x86_64`, `x86`. OS names match the values of
/// [`std::env::consts::OS`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum NativeLibEntry {
    /// `libssl = "/usr/lib/libssl.dylib"` — path-only shorthand.
    /// Equivalent to `{ any = "/usr/lib/libssl.dylib" }`.
    Path(String),
    /// Inline table with platform-selector keys. Any key not matching
    /// the lookup vocabulary above is ignored.
    Map(BTreeMap<String, String>),
}

impl NativeLibEntry {
    /// Pick the best path for the current runtime platform.
    pub fn resolve(&self) -> Option<&str> {
        self.resolve_for(std::env::consts::OS, std::env::consts::ARCH)
    }

    /// Testable variant — caller supplies the OS/arch names.
    pub fn resolve_for(&self, os: &str, arch: &str) -> Option<&str> {
        match self {
            NativeLibEntry::Path(p) => Some(p.as_str()),
            NativeLibEntry::Map(map) => {
                let arch_short = canonical_arch_name(arch);
                let os_arch = format!("{}-{}", os, arch_short);
                map.get(&os_arch)
                    .or_else(|| map.get(os))
                    .or_else(|| map.get("any"))
                    .or_else(|| map.get("path"))
                    .map(String::as_str)
            }
        }
    }
}

/// Map a Rust/LLVM architecture name to the short form hatchfiles use.
/// Only rewrites `aarch64 → arm64`; everything else (`x86_64`, `x86`,
/// `riscv64`, `wasm32`, …) passes through, which matches standard
/// naming already.
fn canonical_arch_name(arch: &str) -> &str {
    match arch {
        "aarch64" => "arm64",
        other => other,
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
    VersionMismatch {
        expected: u32,
        found: u32,
    },
    /// Header advertises a payload length the buffer can't cover.
    TruncatedPayload {
        declared: u64,
        available: usize,
    },
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
    /// Hatch was built for a different target than the runtime
    /// loading it. Carries the bundle's declared target and the
    /// runtime's expected target so the message is actionable.
    WrongTarget {
        bundle: String,
        runtime: String,
    },
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
            HatchError::TruncatedPayload {
                declared,
                available,
            } => write!(
                f,
                "hatch payload truncated: header says {declared} bytes, only {available} available"
            ),
            HatchError::Decompress(e) => write!(f, "zstd decompression failed: {e}"),
            HatchError::MalformedSection(e) => write!(f, "malformed section: {e}"),
            HatchError::ManifestMissing => write!(f, "hatch has no manifest section"),
            HatchError::ManifestParse(e) => write!(f, "hatch.toml parse failed: {e}"),
            HatchError::Unsupported(s) => write!(f, "unsupported: {s}"),
            HatchError::WrongTarget { bundle, runtime } => write!(
                f,
                "hatch built for target '{bundle}' but this runtime is '{runtime}' — \
                 rebuild with `--target {runtime}` (or matching family)"
            ),
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
/// Encode a `Hatch` into the on-disk byte stream. Compresses
/// the payload with zstd when the `host` feature is enabled
/// (the only build that links zstd); on `feature = "host"` off
/// builds (i.e. wasm), falls back to an uncompressed payload
/// with `FLAG_ZSTD` cleared. `load()` handles both shapes, so a
/// host-built hatch round-trips through a wasm runtime as long
/// as it was built uncompressed (use `--bundle-target wasm32`).
pub fn emit(hatch: &Hatch) -> Result<Vec<u8>, HatchError> {
    emit_with_options(hatch, EmitOptions::default())
}

/// Knobs that affect `emit()`. Exposed because the builder has
/// to override the default for wasm targets — wasm runtimes
/// can't decompress zstd today (the dep is host-feature gated).
#[derive(Debug, Clone, Copy)]
pub struct EmitOptions {
    /// Compress the payload with zstd. Requires the host build
    /// (zstd is gated). `false` always works.
    pub compress: bool,
}

impl Default for EmitOptions {
    fn default() -> Self {
        // Compress by default on host; on non-host builds (wasm)
        // the zstd dep isn't linked, so the compressed branch
        // wouldn't compile — flip the default off there.
        Self {
            compress: cfg!(feature = "host"),
        }
    }
}

pub fn emit_with_options(hatch: &Hatch, opts: EmitOptions) -> Result<Vec<u8>, HatchError> {
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
    let (body, flags): (Vec<u8>, u8) = if opts.compress {
        #[cfg(feature = "host")]
        {
            let compressed = zstd::encode_all(std::io::Cursor::new(&payload), ZSTD_LEVEL)
                .map_err(|e| HatchError::Encode(format!("zstd encode: {e}")))?;
            (compressed, FLAG_ZSTD)
        }
        #[cfg(not(feature = "host"))]
        {
            return Err(HatchError::Encode(
                "compression requested but this build has no zstd (wasm/no-host)".to_string(),
            ));
        }
    } else {
        (payload, 0)
    };

    let mut out = Vec::with_capacity(16 + body.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.push(flags); // flags
    out.extend_from_slice(&[0u8; 3]); // reserved
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&body);
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
        #[cfg(feature = "host")]
        {
            zstd::decode_all(std::io::Cursor::new(raw_payload))
                .map_err(|e| HatchError::Decompress(e.to_string()))?
        }
        #[cfg(not(feature = "host"))]
        {
            // Compressed payload but this build has no zstd
            // (wasm/no-host). Producers targeting wasm should
            // build with `--bundle-target wasm32-*`, which sets
            // `EmitOptions::compress = false` and clears
            // FLAG_ZSTD on the hatch.
            return Err(HatchError::Decompress(
                "compressed hatch can't be loaded — runtime has no zstd; \
                 rebuild with `--bundle-target wasm32-*` for an uncompressed payload"
                    .to_string(),
            ));
        }
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

        let data_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
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
#[cfg(feature = "host")]
pub fn build_from_source_tree(root: &Path) -> Result<Vec<u8>, HatchError> {
    build_from_source_tree_with_cache(root, None)
}

/// Variant that lets callers override the registry cache directory
/// `hatch build` consults when resolving version-pinned dependencies.
/// `None` falls back to the ambient `cache_root()` — `HATCH_CACHE_DIR`
/// or `$HOME/.hatch/cache`. Tests pass an explicit path to avoid
/// process-wide env var coupling.
#[cfg(feature = "host")]
pub fn build_from_source_tree_with_cache(
    root: &Path,
    cache_dir: Option<&Path>,
) -> Result<Vec<u8>, HatchError> {
    build_from_source_tree_for_target(root, cache_dir, None)
}

/// Most-flexible builder entry point — same as
/// `build_from_source_tree_with_cache`, plus an optional target
/// triple to stamp into the manifest. `None` means "host target"
/// (legacy behaviour). For `wasm32-*` triples,
/// `pack_bundled_native_libs` skips packing `.dylib`/`.so` bytes
/// because wasm runtimes don't dlopen anything; the manifest's
/// `[native_libs]` declarations stay as a *requirement list* the
/// receiving runtime must satisfy via statically-linked plugins.
#[cfg(feature = "host")]
pub fn build_from_source_tree_for_target(
    root: &Path,
    cache_dir: Option<&Path>,
    target: Option<&str>,
) -> Result<Vec<u8>, HatchError> {
    let mut state = BuildState::default();
    let mut bytes = build_recursive(root, &mut state, cache_dir)?;
    if target.is_some() {
        // Re-stamp the manifest's `target` field. Cheaper to
        // decode/re-encode the whole hatch than to thread `target`
        // through every layer of `build_recursive` — this only
        // runs once per top-level build, not per dependency.
        let mut hatch = load(&bytes)?;
        hatch.manifest.target = target.map(|s| s.to_string());
        // For wasm targets, drop any NativeLib sections that
        // `pack_bundled_native_libs` may have packed before we
        // knew the target. The wasm runtime ignores them anyway,
        // and shipping x86_64 dylibs in a wasm hatch wastes
        // bytes + confuses `--inspect`.
        // Wasm targets get an uncompressed payload — the wasm
        // runtime doesn't link zstd, so a compressed bundle
        // would refuse to load there with a decode error.
        let opts = if is_wasm_target(target) {
            hatch
                .sections
                .retain(|s| !matches!(s.kind, SectionKind::NativeLib));
            EmitOptions { compress: false }
        } else {
            EmitOptions::default()
        };
        bytes = emit_with_options(&hatch, opts)?;
    }
    Ok(bytes)
}

/// True iff `target` is a wasm32 family triple. `wasm32` (bare)
/// is the family marker; `wasm32-unknown-unknown` /
/// `wasm32-wasip1` etc. are concrete triples within it. Matched
/// by prefix so future wasm32 sub-triples don't need a code
/// change.
pub fn is_wasm_target(target: Option<&str>) -> bool {
    matches!(target, Some(t) if t == "wasm32" || t.starts_with("wasm32-"))
}

/// Family bucket a target triple falls into. Hatches are
/// loadable on any runtime within the same family — a
/// `wasm32-unknown-unknown` build runs on a `wasm32-wasip1`
/// runtime (same wlbc, same statically-linked plugin set), and a
/// `x86_64-apple-darwin` build runs on `aarch64-apple-darwin`
/// (same wlbc, same dlopen path). Cross-family mismatch is the
/// only thing that errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFamily {
    /// `wasm32` (family marker) and any `wasm32-*` concrete triple.
    Wasm32,
    /// Any host-native triple — `x86_64-*`, `aarch64-*`, etc.
    Native,
}

/// Bucket a triple into a [`TargetFamily`]. `wasm32` and
/// `wasm32-*` map to `Wasm32`; everything else maps to `Native`.
pub fn target_family(target: &str) -> TargetFamily {
    if target == "wasm32" || target.starts_with("wasm32-") {
        TargetFamily::Wasm32
    } else {
        TargetFamily::Native
    }
}

/// True iff the `bundle` target is loadable on a runtime built
/// for `runtime`. Compatibility rules:
///
///   * `bundle == None` (target-agnostic / legacy hatches) — accepted everywhere.
///   * Same family — accepted (see `target_family`).
///   * Different family — `WrongTarget` error.
pub fn check_target_compat(
    bundle: Option<&str>,
    runtime: &str,
) -> Result<(), HatchError> {
    let Some(bundle) = bundle else {
        return Ok(());
    };
    if target_family(bundle) == target_family(runtime) {
        return Ok(());
    }
    Err(HatchError::WrongTarget {
        bundle: bundle.to_string(),
        runtime: runtime.to_string(),
    })
}

/// The target the current binary is running under. Family
/// markers (`"wasm32"`, `"native"`) rather than full triples —
/// the family is what compat checks against, and we don't have
/// a build.rs to capture the concrete triple. Concrete triples
/// can still be *stamped into bundles* via `--target`; the
/// loader's `check_target_compat` family-matches them against
/// this string at install time.
pub fn current_runtime_target() -> &'static str {
    #[cfg(target_arch = "wasm32")]
    {
        "wasm32"
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        "native"
    }
}

/// Per-recursion build state. `active` is the in-progress recursion
/// path — a revisit indicates a true cycle and errors. `cache` holds
/// the encoded bytes of every workspace we've already built; a hit
/// means we're seeing the same dep through a diamond path and can
/// reuse the bytes verbatim instead of rebuilding (and hitting a
/// false-positive cycle error).
#[cfg(feature = "host")]
#[derive(Default)]
struct BuildState {
    active: std::collections::HashSet<std::path::PathBuf>,
    cache: std::collections::HashMap<std::path::PathBuf, Vec<u8>>,
}

/// Internal recursive variant. Distinguishes a true cycle (`a → b →
/// a`, where `a` is *currently being built* on the recursion path)
/// from a diamond dep (`a → b → c` plus `a → c`, where `c` is
/// already-built and just needs to be re-folded). The former errors
/// loudly; the latter returns the cached bytes so the second arm of
/// the diamond doesn't trip the cycle detector.
#[cfg(feature = "host")]
fn build_recursive(
    root: &Path,
    state: &mut BuildState,
    cache_dir: Option<&Path>,
) -> Result<Vec<u8>, HatchError> {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    if state.active.contains(&canonical) {
        return Err(HatchError::Encode(format!(
            "dependency cycle detected at {}",
            root.display()
        )));
    }
    if let Some(cached) = state.cache.get(&canonical) {
        return Ok(cached.clone());
    }
    state.active.insert(canonical.clone());

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
        // Bundle the original source text alongside the compiled
        // bytecode so runtime errors raised inside the module can
        // render through ariadne labels at install time. Adds the
        // source bytes to the artifact (typically <10% of the total
        // — wlbc + native libs dominate) but turns a "Class does
        // not implement 'create(_)'" runtime error into a labelled
        // span pointing at the actual call site.
        sections.push(Section {
            kind: SectionKind::Source,
            name: module_name.clone(),
            data: source.into_bytes(),
        });
        module_names.push(module_name.clone());
    }

    // Manifest: `hatchfile` at project root, or synthesized.
    let mut manifest = match std::fs::read_to_string(root.join(HATCHFILE)) {
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
            description: None,
            modules: module_names.clone(),
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
    };

    rename_entry_to_manifest_name(&mut manifest, &mut sections)?;
    pack_bundled_native_libs(root, &mut manifest, &mut sections)?;
    merge_path_dependencies(root, &mut manifest, &mut sections, state, cache_dir)?;

    let hatch = Hatch { manifest, sections };
    let bytes = emit(&hatch)?;
    // Pop from the active recursion stack and cache the encoded
    // bytes so a diamond revisit (`a → b → c` plus `a → c`) returns
    // these same bytes without rebuilding.
    state.active.remove(&canonical);
    state.cache.insert(canonical, bytes.clone());
    Ok(bytes)
}

/// When `manifest.name` differs from the entry file's module name,
/// rename the entry section so the published module's import name
/// matches how consumers spell the dep. This is what makes
/// `@hatch:assert` work: the package source lives at an ordinary
/// filesystem path (`packages/hatch-assert/assert.wren`) but the
/// module surfaces as `"@hatch:assert"` at install time because
/// that's the package's declared name.
///
/// Only the *entry* module is renamed. Multi-file packages keep
/// their internal file-stem names for non-entry modules; a package
/// named `@hatch:fs` with `fs.wren` + `fs/stat.wren` would install
/// modules `@hatch:fs` and `fs/stat` (or `fs.stat` — whatever the
/// file layout produces).
#[cfg(feature = "host")]
fn rename_entry_to_manifest_name(
    manifest: &mut Manifest,
    sections: &mut [Section],
) -> Result<(), HatchError> {
    // Only rename for *scoped* packages. A bare name like
    // `libcounter` is a perfectly normal package whose entry file
    // can still be `counter.wren`; we shouldn't surprise its
    // consumers by renaming the module. The scoping characters
    // (`:` for `@hatch:*`, `@` and `/` for future scopes) are what
    // signal "this package wants its module imported under the
    // package name."
    let is_scoped = manifest.name.chars().any(|c| matches!(c, ':' | '@' | '/'));
    if !is_scoped || manifest.name == manifest.entry {
        return Ok(());
    }

    let old = manifest.entry.clone();
    let new = manifest.name.clone();

    // If the entry section doesn't exist, the hatchfile is
    // pointing at a module that isn't on disk — surface that now
    // rather than at consumer install time.
    let entry_section = sections
        .iter_mut()
        .find(|s| matches!(s.kind, SectionKind::Wlbc) && s.name == old);
    let Some(section) = entry_section else {
        return Err(HatchError::Encode(format!(
            "manifest.entry '{}' has no matching .wren source",
            old
        )));
    };
    section.name = new.clone();
    // Rename the matching Source section too so error reporting can
    // still find the source by module name.
    if let Some(src_section) = sections
        .iter_mut()
        .find(|s| matches!(s.kind, SectionKind::Source) && s.name == old)
    {
        src_section.name = new.clone();
    }

    // Mirror the rename in the module list so the install loop
    // asks for the new name when it iterates manifest.modules.
    for m in &mut manifest.modules {
        if m == &old {
            *m = new.clone();
        }
    }
    // And in the entry field itself, so consumers inspecting the
    // manifest see consistent naming.
    manifest.entry = new;

    Ok(())
}

/// Recursively resolve every dependency and fold its sections into
/// `sections` / `manifest.modules`. Two resolution modes:
///
/// * `path = "..."` — recursively build the sibling workspace.
/// * `"<version>"` — look up `~/.hatch/cache/<name>-<version>.hatch`
///   (populated by `hatch install`). A cache miss is a hard error —
///   we never silently reach out to the network during `hatch build`.
///
/// Dep modules install *before* this hatch's own modules, so an
/// `import "counter"` in `main.wren` resolves against the dep's
/// already-installed class. Name collisions between modules /
/// sections are rejected loudly so ambiguous imports never slip
/// through to runtime.
/// Resolve a single dependency declaration to the `.hatch` bytes it
/// represents. Handles all three `Dependency` shapes — path (build
/// the sibling workspace), version (registry cache), git (git cache).
///
/// `root` is the directory containing the hatchfile that declared this
/// dep; used to resolve relative `path = "..."` deps. `cache_dir` lets
/// callers pin the registry cache root (tests pass an explicit path;
/// production passes `None` → `$HOME/.hatch/cache`).
#[cfg(feature = "host")]
pub fn resolve_dependency_bytes(
    root: &Path,
    dep_name: &str,
    dep: &Dependency,
    cache_dir: Option<&Path>,
) -> Result<Vec<u8>, HatchError> {
    let mut state = BuildState::default();
    resolve_dep_bytes_inner(root, dep_name, dep, &mut state, cache_dir)
}

#[cfg(feature = "host")]
fn resolve_dep_bytes_inner(
    root: &Path,
    dep_name: &str,
    dep: &Dependency,
    state: &mut BuildState,
    cache_dir: Option<&Path>,
) -> Result<Vec<u8>, HatchError> {
    match dep {
        Dependency::Path { path, .. } => {
            let dep_root = root.join(path);
            build_recursive(&dep_root, state, cache_dir).map_err(|e| {
                HatchError::Encode(format!("failed to build dependency '{}': {}", dep_name, e))
            })
        }
        Dependency::Version(version) => {
            let cached = match cache_dir {
                Some(dir) => crate::hatch_registry::cached_artifact_path_in(dir, dep_name, version),
                None => crate::hatch_registry::cached_artifact_path(dep_name, version)
                    .map_err(|e| HatchError::Encode(e.to_string()))?,
            };
            if !cached.exists() {
                return Err(HatchError::Encode(format!(
                    "dependency '{}@{}' isn't cached. Run `hatch install {}@{}` first.",
                    dep_name, version, dep_name, version
                )));
            }
            Ok(std::fs::read(&cached)?)
        }
        Dependency::Git { git, .. } => {
            let git_ref = dep.git_ref().ok_or_else(|| {
                HatchError::Encode(format!(
                    "git dependency '{}' must specify one of tag / rev / branch",
                    dep_name
                ))
            })?;
            let cache_base = match cache_dir {
                Some(p) => p.to_path_buf(),
                None => crate::hatch_registry::cache_root()
                    .map_err(|e| HatchError::Encode(e.to_string()))?,
            };
            let checkout =
                crate::hatch_registry::cached_git_checkout_path(&cache_base, git, git_ref);
            if !checkout.exists() {
                return Err(HatchError::Encode(format!(
                    "git dependency '{}' ({} @ {}) isn't cached. Run `hatch install {}` first.",
                    dep_name,
                    git,
                    git_ref.as_str(),
                    dep_name
                )));
            }
            build_recursive(&checkout, state, cache_dir).map_err(|e| {
                HatchError::Encode(format!(
                    "failed to build git dependency '{}': {}",
                    dep_name, e
                ))
            })
        }
    }
}

#[cfg(feature = "host")]
fn merge_path_dependencies(
    root: &Path,
    manifest: &mut Manifest,
    sections: &mut Vec<Section>,
    state: &mut BuildState,
    cache_dir: Option<&Path>,
) -> Result<(), HatchError> {
    // Collect into an owned list so we can mutate `manifest.dependencies`
    // while iterating.
    let deps: Vec<(String, Dependency)> = manifest
        .dependencies
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    for (dep_name, dep) in deps {
        let dep_bytes = match &dep {
            Dependency::Path { path, .. } => {
                let dep_root = root.join(path);
                build_recursive(&dep_root, state, cache_dir).map_err(|e| {
                    HatchError::Encode(format!("failed to build dependency '{}': {}", dep_name, e))
                })?
            }
            Dependency::Version(version) => {
                // Resolve via registry cache. `hatch install` populates
                // this; during `hatch build` we refuse to fetch so
                // offline builds remain deterministic.
                let cached = match cache_dir {
                    Some(dir) => {
                        crate::hatch_registry::cached_artifact_path_in(dir, &dep_name, version)
                    }
                    None => crate::hatch_registry::cached_artifact_path(&dep_name, version)
                        .map_err(|e| HatchError::Encode(e.to_string()))?,
                };
                if !cached.exists() {
                    return Err(HatchError::Encode(format!(
                        "dependency '{}@{}' isn't cached. Run `hatch install {}@{}` first.",
                        dep_name, version, dep_name, version
                    )));
                }
                std::fs::read(&cached)?
            }
            Dependency::Git { git, .. } => {
                let git_ref = dep.git_ref().ok_or_else(|| {
                    HatchError::Encode(format!(
                        "git dependency '{}' must specify one of tag / rev / branch",
                        dep_name
                    ))
                })?;
                let cache_base = match cache_dir {
                    Some(p) => p.to_path_buf(),
                    None => crate::hatch_registry::cache_root()
                        .map_err(|e| HatchError::Encode(e.to_string()))?,
                };
                let checkout =
                    crate::hatch_registry::cached_git_checkout_path(&cache_base, git, git_ref);
                if !checkout.exists() {
                    return Err(HatchError::Encode(format!(
                        "git dependency '{}' ({} @ {}) isn't cached. Run `hatch install {}` first.",
                        dep_name,
                        git,
                        git_ref.as_str(),
                        dep_name
                    )));
                }
                // Treat the cached checkout like any path dep:
                // recursively build so transitive deps resolve too.
                build_recursive(&checkout, state, cache_dir).map_err(|e| {
                    HatchError::Encode(format!(
                        "failed to build git dependency '{}': {}",
                        dep_name, e
                    ))
                })?
            }
        };
        let dep_hatch = load(&dep_bytes)?;

        // Prepend dep modules so they install before ours. A name
        // collision means EITHER a true collision (two unrelated
        // packages chose the same module name — must error) OR a
        // diamond dep where the same package appears via two paths
        // and contributes byte-identical sections (silently dedupe).
        // We tell them apart by checking the section bytes against
        // what's already in `sections`.
        let mut new_modules: Vec<String> = Vec::new();
        for mod_name in &dep_hatch.manifest.modules {
            if manifest.modules.contains(mod_name) {
                let dep_section = dep_hatch.sections.iter().find(|s| {
                    matches!(s.kind, SectionKind::Wlbc | SectionKind::NativeLib)
                        && &s.name == mod_name
                });
                let existing = sections.iter().find(|s| {
                    matches!(s.kind, SectionKind::Wlbc | SectionKind::NativeLib)
                        && &s.name == mod_name
                });
                match (dep_section, existing) {
                    (Some(d), Some(e)) if d.kind == e.kind && d.data == e.data => {
                        // Diamond — already bundled identically, skip.
                        continue;
                    }
                    _ => {
                        return Err(HatchError::Encode(format!(
                            "dependency '{}' carries module '{}' that collides with the enclosing hatch",
                            dep_name, mod_name
                        )));
                    }
                }
            }
            new_modules.push(mod_name.clone());
        }
        let mut merged_modules = new_modules;
        merged_modules.extend(std::mem::take(&mut manifest.modules));
        manifest.modules = merged_modules;

        // Carry over Wlbc / NativeLib / Source sections the dep
        // bundled. Diamond-dep sections (same name+kind+bytes) silently
        // dedupe; genuine collisions still error.
        for section in dep_hatch.sections {
            if matches!(
                section.kind,
                SectionKind::Wlbc | SectionKind::NativeLib | SectionKind::Source
            ) {
                if let Some(existing) = sections
                    .iter()
                    .find(|s| s.name == section.name && s.kind == section.kind)
                {
                    if existing.data == section.data {
                        continue; // diamond dedupe
                    }
                    return Err(HatchError::Encode(format!(
                        "dependency '{}' carries section '{:?}/{}' that collides with the enclosing hatch",
                        dep_name, section.kind, section.name
                    )));
                }
                sections.push(section);
            }
        }
        // Fold dep system refs + extra search paths into ours. Local
        // workspace path entries from the dep were already bundled
        // during its own `build_recursive` and won't appear here.
        for (name, entry) in dep_hatch.manifest.native_libs {
            manifest.native_libs.entry(name).or_insert(entry);
        }
        for path in dep_hatch.manifest.native_search_paths {
            if !manifest.native_search_paths.contains(&path) {
                manifest.native_search_paths.push(path);
            }
        }
        // Drop the dep from our manifest — it's bundled now, the
        // loader doesn't need to chase any external reference.
        manifest.dependencies.remove(&dep_name);
    }
    Ok(())
}

/// Walk the workspace's `[native_libs]` entries and bundle each one
/// that references a workspace-relative path. The file's bytes are
/// packed as a `NativeLib` section keyed by the library name; the
/// manifest entry is then removed so downstream loaders rely on the
/// section (not the filesystem path that only existed on the build
/// host). Absolute paths are left untouched — those are system refs
/// and shouldn't be bundled.
///
/// This is host-platform-only for now: the resolver picks the current
/// platform's entry and bundles that one. Cross-platform packaging
/// (multi-platform sections side-by-side) is a follow-up.
#[cfg(feature = "host")]
fn pack_bundled_native_libs(
    root: &Path,
    manifest: &mut Manifest,
    sections: &mut Vec<Section>,
) -> Result<(), HatchError> {
    let mut bundled: Vec<String> = Vec::new();
    for (name, entry) in &manifest.native_libs {
        let Some(path_str) = entry.resolve() else {
            continue; // nothing declared for this host platform — skip
        };
        let path = Path::new(path_str);
        if path.is_absolute() {
            continue; // system ref — don't bundle
        }
        let full = root.join(path);
        let bytes = std::fs::read(&full).map_err(|e| {
            HatchError::Encode(format!(
                "native lib '{}' declared at '{}' could not be read: {}",
                name,
                full.display(),
                e
            ))
        })?;
        sections.push(Section {
            kind: SectionKind::NativeLib,
            name: name.clone(),
            data: bytes,
        });
        bundled.push(name.clone());
    }
    for name in bundled {
        manifest.native_libs.remove(&name);
    }
    Ok(())
}

#[cfg(feature = "host")]
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
            // `*.spec.wren` is the convention for test files. They
            // run under `hatch test`, never ship in built hatches —
            // publishing test code would bloat the artifact and
            // create a runtime dependency on the test runner.
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if stem.ends_with(".spec") {
                continue;
            }
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

#[cfg(feature = "host")]
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
                description: None,
                modules: vec!["main".to_string(), "util".to_string()],
                dependencies: BTreeMap::new(),
                spec_dependencies: BTreeMap::new(),
                native_libs: BTreeMap::new(),
                native_search_paths: Vec::new(),
                plugin_source: None,
                target: None,
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
        // Bare shorthand + inline-table + section-header forms all
        // deserialize into the same enum.
        let text = r#"
name = "x"
version = "0.0.0"
entry = "main"

native_search_paths = ["/opt/custom/lib"]

[native_libs]
libssl  = "/usr/lib/libssl.dylib"
sqlite3 = { path = "vendor/libsqlite3.dylib" }

[native_libs.openssl]
any = "libssl"
macos = "libs/libssl.dylib"
"#;
        let m: Manifest = toml::from_str(text).expect("parse");
        assert_eq!(m.native_search_paths, vec!["/opt/custom/lib"]);
        // Bare shorthand resolves to its string everywhere.
        assert_eq!(
            m.native_libs
                .get("libssl")
                .and_then(|e| e.resolve_for("macos", "aarch64")),
            Some("/usr/lib/libssl.dylib")
        );
        // `path` key still resolves via the legacy alias.
        assert_eq!(
            m.native_libs
                .get("sqlite3")
                .and_then(|e| e.resolve_for("linux", "x86_64")),
            Some("vendor/libsqlite3.dylib")
        );
        // OS-specific key wins over `any`.
        assert_eq!(
            m.native_libs
                .get("openssl")
                .and_then(|e| e.resolve_for("macos", "x86_64")),
            Some("libs/libssl.dylib")
        );
        // Non-matching OS falls back to `any`.
        assert_eq!(
            m.native_libs
                .get("openssl")
                .and_then(|e| e.resolve_for("linux", "x86_64")),
            Some("libssl")
        );
    }

    #[test]
    fn native_lib_resolve_prefers_most_specific_key() {
        // `<os>-<arch>` beats `<os>` beats `any`.
        let text = r#"
[native_libs.zlib]
any = "libs/libz"
macos = "libs/mac/libz.dylib"
"macos-arm64" = "libs/mac-arm64/libz.dylib"
"#;
        let m: Manifest = toml::from_str(&format!(
            "name = \"x\"\nversion = \"0\"\nentry = \"m\"\n{}",
            text
        ))
        .expect("parse");
        let entry = &m.native_libs["zlib"];
        // Exact arch match — `aarch64` should canonicalize to `arm64`.
        assert_eq!(
            entry.resolve_for("macos", "aarch64"),
            Some("libs/mac-arm64/libz.dylib")
        );
        // OS match only (x86_64 on macos isn't listed explicitly).
        assert_eq!(
            entry.resolve_for("macos", "x86_64"),
            Some("libs/mac/libz.dylib")
        );
        // No OS match — fall back to `any`.
        assert_eq!(entry.resolve_for("linux", "x86_64"), Some("libs/libz"));
    }

    #[test]
    fn build_renames_entry_section_to_scoped_manifest_name() {
        // First-party packages declare their name with a scoped
        // prefix (`@hatch:assert`) but the source lives on disk
        // under a shell-safe directory. The build pass must rename
        // the entry section so consumers can `import "@hatch:assert"`
        // at runtime.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(
            root.join("assert.wren"),
            "class Expect {\n  static that(a) { a }\n}",
        )
        .unwrap();
        std::fs::write(
            root.join("hatchfile"),
            r#"name = "@hatch:assert"
version = "0.1.0"
entry = "assert"
"#,
        )
        .unwrap();

        let bytes = build_from_source_tree(root).expect("build");
        let hatch = load(&bytes).unwrap();

        assert_eq!(hatch.manifest.modules, vec!["@hatch:assert"]);
        assert_eq!(hatch.manifest.entry, "@hatch:assert");
        assert!(
            hatch
                .sections
                .iter()
                .any(|s| matches!(s.kind, SectionKind::Wlbc) && s.name == "@hatch:assert"),
            "expected a Wlbc section named '@hatch:assert', got: {:?}",
            hatch
                .sections
                .iter()
                .map(|s| (&s.kind, &s.name))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_excludes_spec_files_from_sections() {
        // `*.spec.wren` is test-only and must not land in the
        // published hatch. `hatch test` loads specs directly from
        // source — no reason to ship bytecode for them.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("assert.wren"), "class Expect {}").unwrap();
        std::fs::write(
            root.join("assert.spec.wren"),
            "System.print(\"would have run a test\")",
        )
        .unwrap();
        std::fs::write(
            root.join("hatchfile"),
            r#"name = "assert"
version = "0.1.0"
entry = "assert"
"#,
        )
        .unwrap();

        let bytes = build_from_source_tree(root).expect("build");
        let hatch = load(&bytes).unwrap();

        // Only the non-spec module should land in sections + modules.
        let section_names: Vec<&str> = hatch
            .sections
            .iter()
            .filter(|s| matches!(s.kind, SectionKind::Wlbc))
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(section_names, vec!["assert"]);
        assert!(
            !hatch.manifest.modules.iter().any(|m| m.contains("spec")),
            "spec leaked into modules list: {:?}",
            hatch.manifest.modules
        );
    }

    #[test]
    fn build_errors_when_manifest_entry_has_no_source() {
        // Author typo — entry points at a file that doesn't exist.
        // Must fail at build, not at consumer install.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("main.wren"), "1").unwrap();
        std::fs::write(
            root.join("hatchfile"),
            r#"name = "@hatch:wrong"
version = "0.1.0"
entry = "nope"
"#,
        )
        .unwrap();

        let result = build_from_source_tree(root);
        match result {
            Err(HatchError::Encode(msg)) => assert!(msg.contains("nope")),
            other => panic!("expected encode error, got {:?}", other),
        }
    }

    #[test]
    fn build_packs_relative_native_libs_and_strips_manifest_entry() {
        // Stand up a fake workspace, declare a relative-path native
        // lib, and confirm `build_from_source_tree` reads the file,
        // emits a NativeLib section with the bytes, and drops the
        // now-bundled entry from the manifest. Absolute-path entries
        // pass through untouched — those are system refs.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        std::fs::write(root.join("main.wren"), "System.print(\"hi\")").unwrap();
        std::fs::create_dir_all(root.join("libs")).unwrap();
        std::fs::write(root.join("libs/libfoo.bin"), b"native-bytes").unwrap();

        let hatchfile = r#"
name = "pkg"
version = "0.1.0"
entry = "main"

[native_libs]
libfoo = "libs/libfoo.bin"
libssl = "/usr/lib/libssl.dylib"
"#;
        std::fs::write(root.join("hatchfile"), hatchfile).unwrap();

        let bytes = build_from_source_tree(root).expect("build");
        let hatch = load(&bytes).expect("reload");

        // Relative entry was bundled: one NativeLib section named
        // "libfoo" whose bytes match what was on disk.
        let lib_section = hatch
            .sections
            .iter()
            .find(|s| matches!(s.kind, SectionKind::NativeLib) && s.name == "libfoo")
            .expect("libfoo NativeLib section");
        assert_eq!(lib_section.data, b"native-bytes");

        // Manifest no longer carries the bundled entry …
        assert!(!hatch.manifest.native_libs.contains_key("libfoo"));
        // … but the absolute-path (system) entry stays so the loader
        // keeps the override.
        assert!(hatch.manifest.native_libs.contains_key("libssl"));
    }

    #[test]
    fn build_bundles_path_dependencies_transitively() {
        // Lay out a tiny two-hatch workspace:
        //   deps/libcounter/  → a dep hatch with a `counter.wren` module
        //   app/              → an app hatch whose hatchfile names the
        //                       dep via `{ path = "../deps/libcounter" }`
        // After building app/, the resulting hatch must carry both
        // modules (dep first in install order), no dangling dependency
        // reference, and remain runnable end-to-end.
        let tmp = tempfile::tempdir().expect("tempdir");
        let lib_root = tmp.path().join("libcounter");
        let app_root = tmp.path().join("app");
        std::fs::create_dir_all(&lib_root).unwrap();
        std::fs::create_dir_all(&app_root).unwrap();

        std::fs::write(
            lib_root.join("counter.wren"),
            "class Counter {\n  static bump(n) { n + 1 }\n}",
        )
        .unwrap();
        std::fs::write(
            lib_root.join("hatchfile"),
            r#"name = "libcounter"
version = "0.1.0"
entry = "counter"
"#,
        )
        .unwrap();

        std::fs::write(
            app_root.join("main.wren"),
            "import \"counter\" for Counter\nSystem.print(Counter.bump(41))",
        )
        .unwrap();
        std::fs::write(
            app_root.join("hatchfile"),
            r#"name = "app"
version = "0.1.0"
entry = "main"

[dependencies]
libcounter = { path = "../libcounter" }
"#,
        )
        .unwrap();

        let bytes = build_from_source_tree(&app_root).expect("build");
        let hatch = load(&bytes).expect("reload");

        // Both modules present, dep first so the install loop hits
        // the class declaration before `main` imports it.
        assert_eq!(hatch.manifest.modules, vec!["counter", "main"]);

        // Dep reference stripped from the manifest — it's bundled now.
        assert!(!hatch.manifest.dependencies.contains_key("libcounter"));

        // Both .wlbc sections present.
        let module_names: Vec<&str> = hatch
            .sections
            .iter()
            .filter(|s| matches!(s.kind, SectionKind::Wlbc))
            .map(|s| s.name.as_str())
            .collect();
        assert!(module_names.contains(&"counter"));
        assert!(module_names.contains(&"main"));

        // And the whole thing actually runs: 41 → 42.
        let mut vm = crate::runtime::vm::VM::new_default();
        vm.output_buffer = Some(String::new());
        let result = vm.interpret_hatch(&bytes);
        assert!(matches!(
            result,
            crate::runtime::engine::InterpretResult::Success
        ));
        assert_eq!(vm.take_output().trim(), "42");
    }

    #[test]
    fn build_resolves_version_deps_from_cache() {
        // Simulate a full "install → build → run" cycle without
        // going over the wire. Pre-populate the registry cache with a
        // `libgreet-0.1.0.hatch` artifact, then build an app whose
        // hatchfile pins it by version. The build-side resolver must
        // find the cached artifact and inline its modules.
        let scratch = tempfile::tempdir().expect("tempdir");
        let cache_dir = scratch.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // 1. Build the library out of a fake workspace, then write
        //    its bytes to the cache under the expected filename.
        let lib_workspace = scratch.path().join("lib_src");
        std::fs::create_dir_all(&lib_workspace).unwrap();
        std::fs::write(
            lib_workspace.join("greet.wren"),
            "class Greet {\n  static hello { \"hi\" }\n}",
        )
        .unwrap();
        std::fs::write(
            lib_workspace.join("hatchfile"),
            r#"name = "libgreet"
version = "0.1.0"
entry = "greet"
"#,
        )
        .unwrap();
        let lib_bytes = build_from_source_tree(&lib_workspace).unwrap();
        std::fs::write(cache_dir.join("libgreet-0.1.0.hatch"), &lib_bytes).unwrap();

        // 2. The app workspace pins `libgreet` by version.
        let app_workspace = scratch.path().join("app");
        std::fs::create_dir_all(&app_workspace).unwrap();
        std::fs::write(
            app_workspace.join("main.wren"),
            "import \"greet\" for Greet\nSystem.print(Greet.hello)",
        )
        .unwrap();
        std::fs::write(
            app_workspace.join("hatchfile"),
            r#"name = "app"
version = "0.1.0"
entry = "main"

[dependencies]
libgreet = "0.1.0"
"#,
        )
        .unwrap();

        // 3. Point the registry cache at our scratch dir, then build.
        let bytes =
            build_from_source_tree_with_cache(&app_workspace, Some(&cache_dir)).expect("build");

        let hatch = load(&bytes).unwrap();
        // Dep module prepended so imports resolve during install loop.
        assert_eq!(hatch.manifest.modules, vec!["greet", "main"]);
        // Dep stripped once folded in.
        assert!(!hatch.manifest.dependencies.contains_key("libgreet"));

        // And the bundled app actually runs.
        let mut vm = crate::runtime::vm::VM::new_default();
        vm.output_buffer = Some(String::new());
        assert!(matches!(
            vm.interpret_hatch(&bytes),
            crate::runtime::engine::InterpretResult::Success
        ));
        assert_eq!(vm.take_output().trim(), "hi");
    }

    #[test]
    fn build_resolves_git_deps_from_cache() {
        // End-to-end: pre-populate the git cache with a checked-out
        // workspace, declare the dep in the consumer's hatchfile, and
        // make sure `hatch build` folds the dep's modules in and the
        // whole thing runs.
        let scratch = tempfile::tempdir().expect("tempdir");
        let cache_dir = scratch.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();

        // The git-dep checkout path — `hatch install` would have
        // populated this by shallow-cloning the remote. We fake it
        // directly so the test doesn't spawn git.
        let git_url = "https://example.invalid/alice/mylib.git";
        let checkout = crate::hatch_registry::cached_git_checkout_path(
            &cache_dir,
            git_url,
            GitRef::Tag("v1.2.3"),
        );
        std::fs::create_dir_all(&checkout).unwrap();
        std::fs::write(
            checkout.join("hatchfile"),
            "name = \"mylib\"\nversion = \"1.2.3\"\nentry = \"mylib\"\n",
        )
        .unwrap();
        std::fs::write(
            checkout.join("mylib.wren"),
            "class MyLib {\n  static answer { 42 }\n}",
        )
        .unwrap();

        // Consumer workspace pins the git dep.
        let app = scratch.path().join("app");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            app.join("main.wren"),
            "import \"mylib\" for MyLib\nSystem.print(MyLib.answer)",
        )
        .unwrap();
        std::fs::write(
            app.join("hatchfile"),
            format!(
                r#"name = "app"
version = "0.1.0"
entry = "main"

[dependencies]
mylib = {{ git = "{}", tag = "v1.2.3" }}
"#,
                git_url
            ),
        )
        .unwrap();

        let bytes = build_from_source_tree_with_cache(&app, Some(&cache_dir)).expect("build");

        let hatch = load(&bytes).unwrap();
        assert_eq!(hatch.manifest.modules, vec!["mylib", "main"]);
        assert!(!hatch.manifest.dependencies.contains_key("mylib"));

        let mut vm = crate::runtime::vm::VM::new_default();
        vm.output_buffer = Some(String::new());
        assert!(matches!(
            vm.interpret_hatch(&bytes),
            crate::runtime::engine::InterpretResult::Success
        ));
        assert_eq!(vm.take_output().trim(), "42");
    }

    #[test]
    fn build_reports_missing_cached_git_dep() {
        // A git dep with no prior `hatch install` must surface a
        // pointed error — no silent network I/O during build.
        let scratch = tempfile::tempdir().expect("tempdir");
        let cache_dir = scratch.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let workspace = scratch.path().join("app");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(workspace.join("main.wren"), "1").unwrap();
        std::fs::write(
            workspace.join("hatchfile"),
            r#"name = "app"
version = "0.1.0"
entry = "main"

[dependencies]
mylib = { git = "https://example.invalid/alice/mylib.git", tag = "v0.1.0" }
"#,
        )
        .unwrap();

        let result = build_from_source_tree_with_cache(&workspace, Some(&cache_dir));
        match result {
            Err(HatchError::Encode(msg)) => {
                assert!(msg.contains("mylib"));
                assert!(msg.contains("hatch install"));
            }
            other => panic!("expected install hint, got {:?}", other),
        }
    }

    #[test]
    fn build_reports_missing_cached_version() {
        // A version-pinned dep with nothing in the cache must surface
        // a pointed error — not silently succeed, not reach the
        // network during `hatch build`.
        let scratch = tempfile::tempdir().expect("tempdir");
        let cache_dir = scratch.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let workspace = scratch.path().join("app");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::write(workspace.join("main.wren"), "1").unwrap();
        std::fs::write(
            workspace.join("hatchfile"),
            r#"name = "app"
version = "0.1.0"
entry = "main"

[dependencies]
ghost = "9.9.9"
"#,
        )
        .unwrap();

        let result = build_from_source_tree_with_cache(&workspace, Some(&cache_dir));
        match result {
            Err(HatchError::Encode(msg)) => {
                assert!(msg.contains("ghost"));
                assert!(msg.contains("hatch install"));
            }
            other => panic!("expected install hint, got {:?}", other),
        }
    }

    #[test]
    fn build_rejects_dep_cycle() {
        // a/hatchfile depends on b, b/hatchfile depends on a — build
        // must bail out rather than spin.
        let tmp = tempfile::tempdir().expect("tempdir");
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("a.wren"), "1").unwrap();
        std::fs::write(b.join("b.wren"), "1").unwrap();
        std::fs::write(
            a.join("hatchfile"),
            "name = \"a\"\nversion = \"0\"\nentry = \"a\"\n[dependencies]\nb = { path = \"../b\" }\n",
        )
        .unwrap();
        std::fs::write(
            b.join("hatchfile"),
            "name = \"b\"\nversion = \"0\"\nentry = \"b\"\n[dependencies]\na = { path = \"../a\" }\n",
        )
        .unwrap();

        let result = build_from_source_tree(&a);
        assert!(matches!(result, Err(HatchError::Encode(msg)) if msg.contains("cycle")));
    }

    /// `a` depends on both `b` and `c`; `b` depends on `c`. The same
    /// path-dep `c` appears twice in the recursion — once via `b`,
    /// once direct from `a`. Used to error with "dependency cycle
    /// detected at .../c" because the visited HashSet didn't
    /// distinguish "currently in progress" from "already built". Now
    /// the second visit hits the build cache and reuses the bytes.
    #[test]
    fn build_accepts_diamond_dependencies() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        let c = tmp.path().join("c");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::create_dir_all(&c).unwrap();
        std::fs::write(a.join("a.wren"), "1").unwrap();
        std::fs::write(b.join("b.wren"), "1").unwrap();
        std::fs::write(c.join("c.wren"), "1").unwrap();
        std::fs::write(
            a.join("hatchfile"),
            "name = \"a\"\nversion = \"0\"\nentry = \"a\"\n[dependencies]\nb = { path = \"../b\" }\nc = { path = \"../c\" }\n",
        )
        .unwrap();
        std::fs::write(
            b.join("hatchfile"),
            "name = \"b\"\nversion = \"0\"\nentry = \"b\"\n[dependencies]\nc = { path = \"../c\" }\n",
        )
        .unwrap();
        std::fs::write(
            c.join("hatchfile"),
            "name = \"c\"\nversion = \"0\"\nentry = \"c\"\n",
        )
        .unwrap();

        let bytes = build_from_source_tree(&a).expect("diamond dep should build cleanly");
        let hatch = load(&bytes).expect("load");
        // `c` should appear once, regardless of arriving via `b` and
        // directly from `a`.
        let c_module_count = hatch.manifest.modules.iter().filter(|m| *m == "c").count();
        assert_eq!(
            c_module_count, 1,
            "c should be deduplicated, not duplicated"
        );
    }

    #[test]
    fn build_errors_when_declared_native_lib_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("main.wren"), "1").unwrap();
        let hatchfile = r#"
name = "pkg"
version = "0.1.0"
entry = "main"

[native_libs]
ghost = "libs/missing.bin"
"#;
        std::fs::write(root.join("hatchfile"), hatchfile).unwrap();

        let result = build_from_source_tree(root);
        assert!(matches!(result, Err(HatchError::Encode(_))));
    }

    #[test]
    fn native_lib_resolve_unknown_keys_are_ignored() {
        // A table with no recognized keys resolves to None — the
        // loader then leaves the library unbound rather than picking
        // some arbitrary entry.
        let text = r#"
[native_libs.mystery]
unsupported = "nope"
notes = "hello"
"#;
        let m: Manifest = toml::from_str(&format!(
            "name = \"x\"\nversion = \"0\"\nentry = \"m\"\n{}",
            text
        ))
        .expect("parse");
        assert_eq!(m.native_libs["mystery"].resolve_for("macos", "arm64"), None);
    }

    #[test]
    fn target_family_buckets_correctly() {
        assert_eq!(target_family("wasm32"), TargetFamily::Wasm32);
        assert_eq!(target_family("wasm32-unknown-unknown"), TargetFamily::Wasm32);
        assert_eq!(target_family("wasm32-wasip1"), TargetFamily::Wasm32);
        assert_eq!(target_family("x86_64-apple-darwin"), TargetFamily::Native);
        assert_eq!(target_family("aarch64-unknown-linux-gnu"), TargetFamily::Native);
        assert_eq!(target_family("native"), TargetFamily::Native);
    }

    #[test]
    fn check_target_compat_accepts_legacy_and_same_family() {
        // Legacy bundle (no target stamp) — accepted everywhere.
        assert!(check_target_compat(None, "native").is_ok());
        assert!(check_target_compat(None, "wasm32").is_ok());

        // Same family — concrete triple loadable on the family marker.
        assert!(check_target_compat(Some("wasm32-unknown-unknown"), "wasm32").is_ok());
        assert!(check_target_compat(Some("wasm32"), "wasm32-wasip1").is_ok());
        assert!(check_target_compat(Some("x86_64-apple-darwin"), "native").is_ok());

        // Cross-family — rejected.
        let err = check_target_compat(Some("wasm32-unknown-unknown"), "native")
            .expect_err("wasm bundle on native runtime should reject");
        assert!(matches!(err, HatchError::WrongTarget { .. }));
        let err = check_target_compat(Some("x86_64-apple-darwin"), "wasm32")
            .expect_err("native bundle on wasm runtime should reject");
        assert!(matches!(err, HatchError::WrongTarget { .. }));
    }

    #[test]
    fn build_for_wasm_target_strips_native_lib_sections_and_stamps_manifest() {
        // Reuse `build_packs_relative_native_libs_and_strips_manifest_entry`'s
        // setup but call the target-aware builder. The wasm path
        // should drop the NativeLib section that the host pack pass
        // would have produced.
        use std::io::Write;

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let lib_dir = root.join("libs");
        std::fs::create_dir(&lib_dir).expect("libs/");
        let mut f = std::fs::File::create(lib_dir.join("openssl.dylib")).expect("create dylib");
        f.write_all(b"FAKE_DYLIB").expect("write dylib");
        std::fs::write(
            root.join("hatchfile"),
            r#"
name = "wasm-target-test"
version = "0.1.0"
entry = "main"

[native_libs]
openssl = "libs/openssl.dylib"
"#,
        )
        .expect("hatchfile");
        std::fs::write(root.join("main.wren"), "System.print(\"hi\")").expect("main.wren");

        let bytes = build_from_source_tree_for_target(root, None, Some("wasm32-unknown-unknown"))
            .expect("build wasm-targeted hatch");
        let hatch = load(&bytes).expect("decode wasm hatch");

        assert_eq!(
            hatch.manifest.target.as_deref(),
            Some("wasm32-unknown-unknown"),
            "manifest should record the build target"
        );
        let has_native = hatch
            .sections
            .iter()
            .any(|s| matches!(s.kind, SectionKind::NativeLib));
        assert!(
            !has_native,
            "wasm target should not pack host-native dylib bytes"
        );
    }

    #[test]
    fn wasm_target_uses_uncompressed_payload_so_no_zstd_needed_at_load() {
        // The whole point of `EmitOptions::compress = false` for
        // wasm targets is that the wasm runtime (built without
        // `feature = "host"`) doesn't link zstd — `load()` would
        // otherwise hit the not-host branch and refuse to decode.
        // Verify the bundled bytes are uncompressed by checking
        // the FLAG_ZSTD bit at offset 9 in the header.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(
            root.join("hatchfile"),
            "name = \"u\"\nversion = \"0.1.0\"\nentry = \"main\"\n",
        )
        .unwrap();
        std::fs::write(root.join("main.wren"), "System.print(\"hi\")").unwrap();

        let bytes =
            build_from_source_tree_for_target(root, None, Some("wasm32-unknown-unknown"))
                .expect("build wasm-targeted");
        // Header layout (cf. format docs at top of file):
        //   bytes[0..5]  = MAGIC ("HATCH")
        //   bytes[5..9]  = VERSION (u32 LE)
        //   bytes[9]     = flags
        assert!(bytes.len() >= 16);
        let flags = bytes[9];
        assert_eq!(
            flags & FLAG_ZSTD,
            0,
            "wasm-targeted hatch must clear FLAG_ZSTD; got flags=0b{:08b}",
            flags
        );

        // Round-trip: load the bytes back and confirm the
        // manifest + Wlbc section made it through.
        let h = load(&bytes).expect("load uncompressed");
        assert_eq!(h.manifest.name, "u");
        assert_eq!(h.manifest.target.as_deref(), Some("wasm32-unknown-unknown"));
        assert!(h.sections.iter().any(|s| matches!(s.kind, SectionKind::Wlbc)));
    }

    #[test]
    fn cross_module_import_resolves_inside_a_hatch() {
        // Two-module hatch where `main` imports a class declared in
        // a sibling `util` module. install_hatch_sections has to
        // honour `manifest.modules` order so `util` is in
        // `engine.modules` by the time `main`'s top-level runs.
        // Same code path the wasm playground exercises through
        // `wlift_wasm::run_hatch`.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(
            root.join("hatchfile"),
            r#"
name = "two-module"
version = "0.1.0"
entry = "main"
modules = ["util", "main"]
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("util.wren"),
            "class Util { static greet(name) { return \"hi, %(name)\" } }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("main.wren"),
            r#"
import "util" for Util
System.print(Util.greet("hatch"))
"#,
        )
        .unwrap();

        let bytes = build_from_source_tree(root).expect("build");
        // Use the public VM install-and-run path so this test
        // exercises the same loader the playground / CLI hit.
        let mut vm = crate::runtime::vm::VM::new(crate::runtime::vm::VMConfig::default());
        vm.output_buffer = Some(String::new());
        let result = vm.interpret_hatch(&bytes);
        let out = vm.output_buffer.clone().unwrap_or_default();
        assert_eq!(
            result,
            crate::runtime::engine::InterpretResult::Success,
            "interpret_hatch should succeed: output={:?}",
            out
        );
        assert!(
            out.contains("hi, hatch"),
            "expected cross-module Util.greet output; got {:?}",
            out
        );
    }

    #[test]
    fn install_target_mismatch_returns_wrong_target() {
        // A manifest stamped with a wasm target should be rejected
        // by `check_target_compat` against a "native" runtime.
        let mut hatch = sample();
        hatch.manifest.target = Some("wasm32-unknown-unknown".to_string());
        let bytes = emit(&hatch).expect("encode");
        let decoded = load(&bytes).expect("decode");
        assert_eq!(decoded.manifest.target.as_deref(), Some("wasm32-unknown-unknown"));
        let err = check_target_compat(decoded.manifest.target.as_deref(), "native")
            .expect_err("cross-family should reject");
        match err {
            HatchError::WrongTarget { bundle, runtime } => {
                assert_eq!(bundle, "wasm32-unknown-unknown");
                assert_eq!(runtime, "native");
            }
            e => panic!("unexpected error: {:?}", e),
        }
    }
}
