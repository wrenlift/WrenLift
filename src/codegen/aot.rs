//! AOT object-file emission — Cranelift's `ObjectModule` swap-in
//! for the production JIT path.
//!
//! Drives the same parser → sema → MIR → CLIF translator the JIT
//! uses, but hands the lowered function to a `cranelift_object::
//! ObjectModule` (writes a `.o` to disk) instead of the
//! `cranelift_jit::JITModule` (mutates executable memory in
//! place). The shared translator lives in
//! `cranelift_backend::cl::lower_mir_to_module` — see Phase-2 of
//! the AOT plan: `cranelift_module::Module` is dyn-safe, so JIT
//! and AOT share the per-instruction lowering as a `&mut dyn
//! Module` parameter without cloning the codegen.
//!
//! Today's surface compiles ONE function — the entry point of
//! the parsed Wren source. Phase 3 walks the whole import graph
//! and links every reachable module's lowered MIR into a single
//! object file. Phase 4 hands the object to a system linker
//! alongside the runtime `staticlib` to land a self-contained
//! executable.
//!
//! IC pointers, code-base, and OSR-entry signals are JIT-only —
//! they patch self-call sites, on-stack replacement entries, and
//! inline-cache snapshots into mutable JIT memory. AOT outputs
//! go through a static linker, so we pass `None` for all three
//! and emit a clean object instead.
//!
//! Behind `feature = "aot"` so the production crate stays
//! untouched. Pulled in via `cargo build --features aot` or by
//! the `wlift build` driver once that lands.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use cranelift_codegen::ir::{types, AbiParam, Function, Signature, UserFuncName};
use cranelift_codegen::isa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ast::Stmt;
use crate::codegen::cranelift_backend::cl::{
    lower_mir_to_module, AotCha, AotDefiningClass, AotLoweringConfig, AotMethodImpl,
};
use crate::intern::Interner;
use crate::mir::builder::lower_module_with_known_classes;
use crate::mir::ModuleMir;
use crate::parse::parser::parse;
use crate::sema::resolve::resolve_with_prelude;

/// Errors the AOT pipeline can surface up to the CLI.
#[derive(Debug)]
pub enum AotError {
    /// The host triple isn't one Cranelift can target — should
    /// never fire on a supported runtime host but surfaced as a
    /// real error so the CLI can print a useful message instead
    /// of panicking.
    UnsupportedTarget(String),
    /// Cranelift couldn't construct an ISA for the resolved
    /// triple (e.g. missing CPU feature, unknown register set).
    Isa(String),
    /// CLIF lowering / object-table operation failed.
    Module(String),
    /// `std::fs::write` couldn't write the object bytes.
    Io(std::io::Error),
    /// Wren parse / sema / MIR-build error chain. Joined into a
    /// single string at the AOT boundary so the CLI's
    /// error-reporting path doesn't need to crack open each
    /// frontend's diagnostic shape.
    Frontend(String),
}

impl std::fmt::Display for AotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AotError::UnsupportedTarget(t) => write!(f, "unsupported target triple: {}", t),
            AotError::Isa(m) => write!(f, "cranelift ISA setup failed: {}", m),
            AotError::Module(m) => write!(f, "cranelift module operation failed: {}", m),
            AotError::Io(e) => write!(f, "writing object file: {}", e),
            AotError::Frontend(m) => write!(f, "wren frontend: {}", m),
        }
    }
}

impl std::error::Error for AotError {}

/// One module visited by the AOT walker.
///
/// Carries the canonical name (the absolute on-disk path for
/// relative imports — same shape `make_module_io` in the CLI driver
/// uses), the lowered MIR, and the per-parse `Interner` whose
/// `SymbolId`s appear inside `mir`. Each parse mints its own
/// interner; downstream lowering passes keep them paired so symbol
/// identity stays consistent across the module-by-module emit.
pub struct AotModule {
    // `name` is informational — kept around so emit-time errors
    // can attribute failures to a specific module path. The
    // emitted symbols use a separate, deterministic naming scheme
    // (`wlift_aot_main` / `wlift_aot_mod_<n>`) to avoid leaking
    // local filesystem paths into the produced object.
    #[allow(dead_code)]
    pub name: String,
    /// Module name the runtime should register this source under
    /// when the AOT bootstrap installs it — the `import "./foo"`
    /// path string from the FIRST importer that pulled this
    /// module in (or `"main"` for the entry). The runtime resolves
    /// imports by name match, so the bootstrap and the runtime
    /// have to agree: bundle a module under the exact spelling
    /// the source's `import` statement uses.
    pub request_name: String,
    /// Raw source bytes, for the bootstrap to embed into the
    /// produced executable. Read once during the walk so a later
    /// edit to the source file doesn't desync the embedded copy
    /// from the lowered MIR.
    pub source: Vec<u8>,
    /// Total module-var slot count (prelude + user-declared). Drives
    /// the per-module `wlift_modvars_<n>` data symbol's size — every
    /// `GetModuleVar(slot)` / `SetModuleVar(slot, _)` in this
    /// module's MIR addresses an offset inside that array, so the
    /// AOT emitter has to declare it `module_var_count * 8` bytes
    /// wide.
    pub module_var_count: usize,
    /// Resolved name per module-var slot. Position in the vec =
    /// slot index in `wlift_modvars_<n>`. The class-install pass
    /// looks up each class's `name` here to find which slot the
    /// installed `*mut ObjClass` should land in.
    pub module_var_names: Vec<String>,
    /// Per-slot import source — `Some(path)` when the slot was
    /// declared via `import "<path>" for <name>`, `None` for
    /// locally-defined or prelude vars. The bootstrap reads this
    /// to know which dependency's modvars to copy from at
    /// startup, replicating the cross-module class-binding the
    /// JIT install loop does inline.
    pub module_var_sources: Vec<Option<String>>,
    pub mir: ModuleMir,
    pub interner: Interner,
}

/// Append `.wren` to a relative-import path **without stripping
/// existing extensions**. Mirrors `with_wren_suffix` in `main.rs` —
/// `Path::with_extension` would replace `.spec` so `import "./x.spec"`
/// would silently redirect to `x.wren`.
fn with_wren_suffix(path: PathBuf) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("wren") {
        return path;
    }
    let mut s = path.into_os_string();
    s.push(".wren");
    PathBuf::from(s)
}

/// Walk the import graph rooted at `entry_path` and return one
/// `AotModule` per reachable on-disk module, in dependency-first
/// order (a module appears after every module it imports — the
/// shape Phase 4 wants when it lowers each in turn into a single
/// `ObjectModule`).
///
/// Reuses the runtime's pipeline verbatim (`parse` →
/// `resolve_with_prelude` → `lower_module_with_known_classes`) so
/// the per-module IR is bit-identical to what the JIT would build
/// on first load. Cross-module field layouts thread through via
/// the same `field_layouts` map `VM::interpret` keeps as state.
///
/// Scoped (`@hatch:foo`) and bare-builtin imports are skipped:
/// they're runtime-resolved today and have no on-disk source to
/// recurse into. Phase 4's link step decides whether to bundle
/// them, surface them as object-file imports, or assume the
/// runtime staticlib provides them.
/// Walk-phase output: every reachable module's MIR plus the
/// bundle-level metadata the bootstrap needs (native search
/// paths, bundled native lib payloads). Driver hands both to
/// `compile_modules_to_object_with_manifest_and_meta`.
pub struct AotWalkResult {
    pub modules: Vec<AotModule>,
    pub bundle: AotBundleMeta,
}

pub fn walk_imports(entry_path: &Path) -> Result<AotWalkResult, AotError> {
    // `.hatch` archive: every `Source` section becomes an
    // `AotModule`, ordered by the manifest's `modules` list with
    // `entry` last. Source-only — `Wlbc`-only modules need wlbc-
    // to-MIR deserialisation, which lands separately.
    let raw_bytes = std::fs::read(entry_path).map_err(AotError::Io)?;
    if crate::hatch::looks_like_hatch(&raw_bytes) {
        return walk_hatch_archive(&raw_bytes);
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut out: Vec<AotModule> = Vec::new();
    let mut field_layouts: HashMap<String, Vec<String>> = HashMap::new();

    let entry_canonical = std::fs::canonicalize(entry_path)
        .map_err(AotError::Io)?
        .to_string_lossy()
        .into_owned();
    let entry_source_bytes = raw_bytes;

    // Build a scoped-import resolver from the entry's hatchfile
    // (if any). Each entry maps an import name like
    // `"@hatch:fmt"` to the absolute path of the dep's entry
    // `.wren` file. Path-linked deps only — registry / git
    // archive deps need `.hatch` cracking, which lands later.
    let scoped_resolver = build_scoped_resolver(entry_path);

    walk_module(
        &entry_canonical,
        "main",
        &entry_source_bytes,
        &mut visited,
        &mut out,
        &mut field_layouts,
        &scoped_resolver,
    )?;

    // Native-lib search paths: collect from the entry's hatchfile
    // and every path-linked dep's hatchfile. Resolved against the
    // hatchfile's own directory so a dep declaring
    // `native_search_paths = ["libs"]` lands as
    // `<dep_dir>/libs`.
    let bundle = AotBundleMeta {
        native_search_paths: collect_native_search_paths(entry_path),
        native_libs: Vec::new(),
    };
    Ok(AotWalkResult {
        modules: out,
        bundle,
    })
}

fn collect_native_search_paths(entry_path: &Path) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let entry_dir = entry_path.parent().unwrap_or(Path::new(".")).to_path_buf();

    // Find the entry's hatchfile (walk up).
    let hatchfile = {
        let mut cursor = Some(entry_dir.clone());
        loop {
            match cursor {
                Some(d) => {
                    let candidate = d.join("hatchfile");
                    if candidate.exists() {
                        break Some(candidate);
                    }
                    cursor = d.parent().map(Path::to_path_buf);
                }
                None => break None,
            }
        }
    };

    let push_resolved = |out: &mut Vec<String>, root: &Path, rel: &str| {
        let p = if Path::new(rel).is_absolute() {
            std::path::PathBuf::from(rel)
        } else {
            root.join(rel)
        };
        if let Some(s) = p.to_str() {
            if !out.iter().any(|existing| existing == s) {
                out.push(s.to_string());
            }
        }
    };

    let Some(hatchfile) = hatchfile else {
        return out;
    };
    let workspace_root = hatchfile.parent().unwrap_or(Path::new("."));

    if let Ok(text) = std::fs::read_to_string(&hatchfile) {
        if let Ok(manifest) = toml::from_str::<crate::hatch::Manifest>(&text) {
            for path in &manifest.native_search_paths {
                push_resolved(&mut out, workspace_root, path);
            }
            // Recurse into path-link deps' hatchfiles to pull
            // their `native_search_paths` too — mirrors the
            // runtime's `apply_hatch_native_manifest_rooted` walk.
            for dep in manifest
                .dependencies
                .values()
                .chain(manifest.spec_dependencies.values())
            {
                let crate::hatch::Dependency::Path { path, .. } = dep else {
                    continue;
                };
                let dep_dir = workspace_root.join(path);
                let dep_hatchfile = dep_dir.join("hatchfile");
                let Ok(dep_text) = std::fs::read_to_string(&dep_hatchfile) else {
                    continue;
                };
                let Ok(dep_manifest) = toml::from_str::<crate::hatch::Manifest>(&dep_text) else {
                    continue;
                };
                for sp in &dep_manifest.native_search_paths {
                    push_resolved(&mut out, &dep_dir, sp);
                }
            }
        }
    }
    out
}

/// Walk a `.hatch` archive: parse via `hatch::load`, take every
/// `SectionKind::Source` payload, parse + sema + MIR-build each
/// in `manifest.modules` order (entry last), and return the
/// resulting `AotModule` list ready for `compile_modules_to_object`.
///
/// `Wlbc`-only modules in the archive are skipped — wlbc-to-MIR
/// deserialisation is a separate codepath; the archive is only
/// AOT-compilable when it includes `Source` sections (the default
/// `hatch build`-produced shape).
fn walk_hatch_archive(bytes: &[u8]) -> Result<AotWalkResult, AotError> {
    use std::collections::HashMap as Map;

    let hatch = crate::hatch::load(bytes)
        .map_err(|e| AotError::Frontend(format!("loading hatch archive: {}", e)))?;

    // Collect bundled native libs (NativeLib sections) — bytes
    // get extracted to a temp file at startup by the bootstrap.
    let mut native_libs: Vec<(String, Vec<u8>)> = Vec::new();
    for section in &hatch.sections {
        if matches!(section.kind, crate::hatch::SectionKind::NativeLib) {
            native_libs.push((section.name.clone(), section.data.clone()));
        }
    }

    // Build a name → source map from Source sections so manifest
    // order can drive the walk.
    let mut sources: Map<String, &[u8]> = Map::new();
    for section in &hatch.sections {
        if matches!(section.kind, crate::hatch::SectionKind::Source) {
            sources.insert(section.name.clone(), section.data.as_slice());
        }
    }

    if sources.is_empty() {
        return Err(AotError::Frontend(format!(
            "hatch archive '{}' has no Source sections (Wlbc-only \
             archives need wlbc-to-MIR deserialisation, not yet wired \
             into AOT)",
            hatch.manifest.name
        )));
    }

    // Order: dependency-first via `manifest.modules`, with the
    // archive's `entry` module appearing last so the bootstrap
    // dispatches it last (matches the .wren walker convention).
    let mut order: Vec<String> = hatch
        .manifest
        .modules
        .iter()
        .filter(|n| sources.contains_key(*n))
        .cloned()
        .collect();
    if order.is_empty() {
        // Manifest didn't list modules — fall back to alphabetical.
        order.extend(sources.keys().cloned());
    }
    let entry = hatch.manifest.entry.clone();
    order.retain(|n| n != &entry);
    if sources.contains_key(&entry) {
        order.push(entry);
    }

    let mut out: Vec<AotModule> = Vec::with_capacity(order.len());
    let mut field_layouts: HashMap<String, Vec<String>> = HashMap::new();
    for name in &order {
        let Some(source_bytes) = sources.get(name) else {
            continue;
        };
        out.push(build_aot_module_from_source(
            name,
            source_bytes,
            &mut field_layouts,
        )?);
    }
    Ok(AotWalkResult {
        modules: out,
        bundle: AotBundleMeta {
            native_search_paths: hatch.manifest.native_search_paths.clone(),
            native_libs,
        },
    })
}

/// Parse → sema → MIR-build a single in-memory source. Same
/// pipeline `walk_module` runs per `.wren` file, factored out so
/// the `.hatch`-archive walker can reuse it without recursing
/// into relative imports.
fn build_aot_module_from_source(
    request_name: &str,
    source_bytes: &[u8],
    field_layouts: &mut HashMap<String, Vec<String>>,
) -> Result<AotModule, AotError> {
    let source = std::str::from_utf8(source_bytes)
        .map_err(|e| AotError::Frontend(format!("{} is not valid UTF-8: {}", request_name, e)))?;
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        return Err(AotError::Frontend(
            parsed
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let mut interner: Interner = parsed.interner;
    let prelude_syms: Vec<crate::intern::SymbolId> = crate::sema::PRELUDE_NAMES
        .iter()
        .map(|n| interner.intern(n))
        .collect();
    let resolved = resolve_with_prelude(&parsed.module, &interner, &prelude_syms);
    if !resolved.errors.is_empty() {
        return Err(AotError::Frontend(
            resolved
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let module_var_count = resolved.module_vars.len();
    let module_var_names: Vec<String> = resolved
        .module_vars
        .iter()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect();
    let module_var_sources = resolved.module_var_sources.clone();
    let (module_mir, new_layouts) =
        lower_module_with_known_classes(&parsed.module, &mut interner, &resolved, field_layouts);
    for (k, v) in new_layouts {
        field_layouts.insert(k, v);
    }

    Ok(AotModule {
        name: request_name.to_string(),
        request_name: request_name.to_string(),
        source: source_bytes.to_vec(),
        module_var_count,
        module_var_names,
        module_var_sources,
        mir: module_mir,
        interner,
    })
}

/// Map from scoped import name (e.g. `"@hatch:fmt"`) to the
/// absolute path of the dep's entry `.wren` file. Built once at
/// walk start from the entry's hatchfile.
type ScopedResolver = HashMap<String, PathBuf>;

fn build_scoped_resolver(entry_path: &Path) -> ScopedResolver {
    let mut map = HashMap::new();
    let entry_dir = entry_path.parent().unwrap_or(Path::new(".")).to_path_buf();

    // Walk up from the entry's dir looking for a hatchfile.
    let mut cursor = Some(entry_dir);
    let hatchfile = loop {
        match cursor {
            Some(d) => {
                let candidate = d.join("hatchfile");
                if candidate.exists() {
                    break Some(candidate);
                }
                cursor = d.parent().map(Path::to_path_buf);
            }
            None => break None,
        }
    };

    let Some(hatchfile) = hatchfile else {
        return map;
    };
    let workspace_root = match hatchfile.parent() {
        Some(p) => p.to_path_buf(),
        None => return map,
    };

    let text = match std::fs::read_to_string(&hatchfile) {
        Ok(t) => t,
        Err(_) => return map,
    };
    let manifest: crate::hatch::Manifest = match toml::from_str(&text) {
        Ok(m) => m,
        Err(_) => return map,
    };

    // For every path-linked dep, read its own hatchfile to find
    // its `[package].entry` module name → resolve to a `.wren`
    // file under the dep's directory.
    for (name, dep) in manifest
        .dependencies
        .iter()
        .chain(manifest.spec_dependencies.iter())
    {
        let crate::hatch::Dependency::Path { path, .. } = dep else {
            continue;
        };
        let dep_dir = workspace_root.join(path);
        let dep_hatchfile = dep_dir.join("hatchfile");
        let Ok(dep_text) = std::fs::read_to_string(&dep_hatchfile) else {
            continue;
        };
        let dep_manifest: crate::hatch::Manifest = match toml::from_str(&dep_text) {
            Ok(m) => m,
            Err(_) => continue,
        };
        // `entry` is a module name (e.g. `"fmt"`). The `.wren`
        // file lives at `<dep_dir>/<entry>.wren`.
        let entry_file = dep_dir.join(format!("{}.wren", dep_manifest.entry));
        if entry_file.exists() {
            map.insert(name.clone(), entry_file);
        }
    }
    map
}

fn walk_module(
    canonical_name: &str,
    request_name: &str,
    source_bytes: &[u8],
    visited: &mut HashSet<String>,
    out: &mut Vec<AotModule>,
    field_layouts: &mut HashMap<String, Vec<String>>,
    scoped_resolver: &ScopedResolver,
) -> Result<(), AotError> {
    if !visited.insert(canonical_name.to_string()) {
        return Ok(());
    }

    let source = std::str::from_utf8(source_bytes)
        .map_err(|e| AotError::Frontend(format!("{} is not valid UTF-8: {}", canonical_name, e)))?;
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        return Err(AotError::Frontend(
            parsed
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let mut interner: Interner = parsed.interner;
    let prelude_syms: Vec<crate::intern::SymbolId> = crate::sema::PRELUDE_NAMES
        .iter()
        .map(|n| interner.intern(n))
        .collect();
    let resolved = resolve_with_prelude(&parsed.module, &interner, &prelude_syms);
    if !resolved.errors.is_empty() {
        return Err(AotError::Frontend(
            resolved
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    // Recurse into relative imports first — depth-first ordering
    // gives us dependency-before-importer in `out`, which matches
    // the order Phase 4 wants when feeding modules into the
    // shared `ObjectModule`.
    let module_dir = Path::new(canonical_name).parent().map(Path::to_path_buf);
    for stmt in &parsed.module {
        let Stmt::Import {
            module: spanned, ..
        } = &stmt.0
        else {
            continue;
        };
        let req = &spanned.0;
        let candidate: PathBuf = if req.starts_with("./") || req.starts_with("../") {
            // Relative import: resolve against the importer's
            // directory, fall through to the read+walk below.
            let Some(base_dir) = module_dir.as_ref() else {
                continue;
            };
            with_wren_suffix(base_dir.join(Path::new(req)))
        } else if let Some(scoped_path) = scoped_resolver.get(req) {
            // Scoped import (`@hatch:foo`) resolved via the
            // entry's hatchfile [dependencies] table.
            scoped_path.clone()
        } else {
            // Unresolved scoped/bare import — leaves the slot
            // null and the AOT body will surface the error at
            // first use. Same behaviour as the runtime when an
            // import goes unresolved.
            continue;
        };
        let imported_canonical = std::fs::canonicalize(&candidate)
            .map_err(AotError::Io)?
            .to_string_lossy()
            .into_owned();
        if visited.contains(&imported_canonical) {
            continue;
        }
        let imported_bytes = std::fs::read(&candidate).map_err(AotError::Io)?;
        walk_module(
            &imported_canonical,
            req,
            &imported_bytes,
            visited,
            out,
            field_layouts,
            scoped_resolver,
        )?;
    }

    let module_var_count = resolved.module_vars.len();
    let module_var_names: Vec<String> = resolved
        .module_vars
        .iter()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect();
    let module_var_sources = resolved.module_var_sources.clone();
    let (module_mir, new_layouts) =
        lower_module_with_known_classes(&parsed.module, &mut interner, &resolved, field_layouts);
    for (k, v) in new_layouts {
        field_layouts.insert(k, v);
    }

    out.push(AotModule {
        name: canonical_name.to_string(),
        request_name: request_name.to_string(),
        source: source_bytes.to_vec(),
        module_var_count,
        module_var_names,
        module_var_sources,
        mir: module_mir,
        interner,
    });
    Ok(())
}

/// Build a host-targeted `ObjectModule` with the AOT pipeline's
/// settings (PIC + speed). Shared by both single-source and
/// path-walking entry points.
fn make_object_module() -> Result<ObjectModule, AotError> {
    let triple = target_lexicon::Triple::host();

    // ISA settings.
    //
    // `is_pic = true` so the produced object can be linked into a
    // shared library or PIE without runtime relocation issues —
    // distribution-friendly default.
    //
    // `opt_level = "speed"` runs Cranelift's full optimisation
    // pipeline (alternatives: `none`/`speed_and_size`).
    //
    // `enable_alias_analysis = true` keeps load/store dedup
    // working — currently the default, called out so a future
    // Cranelift change can't silently regress it.
    //
    // `enable_probestack = false` mirrors the JIT path — macOS
    // aarch64's inline probestack fires false SIGSEGVs on
    // legitimately-deep stacks.
    //
    // `enable_verifier = true` is cheap on AOT (single-shot
    // codegen, not a hot loop) and catches a class of CLIF bugs
    // before they ship as miscompiles.
    //
    // `unwind_info = true` so the linker emits .eh_frame /
    // compact-unwind sections — backtraces from inside AOT'd
    // code show real frames instead of stopping at `_main`.
    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "true")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    flag_builder
        .set("enable_alias_analysis", "true")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    flag_builder
        .set("enable_probestack", "false")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    flag_builder
        .set("enable_verifier", "true")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    flag_builder
        .set("unwind_info", "true")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    let flags = settings::Flags::new(flag_builder);

    let isa_builder =
        isa::lookup(triple.clone()).map_err(|_| AotError::UnsupportedTarget(triple.to_string()))?;
    let isa = isa_builder
        .finish(flags)
        .map_err(|e| AotError::Isa(e.to_string()))?;

    // Object module: the JIT swap-in. Same `Module` trait, same
    // `define_function`-shaped API the future shared codegen
    // uses; only the backing storage differs.
    let object_builder = ObjectBuilder::new(
        isa,
        b"wlift_aot".to_vec(),
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| AotError::Module(e.to_string()))?;
    Ok(ObjectModule::new(object_builder))
}

/// Declare + lower one Wren module's top-level into the shared
/// `ObjectModule` under `fn_symbol`, with module-var reads/writes
/// rerouted through the per-module `modvars_symbol` data array.
///
/// Top-level fn: every Wren value is NaN-boxed as a `u64` —
/// signature is `(arity × i64) -> i64`. The AOT lowering emits
/// `GetModuleVar(slot)` / `SetModuleVar(slot, _)` as direct
/// `global_value` + `load`/`store` against `modvars_symbol`,
/// killing the runtime helper call entirely. The data symbol is
/// declared up front so the lowering can resolve its `DataId`
/// without reaching back into the driver.
/// Lower one MIR function under the active AOT config and define
/// it under `symbol`. Shared by top-level, every class method,
/// and every closure body — they all use the same per-module
/// `modvars` / `consts` data symbols (the lowering branches keyed
/// off `aot_cfg`), so factoring this out keeps the per-function
/// loop in `emit_aot_module` from re-declaring data per emission.
/// Per-function metadata captured during emit. Fed back into the
/// bootstrap so each AOT function's code range gets registered with
/// the engine alongside its safepoint live-roots — without that,
/// the GC stack walker can't find AOT frames and any GC fired
/// from a runtime helper sweeps live spilled values.
struct EmittedFnMeta {
    code_size: u32,
    /// MIR arity (including receiver). Carried through so the
    /// bootstrap can re-import the function symbol with the same
    /// signature `emit_aot_function` declared — Cranelift errors
    /// `declare_function` on mismatched signatures, even when we
    /// only want the address via `func_addr`.
    arity: u8,
    /// `None` when Cranelift emitted no user stack maps for this
    /// function (e.g. a top-level body that never calls back into
    /// runtime helpers). The bootstrap still registers the code
    /// range so frame walks find it; the metadata's safepoints
    /// list is just empty in that case.
    native_meta: Option<crate::codegen::native_meta::NativeFrameMetadata>,
}

fn emit_aot_function(
    module: &mut ObjectModule,
    interner: &Interner,
    mir: &crate::mir::MirFunction,
    aot_cfg: &AotLoweringConfig,
    symbol: &str,
    defining_class: Option<AotDefiningClass>,
) -> Result<EmittedFnMeta, AotError> {
    let mut sig = Signature::new(module.target_config().default_call_conv);
    let arity = mir.arity as usize;
    for _ in 0..arity {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));

    let func_id = module
        .declare_function(symbol, Linkage::Export, &sig)
        .map_err(|e| AotError::Module(e.to_string()))?;

    *aot_cfg.current_defining_class.borrow_mut() = defining_class;
    *aot_cfg.current_closure_ptr_var.borrow_mut() = None;

    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        lower_mir_to_module(mir, interner, &mut builder, module, Some(aot_cfg))
            .map_err(AotError::Module)?;
        builder.seal_all_blocks();
        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| AotError::Module(e.to_string()))?;

    // Capture safepoint metadata + code size before clearing the
    // context. Same path the JIT's `compile_mir` uses — the
    // `CompiledCode` is only valid until `clear_context` so we
    // pull what we need first.
    let compiled = ctx.compiled_code().expect("compiled_code post define_function");
    let code_size = compiled.code_info().total_size;
    let native_meta =
        crate::codegen::cranelift_backend::cl::native_meta_from_cranelift(compiled);

    module.clear_context(&mut ctx);

    *aot_cfg.current_defining_class.borrow_mut() = None;
    *aot_cfg.current_closure_ptr_var.borrow_mut() = None;
    Ok(EmittedFnMeta {
        code_size,
        arity: mir.arity,
        native_meta,
    })
}

/// Lower every reachable function in `aot_mod`'s MIR — top-level,
/// every class's methods, and every closure body — into the
/// shared `ObjectModule`. Each function gets its own exported
/// symbol; all of them share the per-module `wlift_modvars_<n>`
/// and `wlift_consts_<n>` data symbols declared here.
///
/// Symbol layout (per module):
/// - `<top_level_symbol>` — the top-level body
/// - `<top_level_symbol>__method_<class_idx>_<method_idx>` — class methods
/// - `<top_level_symbol>__closure_<closure_idx>` — closure bodies
///
/// The `__` separator keeps them grep-friendly inside `nm` output
/// and avoids colliding with whatever names the source itself uses.
struct EmitAotResult {
    const_texts: Vec<String>,
    symbol_names: Vec<String>,
    classes: Vec<AotClassManifest>,
    closures: Vec<AotClosureManifest>,
    /// Per-function code-range + safepoint metadata, keyed by the
    /// emitted symbol name. Bootstrap reads this to call
    /// `wlift_aot_register_code_range(fn_ptr, code_size, safepoints)`
    /// at startup so the GC stack walker can find AOT frames.
    fn_metas: Vec<(String, EmittedFnMeta)>,
}

#[derive(Debug, Clone)]
pub struct AotClosureManifest {
    pub fn_symbol: String,
    pub arity: u8,
}

fn emit_aot_module(
    module: &mut ObjectModule,
    aot_mod: &AotModule,
    fn_symbol: &str,
    modvars_symbol: &str,
    consts_symbol: &str,
    symbols_symbol: &str,
    cha: &AotCha,
) -> Result<EmitAotResult, AotError> {
    // Per-module .bss for module vars. `var_count == 0` is rare
    // (a module that defines no top-level names — pure imports
    // wouldn't emit any GetModuleVar / SetModuleVar) but the
    // emit must still produce a valid 0-length placeholder so
    // the `DataId` resolves cleanly.
    let var_bytes = aot_mod.module_var_count.saturating_mul(8).max(8);
    let modvars_data = module
        .declare_data(modvars_symbol, Linkage::Export, true, false)
        .map_err(|e| AotError::Module(e.to_string()))?;
    let mut data_desc = DataDescription::new();
    data_desc.define_zeroinit(var_bytes);
    module
        .define_data(modvars_data, &data_desc)
        .map_err(|e| AotError::Module(e.to_string()))?;

    // Per-module string-constant slot table. Declared up front so
    // the lowering can resolve the `DataId`; sized + defined after
    // every function in the module has been lowered.
    let consts_data = module
        .declare_data(consts_symbol, Linkage::Export, true, false)
        .map_err(|e| AotError::Module(e.to_string()))?;

    // Per-module symbol-remap slot table. Same shape — declared
    // up front, sized + defined post-lowering.
    let symbols_data = module
        .declare_data(symbols_symbol, Linkage::Export, true, false)
        .map_err(|e| AotError::Module(e.to_string()))?;

    // Per-module closure FuncId slot table. One slot per closure
    // in `aot_mod.mir.closures`, populated at startup by
    // `wlift_aot_register_closure` calls.
    let closures_symbol = format!("{}__closures_data", fn_symbol);
    let closures_data = module
        .declare_data(&closures_symbol, Linkage::Export, true, false)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let aot_cfg = AotLoweringConfig {
        modvars_data,
        consts_data,
        const_strings: std::cell::RefCell::new(Vec::new()),
        symbols_data,
        symbol_remap: std::cell::RefCell::new(Vec::new()),
        closures_data,
        cha: Some(cha as *const AotCha),
        current_defining_class: std::cell::RefCell::new(None),
        current_closure_ptr_var: std::cell::RefCell::new(None),
    };

    let mut fn_metas: Vec<(String, EmittedFnMeta)> = Vec::new();

    // Top-level body — the entry point Phase 7's init pass calls
    // last, after every dependency module's top-level has run.
    let top_meta = emit_aot_function(
        module,
        &aot_mod.interner,
        &aot_mod.mir.top_level,
        &aot_cfg,
        fn_symbol,
        None,
    )?;
    fn_metas.push((fn_symbol.to_string(), top_meta));

    // Each class's methods. Methods share the module's `modvars`
    // and `consts` symbols. Per-method symbols flow back into
    // the class manifest the bootstrap consumes to install the
    // class via `wlift_aot_install_class` — same FuncId-shaped
    // dispatch as the JIT install path, but the method bodies
    // are AOT-emitted symbols instead of JIT-tier-up MIR.
    let mut classes_manifest: Vec<AotClassManifest> = Vec::with_capacity(aot_mod.mir.classes.len());
    for (class_idx, class) in aot_mod.mir.classes.iter().enumerate() {
        let class_name = aot_mod.interner.resolve(class.name).to_string();
        let parent_slot = class.superclass.and_then(|sym| {
            let parent_name = aot_mod.interner.resolve(sym);
            aot_mod
                .module_var_names
                .iter()
                .position(|n| n == parent_name)
                .map(|s| s as u32)
        });
        let slot = aot_mod
            .module_var_names
            .iter()
            .position(|n| n == &class_name)
            .unwrap_or(usize::MAX);

        let mut methods_manifest: Vec<AotMethodManifest> = Vec::with_capacity(class.methods.len());
        let defining_class_for_methods = if slot != usize::MAX {
            Some(AotDefiningClass {
                modvars_symbol: modvars_symbol.to_string(),
                slot: slot as u32,
            })
        } else {
            None
        };
        for (method_idx, method) in class.methods.iter().enumerate() {
            let sym = format!("{}__method_{}_{}", fn_symbol, class_idx, method_idx);
            let meta = emit_aot_function(
                module,
                &aot_mod.interner,
                &method.mir,
                &aot_cfg,
                &sym,
                defining_class_for_methods.clone(),
            )?;
            fn_metas.push((sym.clone(), meta));
            methods_manifest.push(AotMethodManifest {
                signature: method.signature.clone(),
                fn_symbol: sym,
                arity: method.mir.arity,
                is_static: method.is_static,
                is_constructor: method.is_constructor,
            });
        }

        let foreign_methods: Vec<AotForeignMethodManifest> = class
            .foreign_methods
            .iter()
            .map(|fm| AotForeignMethodManifest {
                signature: fm.signature.clone(),
                symbol: fm.symbol.clone(),
                is_static: fm.is_static,
            })
            .collect();

        if slot != usize::MAX {
            classes_manifest.push(AotClassManifest {
                name: class_name,
                parent_slot,
                num_fields: class.num_fields,
                slot: slot as u32,
                methods: methods_manifest,
                foreign_library: class.native_library.clone(),
                foreign_methods,
            });
        }
    }

    // Closure bodies — heap-allocated at runtime; the bootstrap
    // registers each one via `wlift_aot_register_closure` to
    // claim a runtime FuncId, populating the per-module slot
    // table the AOT body's MakeClosure reads against.
    let mut closure_manifest: Vec<AotClosureManifest> =
        Vec::with_capacity(aot_mod.mir.closures.len());
    for (closure_idx, closure_mir) in aot_mod.mir.closures.iter().enumerate() {
        let sym = format!("{}__closure_{}", fn_symbol, closure_idx);
        let meta =
            emit_aot_function(module, &aot_mod.interner, closure_mir, &aot_cfg, &sym, None)?;
        fn_metas.push((sym.clone(), meta));
        closure_manifest.push(AotClosureManifest {
            fn_symbol: sym,
            arity: closure_mir.arity,
        });
    }

    // Define the consts + symbols + closures slot arrays now that
    // we know how many entries each holds. Zero-fill — populated
    // at startup by the per-module init pass.
    let const_strings = aot_cfg.const_strings.into_inner();
    let const_count = const_strings.len();
    let const_bytes = const_count.saturating_mul(8).max(8);
    let mut consts_desc = DataDescription::new();
    consts_desc.define_zeroinit(const_bytes);
    module
        .define_data(consts_data, &consts_desc)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let symbol_remap = aot_cfg.symbol_remap.into_inner();
    let sym_count = symbol_remap.len();
    let sym_bytes = sym_count.saturating_mul(8).max(8);
    let mut sym_desc = DataDescription::new();
    sym_desc.define_zeroinit(sym_bytes);
    module
        .define_data(symbols_data, &sym_desc)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let closures_count = closure_manifest.len();
    let closures_bytes = closures_count.saturating_mul(8).max(8);
    let mut closures_desc = DataDescription::new();
    closures_desc.define_zeroinit(closures_bytes);
    module
        .define_data(closures_data, &closures_desc)
        .map_err(|e| AotError::Module(e.to_string()))?;

    Ok(EmitAotResult {
        const_texts: const_strings.into_iter().map(|(_, t)| t).collect(),
        symbol_names: symbol_remap.into_iter().map(|(_, t)| t).collect(),
        classes: classes_manifest,
        closures: closure_manifest,
        fn_metas,
    })
}

/// Run an `AotModule` straight from a single in-memory source —
/// the parse + sema + MIR-build that `walk_module` runs per file,
/// minus the import recursion. Used by `compile_to_object` for the
/// single-module emit and by the test harness.
fn build_single_aot_module(source: &str, name: &str) -> Result<AotModule, AotError> {
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        return Err(AotError::Frontend(
            parsed
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let mut interner: Interner = parsed.interner;
    let prelude_syms: Vec<crate::intern::SymbolId> = crate::sema::PRELUDE_NAMES
        .iter()
        .map(|n| interner.intern(n))
        .collect();
    let resolved = resolve_with_prelude(&parsed.module, &interner, &prelude_syms);
    if !resolved.errors.is_empty() {
        return Err(AotError::Frontend(
            resolved
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let field_layouts = std::collections::HashMap::new();
    let (module_mir, _new_layouts) =
        lower_module_with_known_classes(&parsed.module, &mut interner, &resolved, &field_layouts);

    let module_var_names: Vec<String> = resolved
        .module_vars
        .iter()
        .map(|sym| interner.resolve(*sym).to_string())
        .collect();
    let module_var_sources = resolved.module_var_sources.clone();
    Ok(AotModule {
        name: name.to_string(),
        request_name: name.to_string(),
        source: source.as_bytes().to_vec(),
        module_var_count: resolved.module_vars.len(),
        module_var_names,
        module_var_sources,
        mir: module_mir,
        interner,
    })
}

/// Compile Wren `source` to a native object file at `output`.
///
/// Single-module entry point — no import resolution. The source is
/// parsed, MIR-built, and lowered into one `ObjectModule` whose
/// only export is `wlift_aot_main`. For multi-file programs use
/// [`compile_path_to_object`].
pub fn compile_to_object(source: &str, output: &Path) -> Result<(), AotError> {
    let aot_mod = build_single_aot_module(source, "<inline>")?;
    let cha = build_cha(std::slice::from_ref(&aot_mod), 0);
    let mut module = make_object_module()?;
    emit_aot_module(
        &mut module,
        &aot_mod,
        "wlift_aot_main",
        "wlift_modvars_main",
        "wlift_consts_main",
        "wlift_symbols_main",
        &cha,
    )?;
    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|e| AotError::Module(e.to_string()))?;
    std::fs::write(output, &bytes).map_err(AotError::Io)?;
    Ok(())
}

/// Compile a Wren program rooted at `entry_path` to a native
/// object file at `output`, walking the import graph and emitting
/// every reachable module's top-level into the same object.
///
/// Output symbols:
/// - The entry module's top-level is exported as `wlift_aot_main`
///   (so the runtime / a future linker shim can find the program
///   entry point unconditionally).
/// - Each imported module's top-level is exported as
///   `wlift_aot_mod_<n>` where `n` is the dependency-first index
///   in the walker output. Stable for a given source tree but not
///   stable across edits — Phase 5's runtime shim will discover
///   them by walking the symbol table at startup.
///
/// Modules are walked + lowered in dependency-first order so that
/// when Phase 5 wires up a startup shim that runs each top-level,
/// imported modules are initialised before the importer's body
/// runs — matching the runtime's `interpret` recursion order.
pub fn compile_path_to_object(entry_path: &Path, output: &Path) -> Result<(), AotError> {
    let result = walk_imports(entry_path)?;
    compile_modules_to_object(&result.modules, output)
}

/// Per-module shape the bootstrap generator needs to wire each
/// AOT-emitted module into the runtime at startup. Returned from
/// [`compile_modules_to_object_with_manifest`] alongside the
/// produced `.o` so the driver can emit a C shim that:
///
/// 1. Calls `wlift_aot_init_prelude` against `modvars_symbol` +
///    `modvars_count` (populates the prelude class slots).
/// 2. For each entry in `const_texts`, allocates an `ObjString`
///    via `wlift_aot_alloc_const_string` and writes the result
///    into `consts_symbol[i]`.
/// 3. Sets `JitContext` via `wlift_aot_enter` with this
///    manifest's `module_name` + modvars pointer.
/// 4. Calls `fn_symbol()` (the AOT-lowered top-level body).
/// 5. Restores via `wlift_aot_exit`.
#[derive(Debug, Clone)]
pub struct AotMethodManifest {
    /// Wren method signature (e.g. `"greet(_)"`, `"value"`).
    pub signature: String,
    /// AOT-emitted symbol the bootstrap hands to
    /// `wlift_aot_install_class` as the method body's function
    /// pointer.
    pub fn_symbol: String,
    /// Wren-visible parameter count (excludes the implicit
    /// receiver). The dispatch helper passes args as positional
    /// `u64`s; the AOT body's signature is
    /// `(receiver, arg0, …, arg_{arity-1}) -> u64`.
    pub arity: u8,
    pub is_static: bool,
    pub is_constructor: bool,
}

#[derive(Debug, Clone)]
pub struct AotClassManifest {
    pub name: String,
    /// Module-var slot of the parent class. Resolved at AOT-build
    /// time against this module's own var list — every class the
    /// resolver knows about (core prelude classes + imported user
    /// classes + same-module user classes) has a slot, so the
    /// install path just loads `modvars[parent_slot]` for the
    /// parent pointer. `None` means no superclass declared
    /// (defaults to `Object` at install time).
    pub parent_slot: Option<u32>,
    pub num_fields: u16,
    /// Module-var slot the bootstrap writes the installed class
    /// pointer into so AOT bodies' `GetModuleVar(slot)` reads
    /// the right `*mut ObjClass` at the use site.
    pub slot: u32,
    pub methods: Vec<AotMethodManifest>,
    /// Value of `#!native = "<libname>"` on a `foreign class`.
    /// `None` for pure-Wren classes. Bootstrap calls
    /// `wlift_aot_bind_foreign_class` after `install_class` to
    /// dlopen the library + bind every entry in `foreign_methods`.
    pub foreign_library: Option<String>,
    /// Stub records for `foreign` methods declared inside a
    /// `foreign class`. The runtime resolves each via dlsym at
    /// install time and binds the resulting `extern "C"` fn
    /// pointer into the class method table.
    pub foreign_methods: Vec<AotForeignMethodManifest>,
}

#[derive(Debug, Clone)]
pub struct AotForeignMethodManifest {
    pub signature: String,
    /// `#!symbol = "..."` override. `None` falls back to the
    /// signature's base name (Wren's default convention).
    pub symbol: Option<String>,
    pub is_static: bool,
}

#[derive(Debug, Clone)]
pub struct AotManifest {
    pub fn_symbol: String,
    pub modvars_symbol: String,
    pub modvars_count: usize,
    pub consts_symbol: String,
    pub const_texts: Vec<String>,
    /// Symbol-remap slot table — `wlift_symbols_<n>`. The
    /// bootstrap re-interns each name in the VM's interner at
    /// startup, writing the resulting `u32`-padded `SymbolId`
    /// into slot `i`. Lowering issues `load symbols[slot]`
    /// instead of a baked `iconst sym_idx` for every site that
    /// passes a SymbolId to a runtime helper (Call's method,
    /// is-type's class symbol, …).
    pub symbols_symbol: String,
    pub symbol_names: Vec<String>,
    pub module_name: String,
    pub classes: Vec<AotClassManifest>,
    /// Cross-module import bindings — each entry directs the
    /// bootstrap to copy `source_modvars[source_slot]` into
    /// this module's `modvars[target_slot]` after the source
    /// module's classes have been installed. Replicates the
    /// JIT install loop's cross-module value-resolution pass
    /// without dragging in the engine's `find_imported_var`.
    pub imports: Vec<AotImportBinding>,
    /// Closure bodies (one per `MakeClosure` target in this
    /// module's MIR). The bootstrap calls
    /// `wlift_aot_register_closure` per entry at startup,
    /// writing the resulting FuncId into the closure slot
    /// table the AOT body's `MakeClosure` reads against.
    pub closures: Vec<AotClosureManifest>,
    /// Symbol of the `wlift_closures_<n>` slot table.
    pub closures_symbol: String,
    /// Per-AOT-function safepoint metadata captured during emit.
    /// Used by the bootstrap to call `wlift_aot_register_code_range`
    /// once per function so the GC stack walker can find AOT
    /// frames + map return addresses to live-root spill offsets.
    pub fn_metas: Vec<AotFnMetaManifest>,
    /// Bare-builtin imports the walker can't follow into source
    /// (no `.wren` file for `"socket"`, `"fs"`, ...). Bootstrap
    /// resolves each at startup via `wlift_aot_resolve_runtime_import`.
    pub runtime_imports: Vec<AotRuntimeImport>,
}

/// Per-function code-range + safepoint info the bootstrap embeds
/// in the AOT object as `Linkage::Local` data so it can call
/// `wlift_aot_register_code_range` at startup. Code size is
/// captured here (Cranelift's `CompiledCode.code_info().total_size`)
/// because once the object linker rewrites symbols at link time
/// the runtime can't recover per-function sizes by name alone.
#[derive(Debug, Clone)]
pub struct AotFnMetaManifest {
    pub fn_symbol: String,
    pub code_size: u32,
    /// MIR arity (including the implicit receiver). The bootstrap
    /// re-imports the function symbol with this arity so Cranelift's
    /// `declare_function` accepts the call as compatible with what
    /// `emit_aot_function` originally declared.
    pub arity: u8,
    /// `frame_to_fp_offset` — the same anchor `native_meta_from_cranelift`
    /// already baked into each safepoint's spill offsets, retained
    /// here for completeness even though the lowering side has
    /// already turned each spill offset FP-relative.
    pub fp_anchor: i64,
    pub safepoints: Vec<AotSafepointEntry>,
}

#[derive(Debug, Clone)]
pub struct AotSafepointEntry {
    /// Offset of the safepoint's return-address-after-call from
    /// the function's start, used by the GC scanner to look up
    /// the active safepoint at GC time.
    pub code_offset: u32,
    /// FP-relative spill offsets of every live root at this
    /// safepoint. The scanner reads `*(jit_fp + offset)` per
    /// entry and adds it to the root set if it's an object Value.
    pub root_fp_offsets: Vec<i32>,
}

/// Top-level AOT bundle metadata that doesn't fit per-module:
/// native-library search paths (where `foreign class
/// #!native = "<lib>"` looks for dlopen targets) and bundled
/// `.hatch` `NativeLib`-section payloads. The bootstrap drains
/// these once at startup, before any class install.
#[derive(Debug, Clone, Default)]
pub struct AotBundleMeta {
    /// Absolute paths the foreign loader's `library_candidates`
    /// search ought to try ahead of the OS ambient search.
    /// Resolved at AOT-build time from each path-linked dep's
    /// `manifest.native_search_paths` (workspace-rooted).
    pub native_search_paths: Vec<String>,
    /// Bundled dynamic-library payloads from `.hatch` `NativeLib`
    /// sections. The bootstrap writes each to a temp path and
    /// registers `(name, path)` with `vm.native_lib_paths` so
    /// the foreign loader picks up the bundled copy.
    pub native_libs: Vec<(String, Vec<u8>)>,
}

#[derive(Debug, Clone)]
pub struct AotImportBinding {
    pub target_slot: u32,
    pub source_modvars_symbol: String,
    pub source_slot: u32,
}

/// Bare-builtin import (`import "socket" for SocketCore`,
/// `import "fs" for FS`, ...). The walker skips these — they have
/// no on-disk source — so the bootstrap calls
/// `wlift_aot_resolve_runtime_import` once per binding at startup
/// to look up the value through the runtime VM and write it
/// into modvars[target_slot].
#[derive(Debug, Clone)]
pub struct AotRuntimeImport {
    pub target_slot: u32,
    pub module_name: String,
    pub var_name: String,
}

/// Lower a pre-walked list of modules into one native object file
/// at `output`. Convenience wrapper over
/// [`compile_modules_to_object_with_manifest`] for callers that
/// only need the `.o`.
pub fn compile_modules_to_object(modules: &[AotModule], output: &Path) -> Result<(), AotError> {
    compile_modules_to_object_with_manifest(modules, output).map(|_| ())
}

/// Scan a method's MIR for instructions that read
/// `JitContext.defining_class` at runtime. Currently:
///
/// * `SuperCall` — `wren_super_call_N` walks up
///   `defining_class.superclass` to resolve the target.
/// * `GetStaticField` / `SetStaticField` — read or write a
///   field on `defining_class` directly.
///
/// `dispatch_method`'s `Closure` arm sets `defining_class` per
/// call before entering the body, but the AOT CHA direct-call
/// path skips that setup to avoid the per-call context swap.
/// Excluding these methods from devirt keeps the runtime's
/// implicit-context contract intact at the cost of one
/// `wren_call_N` per affected site.
fn method_uses_defining_class(mir: &crate::mir::MirFunction) -> bool {
    use crate::mir::Instruction;
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            if matches!(
                inst,
                Instruction::SuperCall { .. }
                    | Instruction::GetStaticField(_)
                    | Instruction::SetStaticField(_, _)
            ) {
                return true;
            }
        }
    }
    false
}

/// Build the whole-program method table — every (signature →
/// `Vec<AotMethodImpl>`) pair across all walked modules — once
/// up front. The lowering threads a borrow of this into
/// `AotLoweringConfig` so each Call site can devirtualize.
fn build_cha(modules: &[AotModule], last_idx: usize) -> AotCha {
    let mut by_sig: HashMap<String, Vec<AotMethodImpl>> = HashMap::new();

    for (idx, aot_mod) in modules.iter().enumerate() {
        let fn_prefix = if idx == last_idx {
            "wlift_aot_main".to_string()
        } else {
            format!("wlift_aot_mod_{}", idx)
        };
        let modvars_symbol = if idx == last_idx {
            "wlift_modvars_main".to_string()
        } else {
            format!("wlift_modvars_{}", idx)
        };

        for (c_idx, class_mir) in aot_mod.mir.classes.iter().enumerate() {
            let class_name = aot_mod.interner.resolve(class_mir.name).to_string();
            let class_slot_opt = aot_mod
                .module_var_names
                .iter()
                .position(|n| n == &class_name);
            let Some(class_slot) = class_slot_opt else {
                continue;
            };
            for (m_idx, method) in class_mir.methods.iter().enumerate() {
                // Skip methods that issue a SuperCall — those need
                // `JitContext.defining_class` to be set to *this*
                // method's class (the runtime's dispatch_method
                // does it before calling the closure body); the
                // CHA direct-call path doesn't set it. Routing
                // through `wren_call_N` keeps super dispatch
                // correct at the cost of devirt for these
                // methods. Same call covers methods that read
                // static fields off the defining class, since
                // those also consult `defining_class`.
                if method_uses_defining_class(&method.mir) {
                    continue;
                }
                let fn_symbol = format!("{}__method_{}_{}", fn_prefix, c_idx, m_idx);
                let trivial =
                    crate::runtime::engine::ExecutionEngine::mir_trivial_getter_field(&method.mir);
                by_sig
                    .entry(method.signature.clone())
                    .or_default()
                    .push(AotMethodImpl {
                        class_name: class_name.clone(),
                        fn_symbol,
                        arity: method.mir.arity,
                        trivial_getter_field: trivial,
                        class_modvars_symbol: modvars_symbol.clone(),
                        class_slot: class_slot as u32,
                    });
            }
        }
    }

    AotCha { by_sig }
}

/// Lower a pre-walked list of modules into one native object file
/// at `output`, returning a per-module manifest. The same `.o`
/// also carries a Cranelift-emitted `main` function that
/// orchestrates startup — runtime entry imports, per-module init
/// data, AOT body dispatch — so the produced object links
/// directly with `libwren_lift.a` into a runnable executable
/// without any C bootstrap source.
///
/// Modules must already be in dependency-first order; the entry
/// is the last element. Symbol naming: entry → `wlift_aot_main`,
/// dependencies → `wlift_aot_mod_<n>`.
pub fn compile_modules_to_object_with_manifest(
    modules: &[AotModule],
    output: &Path,
) -> Result<Vec<AotManifest>, AotError> {
    compile_walk_to_object_with_manifest(modules, &AotBundleMeta::default(), output)
}

/// Like [`compile_modules_to_object_with_manifest`] but threads
/// bundle-level metadata (native search paths + bundled lib
/// payloads) into the bootstrap so foreign-class binding works
/// at runtime.
pub fn compile_walk_to_object_with_manifest(
    modules: &[AotModule],
    bundle: &AotBundleMeta,
    output: &Path,
) -> Result<Vec<AotManifest>, AotError> {
    if modules.is_empty() {
        return Err(AotError::Frontend("no modules to emit".into()));
    }

    let mut module = make_object_module()?;
    let last_idx = modules.len() - 1;

    // Build the whole-program method table (CHA) up front so the
    // Call-site lowering can devirtualize. Same module-naming
    // convention the per-module emit loop below uses, so the
    // class_modvars_symbol + class_slot fields point at the
    // exact data the bootstrap installs the class pointer into.
    let cha = build_cha(modules, last_idx);

    let mut manifests: Vec<AotManifest> = Vec::with_capacity(modules.len());
    for (idx, aot_mod) in modules.iter().enumerate() {
        let (fn_symbol, modvars_symbol, consts_symbol, symbols_symbol) = if idx == last_idx {
            (
                "wlift_aot_main".to_string(),
                "wlift_modvars_main".to_string(),
                "wlift_consts_main".to_string(),
                "wlift_symbols_main".to_string(),
            )
        } else {
            (
                format!("wlift_aot_mod_{}", idx),
                format!("wlift_modvars_{}", idx),
                format!("wlift_consts_{}", idx),
                format!("wlift_symbols_{}", idx),
            )
        };
        let emitted = emit_aot_module(
            &mut module,
            aot_mod,
            &fn_symbol,
            &modvars_symbol,
            &consts_symbol,
            &symbols_symbol,
            &cha,
        )?;
        let closures_symbol = format!("{}__closures_data", fn_symbol);
        // Convert the per-fn captured `EmittedFnMeta` into the
        // manifest shape the bootstrap consumes. Functions that
        // had no Cranelift safepoints (e.g. trivial leaf bodies
        // or top-level boots that don't call helpers) still get
        // an entry with an empty safepoints list so their code
        // range is registered for frame-walking.
        let fn_metas: Vec<AotFnMetaManifest> = emitted
            .fn_metas
            .into_iter()
            .map(|(sym, meta)| {
                let (fp_anchor, safepoints) = match meta.native_meta {
                    Some(nm) => {
                        let safepoints = nm
                            .safepoints
                            .into_iter()
                            .map(|sp| AotSafepointEntry {
                                code_offset: sp.code_offset,
                                root_fp_offsets: sp
                                    .live_roots
                                    .into_iter()
                                    .filter_map(|r| match r.location {
                                        crate::codegen::native_meta::RootLocation::Spill(o) => {
                                            Some(o)
                                        }
                                        _ => None,
                                    })
                                    .collect(),
                            })
                            .collect();
                        (0, safepoints)
                    }
                    None => (0, Vec::new()),
                };
                AotFnMetaManifest {
                    fn_symbol: sym,
                    code_size: meta.code_size,
                    arity: meta.arity,
                    fp_anchor,
                    safepoints,
                }
            })
            .collect();
        manifests.push(AotManifest {
            fn_symbol,
            modvars_symbol,
            modvars_count: aot_mod.module_var_count,
            consts_symbol,
            const_texts: emitted.const_texts,
            symbols_symbol,
            symbol_names: emitted.symbol_names,
            module_name: aot_mod.request_name.clone(),
            classes: emitted.classes,
            imports: Vec::new(),
            closures: emitted.closures,
            closures_symbol,
            fn_metas,
            runtime_imports: Vec::new(),
        });
    }

    // Resolve cross-module imports: each module's resolver gave us
    // a per-slot `Option<source_path>`. Walk those once we have
    // every manifest in hand so a `Some(path)` can resolve to the
    // dependency's modvars symbol + slot, and the bootstrap emits
    // a direct copy without any runtime name lookup.
    //
    // Sources that don't match any emitted manifest are bare-builtin
    // imports (`"socket"`, `"fs"`, `"meta"`, ...) the walker
    // skipped — those get an `AotRuntimeImport` entry the bootstrap
    // resolves at startup via `wlift_aot_resolve_runtime_import`.
    for (idx, aot_mod) in modules.iter().enumerate() {
        let mut bindings: Vec<AotImportBinding> = Vec::new();
        let mut runtime_imports: Vec<AotRuntimeImport> = Vec::new();
        for (slot, source) in aot_mod.module_var_sources.iter().enumerate() {
            let Some(source_path) = source else { continue };
            let var_name = &aot_mod.module_var_names[slot];

            // Match against any earlier-installed module sharing
            // the import path. Dependency-first walker order
            // guarantees the source has been emitted by the time
            // we get here.
            let matched = manifests
                .iter()
                .take(idx)
                .find(|src| &src.module_name == source_path);
            if let Some(src) = matched {
                if let Some(src_class) = src.classes.iter().find(|c| &c.name == var_name) {
                    bindings.push(AotImportBinding {
                        target_slot: slot as u32,
                        source_modvars_symbol: src.modvars_symbol.clone(),
                        source_slot: src_class.slot,
                    });
                }
            } else {
                // No emitted module under this name → bare-builtin
                // (or a registry/git dep the walker couldn't reach).
                // Defer resolution to runtime.
                runtime_imports.push(AotRuntimeImport {
                    target_slot: slot as u32,
                    module_name: source_path.clone(),
                    var_name: var_name.clone(),
                });
            }
        }
        manifests[idx].imports = bindings;
        manifests[idx].runtime_imports = runtime_imports;
    }

    emit_aot_bootstrap_main(&mut module, &manifests, bundle)?;

    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|e| AotError::Module(e.to_string()))?;
    std::fs::write(output, &bytes).map_err(AotError::Io)?;
    Ok(manifests)
}

/// Locate `libwren_lift.a` (or `wren_lift.lib` on Windows) for
/// the linker step that turns an AOT-emitted `.o` into a
/// runnable executable. Lookup order:
///
/// 1. `WLIFT_STATICLIB` env var pointing at an explicit path.
/// 2. Sibling of the running executable (CLI install case).
/// 3. `target/{release,debug}/` from the current working
///    directory (developer checkout case).
///
/// Returns `None` if no candidate file exists; callers should
/// surface a `cargo build --release --features aot` hint.
pub fn locate_runtime_staticlib() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WLIFT_STATICLIB") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }

    let staticlib_name = if cfg!(target_os = "windows") {
        "wren_lift.lib"
    } else {
        "libwren_lift.a"
    };

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(staticlib_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    for profile in ["release", "debug"] {
        let candidate = PathBuf::from("target").join(profile).join(staticlib_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Link a previously-emitted AOT object file against
/// `libwren_lift.a` into a standalone executable at `out`. The
/// command is the same one the `wlift --aot` CLI runs: invoke
/// `$CC` (defaults to `cc`) with the object, the staticlib, and
/// the platform-specific system libraries the runtime depends
/// on (pthread, dl, m everywhere; CoreFoundation + Security on
/// macOS).
///
/// On macOS arm64, also runs an ad-hoc `codesign -f -s -` over
/// the produced binary so the OS doesn't refuse to execute the
/// freshly-linked file (Gatekeeper rejects unsigned arm64 Mach-O
/// binaries with EBADARCH on recent macOS releases). The codesign
/// step is best-effort; failures are surfaced but don't poison
/// the link result for callers that are happy to ship unsigned.
pub fn link_executable(obj: &Path, staticlib: &Path, out: &Path) -> Result<(), AotError> {
    use std::process::Command;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut cmd = Command::new(&cc);
    cmd.arg(obj).arg(staticlib).arg("-o").arg(out);
    if cfg!(target_os = "macos") {
        cmd.arg("-lpthread")
            .arg("-ldl")
            .arg("-lm")
            .arg("-framework")
            .arg("CoreFoundation")
            .arg("-framework")
            .arg("Security");
    } else if cfg!(target_os = "linux") {
        cmd.arg("-lpthread").arg("-ldl").arg("-lm");
    }

    let status = cmd.status().map_err(|e| {
        AotError::Module(format!(
            "invoking `{}`: {}; set CC to a compiler binary if `cc` isn't on PATH",
            cc, e
        ))
    })?;
    if !status.success() {
        return Err(AotError::Module(format!("linker exited with {}", status)));
    }

    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        let _ = Command::new("codesign")
            .args(["-f", "-s", "-"])
            .arg(out)
            .status();
    }

    Ok(())
}

/// Emit the program entry — `main(argc, argv)` — directly into
/// the AOT object via Cranelift, alongside the per-module name
/// strings and const-text bytes the entry references. The
/// produced `.o` then links with `libwren_lift.a` in a single
/// system-linker call, no C bootstrap source involved.
///
/// Shape of the emitted body, per manifest entry:
///
/// 1. `wlift_aot_init_prelude(vm, modvars_<i>, count_<i>)` —
///    populates the prelude class slots so the AOT body can
///    `GetModuleVar` for `System` / `Num` / etc. as direct
///    `.bss` loads.
/// 2. For each const text: `consts_<i>[k] = wlift_aot_alloc_const_string(...)`.
/// 3. `wlift_aot_enter(vm, modvars_<i>, count_<i>, name_<i>, len, &saved)`.
/// 4. `wlift_aot_<symbol>()` — runs the module's top-level.
/// 5. `wlift_aot_exit(&saved)`.
///
/// `saved` is a 128-byte stack slot (`JitContext` is 80 bytes
/// today; the headroom keeps the bootstrap stable across
/// runtime additions).
fn emit_aot_bootstrap_main(
    module: &mut ObjectModule,
    manifests: &[AotManifest],
    bundle: &AotBundleMeta,
) -> Result<(), AotError> {
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::{InstBuilder, MemFlags, StackSlotData, StackSlotKind};
    use cranelift_module::DataDescription;

    let ptr_ty = module.target_config().pointer_type();
    let cc = module.target_config().default_call_conv;

    let declare_import = |module: &mut ObjectModule,
                          name: &str,
                          params: &[types::Type],
                          ret: Option<types::Type>|
     -> Result<cranelift_module::FuncId, AotError> {
        let mut sig = Signature::new(cc);
        for p in params {
            sig.params.push(AbiParam::new(*p));
        }
        if let Some(r) = ret {
            sig.returns.push(AbiParam::new(r));
        }
        module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| AotError::Module(e.to_string()))
    };

    let wren_new_vm = declare_import(module, "wrenNewVM", &[ptr_ty], Some(ptr_ty))?;
    let wren_free_vm = declare_import(module, "wrenFreeVM", &[ptr_ty], None)?;
    let init_prelude = declare_import(
        module,
        "wlift_aot_init_prelude",
        &[ptr_ty, ptr_ty, ptr_ty],
        Some(types::I32),
    )?;
    let alloc_const = declare_import(
        module,
        "wlift_aot_alloc_const_string",
        &[ptr_ty, ptr_ty, ptr_ty],
        Some(types::I64),
    )?;
    let intern_symbol = declare_import(
        module,
        "wlift_aot_intern_symbol",
        &[ptr_ty, ptr_ty, ptr_ty],
        Some(types::I64),
    )?;
    let register_closure = declare_import(
        module,
        "wlift_aot_register_closure",
        &[ptr_ty, types::I8, ptr_ty],
        Some(types::I64),
    )?;
    let register_code_range = declare_import(
        module,
        "wlift_aot_register_code_range",
        &[
            ptr_ty,     // vm
            ptr_ty,     // fn_ptr
            types::I32, // code_size
            ptr_ty,     // safepoints desc
            types::I32, // safepoints count
            ptr_ty,     // roots
        ],
        Some(types::I32),
    )?;
    let resolve_runtime_import = declare_import(
        module,
        "wlift_aot_resolve_runtime_import",
        &[
            ptr_ty, // vm
            ptr_ty, // modvars
            ptr_ty, // slot
            ptr_ty, // module_name
            ptr_ty, // module_name_len
            ptr_ty, // var_name
            ptr_ty, // var_name_len
        ],
        Some(types::I32),
    )?;
    let install_class = declare_import(
        module,
        "wlift_aot_install_class",
        &[
            ptr_ty,     // vm
            ptr_ty,     // modvars
            ptr_ty,     // slot
            ptr_ty,     // name
            ptr_ty,     // name_len
            ptr_ty,     // parent_slot
            types::I32, // num_fields
            ptr_ty,     // methods desc
            ptr_ty,     // methods count
        ],
        Some(types::I32),
    )?;
    let bind_foreign_class = declare_import(
        module,
        "wlift_aot_bind_foreign_class",
        &[
            ptr_ty, // vm
            ptr_ty, // modvars
            ptr_ty, // slot
            ptr_ty, // lib_name
            ptr_ty, // lib_name_len
            ptr_ty, // methods desc
            ptr_ty, // methods count
        ],
        Some(types::I32),
    )?;
    let add_native_search_path = declare_import(
        module,
        "wlift_aot_add_native_search_path",
        &[ptr_ty, ptr_ty, ptr_ty],
        Some(types::I32),
    )?;
    let install_native_lib = declare_import(
        module,
        "wlift_aot_install_native_lib",
        &[ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty],
        Some(types::I32),
    )?;
    let enter_ctx = declare_import(
        module,
        "wlift_aot_enter",
        &[ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty],
        None,
    )?;
    let exit_ctx = declare_import(module, "wlift_aot_exit", &[ptr_ty], None)?;

    // Emit per-module name + per-const text byte arrays as Data
    // symbols. Each gets a trailing 0 so it's also valid as a
    // C-string pointer if any future helper wants one.
    let define_bytes = |module: &mut ObjectModule,
                        name: &str,
                        bytes: &[u8]|
     -> Result<cranelift_module::DataId, AotError> {
        let id = module
            .declare_data(name, Linkage::Local, false, false)
            .map_err(|e| AotError::Module(e.to_string()))?;
        let mut desc = DataDescription::new();
        let mut payload = bytes.to_vec();
        payload.push(0);
        desc.define(payload.into_boxed_slice());
        module
            .define_data(id, &desc)
            .map_err(|e| AotError::Module(e.to_string()))?;
        Ok(id)
    };

    // Per-manifest fn-import declarations + name/text data.
    struct ClassData {
        name_id: cranelift_module::DataId,
        name_len: usize,
        // `usize::MAX` sentinel = no superclass declared (defaults
        // to Object). Otherwise an index into this module's
        // modvars; install reads `modvars[parent_slot]` for the
        // superclass pointer, populated earlier by `init_prelude`
        // (core classes) or the import-copy step (cross-module).
        parent_slot: usize,
        slot: u32,
        num_fields: u16,
        // Per-method: (sig DataId, sig_len, body FuncId, arity, flags).
        // The bootstrap stack-allocates an array of `WliftAotMethodDesc`s
        // and fills it from these references, then hands the array to
        // `wlift_aot_install_class`.
        methods: Vec<(
            cranelift_module::DataId,
            usize,
            cranelift_module::FuncId,
            u8,
            u8,
        )>,
        /// Foreign-class binding info, present iff the class
        /// declared `#!native = "..."`. Bootstrap calls
        /// `wlift_aot_bind_foreign_class` after install with the
        /// dlopen library name + each `foreign` method's dlsym
        /// descriptor.
        foreign: Option<ClassForeignData>,
    }

    struct ClassForeignData {
        lib_id: cranelift_module::DataId,
        lib_len: usize,
        /// Per `foreign` method: (sig DataId, sig_len, optional
        /// `#!symbol` override DataId, sym_len, flags). `flags`
        /// bit-0 = is_static. The bootstrap stack-allocates an
        /// array of `WliftAotForeignMethodDesc`s and writes
        /// these fields in.
        methods: Vec<(
            cranelift_module::DataId,
            usize,
            Option<cranelift_module::DataId>,
            usize,
            u8,
        )>,
    }
    struct ManifestData {
        fn_id: cranelift_module::FuncId,
        modvars_id: cranelift_module::DataId,
        consts_id: cranelift_module::DataId,
        symbols_id: cranelift_module::DataId,
        name_id: cranelift_module::DataId,
        const_text_ids: Vec<cranelift_module::DataId>,
        symbol_name_ids: Vec<cranelift_module::DataId>,
        modvars_count: usize,
        name_len: usize,
        const_lens: Vec<usize>,
        symbol_lens: Vec<usize>,
        classes: Vec<ClassData>,
        // Per-import: (target_slot, source modvars DataId, source_slot).
        imports: Vec<(u32, cranelift_module::DataId, u32)>,
        // Per-closure: (body FuncId import, arity). Bootstrap
        // resolves each FuncId's address and registers via
        // `wlift_aot_register_closure`, writing the returned
        // engine FuncId into `closures_data[i]`.
        closures_data_id: cranelift_module::DataId,
        closures: Vec<(cranelift_module::FuncId, u8)>,
        /// Per-AOT-function frame metadata: (fn_id import,
        /// code_size, safepoints desc DataId, safepoint count,
        /// roots DataId). Bootstrap calls
        /// `wlift_aot_register_code_range` per entry so the GC
        /// stack walker can map AOT frames' return addresses to
        /// live-root spill offsets.
        fn_metas: Vec<(
            cranelift_module::FuncId,
            u32,
            cranelift_module::DataId,
            u32,
            cranelift_module::DataId,
        )>,
        /// Bare-builtin runtime imports — bootstrap resolves each
        /// at startup via `wlift_aot_resolve_runtime_import`.
        /// Tuple: (target_slot, module_name DataId, mod_len,
        /// var_name DataId, var_len).
        runtime_imports: Vec<(
            u32,
            cranelift_module::DataId,
            usize,
            cranelift_module::DataId,
            usize,
        )>,
    }
    let mut per_module: Vec<ManifestData> = Vec::with_capacity(manifests.len());
    for (i, m) in manifests.iter().enumerate() {
        let mut fn_sig = Signature::new(cc);
        fn_sig.returns.push(AbiParam::new(types::I64));
        let fn_id = module
            .declare_function(&m.fn_symbol, Linkage::Import, &fn_sig)
            .map_err(|e| AotError::Module(e.to_string()))?;
        // declare_data is idempotent — Cranelift hands back the
        // existing DataId for the modvars/consts/symbols symbols
        // already defined by emit_aot_module.
        let modvars_id = module
            .declare_data(&m.modvars_symbol, Linkage::Export, true, false)
            .map_err(|e| AotError::Module(e.to_string()))?;
        let consts_id = module
            .declare_data(&m.consts_symbol, Linkage::Export, true, false)
            .map_err(|e| AotError::Module(e.to_string()))?;
        let symbols_id = module
            .declare_data(&m.symbols_symbol, Linkage::Export, true, false)
            .map_err(|e| AotError::Module(e.to_string()))?;
        let name_id = define_bytes(
            module,
            &format!("wlift_modname_{}", i),
            m.module_name.as_bytes(),
        )?;
        let mut const_text_ids = Vec::with_capacity(m.const_texts.len());
        let mut const_lens = Vec::with_capacity(m.const_texts.len());
        for (k, text) in m.const_texts.iter().enumerate() {
            let id = define_bytes(
                module,
                &format!("wlift_const_text_{}_{}", i, k),
                text.as_bytes(),
            )?;
            const_text_ids.push(id);
            const_lens.push(text.len());
        }
        let mut symbol_name_ids = Vec::with_capacity(m.symbol_names.len());
        let mut symbol_lens = Vec::with_capacity(m.symbol_names.len());
        for (k, name) in m.symbol_names.iter().enumerate() {
            let id = define_bytes(
                module,
                &format!("wlift_symname_{}_{}", i, k),
                name.as_bytes(),
            )?;
            symbol_name_ids.push(id);
            symbol_lens.push(name.len());
        }

        // Per-class metadata: class name, parent name, method
        // signatures + body function ids. The bootstrap builds
        // per-class descriptor arrays on the stack at startup
        // (each `WliftAotMethodDesc` is 32 bytes) and hands the
        // pointer to `wlift_aot_install_class`.
        let mut classes = Vec::with_capacity(m.classes.len());
        for (c_idx, class) in m.classes.iter().enumerate() {
            let name_id = define_bytes(
                module,
                &format!("wlift_classname_{}_{}", i, c_idx),
                class.name.as_bytes(),
            )?;
            let parent_slot = class.parent_slot.map(|s| s as usize).unwrap_or(usize::MAX);
            let mut methods = Vec::with_capacity(class.methods.len());
            for (m_idx, method) in class.methods.iter().enumerate() {
                let sig_id = define_bytes(
                    module,
                    &format!("wlift_methodsig_{}_{}_{}", i, c_idx, m_idx),
                    method.signature.as_bytes(),
                )?;
                let mut body_sig = Signature::new(cc);
                for _ in 0..(method.arity as usize) {
                    body_sig.params.push(AbiParam::new(types::I64));
                }
                body_sig.returns.push(AbiParam::new(types::I64));
                let body_id = module
                    .declare_function(&method.fn_symbol, Linkage::Import, &body_sig)
                    .map_err(|e| AotError::Module(e.to_string()))?;
                let mut flags: u8 = 0;
                if method.is_static {
                    flags |= 1;
                }
                if method.is_constructor {
                    flags |= 2;
                }
                methods.push((sig_id, method.signature.len(), body_id, method.arity, flags));
            }

            // Foreign-class binding data — only populated when
            // the class declared `#!native = "..."`. The bootstrap
            // calls `wlift_aot_bind_foreign_class` after the
            // standard `install_class` so `modvars[slot]` already
            // points at the freshly-allocated `*mut ObjClass`.
            let foreign = if let Some(lib_name) = &class.foreign_library {
                let lib_id = define_bytes(
                    module,
                    &format!("wlift_classnative_{}_{}", i, c_idx),
                    lib_name.as_bytes(),
                )?;
                let mut fmethods = Vec::with_capacity(class.foreign_methods.len());
                for (f_idx, fm) in class.foreign_methods.iter().enumerate() {
                    let sig_id = define_bytes(
                        module,
                        &format!("wlift_fsig_{}_{}_{}", i, c_idx, f_idx),
                        fm.signature.as_bytes(),
                    )?;
                    let (sym_id, sym_len) = if let Some(sym) = &fm.symbol {
                        let id = define_bytes(
                            module,
                            &format!("wlift_fsym_{}_{}_{}", i, c_idx, f_idx),
                            sym.as_bytes(),
                        )?;
                        (Some(id), sym.len())
                    } else {
                        (None, 0)
                    };
                    let flags: u8 = if fm.is_static { 1 } else { 0 };
                    fmethods.push((sig_id, fm.signature.len(), sym_id, sym_len, flags));
                }
                Some(ClassForeignData {
                    lib_id,
                    lib_len: lib_name.len(),
                    methods: fmethods,
                })
            } else {
                None
            };

            classes.push(ClassData {
                name_id,
                name_len: class.name.len(),
                parent_slot,
                slot: class.slot,
                num_fields: class.num_fields,
                methods,
                foreign,
            });
        }
        // Per-import: resolve each binding's source modvars
        // symbol back to a `DataId` for `declare_data_in_func`
        // inside main(). `declare_data` is idempotent — we reuse
        // the existing entry rather than redefining.
        let mut imports = Vec::with_capacity(m.imports.len());
        for binding in &m.imports {
            let src_id = module
                .declare_data(&binding.source_modvars_symbol, Linkage::Export, true, false)
                .map_err(|e| AotError::Module(e.to_string()))?;
            imports.push((binding.target_slot, src_id, binding.source_slot));
        }

        // Per-closure: declare each body fn import + carry arity.
        let closures_data_id = module
            .declare_data(&m.closures_symbol, Linkage::Export, true, false)
            .map_err(|e| AotError::Module(e.to_string()))?;
        let mut closures = Vec::with_capacity(m.closures.len());
        for closure in &m.closures {
            let mut sig = Signature::new(cc);
            for _ in 0..(closure.arity as usize) {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));
            let body_id = module
                .declare_function(&closure.fn_symbol, Linkage::Import, &sig)
                .map_err(|e| AotError::Module(e.to_string()))?;
            closures.push((body_id, closure.arity));
        }

        // Per-function safepoint metadata. Each AOT function gets
        // two `Linkage::Local` data symbols:
        //   `<fn_sym>__sps`   — array of WliftAotSafepointDesc
        //                       (3 × u32 per safepoint = 12 bytes)
        //   `<fn_sym>__roots` — flat i32 array, indexed by
        //                       safepoint.roots_start..count
        // Plus a fresh `Linkage::Import` `FuncId` for the
        // function symbol so the bootstrap can take its address.
        let mut fn_metas: Vec<(
            cranelift_module::FuncId,
            u32,
            cranelift_module::DataId,
            u32,
            cranelift_module::DataId,
        )> = Vec::with_capacity(m.fn_metas.len());
        for fn_meta in &m.fn_metas {
            let mut sps_bytes: Vec<u8> = Vec::with_capacity(fn_meta.safepoints.len() * 12);
            let mut roots_bytes: Vec<u8> = Vec::new();
            let mut roots_cursor: u32 = 0;
            for sp in &fn_meta.safepoints {
                sps_bytes.extend_from_slice(&sp.code_offset.to_le_bytes());
                sps_bytes.extend_from_slice(&roots_cursor.to_le_bytes());
                let count = sp.root_fp_offsets.len() as u32;
                sps_bytes.extend_from_slice(&count.to_le_bytes());
                for off in &sp.root_fp_offsets {
                    roots_bytes.extend_from_slice(&off.to_le_bytes());
                }
                roots_cursor += count;
            }
            // `define_data` needs at least 1 byte; pad with a
            // single zero when the function has no safepoints.
            if sps_bytes.is_empty() {
                sps_bytes.push(0);
            }
            if roots_bytes.is_empty() {
                roots_bytes.push(0);
            }
            let sps_id = module
                .declare_data(
                    &format!("{}__sps", fn_meta.fn_symbol),
                    Linkage::Local,
                    false,
                    false,
                )
                .map_err(|e| AotError::Module(e.to_string()))?;
            let mut sps_desc = DataDescription::new();
            sps_desc.define(sps_bytes.into_boxed_slice());
            module
                .define_data(sps_id, &sps_desc)
                .map_err(|e| AotError::Module(e.to_string()))?;
            let roots_id = module
                .declare_data(
                    &format!("{}__roots", fn_meta.fn_symbol),
                    Linkage::Local,
                    false,
                    false,
                )
                .map_err(|e| AotError::Module(e.to_string()))?;
            let mut roots_desc = DataDescription::new();
            roots_desc.define(roots_bytes.into_boxed_slice());
            module
                .define_data(roots_id, &roots_desc)
                .map_err(|e| AotError::Module(e.to_string()))?;

            // Re-import the function by symbol so we can take its
            // address (the original FuncId from `emit_aot_function`
            // lives inside that module-emit context; here we need
            // a fresh import). Match the AOT body's actual
            // signature — Cranelift's `declare_function` is only
            // idempotent for matching signatures, otherwise it
            // returns the "incompatible signature" module error.
            let mut fn_sig = Signature::new(cc);
            for _ in 0..(fn_meta.arity as usize) {
                fn_sig.params.push(AbiParam::new(types::I64));
            }
            fn_sig.returns.push(AbiParam::new(types::I64));
            let fn_id_import = module
                .declare_function(&fn_meta.fn_symbol, Linkage::Import, &fn_sig)
                .map_err(|e| AotError::Module(e.to_string()))?;
            fn_metas.push((
                fn_id_import,
                fn_meta.code_size,
                sps_id,
                fn_meta.safepoints.len() as u32,
                roots_id,
            ));
        }

        // Per-runtime-import: emit a name + var-name byte blob for
        // each bare-builtin import so the bootstrap can call
        // `wlift_aot_resolve_runtime_import(vm, modvars, slot,
        // mod_name_addr, mod_len, var_name_addr, var_len)` at
        // startup.
        let mut runtime_imports: Vec<(
            u32,
            cranelift_module::DataId,
            usize,
            cranelift_module::DataId,
            usize,
        )> = Vec::with_capacity(m.runtime_imports.len());
        for (k, ri) in m.runtime_imports.iter().enumerate() {
            let mod_id = define_bytes(
                module,
                &format!("wlift_runtime_mod_{}_{}", i, k),
                ri.module_name.as_bytes(),
            )?;
            let var_id = define_bytes(
                module,
                &format!("wlift_runtime_var_{}_{}", i, k),
                ri.var_name.as_bytes(),
            )?;
            runtime_imports.push((
                ri.target_slot,
                mod_id,
                ri.module_name.len(),
                var_id,
                ri.var_name.len(),
            ));
        }

        per_module.push(ManifestData {
            fn_id,
            modvars_id,
            consts_id,
            symbols_id,
            name_id,
            const_text_ids,
            symbol_name_ids,
            modvars_count: m.modvars_count,
            name_len: m.module_name.len(),
            const_lens,
            symbol_lens,
            classes,
            imports,
            closures_data_id,
            closures,
            fn_metas,
            runtime_imports,
        });
    }

    // Bundle-level data: per-search-path name + per-bundled-lib
    // payload. Declared up-front so main() can address them via
    // `declare_data_in_func`.
    let mut search_path_ids: Vec<(cranelift_module::DataId, usize)> =
        Vec::with_capacity(bundle.native_search_paths.len());
    for (k, p) in bundle.native_search_paths.iter().enumerate() {
        let id = define_bytes(module, &format!("wlift_searchpath_{}", k), p.as_bytes())?;
        search_path_ids.push((id, p.len()));
    }
    let mut native_lib_ids: Vec<(
        cranelift_module::DataId,
        usize,
        cranelift_module::DataId,
        usize,
    )> = Vec::with_capacity(bundle.native_libs.len());
    for (k, (name, payload)) in bundle.native_libs.iter().enumerate() {
        let name_id = define_bytes(
            module,
            &format!("wlift_nativelib_name_{}", k),
            name.as_bytes(),
        )?;
        let payload_id = define_bytes(module, &format!("wlift_nativelib_data_{}", k), payload)?;
        native_lib_ids.push((name_id, name.len(), payload_id, payload.len()));
    }

    // ── main(argc, argv) -> i32 ──
    let mut main_sig = Signature::new(cc);
    main_sig.params.push(AbiParam::new(types::I32)); // argc
    main_sig.params.push(AbiParam::new(ptr_ty)); // argv
    main_sig.returns.push(AbiParam::new(types::I32));
    let main_id = module
        .declare_function("main", Linkage::Export, &main_sig)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, main_id.as_u32()), main_sig);
    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_err_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);

        // Stack slot for the JitContext save buffer (128 bytes —
        // JitContext is 80 today, the headroom keeps the
        // bootstrap stable across runtime additions).
        let saved_slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            128,
            3, // 8-byte alignment (log2)
        ));

        // Resolve runtime imports inside main's scope.
        let new_vm_ref = module.declare_func_in_func(wren_new_vm, builder.func);
        let free_vm_ref = module.declare_func_in_func(wren_free_vm, builder.func);
        let init_prelude_ref = module.declare_func_in_func(init_prelude, builder.func);
        let alloc_const_ref = module.declare_func_in_func(alloc_const, builder.func);
        let intern_symbol_ref = module.declare_func_in_func(intern_symbol, builder.func);
        let register_closure_ref = module.declare_func_in_func(register_closure, builder.func);
        let register_code_range_ref =
            module.declare_func_in_func(register_code_range, builder.func);
        let resolve_runtime_import_ref =
            module.declare_func_in_func(resolve_runtime_import, builder.func);
        let install_class_ref = module.declare_func_in_func(install_class, builder.func);
        let bind_foreign_class_ref = module.declare_func_in_func(bind_foreign_class, builder.func);
        let add_native_search_path_ref =
            module.declare_func_in_func(add_native_search_path, builder.func);
        let install_native_lib_ref = module.declare_func_in_func(install_native_lib, builder.func);
        let enter_ref = module.declare_func_in_func(enter_ctx, builder.func);
        let exit_ref = module.declare_func_in_func(exit_ctx, builder.func);

        // entry: vm = wrenNewVM(NULL); brif vm == 0 → err else body.
        builder.switch_to_block(entry_block);
        let null_cfg = builder.ins().iconst(ptr_ty, 0);
        let vm_call = builder.ins().call(new_vm_ref, &[null_cfg]);
        let vm = builder.inst_results(vm_call)[0];
        let zero = builder.ins().iconst(ptr_ty, 0);
        let is_null = builder.ins().icmp(IntCC::Equal, vm, zero);
        builder
            .ins()
            .brif(is_null, exit_err_block, &[], body_block, &[vm.into()]);

        // body: per-module init + dispatch loop, then free + return 0.
        builder.append_block_param(body_block, ptr_ty);
        builder.switch_to_block(body_block);
        let vm = builder.block_params(body_block)[0];
        let saved_addr = builder.ins().stack_addr(ptr_ty, saved_slot, 0);

        // Bundle-level init (search paths + bundled native libs).
        // Runs before any class install so foreign-binding's
        // dlopen sees the right candidate dirs.
        for (path_id, path_len) in &search_path_ids {
            let gv = module.declare_data_in_func(*path_id, builder.func);
            let addr = builder.ins().global_value(ptr_ty, gv);
            let len = builder.ins().iconst(ptr_ty, *path_len as i64);
            let _ = builder
                .ins()
                .call(add_native_search_path_ref, &[vm, addr, len]);
        }
        for (name_id, name_len, payload_id, payload_len) in &native_lib_ids {
            let name_gv = module.declare_data_in_func(*name_id, builder.func);
            let payload_gv = module.declare_data_in_func(*payload_id, builder.func);
            let name_addr = builder.ins().global_value(ptr_ty, name_gv);
            let payload_addr = builder.ins().global_value(ptr_ty, payload_gv);
            let name_len_v = builder.ins().iconst(ptr_ty, *name_len as i64);
            let payload_len_v = builder.ins().iconst(ptr_ty, *payload_len as i64);
            let _ = builder.ins().call(
                install_native_lib_ref,
                &[vm, name_addr, name_len_v, payload_addr, payload_len_v],
            );
        }

        for m in &per_module {
            let modvars_gv = module.declare_data_in_func(m.modvars_id, builder.func);
            let consts_gv = module.declare_data_in_func(m.consts_id, builder.func);
            let symbols_gv = module.declare_data_in_func(m.symbols_id, builder.func);
            let name_gv = module.declare_data_in_func(m.name_id, builder.func);

            let modvars_addr = builder.ins().global_value(ptr_ty, modvars_gv);
            let consts_addr = builder.ins().global_value(ptr_ty, consts_gv);
            let symbols_addr = builder.ins().global_value(ptr_ty, symbols_gv);
            let name_addr = builder.ins().global_value(ptr_ty, name_gv);

            let modvars_count = builder.ins().iconst(ptr_ty, m.modvars_count as i64);
            let _ = builder
                .ins()
                .call(init_prelude_ref, &[vm, modvars_addr, modvars_count]);

            for (k, text_id) in m.const_text_ids.iter().enumerate() {
                let text_gv = module.declare_data_in_func(*text_id, builder.func);
                let text_addr = builder.ins().global_value(ptr_ty, text_gv);
                let len = builder.ins().iconst(ptr_ty, m.const_lens[k] as i64);
                let alloc_call = builder.ins().call(alloc_const_ref, &[vm, text_addr, len]);
                let str_val = builder.inst_results(alloc_call)[0];
                builder
                    .ins()
                    .store(MemFlags::trusted(), str_val, consts_addr, (k as i32) * 8);
            }

            for (k, name_id) in m.symbol_name_ids.iter().enumerate() {
                let sym_name_gv = module.declare_data_in_func(*name_id, builder.func);
                let sym_name_addr = builder.ins().global_value(ptr_ty, sym_name_gv);
                let len = builder.ins().iconst(ptr_ty, m.symbol_lens[k] as i64);
                let intern_call = builder
                    .ins()
                    .call(intern_symbol_ref, &[vm, sym_name_addr, len]);
                let sym_val = builder.inst_results(intern_call)[0];
                builder
                    .ins()
                    .store(MemFlags::trusted(), sym_val, symbols_addr, (k as i32) * 8);
            }

            // Register each closure body — `wren_make_closure_*`
            // expects a runtime FuncId; AOT bakes a build-time
            // index into `MakeClosure`, then loads the FuncId
            // from this slot table at run time.
            let closures_gv = module.declare_data_in_func(m.closures_data_id, builder.func);
            let closures_addr = builder.ins().global_value(ptr_ty, closures_gv);
            for (k, (body_id, arity)) in m.closures.iter().enumerate() {
                let body_ref = module.declare_func_in_func(*body_id, builder.func);
                let body_addr = builder.ins().func_addr(ptr_ty, body_ref);
                let arity_val = builder.ins().iconst(types::I8, *arity as i64);
                let reg_call = builder
                    .ins()
                    .call(register_closure_ref, &[vm, arity_val, body_addr]);
                let func_id_val = builder.inst_results(reg_call)[0];
                builder.ins().store(
                    MemFlags::trusted(),
                    func_id_val,
                    closures_addr,
                    (k as i32) * 8,
                );
            }

            // Register each AOT function's code range + safepoint
            // metadata so the GC stack walker can find AOT frames
            // by return-address lookup. Without this every GC
            // fired from inside an alloc helper sweeps live spill
            // slots in AOT frames and leaves the body running
            // against freed memory.
            for (fn_id_import, code_size, sps_id, sps_count, roots_id) in &m.fn_metas {
                let fn_ref = module.declare_func_in_func(*fn_id_import, builder.func);
                let fn_addr = builder.ins().func_addr(ptr_ty, fn_ref);
                let code_size_val = builder.ins().iconst(types::I32, *code_size as i64);
                let sps_gv = module.declare_data_in_func(*sps_id, builder.func);
                let sps_addr = builder.ins().global_value(ptr_ty, sps_gv);
                let sps_count_val = builder.ins().iconst(types::I32, *sps_count as i64);
                let roots_gv = module.declare_data_in_func(*roots_id, builder.func);
                let roots_addr = builder.ins().global_value(ptr_ty, roots_gv);
                let _ = builder.ins().call(
                    register_code_range_ref,
                    &[vm, fn_addr, code_size_val, sps_addr, sps_count_val, roots_addr],
                );
            }

            // Resolve bare-builtin imports — `import "socket" for
            // SocketCore`, `import "fs" for FS`, ... The walker
            // skipped these (no on-disk source); the bootstrap
            // asks the runtime to load the module + look up the
            // binding by name and write it into the modvars slot
            // the AOT body reads through `GetModuleVar`.
            //
            // Runs BEFORE class install so a user class with a
            // bare-builtin parent (e.g. `class X is FS { }`) sees
            // a populated parent slot in modvars.
            for (target_slot, mod_id, mod_len, var_id, var_len) in &m.runtime_imports {
                let mod_gv = module.declare_data_in_func(*mod_id, builder.func);
                let mod_addr = builder.ins().global_value(ptr_ty, mod_gv);
                let mod_len_v = builder.ins().iconst(ptr_ty, *mod_len as i64);
                let var_gv = module.declare_data_in_func(*var_id, builder.func);
                let var_addr = builder.ins().global_value(ptr_ty, var_gv);
                let var_len_v = builder.ins().iconst(ptr_ty, *var_len as i64);
                let slot_v = builder.ins().iconst(ptr_ty, *target_slot as i64);
                let _ = builder.ins().call(
                    resolve_runtime_import_ref,
                    &[vm, modvars_addr, slot_v, mod_addr, mod_len_v, var_addr, var_len_v],
                );
            }

            // Cross-module import copies. Earlier manifest loop
            // order means the source module's classes are already
            // installed in its modvars by the time this runs; we
            // just pull each pointer across by slot.
            for (target_slot, src_id, source_slot) in &m.imports {
                let src_gv = module.declare_data_in_func(*src_id, builder.func);
                let src_addr = builder.ins().global_value(ptr_ty, src_gv);
                let val = builder.ins().load(
                    types::I64,
                    MemFlags::trusted(),
                    src_addr,
                    (*source_slot as i32) * 8,
                );
                builder.ins().store(
                    MemFlags::trusted(),
                    val,
                    modvars_addr,
                    (*target_slot as i32) * 8,
                );
            }

            // Install user classes — each gets a stack-allocated
            // descriptor array (32 bytes per method) the bootstrap
            // hands to `wlift_aot_install_class`. The runtime
            // builds the `*mut ObjClass`, registers each method
            // body via `register_aot_function`, and writes the
            // class pointer into modvars[slot] so AOT bodies'
            // `GetModuleVar` reads it directly.
            for class in &m.classes {
                let descs_size = (class.methods.len() * 32).max(8) as u32;
                let descs_slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    descs_size,
                    3,
                ));
                let descs_addr = builder.ins().stack_addr(ptr_ty, descs_slot, 0);

                for (m_idx, (sig_id, sig_len, body_id, arity, flags)) in
                    class.methods.iter().enumerate()
                {
                    let sig_gv = module.declare_data_in_func(*sig_id, builder.func);
                    let sig_addr = builder.ins().global_value(ptr_ty, sig_gv);
                    let body_ref = module.declare_func_in_func(*body_id, builder.func);
                    let body_addr = builder.ins().func_addr(ptr_ty, body_ref);

                    let off = (m_idx * 32) as i32;
                    builder
                        .ins()
                        .store(MemFlags::trusted(), sig_addr, descs_addr, off);
                    let len_val = builder.ins().iconst(ptr_ty, *sig_len as i64);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), len_val, descs_addr, off + 8);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), body_addr, descs_addr, off + 16);
                    let arity_val = builder.ins().iconst(types::I8, *arity as i64);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), arity_val, descs_addr, off + 24);
                    let flags_val = builder.ins().iconst(types::I8, *flags as i64);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), flags_val, descs_addr, off + 25);
                }

                let class_name_gv = module.declare_data_in_func(class.name_id, builder.func);
                let class_name_addr = builder.ins().global_value(ptr_ty, class_name_gv);
                let parent_slot_val = builder.ins().iconst(ptr_ty, class.parent_slot as i64);
                let slot_val = builder.ins().iconst(ptr_ty, class.slot as i64);
                let class_name_len = builder.ins().iconst(ptr_ty, class.name_len as i64);
                let num_fields_val = builder.ins().iconst(types::I32, class.num_fields as i64);
                let methods_count = builder.ins().iconst(ptr_ty, class.methods.len() as i64);

                let _ = builder.ins().call(
                    install_class_ref,
                    &[
                        vm,
                        modvars_addr,
                        slot_val,
                        class_name_addr,
                        class_name_len,
                        parent_slot_val,
                        num_fields_val,
                        descs_addr,
                        methods_count,
                    ],
                );

                // Foreign-class binding — only emitted when the
                // class declared `#!native = "..."`. Builds a
                // descriptor array on the stack (24 bytes per
                // method: sig*8 + sig_len*8 + symbol*8 +
                // sym_len*8 + flags*1 padded to 32) and hands it
                // to `wlift_aot_bind_foreign_class`.
                if let Some(foreign) = &class.foreign {
                    // `WliftAotForeignMethodDesc` is repr(C) — 5
                    // 8-byte fields (sig*, sig_len, symbol*,
                    // symbol_len, flags-padded-to-8) = 40 bytes.
                    let f_size = (foreign.methods.len() * 40).max(8) as u32;
                    let f_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        f_size,
                        3,
                    ));
                    let f_addr = builder.ins().stack_addr(ptr_ty, f_slot, 0);

                    for (k, (sig_id, sig_len, sym_opt, sym_len, flags)) in
                        foreign.methods.iter().enumerate()
                    {
                        let sig_gv = module.declare_data_in_func(*sig_id, builder.func);
                        let sig_addr = builder.ins().global_value(ptr_ty, sig_gv);
                        let sym_addr = match sym_opt {
                            Some(id) => {
                                let sgv = module.declare_data_in_func(*id, builder.func);
                                builder.ins().global_value(ptr_ty, sgv)
                            }
                            None => builder.ins().iconst(ptr_ty, 0),
                        };
                        let off = (k * 40) as i32;
                        builder
                            .ins()
                            .store(MemFlags::trusted(), sig_addr, f_addr, off);
                        let len_v = builder.ins().iconst(ptr_ty, *sig_len as i64);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), len_v, f_addr, off + 8);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), sym_addr, f_addr, off + 16);
                        let sym_len_v = builder.ins().iconst(ptr_ty, *sym_len as i64);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), sym_len_v, f_addr, off + 24);
                        let flags_v = builder.ins().iconst(types::I8, *flags as i64);
                        builder
                            .ins()
                            .store(MemFlags::trusted(), flags_v, f_addr, off + 32);
                    }

                    let lib_gv = module.declare_data_in_func(foreign.lib_id, builder.func);
                    let lib_addr = builder.ins().global_value(ptr_ty, lib_gv);
                    let lib_len_v = builder.ins().iconst(ptr_ty, foreign.lib_len as i64);
                    let f_count = builder.ins().iconst(ptr_ty, foreign.methods.len() as i64);
                    let _ = builder.ins().call(
                        bind_foreign_class_ref,
                        &[
                            vm,
                            modvars_addr,
                            slot_val,
                            lib_addr,
                            lib_len_v,
                            f_addr,
                            f_count,
                        ],
                    );
                }
            }

            let name_len = builder.ins().iconst(ptr_ty, m.name_len as i64);
            let _ = builder.ins().call(
                enter_ref,
                &[
                    vm,
                    modvars_addr,
                    modvars_count,
                    name_addr,
                    name_len,
                    saved_addr,
                ],
            );

            let fn_ref = module.declare_func_in_func(m.fn_id, builder.func);
            let _ = builder.ins().call(fn_ref, &[]);

            let _ = builder.ins().call(exit_ref, &[saved_addr]);
        }

        let _ = builder.ins().call(free_vm_ref, &[vm]);
        let zero_ret = builder.ins().iconst(types::I32, 0);
        builder.ins().return_(&[zero_ret]);

        // err: return 70 (RuntimeError exit code).
        builder.switch_to_block(exit_err_block);
        let err_val = builder.ins().iconst(types::I32, 70);
        builder.ins().return_(&[err_val]);

        builder.seal_all_blocks();
        builder.finalize();
    }

    module
        .define_function(main_id, &mut ctx)
        .map_err(|e| AotError::Module(e.to_string()))?;
    module.clear_context(&mut ctx);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase-1 deliverable from the AOT plan: emit a `.o` and
    /// confirm `nm` (or its API equivalent) sees the mangled
    /// symbol. Avoids shelling out to `nm` itself by re-parsing
    /// the bytes through the same `object` crate Cranelift uses
    /// internally — keeps the test self-contained on every host.
    #[test]
    fn emits_object_with_exported_symbol() {
        // Tempfile guarded path so the test cleans up even when
        // assertions abort.
        let tmp = tempfile::Builder::new()
            .prefix("wlift_aot_phase1_")
            .suffix(".o")
            .tempfile()
            .expect("tempfile");
        let path = tmp.path().to_path_buf();
        compile_to_object("class App { static run() { return 42 } }", &path)
            .expect("compile_to_object");

        let bytes = std::fs::read(&path).expect("read object");
        assert!(!bytes.is_empty(), "object file is empty");

        // Parse the produced object and walk its symbol table.
        // `cranelift-object` produces ELF on Linux, Mach-O on
        // macOS, and PE/COFF on Windows — `object::File::parse`
        // handles all three transparently.
        use object::{Object, ObjectSymbol};
        let obj = object::File::parse(&*bytes).expect("parse object");
        let found = obj.symbols().any(|sym| {
            sym.name()
                .map(|n| n == "wlift_aot_main" || n == "_wlift_aot_main")
                .unwrap_or(false)
        });
        assert!(
            found,
            "expected wlift_aot_main in symbol table; got {:?}",
            obj.symbols()
                .filter_map(|s| s.name().ok().map(str::to_string))
                .collect::<Vec<_>>()
        );
    }

    /// Phase-3 walker: a 2-file fixture (`entry.wren` imports
    /// `./helper`) produces two `AotModule` entries in
    /// dependency-first order, with each module's MIR carrying the
    /// expected top-level entry function.
    #[test]
    fn walk_imports_collects_reachable_modules() {
        use std::io::Write;

        let dir = tempfile::Builder::new()
            .prefix("wlift_aot_phase3_")
            .tempdir()
            .expect("tempdir");
        let entry_path = dir.path().join("entry.wren");
        let helper_path = dir.path().join("helper.wren");

        // Two trivial modules — the import is what matters; the
        // bodies just have to parse + resolve cleanly.
        std::fs::File::create(&helper_path)
            .and_then(|mut f| f.write_all(b"class Helper { static value() { return 7 } }\n"))
            .expect("write helper");
        std::fs::File::create(&entry_path)
            .and_then(|mut f| f.write_all(b"import \"./helper\"\nvar x = 1\n"))
            .expect("write entry");

        let modules = walk_imports(&entry_path).expect("walk_imports").modules;
        assert_eq!(
            modules.len(),
            2,
            "expected entry + helper, got {} modules: {:?}",
            modules.len(),
            modules.iter().map(|m| &m.name).collect::<Vec<_>>()
        );

        // Helper canonical paths use the symlink-resolved temp dir,
        // so compare against `canonicalize`'d expectations.
        let helper_canonical = std::fs::canonicalize(&helper_path).unwrap();
        let entry_canonical = std::fs::canonicalize(&entry_path).unwrap();
        assert_eq!(
            modules[0].name,
            helper_canonical.to_string_lossy(),
            "dependency must come before importer in walk order"
        );
        assert_eq!(modules[1].name, entry_canonical.to_string_lossy());
    }

    /// A `./foo` and `../<dir>/foo` referring to the same on-disk
    /// file dedupe to a single `AotModule`. Mirrors the runtime's
    /// canonicalising resolver — without dedupe we'd emit two
    /// copies of the same function and break class identity at
    /// link time the same way two ModuleEntry copies break `is`
    /// at runtime.
    #[test]
    fn walk_imports_dedupes_aliased_paths() {
        use std::io::Write;

        let dir = tempfile::Builder::new()
            .prefix("wlift_aot_phase3_dedupe_")
            .tempdir()
            .expect("tempdir");
        // entry.wren and a.wren both import the same shared.wren
        // — entry directly via "./shared", a.wren via the same
        // "./shared". Two importers, one helper.
        let shared = dir.path().join("shared.wren");
        let a = dir.path().join("a.wren");
        let entry = dir.path().join("entry.wren");

        std::fs::File::create(&shared)
            .and_then(|mut f| f.write_all(b"class Shared {}\n"))
            .expect("write shared");
        std::fs::File::create(&a)
            .and_then(|mut f| f.write_all(b"import \"./shared\"\nvar a = 1\n"))
            .expect("write a");
        std::fs::File::create(&entry)
            .and_then(|mut f| f.write_all(b"import \"./shared\"\nimport \"./a\"\nvar e = 1\n"))
            .expect("write entry");

        let modules = walk_imports(&entry).expect("walk_imports").modules;
        let names: Vec<&String> = modules.iter().map(|m| &m.name).collect();
        let shared_canonical = std::fs::canonicalize(&shared)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let count = names.iter().filter(|n| ***n == shared_canonical).count();
        assert_eq!(
            count,
            1,
            "shared.wren must appear exactly once across {} modules: {:?}",
            modules.len(),
            names
        );
        assert_eq!(modules.len(), 3, "expected shared + a + entry");
    }

    /// Phase-4 multi-module emit: a 2-file fixture produces one
    /// `.o` containing **both** `wlift_aot_main` (entry) and
    /// `wlift_aot_mod_0` (helper, the dependency). The shared
    /// codegen runs unchanged across modules — only the entry
    /// symbol differs.
    #[test]
    fn compile_path_emits_per_module_symbols() {
        use std::io::Write;

        let dir = tempfile::Builder::new()
            .prefix("wlift_aot_phase4_")
            .tempdir()
            .expect("tempdir");
        let entry_path = dir.path().join("entry.wren");
        let helper_path = dir.path().join("helper.wren");

        std::fs::File::create(&helper_path)
            .and_then(|mut f| f.write_all(b"class Helper { static value() { return 7 } }\n"))
            .expect("write helper");
        std::fs::File::create(&entry_path)
            .and_then(|mut f| f.write_all(b"import \"./helper\"\nvar x = 1\n"))
            .expect("write entry");

        let obj_path = dir.path().join("out.o");
        compile_path_to_object(&entry_path, &obj_path).expect("compile_path_to_object");

        let bytes = std::fs::read(&obj_path).expect("read object");
        assert!(!bytes.is_empty(), "object file is empty");

        use object::{Object, ObjectSymbol};
        let obj = object::File::parse(&*bytes).expect("parse object");
        let names: Vec<String> = obj
            .symbols()
            .filter_map(|s| s.name().ok().map(str::to_string))
            .collect();
        let has = |needle: &str| {
            names
                .iter()
                .any(|n| n == needle || n == &format!("_{}", needle))
        };
        assert!(
            has("wlift_aot_main"),
            "expected wlift_aot_main; got {:?}",
            names
        );
        assert!(
            has("wlift_aot_mod_0"),
            "expected wlift_aot_mod_0 (helper top-level); got {:?}",
            names
        );
    }

    /// Static fields touch the new explicit-class lowering: a
    /// class with a static getter + setter compiles to method
    /// bodies that reference `wlift_aot_get_static_field` /
    /// `wlift_aot_set_static_field` (linker-resolved against
    /// the runtime staticlib at build time), instead of the
    /// `wren_get/set_static_field` JIT helpers that read the
    /// uninitialised TLS `defining_class` slot.
    #[test]
    fn static_field_emits_explicit_class_helpers() {
        let tmp = tempfile::Builder::new()
            .prefix("wlift_aot_static_field_")
            .suffix(".o")
            .tempfile()
            .expect("tempfile");
        let path = tmp.path().to_path_buf();
        compile_to_object(
            "class Counter {\n  \
                static n { __n }\n  \
                static incr() { __n = (__n == null ? 0 : __n) + 1 }\n\
             }\n",
            &path,
        )
        .expect("compile_to_object");

        let bytes = std::fs::read(&path).expect("read object");
        use object::{Object, ObjectSymbol};
        let obj = object::File::parse(&*bytes).expect("parse object");
        let names: Vec<String> = obj
            .symbols()
            .filter_map(|s| s.name().ok().map(str::to_string))
            .collect();
        let has = |needle: &str| {
            names
                .iter()
                .any(|n| n == needle || n == &format!("_{}", needle))
        };
        assert!(
            has("wlift_aot_get_static_field"),
            "expected import of wlift_aot_get_static_field; got {:?}",
            names
        );
        assert!(
            has("wlift_aot_set_static_field"),
            "expected import of wlift_aot_set_static_field; got {:?}",
            names
        );
        assert!(
            !has("wren_get_static_field"),
            "should not import wren_get_static_field from a class method body \
             — that path TLS-reads JitContext.defining_class which AOT never \
             populates; got {:?}",
            names
        );
    }
}
