use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, ValueEnum};

use wren_lift::diagnostics::Severity;
use wren_lift::mir::opt::{
    self, constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm, sra::Sra,
    MirPass,
};
use wren_lift::parse::{lexer, parser};
use wren_lift::runtime::engine::{ExecutionMode, InterpretResult};
use wren_lift::runtime::gc_trait::GcStrategy;
use wren_lift::runtime::vm::{VMConfig, VM};
use wren_lift::sema;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// WrenLift — Lightning fast JIT runtime for the Wren programming language.
#[derive(Parser)]
#[command(name = "wlift", version, about)]
struct Cli {
    /// Wren source file to compile/run. Omit to start the REPL.
    file: Option<String>,

    /// Compilation target.
    #[arg(long, value_enum, default_value_t = Target::Native)]
    target: Target,

    /// Execution mode.
    #[arg(long, value_enum, default_value_t = Mode::Tiered)]
    mode: Mode,

    /// Output file path (for WASM target).
    #[arg(short, long)]
    output: Option<String>,

    /// Show lexer output.
    #[arg(long)]
    dump_tokens: bool,

    /// Show parsed AST.
    #[arg(long)]
    dump_ast: bool,

    /// Show MIR before optimization.
    #[arg(long)]
    dump_mir: bool,

    /// Show MIR after optimization.
    #[arg(long)]
    dump_opt: bool,

    /// Show generated machine code.
    #[arg(long)]
    dump_asm: bool,

    /// Skip optimization passes.
    #[arg(long)]
    no_opt: bool,

    /// Print GC statistics after execution.
    #[arg(long)]
    gc_stats: bool,

    /// Report which registered functions are unreachable from any
    /// module's top-level after compilation. Conservative: assumes no
    /// reflection, `Meta.eval`, or other runtime source-to-MIR path.
    #[arg(long)]
    tree_shake_stats: bool,

    /// Compile the input `.wren` source into a portable `.wlbc`
    /// bytecode cache at the given path and exit without running.
    /// Subsequent launches can pass the `.wlbc` path instead of the
    /// source file to skip parse / sema / MIR-build / optimize.
    #[arg(long, value_name = "OUT_PATH")]
    build: Option<String>,

    /// Compile a source tree (the positional `file` argument is used
    /// as the root directory) into a `.hatch` distribution package at
    /// the given path and exit. Every `.wren` file under the tree
    /// becomes a module; the name comes from its path relative to the
    /// root (slashes → dots). If the tree contains a `hatch.toml` it
    /// is used as-is; otherwise a minimal manifest is synthesised.
    #[arg(long, value_name = "OUT_PATH")]
    bundle: Option<String>,

    /// Compile the input `.wren` source tree into a self-contained
    /// native executable at the given path and exit. Produces an AOT
    /// object via the Cranelift `ObjectModule` pipeline (every
    /// reachable module's top-level lowered into one `.o`), then
    /// invokes the system `cc` to link it with the WrenLift runtime
    /// staticlib. The resulting binary embeds the program source and
    /// runs it through the bundled interpreter — the AOT'd native
    /// code is included in the binary's symbol table for forthcoming
    /// dispatch.
    ///
    /// Requires the `aot` cargo feature at WrenLift build time. The
    /// runtime staticlib path is read from `WLIFT_STATICLIB`, or
    /// auto-discovered next to the running `wlift` binary
    /// (`<exe_dir>/libwren_lift.a` / `target/<profile>/libwren_lift.a`).
    #[arg(long, value_name = "OUT_PATH")]
    aot: Option<String>,

    /// Target triple for `--bundle`. Defaults to host-family
    /// (recorded as `target = "native"` in the manifest). Pass
    /// `wasm32` (family marker) or a concrete `wasm32-*` triple
    /// (e.g. `wasm32-unknown-unknown`, `wasm32-wasip1`) to
    /// produce a hatch loadable by the wasm runtime; the builder
    /// skips packing host `.dylib`/`.so` bytes for those targets
    /// since wasm runtimes use statically-linked plugins.
    /// Loader-side `check_target_compat` rejects cross-family
    /// mismatches at install time. Distinct from `--target`
    /// (which selects the codegen backend for direct execution).
    #[arg(long, value_name = "TRIPLE", requires = "bundle")]
    bundle_target: Option<String>,

    /// Print the manifest + section listing of a `.hatch` package and
    /// exit without running. Accepts the positional `file` argument
    /// as the hatch path.
    #[arg(long)]
    inspect: bool,

    /// Generate HTML documentation for a source tree (the positional
    /// `file` argument is the root directory) and exit. One HTML file
    /// per module gets written under the given output directory; an
    /// `index.html` lists every module. Doc bodies come from `///` and
    /// `//!` comments, rendered as CommonMark.
    #[arg(long, value_name = "OUT_DIR")]
    docs: Option<String>,

    /// Maximum interpreter steps before aborting. Pass `0` to disable
    /// the limit entirely — recommended for long-running servers
    /// where the default cap (1B interp / 10B tiered) caps out after
    /// ~10-30 minutes of polling-loop instructions.
    #[arg(long)]
    step_limit: Option<usize>,

    /// Baseline warmup threshold before optimize-tier compilation.
    #[arg(long)]
    opt_threshold: Option<u32>,

    /// Garbage collector strategy.
    #[arg(long, value_enum, default_value_t = GcMode::Generational)]
    gc: GcMode,

    /// Enable SIGUSR1-driven in-process hot reload.
    ///
    /// When passed, `wlift` installs a SIGUSR1 handler that flips a
    /// reload-pending flag; the interpreter checks the flag at safe-
    /// points and re-installs any user module whose on-disk mtime
    /// has advanced since install. Designed for the `hatch` dev
    /// supervisor — it watches the workspace and signals the
    /// `wlift` child on every file save.
    #[arg(long)]
    watch: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum GcMode {
    /// Generational nursery + old gen mark-sweep (default).
    Generational,
    /// Allocate-only, free on drop. Best for short-lived scripts / benchmarks.
    Arena,
    /// Simple non-generational mark-sweep.
    MarkSweep,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Target {
    Native,
    Wasm,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Walk MIR directly. Never JIT-compile.
    Interpreter,
    /// Start interpreted, JIT-compile hot functions.
    Tiered,
    /// Compile everything to native before execution.
    Jit,
}

impl From<Mode> for ExecutionMode {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Interpreter => ExecutionMode::Interpreter,
            Mode::Tiered => ExecutionMode::Tiered,
            Mode::Jit => ExecutionMode::Jit,
        }
    }
}

// ---------------------------------------------------------------------------
// VM setup
// ---------------------------------------------------------------------------

fn make_vm(cli: &Cli) -> VM {
    make_vm_with_loader(cli, None)
}

/// Install signal-driven hot reload when `--watch` was passed.
///
/// Idempotent — calling again from a different VM still binds the
/// same handler. Safe to invoke before or after VM construction.
fn maybe_install_reload_signal(cli: &Cli) {
    if cli.watch {
        wren_lift::runtime::vm::install_reload_signal_handler();
    }
}

fn make_vm_with_loader(cli: &Cli, source_dir: Option<PathBuf>) -> VM {
    let mode = cli.mode.into();
    let step_limit = cli.step_limit.unwrap_or(match mode {
        ExecutionMode::Interpreter => 1_000_000_000,
        _ => 10_000_000_000, // tiered/jit: 10x headroom since JIT code doesn't count steps
    });
    let gc_strategy = match cli.gc {
        GcMode::Generational => GcStrategy::Generational,
        GcMode::Arena => GcStrategy::Arena,
        GcMode::MarkSweep => GcStrategy::MarkSweep,
    };
    let (load_module_fn, resolve_module_fn) = match source_dir {
        Some(dir) => {
            let (l, r) = make_module_io(dir);
            (Some(l), Some(r))
        }
        None => (None, None),
    };
    let config = VMConfig {
        execution_mode: mode,
        step_limit,
        gc_strategy,
        opt_threshold: cli
            .opt_threshold
            .unwrap_or(VMConfig::default().opt_threshold),
        load_module_fn,
        resolve_module_fn,
        ..VMConfig::default()
    };
    VM::new(config)
}

// ---------------------------------------------------------------------------
// Module loader + spec-dep pre-installer
// ---------------------------------------------------------------------------

/// Append `.wren` to a relative-import candidate path **without
/// stripping any existing extension**.
///
/// `Path::with_extension("wren")` *replaces* the trailing `.X`
/// segment, so `import "./foo.spec"` would canonicalise to
/// `foo.wren` instead of `foo.spec.wren`. That blocks the
/// `hatch test` per-test wrapper pattern (a runner file imports
/// `./<spec basename>` to drive a `*.spec.wren` source) and
/// silently redirects loads onto a same-stem sibling file.
///
/// Behaviour:
/// - Path already ends in `.wren` → leave untouched.
/// - Anything else (no extension, `.spec`, `.foo.bar`, …) →
///   append `.wren` so `foo` → `foo.wren` and `foo.spec` →
///   `foo.spec.wren`.
fn with_wren_suffix(path: PathBuf) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("wren") {
        return path;
    }
    let mut s = path.into_os_string();
    s.push(".wren");
    PathBuf::from(s)
}

/// Build the loader + resolver pair that the VM's import flow uses.
///
/// The resolver canonicalises relative imports (`./foo`, `../foo`)
/// to absolute on-disk paths so two different relative spellings
/// of the same file collapse to a single ModuleEntry — without
/// this, `./live` and `../live` register as two distinct modules,
/// each with its own copy of the file's classes, and `is`-checks
/// across the boundary fail.
///
/// The loader returns the source text for the canonical module
/// name. Both share an internal `dirs` registry so that an import
/// from inside an already-loaded module resolves against THAT
/// module's directory, not the root entry's — pinning everything to
/// the root would break chains like `examples/foo.wren → ../web →
/// ./css`, where `./css` must look next to `web.wren`.
///
/// Scoped names (`@hatch:foo`) and missing files leave the
/// canonical name unchanged — the engine falls back to its
/// builtin / pre-installed module table.
#[allow(clippy::type_complexity)]
fn make_module_io(
    running_file_dir: PathBuf,
) -> (
    Box<dyn Fn(&str, &str) -> Option<String>>,
    Box<dyn Fn(&str, &str) -> Option<String>>,
) {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    // dirs maps a CANONICAL module name -> the directory that name
    // lives in. Resolving a relative import against an already-known
    // canonical module just looks up its dir here.
    let dirs: Rc<RefCell<HashMap<String, PathBuf>>> = Rc::new(RefCell::new(HashMap::new()));
    // Root entry has empty importer name — record root dir.
    dirs.borrow_mut()
        .insert(String::new(), running_file_dir.clone());

    let resolve_to_canonical = {
        let dirs = Rc::clone(&dirs);
        let root_dir = running_file_dir.clone();
        Box::new(move |name: &str, from: &str| -> Option<String> {
            // Scoped names and bare-non-relative pass through as-is
            // — they're builtin modules or scoped packages keyed by
            // their canonical scope-string already.
            if !(name.starts_with("./") || name.starts_with("../")) {
                return None;
            }
            let base_dir = dirs
                .borrow()
                .get(from)
                .cloned()
                .unwrap_or_else(|| root_dir.clone());
            let rel = Path::new(name);
            let candidate = with_wren_suffix(base_dir.join(rel));
            // Try fs::canonicalize for a normalised absolute path
            // (resolves `..` segments and symlinks). Fall back to
            // `candidate.to_string_lossy()` if the file doesn't
            // exist — the loader will surface the missing-file
            // error in that case rather than masking it here.
            let canonical = std::fs::canonicalize(&candidate)
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| candidate.to_string_lossy().into_owned());
            Some(canonical)
        }) as Box<dyn Fn(&str, &str) -> Option<String>>
    };

    let load_source = {
        let dirs = Rc::clone(&dirs);
        let root_dir = running_file_dir.clone();
        // Wrap raw file reads through `cfg::apply` so `#!native` /
        // `#!wasm` attribute lines on imports get filtered out at
        // load time, the same way `hatch build` filters them at
        // bundle time. Without this, running `wlift pkg.spec.wren`
        // directly hits a parser error on any package source that
        // gates imports per-target — which is the canonical shape
        // for cross-target packages (e.g. @hatch:assets, @hatch:gpu).
        // `None` selects the host arm (drop `#!wasm`, keep `#!native`).
        let host_filter = |text: String| -> String { wren_lift::parse::cfg::apply(&text, None) };
        Box::new(move |name: &str, from: &str| -> Option<String> {
            // The VM passes the canonical name back here for
            // relative imports (resolver returned an absolute
            // path) — the file is reachable directly.
            if Path::new(name).is_absolute() {
                let p = Path::new(name);
                let text = fs::read_to_string(p).ok()?;
                if let Some(parent) = p.parent() {
                    dirs.borrow_mut()
                        .insert(name.to_string(), parent.to_path_buf());
                }
                return Some(host_filter(text));
            }

            // Otherwise resolve against the importer's directory
            // (or root if not yet recorded).
            let base_dir = dirs
                .borrow()
                .get(from)
                .cloned()
                .unwrap_or_else(|| root_dir.clone());

            if name.starts_with("./") || name.starts_with("../") {
                let candidate = with_wren_suffix(base_dir.join(name));
                let text = fs::read_to_string(&candidate).ok()?;
                if let Some(parent) = candidate.parent() {
                    dirs.borrow_mut()
                        .insert(name.to_string(), parent.to_path_buf());
                }
                return Some(host_filter(text));
            }
            // Bare names (no scope chars) — sibling file in the
            // same dir as the importer.
            let is_scoped = name.chars().any(|c| matches!(c, ':' | '@' | '/'));
            if is_scoped {
                return None;
            }
            let candidate = base_dir.join(format!("{}.wren", name));
            let text = fs::read_to_string(&candidate).ok()?;
            if let Some(parent) = candidate.parent() {
                dirs.borrow_mut()
                    .insert(name.to_string(), parent.to_path_buf());
            }
            Some(host_filter(text))
        }) as Box<dyn Fn(&str, &str) -> Option<String>>
    };

    (load_source, resolve_to_canonical)
}

/// Look for a `hatchfile` next to (or above) the running file; if one
/// exists, resolve every `[spec-dependencies]` entry through the same
/// machinery `hatch build` uses (path → recursive build, version →
/// `~/.hatch/cache/...`, git → cached checkout) and install each
/// resulting `.hatch` into the VM. Imports like `@hatch:test` then hit
/// an already-loaded module.
fn preinstall_spec_dependencies(vm: &mut VM, source_dir: &Path) -> Result<(), String> {
    let Some(hatchfile) = find_hatchfile(source_dir) else {
        return Ok(());
    };
    let text = fs::read_to_string(&hatchfile)
        .map_err(|e| format!("reading {}: {}", hatchfile.display(), e))?;
    let manifest: wren_lift::hatch::Manifest =
        toml::from_str(&text).map_err(|e| format!("parsing {}: {}", hatchfile.display(), e))?;

    let workspace_root = hatchfile.parent().unwrap_or(Path::new("."));

    // Apply the package's own native-lib declarations to the VM so
    // spec runs resolve `#!native = "libname"` without needing the
    // source tree to be packaged into a `.hatch` first. Relative
    // paths anchor on the hatchfile's directory.
    vm.apply_hatch_native_manifest_rooted(&manifest, workspace_root);

    // Install `[dependencies]` too — when the spec imports the
    // package under test and that package in turn imports a real
    // dep, the dep has to already be resolvable. Specs declared in
    // `[spec-dependencies]` layer on top.
    for (dep_name, dep) in manifest
        .dependencies
        .iter()
        .chain(manifest.spec_dependencies.iter())
    {
        let bytes = wren_lift::hatch::resolve_dependency_bytes(workspace_root, dep_name, dep, None)
            .map_err(|e| format!("resolving dep '{}': {}", dep_name, e))?;
        match vm.install_hatch_modules(&bytes) {
            InterpretResult::Success => {}
            InterpretResult::CompileError => {
                return Err(format!("compile error installing dep '{}'", dep_name));
            }
            InterpretResult::RuntimeError => {
                return Err(format!("runtime error installing dep '{}'", dep_name));
            }
        }
    }
    Ok(())
}

/// Walk upward from `start` until a `hatchfile` is found. Returns the
/// absolute path to the hatchfile, or `None` if none exists.
fn find_hatchfile(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start.to_path_buf());
    while let Some(d) = dir {
        let candidate = d.join("hatchfile");
        if candidate.exists() {
            return Some(candidate);
        }
        dir = d.parent().map(Path::to_path_buf);
    }
    None
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn run_file(source: &str, filename: &str, cli: &Cli) {
    // --- Debug dump paths (don't need VM) ---

    // 1. Lex
    if cli.dump_tokens {
        let (lexemes, _docs, errors) = lexer::lex(source);
        for err in &errors {
            err.eprint(source);
        }
        for lex in &lexemes {
            println!(
                "{:>4}..{:<4}  {:?}  {:?}",
                lex.span.start, lex.span.end, lex.token, lex.text
            );
        }
        if !errors.is_empty() {
            process::exit(1);
        }
        return;
    }

    // 2. Parse
    let parse_result = parser::parse(source);
    let has_parse_errors = parse_result
        .errors
        .iter()
        .any(|d| d.severity == Severity::Error);

    if has_parse_errors {
        for err in &parse_result.errors {
            err.eprint(source);
        }
        let n = parse_result
            .errors
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count();
        eprintln!(
            "{}: {} parse error{}",
            filename,
            n,
            if n == 1 { "" } else { "s" }
        );
        process::exit(1);
    }

    if cli.dump_ast {
        for stmt in &parse_result.module {
            println!("{:#?}", stmt);
        }
        return;
    }

    // For dump_mir/dump_opt/dump_asm/wasm, use the manual pipeline
    if cli.dump_mir || cli.dump_opt || cli.dump_asm || cli.target == Target::Wasm {
        run_manual_pipeline(source, filename, cli, parse_result);
        return;
    }

    // --- Execution path: route through VM ---

    let source_dir = Path::new(filename)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    maybe_install_reload_signal(cli);
    let mut vm = make_vm_with_loader(cli, Some(source_dir.clone()));

    // Resolve `[spec-dependencies]` declared in a sibling `hatchfile`
    // through the ambient hatch cache and install them so imports like
    // `@hatch:test` find an already-loaded module.
    if let Err(e) = preinstall_spec_dependencies(&mut vm, &source_dir) {
        eprintln!("error: {}", e);
        process::exit(1);
    }

    let module_name = filename.strip_suffix(".wren").unwrap_or(filename);

    match vm.interpret(module_name, source) {
        InterpretResult::Success => {}
        InterpretResult::CompileError => {
            process::exit(65);
        }
        InterpretResult::RuntimeError => {
            process::exit(70);
        }
    }

    if cli.tree_shake_stats {
        let report = wren_lift::mir::opt::tree_shake::analyse(&vm.engine);
        eprintln!("--- Tree-shake ---");
        eprintln!("  total functions: {}", report.total);
        eprintln!("  reachable:       {}", report.reachable);
        eprintln!("  unreachable:     {}", report.dead.len());
        for id in &report.dead {
            let name = vm
                .engine
                .get_mir(*id)
                .map(|mir| vm.interner.resolve(mir.name).to_string())
                .unwrap_or_else(|| "<missing mir>".to_string());
            eprintln!("    FuncId({}) {}", id.0, name);
        }
    }

    if cli.gc_stats {
        let stats = vm.gc.stats();
        eprintln!("--- GC Stats ---");
        eprintln!("  minor collections: {}", stats.minor_collections);
        eprintln!("  major collections: {}", stats.major_collections);
        eprintln!("  objects allocated:  {}", stats.objects_allocated);
        eprintln!("  objects freed:      {}", stats.objects_freed);
        eprintln!("  objects promoted:   {}", stats.objects_promoted);
        eprintln!("  peak objects:       {}", stats.peak_objects);
        eprintln!("  total allocated:    {} KB", stats.total_allocated / 1024);
        eprintln!("  total freed:        {} KB", stats.total_freed / 1024);
        eprintln!(
            "  gc time:            {:.3}s",
            stats.gc_time_ns as f64 / 1e9
        );
    }
}

/// Manual pipeline for debug dumps and WASM codegen.
fn run_manual_pipeline(
    source: &str,
    filename: &str,
    cli: &Cli,
    parse_result: wren_lift::parse::parser::ParseResult,
) {
    // Semantic analysis
    let mut interner = parse_result.interner;
    let resolve_result = sema::resolve::resolve(&parse_result.module, &interner);

    if !resolve_result.errors.is_empty() {
        let has_sema_errors = resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error);
        for err in &resolve_result.errors {
            err.eprint(source);
        }
        if has_sema_errors {
            let n = resolve_result
                .errors
                .iter()
                .filter(|d| d.severity == Severity::Error)
                .count();
            eprintln!(
                "{}: {} semantic error{}",
                filename,
                n,
                if n == 1 { "" } else { "s" }
            );
            process::exit(1);
        }
    }

    // Lower to MIR
    let mut module_mir =
        wren_lift::mir::builder::lower_module(&parse_result.module, &mut interner, &resolve_result);
    let mir = &mut module_mir.top_level;

    if cli.dump_mir {
        println!("{}", mir.pretty_print(&interner));
        for class in &module_mir.classes {
            println!("\n=== class {} ===", interner.resolve(class.name));
            for method in &class.methods {
                println!("\n--- method {} ---", method.signature);
                println!("{}", method.mir.pretty_print(&interner));
            }
        }
        return;
    }

    // Optimize
    if !cli.no_opt {
        run_opt_pipeline(mir, &interner);
        for class in &mut module_mir.classes {
            for method in &mut class.methods {
                run_opt_pipeline(&mut method.mir, &interner);
            }
        }
    }

    if cli.dump_opt {
        println!("{}", mir.pretty_print(&interner));
        for class in &module_mir.classes {
            println!("\n=== class {} ===", interner.resolve(class.name));
            for method in &class.methods {
                println!("\n--- method {} ---", method.signature);
                println!("{}", method.mir.pretty_print(&interner));
            }
        }
        return;
    }

    // Code generation
    match cli.target {
        Target::Wasm => {
            let wasm_module = match wren_lift::codegen::wasm::emit_mir(mir) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: WASM codegen failed: {}", e);
                    process::exit(1);
                }
            };

            let output_path = cli.output.as_deref().unwrap_or("output.wasm");
            match fs::write(output_path, &wasm_module.bytes) {
                Ok(_) => println!("Wrote {} bytes to {}", wasm_module.bytes.len(), output_path),
                Err(e) => {
                    eprintln!("error: failed to write '{}': {}", output_path, e);
                    process::exit(1);
                }
            }
        }
        Target::Native => {
            let mach_func = wren_lift::codegen::lower_mir(mir);

            if cli.dump_asm {
                println!("{}", mach_func.display());
            }
        }
    }
}

fn run_opt_pipeline(mir: &mut wren_lift::mir::MirFunction, interner: &wren_lift::intern::Interner) {
    let constfold = ConstFold;
    let dce = Dce;
    let cse = Cse::default();
    let type_spec = TypeSpecialize::with_math(interner);
    let licm = Licm;
    let sra = Sra;

    let passes: Vec<&dyn MirPass> = vec![
        &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
    ];
    opt::run_to_fixpoint(mir, &passes, 10);
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

fn run_repl() {
    let cli_args: Vec<String> = std::env::args().collect();
    let cli = Cli::parse_from(&cli_args);

    println!("WrenLift REPL (type Ctrl-D to exit)");
    println!("Mode: {:?}", ExecutionMode::from(cli.mode));
    println!();

    let mut vm = make_vm(&cli);

    let stdin = io::stdin();
    let mut line_num: u32 = 0;
    loop {
        print!("> ");
        if io::stdout().flush().is_err() {
            break;
        }

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("error: {}", e);
                break;
            }
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        line_num += 1;
        let module_name = format!("repl_{}", line_num);

        match vm.interpret(&module_name, line) {
            InterpretResult::Success => {}
            InterpretResult::CompileError => {
                // Error already printed by vm.interpret
            }
            InterpretResult::RuntimeError => {
                // Error already printed by vm.interpret
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Compile a source file to a `.wlbc` bytecode cache and write it.
///
/// `--build` short-circuits execution — no fiber is run. If compilation
/// fails, diagnostics are printed to stderr and the process exits with
/// a compile-error code so shells / build systems can branch on it.
fn build_bytecode_cache(source: &str, filename: &str, out_path: &str, cli: &Cli) {
    // Route through the same parse / sema / MIR / opt passes the VM
    // would run on `interpret`; we just stop before installing the
    // module and serialize the compiled artifact instead.
    let mut vm = make_vm(cli);
    let bytes = match vm.compile_source_to_blob(source) {
        Ok(b) => b,
        Err(InterpretResult::CompileError) => process::exit(65),
        Err(_) => process::exit(70),
    };
    if let Err(e) = fs::write(out_path, &bytes) {
        eprintln!("error: cannot write '{}': {}", out_path, e);
        process::exit(1);
    }
    eprintln!(
        "built {} bytes from {} → {}",
        bytes.len(),
        filename,
        out_path
    );
}

/// Walk a source tree, compile every `.wren` file, write the result
/// as a `.hatch` package. `root` is the positional `file` argument.
/// AOT build driver: emit the input's import-graph as a single
/// object via Cranelift, then link it with the WrenLift runtime
/// staticlib + a generated C bootstrap into a self-contained
/// executable.
///
/// Bootstrap shape: a tiny `main()` that calls
/// `wlift_run_embedded(NULL, WLIFT_AOT_SOURCE)` — the runtime
/// helper takes care of VM construction, interpretation, and
/// exit-code mapping. The program source is embedded as a
/// `static const char[]` with literal byte values, so any source
/// content survives the C-string escape rules.
///
/// Locates the runtime staticlib via the `WLIFT_STATICLIB` env
/// var first; falls back to `<wlift's exe dir>/libwren_lift.a`
/// and then `target/{release,debug}/libwren_lift.a` relative to
/// the current working directory. Documents both options in the
/// error path so a fresh `cargo install`-installed user knows how
/// to point at one.
#[cfg(feature = "aot")]
fn aot_build_executable(input: &str, out_path: &str) {
    use std::process::Command;

    let entry_path = std::path::PathBuf::from(input);
    if !entry_path.is_file() {
        eprintln!(
            "error: --aot expects a `.wren` source file or `.hatch` archive (got '{}')",
            input
        );
        process::exit(1);
    }

    // Walk imports up front so the same dependency-first list
    // drives both the object emit (one CLIF function per module)
    // and the bootstrap shim (embeds each module's source bytes
    // under its as-written `import "..."` name).
    let walk_result = match wren_lift::codegen::aot::walk_imports(&entry_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: AOT import walk failed: {}", e);
            process::exit(1);
        }
    };

    let staticlib_path = match locate_wren_lift_staticlib() {
        Some(p) => p,
        None => {
            eprintln!("error: could not locate libwren_lift.a — set WLIFT_STATICLIB to its");
            eprintln!("       full path, or run `wlift --aot` from a checkout where");
            eprintln!("       `cargo build --release --features aot` has been executed.");
            process::exit(1);
        }
    };

    let work_dir = match tempfile::Builder::new().prefix("wlift_aot_").tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: tempdir: {}", e);
            process::exit(1);
        }
    };
    let obj_path = work_dir.path().join("program.o");

    // Cranelift emits everything — every Wren function, the
    // per-module data symbols, AND the program entry (`main`)
    // that drives runtime init + dispatch — into one object.
    // The link step is a pure linker call against the runtime
    // staticlib; no C bootstrap source involved.
    let _manifests = match wren_lift::codegen::aot::compile_walk_to_object_with_manifest(
        &walk_result.modules,
        &walk_result.bundle,
        &obj_path,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: AOT object emit failed: {}", e);
            process::exit(1);
        }
    };
    if std::env::var_os("WLIFT_AOT_KEEP_OBJ").is_some() {
        let keep = std::path::PathBuf::from(format!("{}.o", out_path));
        let _ = std::fs::copy(&obj_path, &keep);
        eprintln!("wlift: kept object at {}", keep.display());
    }

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut cmd = Command::new(&cc);
    cmd.arg(&obj_path)
        .arg(&staticlib_path)
        .arg("-o")
        .arg(out_path);
    // System libraries the runtime staticlib pulls in (pthread,
    // dl, math, libc++ runtime). macOS frameworks live behind
    // `-framework`. Windows would land later — `wlift build`
    // there needs lib.exe / link.exe, separate driver story.
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

    let status = match cmd.status() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: invoking `{}`: {}", cc, e);
            eprintln!(
                "       Set CC to a compiler binary (e.g. `gcc`, `clang`) if `cc` isn't on PATH."
            );
            process::exit(1);
        }
    };
    if !status.success() {
        eprintln!("error: linker exited with {}", status);
        process::exit(1);
    }
    eprintln!("wlift: produced {}", out_path);
}

#[cfg(not(feature = "aot"))]
fn aot_build_executable(_input: &str, _out_path: &str) {
    eprintln!(
        "error: --aot requires `wlift` to be built with `--features aot`. \
         Rebuild via `cargo install --features aot wren_lift` or a source checkout \
         that ships the AOT pipeline."
    );
    process::exit(1);
}

/// Walk a small set of well-known locations for the WrenLift
/// runtime staticlib (`libwren_lift.a`). Returns the first
/// existing file. Order: `WLIFT_STATICLIB` env var (explicit
/// override) → next to `wlift` itself → `target/release/...` →
/// `target/debug/...` from CWD.
#[cfg(feature = "aot")]
fn locate_wren_lift_staticlib() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("WLIFT_STATICLIB") {
        let path = std::path::PathBuf::from(p);
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
        let candidate = std::path::PathBuf::from("target")
            .join(profile)
            .join(staticlib_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// `target` (when `Some`) is stamped into the manifest's `target`
/// field and gates wasm-specific build behavior (skip packing
/// host `.dylib`/`.so` bytes); `None` preserves legacy behavior
/// (target-agnostic bundle).
fn build_hatch_package(root: &str, out_path: &str, target: Option<&str>) {
    let root_path = std::path::PathBuf::from(root);
    if !root_path.is_dir() {
        eprintln!(
            "error: --bundle expects the positional argument to be a directory (got '{}')",
            root
        );
        process::exit(1);
    }
    let bytes = match wren_lift::hatch::build_from_source_tree_for_target(&root_path, None, target)
    {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(65);
        }
    };
    if let Err(e) = fs::write(out_path, &bytes) {
        eprintln!("error: cannot write '{}': {}", out_path, e);
        process::exit(1);
    }
    let target_note = target
        .map(|t| format!(" [target={}]", t))
        .unwrap_or_default();
    eprintln!(
        "bundled {} bytes from {} → {}{}",
        bytes.len(),
        root,
        out_path,
        target_note
    );
}

/// Walk a source root (file or directory), parse every `.wren`
/// file, collect `///` and `//!` comments alongside the typed AST,
/// and emit one HTML page per module plus an `index.html` linking
/// them. Doc bodies render as CommonMark.
///
/// Layout in `<out_dir>/`:
///
/// ```text
/// out_dir/
///   index.html         — module list + any module-level summary
///   <module>.html      — one page per .wren file (slashes → dashes)
/// ```
fn generate_docs(root: &str, out_dir: &str) {
    let root_path = std::path::PathBuf::from(root);
    let out_path = std::path::PathBuf::from(out_dir);
    if let Err(e) = fs::create_dir_all(&out_path) {
        eprintln!("error: cannot create '{}': {}", out_dir, e);
        process::exit(1);
    }

    // .hatch bundle input: extract source sections, run the
    // collector per module. Lets the publish pipeline emit docs
    // for any registry artefact without a source checkout.
    if root_path.is_file() {
        if let Ok(bytes) = fs::read(&root_path) {
            if wren_lift::hatch::looks_like_hatch(&bytes) {
                generate_docs_from_hatch(&bytes, &out_path);
                return;
            }
        }
    }

    // Collect every .wren file under the root. Single-file inputs
    // pass through unchanged. Order: alphabetical by qualified
    // module name so the manifest reads predictably.
    let mut wren_files: Vec<(String, std::path::PathBuf)> = Vec::new();
    if root_path.is_file() {
        let name = root_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module")
            .to_string();
        wren_files.push((name, root_path.clone()));
    } else if root_path.is_dir() {
        walk_wren_files(&root_path, &root_path, &mut wren_files);
    } else {
        eprintln!("error: '{}' is neither a file nor a directory", root);
        process::exit(1);
    }
    wren_files.sort_by(|a, b| a.0.cmp(&b.0));

    // Use the source-tree directory name as the package name for
    // the manifest. The wrenlift.com docs viewer can override this
    // by reading the `name` from a sibling hatchfile if it cares.
    let pkg_name = root_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("package")
        .to_string();

    let mut modules: Vec<wren_lift::docs::ModuleDoc> = Vec::new();
    for (name, path) in &wren_files {
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("warning: skipping '{}': {}", path.display(), e);
                continue;
            }
        };
        modules.push(emit_module(name, &source, &out_path));
    }

    write_manifest(&pkg_name, &modules, &out_path);
    eprintln!(
        "wrote {} module(s) + manifest.json → {}",
        modules.len(),
        out_dir
    );
}

/// Same flow as `generate_docs` but driven by an in-memory hatch
/// bundle's `Source` sections. The bundle's manifest provides the
/// package name + version directly.
fn generate_docs_from_hatch(bytes: &[u8], out_path: &std::path::Path) {
    let hatch = match wren_lift::hatch::load(bytes) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: hatch load failed: {}", e);
            process::exit(1);
        }
    };
    let pkg_name = hatch.manifest.name.clone();

    let mut modules: Vec<wren_lift::docs::ModuleDoc> = Vec::new();
    for section in &hatch.sections {
        if !matches!(section.kind, wren_lift::hatch::SectionKind::Source) {
            continue;
        }
        let source = match std::str::from_utf8(&section.data) {
            Ok(s) => s,
            Err(_) => {
                eprintln!(
                    "warning: skipping non-UTF-8 source section '{}'",
                    section.name
                );
                continue;
            }
        };
        modules.push(emit_module(&section.name, source, out_path));
    }

    write_manifest(&pkg_name, &modules, out_path);
    eprintln!(
        "wrote {} module(s) + manifest.json → {} [package={}]",
        modules.len(),
        out_path.display(),
        pkg_name,
    );
}

/// Parse a single source string, run the doc collector, and write
/// `<out>/<name>.json`. Returns the collected `ModuleDoc` so the
/// caller can also write the package-level manifest.
fn emit_module(name: &str, source: &str, out_path: &std::path::Path) -> wren_lift::docs::ModuleDoc {
    let pr = wren_lift::parse::parser::parse(source);
    let module = wren_lift::docs::collect_module(name, source, &pr.module, &pr.docs, &pr.interner);
    let json = wren_lift::docs::render_module_json(&module);
    let json_path = out_path.join(format!("{}.json", name));
    if let Err(e) = fs::write(&json_path, json) {
        eprintln!("error: cannot write '{}': {}", json_path.display(), e);
        process::exit(1);
    }
    module
}

fn write_manifest(
    pkg_name: &str,
    modules: &[wren_lift::docs::ModuleDoc],
    out_path: &std::path::Path,
) {
    let manifest_json = wren_lift::docs::render_manifest(pkg_name, modules);
    let manifest_path = out_path.join("manifest.json");
    if let Err(e) = fs::write(&manifest_path, manifest_json) {
        eprintln!("error: cannot write '{}': {}", manifest_path.display(), e);
        process::exit(1);
    }
}

/// Walk `dir` recursively, collecting (qualified-name, path) pairs
/// for every `.wren` file. The qualified name is the path relative
/// to `root` with slashes replaced by dashes and the `.wren`
/// extension stripped — same convention as the bundle builder.
fn walk_wren_files(
    root: &std::path::Path,
    dir: &std::path::Path,
    out: &mut Vec<(String, std::path::PathBuf)>,
) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Skip hidden + build artefacts.
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.') || n == "target" || n == "node_modules")
            .unwrap_or(false)
        {
            continue;
        }
        if path.is_dir() {
            walk_wren_files(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("wren") {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let mut name = rel.with_extension("").to_string_lossy().to_string();
            name = name.replace(std::path::MAIN_SEPARATOR, "-");
            out.push((name, path.clone()));
        }
    }
}

/// Parse a `.hatch` byte stream and print its manifest + section
/// listing to stdout. Non-zero exit on format errors.
fn inspect_hatch(bytes: &[u8]) {
    let hatch = match wren_lift::hatch::load(bytes) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(65);
        }
    };
    println!("hatch: {} {}", hatch.manifest.name, hatch.manifest.version);
    println!("  entry:   {}", hatch.manifest.entry);
    println!("  modules: {}", hatch.manifest.modules.join(", "));
    if !hatch.manifest.dependencies.is_empty() {
        println!("  dependencies:");
        for (name, dep) in &hatch.manifest.dependencies {
            match dep {
                wren_lift::hatch::Dependency::Version(v) => println!("    {} = {}", name, v),
                wren_lift::hatch::Dependency::Path { path, version } => match version {
                    Some(v) => println!(
                        "    {} = {{ path = \"{}\", version = \"{}\" }}",
                        name, path, v
                    ),
                    None => println!("    {} = {{ path = \"{}\" }}", name, path),
                },
                wren_lift::hatch::Dependency::Git {
                    git,
                    tag,
                    rev,
                    branch,
                    url,
                } => {
                    let r = tag
                        .as_deref()
                        .map(|t| format!("tag = \"{}\"", t))
                        .or_else(|| rev.as_deref().map(|r| format!("rev = \"{}\"", r)))
                        .or_else(|| branch.as_deref().map(|b| format!("branch = \"{}\"", b)))
                        .unwrap_or_else(|| "ref = <none>".to_string());
                    let url_extra = url
                        .as_deref()
                        .map(|u| format!(", url = \"{}\"", u))
                        .unwrap_or_default();
                    println!("    {} = {{ git = \"{}\", {}{} }}", name, git, r, url_extra);
                }
                wren_lift::hatch::Dependency::Url { url } => {
                    println!("    {} = {{ url = \"{}\" }}", name, url);
                }
            }
        }
    }
    println!("  sections:");
    for section in &hatch.sections {
        println!(
            "    {:>8?}  {:>10} bytes  {}",
            section.kind,
            section.data.len(),
            section.name
        );
    }
}

/// Load + run a `.hatch` package.
fn run_hatch(bytes: &[u8], cli: &Cli) {
    let mut vm = make_vm(cli);
    match vm.interpret_hatch(bytes) {
        InterpretResult::Success => {}
        InterpretResult::CompileError => process::exit(65),
        InterpretResult::RuntimeError => process::exit(70),
    }
    if cli.gc_stats {
        let stats = vm.gc.stats();
        eprintln!("--- GC Stats ---");
        eprintln!("  minor collections: {}", stats.minor_collections);
        eprintln!("  major collections: {}", stats.major_collections);
    }
}

/// Load + run a `.wlbc` bytecode cache.
fn run_bytecode(bytes: &[u8], filename: &str, cli: &Cli) {
    let mut vm = make_vm(cli);
    // Strip directory + extension so module_name matches what `interpret`
    // would have used for the same source file. Keeps behaviour stable
    // if a runtime error points at module name.
    let module_name = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
        .strip_suffix(".wren")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            std::path::Path::new(filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(filename)
                .to_string()
        });
    match vm.interpret_bytecode(&module_name, bytes) {
        InterpretResult::Success => {}
        InterpretResult::CompileError => process::exit(65),
        InterpretResult::RuntimeError => process::exit(70),
    }
    if cli.gc_stats {
        let stats = vm.gc.stats();
        eprintln!("--- GC Stats ---");
        eprintln!("  minor collections: {}", stats.minor_collections);
        eprintln!("  major collections: {}", stats.major_collections);
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.file {
        Some(filename) => {
            // `--bundle` treats the positional argument as a source
            // tree root rather than a file — resolve it before the
            // file-read path below.
            if let Some(out_path) = &cli.bundle {
                build_hatch_package(filename, out_path, cli.bundle_target.as_deref());
                return;
            }

            // `--aot` walks the import graph rooted at `filename`,
            // emits a single object via Cranelift, and links it
            // with the runtime staticlib into a self-contained
            // executable. Reads the source through the same path
            // run_file does (UTF-8) so non-text files surface a
            // sensible error before the AOT pipeline starts.
            if let Some(out_path) = &cli.aot {
                aot_build_executable(filename, out_path);
                return;
            }

            // `--docs` likewise treats the positional argument as a
            // root path (file or directory). Generates an HTML page
            // per module under <out>/.
            if let Some(out_dir) = &cli.docs {
                generate_docs(filename, out_dir);
                return;
            }

            // Read the file as raw bytes first so we can sniff the
            // `.wlbc` / `.hatch` magic and route the right path
            // without trying to UTF-8-decode a binary blob.
            let bytes = match fs::read(filename) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("error: cannot read '{}': {}", filename, e);
                    process::exit(1);
                }
            };

            if wren_lift::hatch::looks_like_hatch(&bytes) {
                if cli.inspect {
                    inspect_hatch(&bytes);
                    return;
                }
                run_hatch(&bytes, &cli);
                return;
            }
            if wren_lift::serialize::looks_like_wlbc(&bytes) {
                run_bytecode(&bytes, filename, &cli);
                return;
            }

            let source = match std::str::from_utf8(&bytes) {
                Ok(s) => s.to_string(),
                Err(e) => {
                    eprintln!("error: '{}' is not valid UTF-8: {}", filename, e);
                    process::exit(1);
                }
            };

            if let Some(out_path) = &cli.build {
                build_bytecode_cache(&source, filename, out_path, &cli);
                return;
            }

            if cli.inspect {
                eprintln!("error: --inspect requires a .hatch file");
                process::exit(1);
            }

            run_file(&source, filename, &cli);
        }
        None => {
            if cli.dump_tokens || cli.dump_ast || cli.dump_mir || cli.dump_opt || cli.dump_asm {
                eprintln!("error: dump flags require a source file");
                process::exit(1);
            }
            run_repl();
        }
    }
}

#[cfg(test)]
mod resolver_tests {
    use super::with_wren_suffix;
    use std::path::PathBuf;

    #[test]
    fn appends_wren_when_missing() {
        assert_eq!(
            with_wren_suffix(PathBuf::from("/tmp/foo")),
            PathBuf::from("/tmp/foo.wren")
        );
    }

    #[test]
    fn preserves_spec_basename() {
        // The bug: `Path::with_extension("wren")` would strip
        // `.spec` and produce `/tmp/assert.wren`, redirecting
        // every `import "./<name>.spec"` to a same-stem sibling.
        assert_eq!(
            with_wren_suffix(PathBuf::from("/tmp/assert.spec")),
            PathBuf::from("/tmp/assert.spec.wren")
        );
    }

    #[test]
    fn idempotent_when_already_wren() {
        assert_eq!(
            with_wren_suffix(PathBuf::from("/tmp/foo.wren")),
            PathBuf::from("/tmp/foo.wren")
        );
    }

    #[test]
    fn preserves_multi_dot_basenames() {
        assert_eq!(
            with_wren_suffix(PathBuf::from("/tmp/foo.bar.baz")),
            PathBuf::from("/tmp/foo.bar.baz.wren")
        );
    }
}
