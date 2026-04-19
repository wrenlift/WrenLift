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

    /// Print the manifest + section listing of a `.hatch` package and
    /// exit without running. Accepts the positional `file` argument
    /// as the hatch path.
    #[arg(long)]
    inspect: bool,

    /// Maximum interpreter steps before aborting.
    /// Defaults to 1B (interpreter) or 10B (tiered/jit).
    #[arg(long)]
    step_limit: Option<usize>,

    /// Baseline warmup threshold before optimize-tier compilation.
    #[arg(long)]
    opt_threshold: Option<u32>,

    /// Garbage collector strategy.
    #[arg(long, value_enum, default_value_t = GcMode::Generational)]
    gc: GcMode,
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
    let load_module_fn = source_dir.map(make_module_loader);
    let config = VMConfig {
        execution_mode: mode,
        step_limit,
        gc_strategy,
        opt_threshold: cli
            .opt_threshold
            .unwrap_or(VMConfig::default().opt_threshold),
        load_module_fn,
        ..VMConfig::default()
    };
    VM::new(config)
}

// ---------------------------------------------------------------------------
// Module loader + spec-dep pre-installer
// ---------------------------------------------------------------------------

/// Build a `load_module_fn` that resolves filesystem-relative imports
/// (`./foo`, `../foo`, bare `foo`) against the running file's
/// directory. Scoped imports like `@hatch:test` are *not* handled here —
/// they must already be installed into the VM (see
/// [`preinstall_spec_dependencies`]), so this loader only covers
/// imports that sit next to the source file on disk.
fn make_module_loader(running_file_dir: PathBuf) -> Box<dyn Fn(&str) -> Option<String>> {
    Box::new(move |name: &str| -> Option<String> {
        if name.starts_with("./") || name.starts_with("../") {
            let rel = Path::new(name);
            let candidate = running_file_dir.join(rel).with_extension("wren");
            return fs::read_to_string(candidate).ok();
        }
        // Bare names (no scope chars) → sibling file in the same dir.
        let is_scoped = name.chars().any(|c| matches!(c, ':' | '@' | '/'));
        if is_scoped {
            // Scoped imports must have been pre-installed — returning
            // None here surfaces a clear "Could not load module" error
            // rather than silently finding the wrong file.
            return None;
        }
        let candidate = running_file_dir.join(format!("{}.wren", name));
        fs::read_to_string(candidate).ok()
    })
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
    for (dep_name, dep) in &manifest.spec_dependencies {
        let bytes = wren_lift::hatch::resolve_dependency_bytes(workspace_root, dep_name, dep, None)
            .map_err(|e| format!("resolving spec-dep '{}': {}", dep_name, e))?;
        match vm.install_hatch_modules(&bytes) {
            InterpretResult::Success => {}
            InterpretResult::CompileError => {
                return Err(format!("compile error installing spec-dep '{}'", dep_name));
            }
            InterpretResult::RuntimeError => {
                return Err(format!("runtime error installing spec-dep '{}'", dep_name));
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
        let (lexemes, errors) = lexer::lex(source);
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
    let cse = Cse;
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
fn build_hatch_package(root: &str, out_path: &str) {
    let root_path = std::path::PathBuf::from(root);
    if !root_path.is_dir() {
        eprintln!(
            "error: --bundle expects the positional argument to be a directory (got '{}')",
            root
        );
        process::exit(1);
    }
    let bytes = match wren_lift::hatch::build_from_source_tree(&root_path) {
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
    eprintln!("bundled {} bytes from {} → {}", bytes.len(), root, out_path);
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
                } => {
                    let r = tag
                        .as_deref()
                        .map(|t| format!("tag = \"{}\"", t))
                        .or_else(|| rev.as_deref().map(|r| format!("rev = \"{}\"", r)))
                        .or_else(|| branch.as_deref().map(|b| format!("branch = \"{}\"", b)))
                        .unwrap_or_else(|| "ref = <none>".to_string());
                    println!("    {} = {{ git = \"{}\", {} }}", name, git, r);
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
                build_hatch_package(filename, out_path);
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
