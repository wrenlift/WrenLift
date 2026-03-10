use std::fs;
use std::io::{self, BufRead, Write};
use std::process;

use clap::{Parser, ValueEnum};

use wren_lift::diagnostics::Severity;
use wren_lift::mir::opt::{
    self, constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm, sra::Sra,
    MirPass,
};
use wren_lift::parse::{lexer, parser};
use wren_lift::runtime::engine::{ExecutionMode, InterpretResult};
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

    /// Maximum interpreter steps before aborting.
    /// Defaults to 1B (interpreter) or 10B (tiered/jit).
    #[arg(long)]
    step_limit: Option<usize>,
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
    let mode = cli.mode.into();
    let step_limit = cli.step_limit.unwrap_or(match mode {
        ExecutionMode::Interpreter => 1_000_000_000,
        _ => 10_000_000_000, // tiered/jit: 10x headroom since JIT code doesn't count steps
    });
    let config = VMConfig {
        execution_mode: mode,
        step_limit,
        ..VMConfig::default()
    };
    VM::new(config)
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

    let mut vm = make_vm(cli);
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

    if cli.gc_stats {
        let stats = &vm.gc.stats;
        eprintln!("--- GC Stats ---");
        eprintln!("  minor collections: {}", stats.minor_collections);
        eprintln!("  major collections: {}", stats.major_collections);
        eprintln!("  objects allocated:  {}", stats.objects_allocated);
        eprintln!("  objects freed:      {}", stats.objects_freed);
        eprintln!("  objects promoted:   {}", stats.objects_promoted);
        eprintln!("  peak objects:       {}", stats.peak_objects);
        eprintln!("  total allocated:    {} KB", stats.total_allocated / 1024);
        eprintln!("  total freed:        {} KB", stats.total_freed / 1024);
        eprintln!("  gc time:            {:.3}s", stats.gc_time_ns as f64 / 1e9);
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
        return;
    }

    // Optimize
    if !cli.no_opt {
        run_opt_pipeline(mir, &interner);
    }

    if cli.dump_opt {
        println!("{}", mir.pretty_print(&interner));
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

fn main() {
    let cli = Cli::parse();

    match &cli.file {
        Some(filename) => {
            let source = match fs::read_to_string(filename) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: cannot read '{}': {}", filename, e);
                    process::exit(1);
                }
            };
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
