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
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Target {
    Native,
    Wasm,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn run_file(source: &str, filename: &str, cli: &Cli) {
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

    // 3. Semantic analysis
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

    // 4. Lower to MIR
    let mut mir = wren_lift::mir::builder::lower_module(&parse_result.module, &mut interner);

    if cli.dump_mir {
        println!("{}", mir.pretty_print(&interner));
        return;
    }

    // 5. Optimize
    if !cli.no_opt {
        run_opt_pipeline(&mut mir);
    }

    if cli.dump_opt {
        println!("{}", mir.pretty_print(&interner));
        return;
    }

    // 6. Code generation
    match cli.target {
        Target::Wasm => {
            let wasm_module = match wren_lift::codegen::wasm::emit_mir(&mir) {
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
            let mach_func = wren_lift::codegen::lower_mir(&mir);

            if cli.dump_asm {
                println!("{}", mach_func.display());
                return;
            }

            eprintln!(
                "Compilation successful ({} machine instructions)",
                mach_func.insts.len()
            );
            eprintln!("Native execution not yet available — use --dump-asm to inspect output");
            eprintln!("or --target=wasm to emit a WebAssembly module.");
        }
    }
}

fn run_opt_pipeline(mir: &mut wren_lift::mir::MirFunction) {
    let constfold = ConstFold;
    let dce = Dce;
    let cse = Cse;
    let type_spec = TypeSpecialize;
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
    println!("WrenLift REPL (type Ctrl-D to exit)");
    println!();

    let stdin = io::stdin();
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

        // Parse
        let parse_result = parser::parse(line);
        let has_errors = parse_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error);
        if has_errors {
            for err in &parse_result.errors {
                err.eprint(line);
            }
            continue;
        }

        // Sema
        let mut interner = parse_result.interner;
        let resolve_result = sema::resolve::resolve(&parse_result.module, &interner);
        if resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &resolve_result.errors {
                err.eprint(line);
            }
            continue;
        }

        // Lower + optimize
        let mut mir = wren_lift::mir::builder::lower_module(&parse_result.module, &mut interner);
        run_opt_pipeline(&mut mir);

        // Show optimized MIR for now (until we have native execution)
        println!("{}", mir.pretty_print(&interner));
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
