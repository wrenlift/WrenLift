use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: wren_lift <file.wren>");
        eprintln!("       wren_lift --dump-tokens <file.wren>");
        eprintln!("       wren_lift --dump-ast <file.wren>");
        eprintln!("       wren_lift --dump-mir <file.wren>");
        eprintln!("       wren_lift --target=wasm <file.wren> -o out.wasm");
        process::exit(1);
    }

    let filename = &args[1];
    let source = match fs::read_to_string(filename) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading '{}': {}", filename, e);
            process::exit(1);
        }
    };

    let _ = source;
    eprintln!("WrenLift compiler - not yet implemented");
    process::exit(1);
}
