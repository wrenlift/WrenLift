//! `hatch` — packaging and distribution tooling for wrenlift.
//!
//! Where the `wlift` binary in the wren_lift repo handles run-a-hatch
//! as a minimal bootstrap, `hatch` is the workspace front-end: create,
//! build, run, and (soon) publish / install / search. A wrenlift
//! *workspace* is any directory with a `hatchfile` at its root.

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process;

/// The wrenlift workspace manifest filename. Re-exported from the
/// runtime crate for consistency.
const HATCHFILE: &str = wren_lift::hatch::HATCHFILE;

#[derive(Parser)]
#[command(name = "hatch", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a new wrenlift workspace at the given path.
    ///
    /// Creates the directory (if missing), writes a `hatchfile` with
    /// reasonable defaults, and drops a `main.wren` stub. The result
    /// is a project `hatch build` can pack without further setup.
    Init {
        /// Directory to scaffold. Defaults to the current directory.
        #[arg(value_name = "DIR", default_value = ".")]
        dir: PathBuf,
        /// Package name for the generated `hatchfile`. Defaults to
        /// the directory's basename.
        #[arg(long)]
        name: Option<String>,
    },
    /// Build the current workspace into a `.hatch` artifact.
    ///
    /// Reads `hatchfile` for package metadata and walks the source
    /// tree for `.wren` files. Output goes to `<name>.hatch` in the
    /// workspace root unless `--out` overrides it.
    Build {
        /// Workspace root. Defaults to the current directory.
        #[arg(value_name = "DIR", default_value = ".")]
        dir: PathBuf,
        /// Override the output path.
        #[arg(short, long, value_name = "OUT")]
        out: Option<PathBuf>,
    },
    /// Print a hatch's manifest + section listing without running.
    Inspect {
        #[arg(value_name = "PACKAGE")]
        path: PathBuf,
    },
    /// Run a workspace or an already-built `.hatch`.
    ///
    /// If `target` is a directory (or omitted, meaning `.`) the
    /// workspace is built first, then run. If `target` is a file the
    /// bytes are loaded directly. Dependency hatches are installed
    /// first in `--with` order; the manifest-driven resolver that
    /// walks `[dependencies]` automatically lands with `hatch tidy`.
    Run {
        /// Workspace directory or `.hatch` file. Defaults to `.`.
        #[arg(value_name = "TARGET", default_value = ".")]
        target: PathBuf,
        /// Preload a library hatch before running the main package.
        /// Repeatable.
        #[arg(long = "with", value_name = "PACKAGE")]
        withs: Vec<PathBuf>,
    },
    /// Record a dependency in the current workspace's `hatchfile`.
    /// Planned — placeholder today.
    Add {
        /// Package name (as advertised by a registry or `hatchfile`).
        name: String,
        /// Version constraint. Defaults to "*" (latest).
        #[arg(default_value = "*")]
        version: String,
    },
    /// Pull a package from the hatch registry into the local cache
    /// and record it under `[dependencies]` in the workspace's
    /// `hatchfile`. Accepts `<name>@<version>` (pinned) or a bare
    /// `<name>`, which only works when the hatchfile already lists
    /// the package.
    Install {
        /// Package name, optionally `@<version>`. Omitted means
        /// "fetch every entry already declared in `[dependencies]`".
        #[arg(value_name = "PACKAGE")]
        package: Option<String>,
        /// Workspace directory. Defaults to `.`.
        #[arg(long = "dir", short = 'C', default_value = ".")]
        dir: PathBuf,
    },
    /// Drop a dependency from the current workspace's `hatchfile`.
    /// Planned — placeholder today.
    Remove { name: String },
    /// Refresh the local `hatchfile.lock`: resolve every declared
    /// dependency transitively, download / cache hatches, prune
    /// unused entries. Planned — placeholder today.
    Tidy,
    /// Download + cache a single dependency without modifying the
    /// `hatchfile`. Planned — placeholder today.
    Get { name: String },
    /// Publish the current workspace's hatch to a registry.
    /// Planned — placeholder today.
    Publish,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Init { dir, name } => cmd_init(&dir, name.as_deref()),
        Command::Build { dir, out } => cmd_build(&dir, out.as_deref()),
        Command::Inspect { path } => cmd_inspect(&path),
        Command::Run { target, withs } => cmd_run(&target, &withs),
        Command::Add { name, version } => cmd_stub(&format!(
            "add {name}@{version} — resolver + registry lookups are planned; see the README roadmap"
        )),
        Command::Install { package, dir } => cmd_install(&dir, package.as_deref()),
        Command::Remove { name } => cmd_stub(&format!(
            "remove {name} — needs the resolver to land first; see the README roadmap"
        )),
        Command::Tidy => cmd_stub(
            "tidy — the lockfile + transitive resolver land in a later commit; see the README roadmap",
        ),
        Command::Get { name } => cmd_stub(&format!(
            "get {name} — needs the registry client to land first; see the README roadmap"
        )),
        Command::Publish => cmd_stub(
            "publish — needs the registry client + auth to land first; see the README roadmap",
        ),
    }
}

/// Helper for the not-yet-implemented verbs. Keeps the CLI surface
/// visible so downstream users can see the planned ergonomics while
/// the resolver / registry work progresses.
fn cmd_stub(msg: &str) -> ! {
    eprintln!("hatch: not yet implemented: {}", msg);
    process::exit(2);
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

fn cmd_init(dir: &Path, name_override: Option<&str>) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("error: cannot create '{}': {}", dir.display(), e);
        process::exit(1);
    }

    let hatchfile_path = dir.join(HATCHFILE);
    if hatchfile_path.exists() {
        eprintln!(
            "error: '{}' already exists — refusing to overwrite",
            hatchfile_path.display()
        );
        process::exit(1);
    }

    let name = name_override
        .map(str::to_string)
        .unwrap_or_else(|| default_package_name(dir));
    let hatchfile_contents = format!(
        r#"# wrenlift workspace manifest.
# See https://github.com/wrenlift/hatch for the full reference.

name = "{name}"
version = "0.1.0"
entry = "main"

# Modules are listed in dependency order (imports are resolved against
# already-installed modules during hatch load). `hatch build` lists
# every .wren it discovers here automatically if this field is empty.
modules = ["main"]

# [dependencies]
# libfoo = "0.2"
"#
    );
    if let Err(e) = std::fs::write(&hatchfile_path, hatchfile_contents) {
        eprintln!("error: cannot write '{}': {}", hatchfile_path.display(), e);
        process::exit(1);
    }

    let main_path = dir.join("main.wren");
    if !main_path.exists() {
        let stub = format!(
            "// Entry point for package '{name}'. `hatch run` executes this file.\nSystem.print(\"hello from {name}\")\n"
        );
        if let Err(e) = std::fs::write(&main_path, stub) {
            eprintln!("error: cannot write '{}': {}", main_path.display(), e);
            process::exit(1);
        }
    }

    eprintln!(
        "initialised workspace '{}' in {}",
        name,
        dir.canonicalize()
            .unwrap_or_else(|_| dir.to_path_buf())
            .display()
    );
}

fn default_package_name(dir: &Path) -> String {
    dir.canonicalize()
        .ok()
        .as_deref()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| "package".to_string())
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

fn cmd_build(dir: &Path, out: Option<&Path>) {
    if !dir.is_dir() {
        eprintln!("error: '{}' is not a directory", dir.display());
        process::exit(1);
    }
    let hatchfile_path = dir.join(HATCHFILE);
    if !hatchfile_path.exists() {
        eprintln!(
            "error: no `{HATCHFILE}` in '{}' — run `hatch init` first",
            dir.display()
        );
        process::exit(1);
    }

    let bytes = match wren_lift::hatch::build_from_source_tree(dir) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(65);
        }
    };

    let out_path = out.map(Path::to_path_buf).unwrap_or_else(|| {
        // Name the artifact after the manifest's `name` field when
        // possible, else fall back to the directory basename.
        let name = std::fs::read_to_string(&hatchfile_path)
            .ok()
            .and_then(|text| {
                toml::from_str::<wren_lift::hatch::Manifest>(&text)
                    .ok()
                    .map(|m| m.name)
            })
            .unwrap_or_else(|| default_package_name(dir));
        dir.join(format!("{name}.hatch"))
    });

    if let Err(e) = std::fs::write(&out_path, &bytes) {
        eprintln!("error: cannot write '{}': {}", out_path.display(), e);
        process::exit(1);
    }
    eprintln!(
        "built {} bytes from {} → {}",
        bytes.len(),
        dir.display(),
        out_path.display()
    );
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

fn cmd_inspect(path: &Path) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path.display(), e);
            process::exit(1);
        }
    };
    let hatch = match wren_lift::hatch::load(&bytes) {
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
                    Some(v) => println!("    {} = {{ path = \"{}\", version = \"{}\" }}", name, path, v),
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

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

/// `hatch install <name>@<version>` — fetch the release artifact into
/// the local cache and record the dependency in the workspace's
/// hatchfile. `hatch install` (no arg) walks `[dependencies]` and
/// ensures every pinned-version entry is cached, leaving path entries
/// alone.
fn cmd_install(dir: &Path, package: Option<&str>) {
    let hatchfile_path = dir.join(HATCHFILE);
    let text = match std::fs::read_to_string(&hatchfile_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "error: cannot read '{}': {} (run `hatch init` first?)",
                hatchfile_path.display(),
                e
            );
            process::exit(1);
        }
    };
    let mut doc: toml_edit::DocumentMut = match text.parse() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: cannot parse hatchfile: {}", e);
            process::exit(1);
        }
    };

    let registry = wren_lift::hatch_registry::registry_url();
    let cache_dir = match wren_lift::hatch_registry::cache_root() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    };

    match package {
        Some(spec) => {
            let (name, version) = wren_lift::hatch_registry::split_name_version(spec);
            if let Some(entry) = doc
                .get("dependencies")
                .and_then(toml_edit::Item::as_table_like)
                .and_then(|t| t.get(name))
            {
                if let Some(spec) = extract_git_spec(entry) {
                    // Git-hosted dep: ignore any `@version` the user
                    // typed and use the declared ref. Downgrading
                    // pinned commits via CLI would be surprising.
                    install_git(&cache_dir, name, &spec);
                    println!("installed {} from {}", name, spec.git);
                    return;
                }
            }

            let resolved_version = match version {
                Some(v) => v.to_string(),
                None => match declared_version(&doc, name) {
                    Some(v) => v,
                    None => {
                        eprintln!(
                            "error: no version given and `{name}` isn't in [dependencies]. Use `{name}@<version>`."
                        );
                        process::exit(1);
                    }
                },
            };

            install_one(&cache_dir, &registry, name, &resolved_version);
            record_in_hatchfile(&mut doc, name, &resolved_version);

            if let Err(e) = std::fs::write(&hatchfile_path, doc.to_string()) {
                eprintln!(
                    "error: cannot update '{}': {}",
                    hatchfile_path.display(),
                    e
                );
                process::exit(1);
            }
            println!("installed {}@{}", name, resolved_version);
        }
        None => {
            // Resolve every entry already in [dependencies]: version-
            // pinned ones hit the release cache, git-hosted ones get
            // shallow-cloned. Path deps need nothing — `hatch build`
            // reads them from the filesystem directly.
            let Some(deps) = doc
                .get("dependencies")
                .and_then(toml_edit::Item::as_table_like)
            else {
                println!("no [dependencies] to install");
                return;
            };
            let mut registry_installs: Vec<(String, String)> = Vec::new();
            let mut git_installs: Vec<(String, GitSpec)> = Vec::new();
            for (name, item) in deps.iter() {
                if let Some(spec) = extract_git_spec(item) {
                    git_installs.push((name.to_string(), spec));
                } else if let Some(version) = extract_version(item) {
                    registry_installs.push((name.to_string(), version));
                }
            }
            if registry_installs.is_empty() && git_installs.is_empty() {
                println!("nothing to install (no version-pinned or git dependencies)");
                return;
            }
            for (name, version) in &registry_installs {
                install_one(&cache_dir, &registry, name, version);
                println!("installed {}@{}", name, version);
            }
            for (name, spec) in &git_installs {
                install_git(&cache_dir, name, spec);
                println!("installed {} from {}", name, spec.git);
            }
        }
    }
}

fn install_one(cache_dir: &Path, registry: &str, name: &str, version: &str) {
    match wren_lift::hatch_registry::ensure_in_cache_dir(cache_dir, registry, name, version) {
        Ok(path) => {
            eprintln!("  cached at {}", path.display());
        }
        Err(e) => {
            eprintln!("error: {}", e);
            eprintln!(
                "  (registry = {}, release URL = {})",
                registry,
                wren_lift::hatch_registry::release_url(registry, name, version),
            );
            process::exit(1);
        }
    }
}

fn install_git(cache_dir: &Path, name: &str, spec: &GitSpec) {
    let git_ref = match spec.git_ref() {
        Some(r) => r,
        None => {
            eprintln!(
                "error: git dependency `{name}` needs one of `tag`, `rev`, or `branch`"
            );
            process::exit(1);
        }
    };
    match wren_lift::hatch_registry::ensure_git_checkout(cache_dir, &spec.git, git_ref) {
        Ok(path) => {
            eprintln!("  cached at {}", path.display());
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

/// Minimal owned copy of a `[dependencies.<name>]` git table — keeps
/// the install path decoupled from toml_edit's borrow lifetimes.
struct GitSpec {
    git: String,
    tag: Option<String>,
    rev: Option<String>,
    branch: Option<String>,
}

impl GitSpec {
    fn git_ref(&self) -> Option<wren_lift::hatch::GitRef<'_>> {
        if let Some(r) = self.rev.as_deref() {
            Some(wren_lift::hatch::GitRef::Rev(r))
        } else if let Some(t) = self.tag.as_deref() {
            Some(wren_lift::hatch::GitRef::Tag(t))
        } else {
            self.branch
                .as_deref()
                .map(wren_lift::hatch::GitRef::Branch)
        }
    }
}

fn extract_git_spec(item: &toml_edit::Item) -> Option<GitSpec> {
    let tbl = item.as_table_like()?;
    let git = tbl.get("git")?.as_str()?.to_string();
    Some(GitSpec {
        git,
        tag: tbl.get("tag").and_then(|i| i.as_str()).map(str::to_string),
        rev: tbl.get("rev").and_then(|i| i.as_str()).map(str::to_string),
        branch: tbl.get("branch").and_then(|i| i.as_str()).map(str::to_string),
    })
}

/// Read the declared version of `name` (if any) out of the hatchfile
/// so `hatch install <name>` can use it without the user retyping.
fn declared_version(doc: &toml_edit::DocumentMut, name: &str) -> Option<String> {
    let deps = doc.get("dependencies")?.as_table_like()?;
    let item = deps.get(name)?;
    extract_version(item)
}

fn extract_version(item: &toml_edit::Item) -> Option<String> {
    // `name = "1.0.0"` — bare string dep.
    if let Some(v) = item.as_str() {
        return Some(v.to_string());
    }
    // `name = { version = "1.0.0", path = "..." }` — inline table.
    if let Some(tbl) = item.as_table_like() {
        if let Some(v) = tbl.get("version").and_then(|i| i.as_str()) {
            return Some(v.to_string());
        }
    }
    None
}

/// Merge `name = "version"` into the hatchfile's `[dependencies]`
/// table, preserving surrounding comments + formatting via
/// `toml_edit`.
fn record_in_hatchfile(doc: &mut toml_edit::DocumentMut, name: &str, version: &str) {
    let deps = doc
        .entry("dependencies")
        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
    if let Some(tbl) = deps.as_table_like_mut() {
        tbl.insert(name, toml_edit::value(version));
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(target: &Path, withs: &[PathBuf]) {
    use wren_lift::runtime::engine::InterpretResult;
    use wren_lift::runtime::vm::VM;

    let bytes_owned;
    let main_bytes: &[u8] = if target.is_dir() {
        // Workspace: build from source, then run the bytes in-memory
        // rather than writing a temp file.
        let hatchfile_path = target.join(HATCHFILE);
        if !hatchfile_path.exists() {
            eprintln!(
                "error: no `{HATCHFILE}` in '{}' — run `hatch init` first",
                target.display()
            );
            process::exit(1);
        }
        bytes_owned = match wren_lift::hatch::build_from_source_tree(target) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: {}", e);
                process::exit(65);
            }
        };
        &bytes_owned
    } else {
        bytes_owned = match std::fs::read(target) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: cannot read '{}': {}", target.display(), e);
                process::exit(1);
            }
        };
        &bytes_owned
    };

    let mut vm = VM::new_default();

    // Preload dependency hatches in CLI order. Manifest-driven
    // resolution against a registry lands in later hatch-cli work.
    for dep_path in withs {
        let dep_bytes = match std::fs::read(dep_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: cannot read '{}': {}", dep_path.display(), e);
                process::exit(1);
            }
        };
        match vm.install_hatch_modules(&dep_bytes) {
            InterpretResult::Success => {}
            InterpretResult::CompileError => process::exit(65),
            InterpretResult::RuntimeError => process::exit(70),
        }
    }

    match vm.interpret_hatch(main_bytes) {
        InterpretResult::Success => {}
        InterpretResult::CompileError => process::exit(65),
        InterpretResult::RuntimeError => process::exit(70),
    }
}
