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
    /// Pick a template via `--template`:
    ///   bare — empty hello-world (default)
    ///   web  — @hatch:web app (router, Css, htmx-ready route)
    ///
    /// Other names (`game`, `cli`, ...) fall back to `bare` with a
    /// warning, so you always get a working hatchfile + entry to
    /// build on. Dedicated scaffolds for those land later.
    ///
    /// Examples:
    ///   hatch init mysite --template web
    ///   hatch init mylib            # → bare
    Init {
        /// Directory to scaffold. Defaults to the current directory.
        #[arg(value_name = "DIR", default_value = ".")]
        dir: PathBuf,
        /// Package name for the generated `hatchfile`. Defaults to
        /// the directory's basename.
        #[arg(long)]
        name: Option<String>,
        /// Starter template. Unknown values fall back to `bare`.
        #[arg(long, value_name = "KIND", default_value = "bare")]
        template: String,
    },
    /// Run every *.spec.wren in the workspace and aggregate results.
    Test {
        /// Workspace root. Defaults to the current directory.
        #[arg(value_name = "DIR", default_value = ".")]
        dir: PathBuf,
    },
    /// @hatch:web framework commands — serve, generate, etc.
    ///
    /// Scaffold a new web app via `hatch init <dir> --template web`;
    /// these subcommands operate on an existing one. Future framework
    /// verbs (auth, migrate, deploy) land here too.
    Web {
        #[command(subcommand)]
        command: WebCommand,
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
    /// Register the current workspace's package with the hatch
    /// catalog so others can `hatch find` and pin it. Requires a
    /// prior `hatch login`.
    Publish {
        /// Workspace directory. Defaults to `.`.
        #[arg(long = "dir", short = 'C', default_value = ".")]
        dir: PathBuf,
        /// Override the git URL written to the catalog. Defaults to
        /// the workspace's `origin` remote.
        #[arg(long = "git", value_name = "URL")]
        git: Option<String>,
    },
    /// Look up a package by name and print where it's hosted. Read-
    /// only — works without `hatch login`.
    Find { name: String },
    /// Authenticate against the hatch service. Full GitHub OAuth
    /// device flow is landing in a follow-up; for now this command
    /// accepts a pre-minted JWT via `--token`.
    Login {
        /// Store the supplied JWT directly instead of running the
        /// interactive flow. Lets CI and early testers sign in
        /// while the OAuth implementation is still in flight.
        #[arg(long = "token", value_name = "JWT")]
        token: Option<String>,
    },
    /// Drop the stored credentials file. Safe to run anytime.
    Logout,
}

#[derive(Subcommand)]
enum WebCommand {
    /// Watch the workspace and re-run the entry on every save.
    ///
    /// Spawns `wlift --mode interpreter --step-limit 0 <entry>.wren`,
    /// polls *.wren mtimes every 500ms, and respawns on any change.
    /// Survives a child crash by waiting for the next save. Ctrl-C
    /// kills both via the foreground process group.
    Serve {
        /// Workspace root. Defaults to the current directory.
        #[arg(value_name = "DIR", default_value = ".")]
        dir: PathBuf,
    },
    /// Generate a new route / form / template stub inside an
    /// existing @hatch:web workspace.
    Generate {
        /// What to generate. One of: `route`, `form`, `template`.
        #[arg(value_name = "KIND")]
        kind: String,
        /// Name of the new artifact (e.g. `posts`).
        #[arg(value_name = "NAME")]
        name: String,
        /// Workspace root. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        dir: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Init {
            dir,
            name,
            template,
        } => cmd_init(&dir, name.as_deref(), &template),
        Command::Test { dir } => cmd_test(&dir),
        Command::Web { command } => match command {
            WebCommand::Serve { dir } => cmd_dev(&dir),
            WebCommand::Generate { kind, name, dir } => cmd_web_generate(&kind, &name, &dir),
        },
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
        Command::Publish { dir, git } => cmd_publish(&dir, git.as_deref()),
        Command::Find { name } => cmd_find(&name),
        Command::Login { token } => cmd_login(token.as_deref()),
        Command::Logout => cmd_logout(),
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

fn cmd_init(dir: &Path, name_override: Option<&str>, template: &str) {
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

    // Any template we don't have a dedicated scaffold for falls
    // back to the bare hatchfile + main.wren — the user still gets
    // a working workspace they can iterate from. We just warn so
    // they know there's no framework wiring in this slot yet.
    let effective_template = match template {
        "bare" => {
            scaffold_bare(dir, &name);
            "bare"
        }
        "web" => {
            scaffold_web(dir, &name);
            "web"
        }
        other => {
            eprintln!(
                "warning: template '{}' has no dedicated scaffold yet — using `bare`. \
                 Run `hatch init --template web` for the @hatch:web starter.",
                other
            );
            scaffold_bare(dir, &name);
            "bare"
        }
    };

    let canonical = dir
        .canonicalize()
        .unwrap_or_else(|_| dir.to_path_buf());
    eprintln!(
        "initialised workspace '{}' ({}) in {}",
        name, effective_template, canonical.display()
    );
    if effective_template == "web" {
        eprintln!("  next: cd {} && hatch web serve", dir.display());
    }
}

// ---------------------------------------------------------------------------
// web — framework-specific commands
// ---------------------------------------------------------------------------

fn cmd_web_generate(kind: &str, name: &str, dir: &Path) {
    if !dir.is_dir() || !dir.join(HATCHFILE).is_file() {
        eprintln!(
            "error: '{}' is not a hatch workspace (no hatchfile)",
            dir.display()
        );
        process::exit(1);
    }
    match kind {
        "route" => generate_route(dir, name),
        "form" => generate_form(dir, name),
        "template" => generate_template(dir, name),
        other => {
            eprintln!(
                "error: unknown kind '{}'. expected one of: route, form, template",
                other
            );
            process::exit(1);
        }
    }
}

fn generate_route(dir: &Path, name: &str) {
    let routes_dir = dir.join("routes");
    if !routes_dir.exists() {
        let _ = std::fs::create_dir_all(&routes_dir);
    }
    let path = routes_dir.join(format!("{}.wren", name));
    if path.exists() {
        eprintln!("error: '{}' already exists", path.display());
        process::exit(1);
    }
    let class_name = pascal_case(name);
    let stub = format!(
        r#"// Routes for {name}. Wire into main.wren via:
//
//   import "./routes/{name}" for {class_name}Routes
//   {class_name}Routes.mount(app)

import "@hatch:web" for Response

class {class_name}Routes {{
  static mount(app) {{
    app.get("/{name}") {{|req|
      "<h1>{class_name}</h1>"
    }}

    app.get("/{name}/:id") {{|req|
      "<h1>{class_name} %(req.param("id"))</h1>"
    }}

    app.post("/{name}") {{|req|
      // var data = req.form
      Response.redirect("/{name}")
    }}
  }}
}}
"#
    );
    write_or_die(&path, &stub);
    eprintln!("created {}", path.display());
    eprintln!("  next: import in main.wren and call {}Routes.mount(app)", class_name);
}

fn generate_form(dir: &Path, name: &str) {
    let forms_dir = dir.join("forms");
    if !forms_dir.exists() {
        let _ = std::fs::create_dir_all(&forms_dir);
    }
    let path = forms_dir.join(format!("{}.wren", name));
    if path.exists() {
        eprintln!("error: '{}' already exists", path.display());
        process::exit(1);
    }
    let const_name = name.to_uppercase();
    let stub = format!(
        r#"// {name} form schema — fill in your fields and validators.

import "@hatch:web" for Form, Field

var {const_name}_FORM = Form.new([
  Field.new("email").trim.lowercase
                    .required("Email is required")
                    .email("Looks invalid"),
  Field.new("name").trim
                   .required("Name is required")
                   .maxLength(80, "Too long")
])
"#
    );
    write_or_die(&path, &stub);
    eprintln!("created {}", path.display());
    eprintln!("  use: import \"./forms/{}\" for {}_FORM", name, const_name);
}

fn generate_template(dir: &Path, name: &str) {
    let templates_dir = dir.join("templates");
    if !templates_dir.exists() {
        let _ = std::fs::create_dir_all(&templates_dir);
    }
    let path = templates_dir.join(format!("{}.wren", name));
    if path.exists() {
        eprintln!("error: '{}' already exists", path.display());
        process::exit(1);
    }
    let const_name = name.to_uppercase();
    let stub = format!(
        r#"// {name} — render via @hatch:template.
//
//   import "./templates/{name}" for {const_name}_TPL
//   {const_name}_TPL.render(ctx)

import "@hatch:template" for Template

var {const_name}_TPL = Template.parse("
<h1>{{{{ title }}}}</h1>
<p>{{{{ body }}}}</p>
")
"#
    );
    write_or_die(&path, &stub);
    eprintln!("created {}", path.display());
}

fn pascal_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut upper_next = true;
    for c in s.chars() {
        if c == '_' || c == '-' || c == ' ' {
            upper_next = true;
        } else if upper_next {
            out.extend(c.to_uppercase());
            upper_next = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn scaffold_bare(dir: &Path, name: &str) {
    let hatchfile_path = dir.join(HATCHFILE);
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
    write_or_die(&hatchfile_path, &hatchfile_contents);

    let main_path = dir.join("main.wren");
    if !main_path.exists() {
        let stub = format!(
            "// Entry point for package '{name}'. `hatch run` executes this file.\nSystem.print(\"hello from {name}\")\n"
        );
        write_or_die(&main_path, &stub);
    }
}

fn scaffold_web(dir: &Path, name: &str) {
    let hatchfile_path = dir.join(HATCHFILE);
    let hatchfile_contents = format!(
        r#"# wrenlift workspace manifest — @hatch:web starter.

name = "{name}"
version = "0.1.0"
entry = "main"
description = "A @hatch:web app"

modules = ["main"]

[dependencies]
"@hatch:web" = "0.1"
"#
    );
    write_or_die(&hatchfile_path, &hatchfile_contents);

    let main_path = dir.join("main.wren");
    if !main_path.exists() {
        let stub = format!(
            r#"// Entry point for {name}. Run with:
//
//   wlift --mode interpreter --step-limit 0 main.wren
//
// Then open http://127.0.0.1:3000

import "@hatch:web" for App, Css

var app = App.new()

var page    = Css.tw("font-sans max-w-2xl mx-auto my-10 p-6")
var heading = Css.tw("text-3xl font-bold text-gray-900 mb-4")
var btn     = Css.tw("inline-block bg-blue-500 text-white px-4 py-2 rounded font-semibold")
                .hover("bg-blue-600")

app.get("/") {{|req|
  req.style(page)
  req.style(heading)
  req.style(btn)
  return req.fragmentSheet.styleTag +
    "<div class='" + page.className + "'>" +
    "<h1 class='" + heading.className + "'>{name}</h1>" +
    "<p>Hello from @hatch:web. Edit <code>main.wren</code> to start.</p>" +
    "<p style='margin-top:1rem;'>" +
    "<a class='" + btn.className + "' href='/hi/world'>Try a route</a>" +
    "</p>" +
    "</div>"
}}

app.get("/hi/:who") {{|req|
  if (req.isHx) return "<span>hello %(req.param("who"))</span>"
  return "<h1>Hello, %(req.param("who"))!</h1><p><a href='/'>back</a></p>"
}}

app.listen("127.0.0.1:3000")
"#
        );
        write_or_die(&main_path, &stub);
    }

    let public_dir = dir.join("public");
    if !public_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&public_dir) {
            eprintln!(
                "error: cannot create '{}': {}",
                public_dir.display(),
                e
            );
            process::exit(1);
        }
    }
    let gitkeep = public_dir.join(".gitkeep");
    if !gitkeep.exists() {
        write_or_die(&gitkeep, "");
    }

    let gitignore = dir.join(".gitignore");
    if !gitignore.exists() {
        write_or_die(&gitignore, "*.hatch\n.DS_Store\n");
    }
}

fn write_or_die(path: &Path, contents: &str) {
    if let Err(e) = std::fs::write(path, contents) {
        eprintln!("error: cannot write '{}': {}", path.display(), e);
        process::exit(1);
    }
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
                eprintln!("error: cannot update '{}': {}", hatchfile_path.display(), e);
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
            eprintln!("error: git dependency `{name}` needs one of `tag`, `rev`, or `branch`");
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
            self.branch.as_deref().map(wren_lift::hatch::GitRef::Branch)
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
        branch: tbl
            .get("branch")
            .and_then(|i| i.as_str())
            .map(str::to_string),
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
// login / logout / find / publish
// ---------------------------------------------------------------------------

/// Store a JWT so subsequent `hatch publish` calls can authenticate.
/// Full interactive GitHub OAuth device flow lands in a follow-up;
/// for now we accept a token directly from `--token` so early
/// adopters / CI jobs can exercise the pipeline.
fn cmd_login(token: Option<&str>) {
    let cfg = wren_lift::hatch_service::ServiceConfig::from_env();

    let creds = match token {
        Some(jwt) => {
            // Escape hatch for CI / scripted flows: skip the browser
            // dance entirely and trust a pre-minted token. No
            // refresh token + no known expiry — the request itself
            // will 401 when the JWT expires and the user will need
            // to re-run this command.
            //
            // `--token -` reads from stdin. Common Unix convention
            // for secret-ish inputs so the token never appears in
            // argv (visible via `ps`), and lets shell pipelines
            // hand tokens off without intermediate files.
            let resolved = if jwt == "-" {
                use std::io::Read as _;
                let mut buf = String::new();
                if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                    eprintln!("error: reading token from stdin: {}", e);
                    process::exit(1);
                }
                buf.trim().to_string()
            } else {
                jwt.to_string()
            };
            if resolved.is_empty() {
                eprintln!("error: empty token — nothing to log in with");
                process::exit(1);
            }
            wren_lift::hatch_service::Credentials {
                access_token: resolved,
                refresh_token: None,
                service_url: Some(cfg.url.clone()),
                expires_at: None,
            }
        }
        None => match wren_lift::hatch_service::interactive_login(&cfg) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        },
    };

    match wren_lift::hatch_service::save_credentials(&creds) {
        Ok(path) => println!("logged in (credentials stored at {})", path.display()),
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_logout() {
    match wren_lift::hatch_service::clear_credentials() {
        Ok(true) => println!("logged out"),
        Ok(false) => println!("not logged in"),
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn cmd_find(name: &str) {
    let cfg = wren_lift::hatch_service::ServiceConfig::from_env();
    // Accept `<name>@<version>` to narrow down; otherwise list every
    // published version with the newest first.
    let (bare_name, version_filter) = wren_lift::hatch_registry::split_name_version(name);
    let rows = match version_filter {
        Some(v) => match wren_lift::hatch_service::find_exact(&cfg, bare_name, v) {
            Ok(Some(r)) => vec![r],
            Ok(None) => Vec::new(),
            Err(e) => {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        },
        None => match wren_lift::hatch_service::find_versions(&cfg, bare_name) {
            Ok(rows) => rows,
            Err(e) => {
                eprintln!("error: {}", e);
                process::exit(1);
            }
        },
    };

    if rows.is_empty() {
        eprintln!("hatch: no package '{}' in the catalog", name);
        process::exit(1);
    }

    let latest = &rows[0];
    println!("{}@{} — {}", latest.name, latest.version, latest.git);
    if let Some(desc) = latest.description.as_deref() {
        if !desc.is_empty() {
            println!("  {}", desc);
        }
    }
    if let Some(owner) = latest.owner.as_deref() {
        println!("  owner: {}", owner);
    }
    if rows.len() > 1 {
        println!("  versions:");
        for row in rows.iter().skip(1) {
            println!("    {}", row.version);
        }
    }
    println!(
        "\nTo use, add to your hatchfile:\n  [dependencies]\n  \"{}\" = \"{}\"",
        latest.name, latest.version,
    );
}

fn cmd_publish(dir: &Path, git_override: Option<&str>) {
    // Read the workspace's hatchfile for name + description.
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
    let manifest: wren_lift::hatch::Manifest = match toml::from_str(&text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: cannot parse hatchfile: {}", e);
            process::exit(1);
        }
    };

    // Git URL: explicit --git wins; else probe the workspace's
    // `origin` remote. If neither is available, the user hasn't
    // committed anywhere yet and publishing would produce a
    // dangling catalog entry.
    let git = match git_override {
        Some(g) => g.to_string(),
        None => match detect_origin_remote(dir) {
            Some(g) => g,
            None => {
                eprintln!(
                    "error: workspace has no `origin` git remote and --git wasn't given.\n\
                     Set one with `git remote add origin <URL>` and retry."
                );
                process::exit(1);
            }
        },
    };

    // Load credentials.
    let creds = match wren_lift::hatch_service::load_credentials() {
        Ok(Some(c)) => c,
        Ok(None) => {
            eprintln!("error: not logged in — run `hatch login` first");
            process::exit(1);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    };

    let cfg = wren_lift::hatch_service::ServiceConfig::from_env();
    // Quietly refresh the JWT if it's near expiry so long-idle
    // sessions don't 401 mid-publish. No-op when the token's still
    // good (or when we don't have a refresh token to redeem).
    let creds = match wren_lift::hatch_service::ensure_fresh_credentials(&cfg, creds) {
        Ok(c) => c,
        Err(wren_lift::hatch_service::ServiceError::NotLoggedIn) => {
            eprintln!("error: session expired — run `hatch login` to renew");
            process::exit(1);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    };

    let record = wren_lift::hatch_service::PackageRecord {
        name: manifest.name.clone(),
        version: manifest.version.clone(),
        git: git.clone(),
        description: manifest.description.clone(),
        owner: None, // server sets from JWT
    };

    match wren_lift::hatch_service::publish_package(&cfg, &creds, &record) {
        Ok(()) => {
            println!(
                "published {}@{} → {}",
                record.name, record.version, record.git
            );
        }
        Err(e) => {
            eprintln!("error: {}", e);
            process::exit(1);
        }
    }
}

fn detect_origin_remote(dir: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .current_dir(dir)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if url.is_empty() {
        None
    } else {
        Some(url)
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

// ---------------------------------------------------------------------------
// dev — file watcher + auto-restart
// ---------------------------------------------------------------------------

fn cmd_dev(dir: &Path) {
    if !dir.is_dir() {
        eprintln!("error: '{}' is not a directory", dir.display());
        process::exit(1);
    }
    let entry = read_entry_name(&dir.join(HATCHFILE)).unwrap_or_else(|| "main".to_string());
    let entry_path = dir.join(format!("{}.wren", entry));
    if !entry_path.exists() {
        eprintln!(
            "error: entry '{}' does not exist (set `entry` in {})",
            entry_path.display(),
            HATCHFILE
        );
        process::exit(1);
    }

    let wlift = locate_wlift().unwrap_or_else(|| {
        eprintln!("error: cannot locate `wlift` binary on PATH or alongside this hatch");
        process::exit(1);
    });

    eprintln!(
        "hatch dev: watching {} (entry: {})",
        dir.display(),
        entry_path.display()
    );
    eprintln!("hatch dev: ctrl-c to stop");

    // Ctrl-C is delivered to the foreground process group, so both
    // hatch and the wlift child receive SIGINT and exit cleanly
    // without explicit handling. The child is drop-killed if the
    // parent panics.
    let mut last = scan_wren_mtimes(dir);
    let mut child = spawn_dev_child(&wlift, &entry_path);

    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Drain exit if the child died on its own (compile error,
        // runtime abort) — surface and wait for the next file change.
        if let Ok(Some(status)) = child.try_wait() {
            eprintln!(
                "hatch dev: process exited ({}) — waiting for changes",
                status
            );
            // Wait until something changes, then respawn.
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let now = scan_wren_mtimes(dir);
                if now != last {
                    last = now;
                    child = spawn_dev_child(&wlift, &entry_path);
                    break;
                }
            }
            continue;
        }

        let now = scan_wren_mtimes(dir);
        if now != last {
            eprintln!("hatch dev: change detected — restarting");
            let _ = child.kill();
            let _ = child.wait();
            child = spawn_dev_child(&wlift, &entry_path);
            last = now;
        }
    }
}

fn spawn_dev_child(wlift: &Path, entry: &Path) -> std::process::Child {
    std::process::Command::new(wlift)
        .arg("--mode")
        .arg("interpreter")
        .arg("--step-limit")
        .arg("0")
        .arg(entry)
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("error: cannot spawn wlift: {}", e);
            process::exit(1);
        })
}

fn scan_wren_mtimes(dir: &Path) -> std::collections::BTreeMap<PathBuf, std::time::SystemTime> {
    let mut out = std::collections::BTreeMap::new();
    walk_wren(dir, &mut out);
    out
}

fn walk_wren(
    dir: &Path,
    out: &mut std::collections::BTreeMap<PathBuf, std::time::SystemTime>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            // Skip hidden + the OS junk + common build dirs.
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
        }
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            walk_wren(&path, out);
        } else if ft.is_file()
            && path.extension().and_then(|s| s.to_str()) == Some("wren")
        {
            if let Ok(meta) = entry.metadata() {
                if let Ok(mt) = meta.modified() {
                    out.insert(path, mt);
                }
            }
        }
    }
}

fn read_entry_name(hatchfile: &Path) -> Option<String> {
    // Looking for `entry = "main"` at the top level of the
    // hatchfile. Tolerant of leading whitespace and either single
    // or double quotes; ignores anything inside a `[section]`.
    let text = std::fs::read_to_string(hatchfile).ok()?;
    let mut in_section = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_section = true;
            continue;
        }
        if in_section {
            continue;
        }
        let (k, v) = match line.split_once('=') {
            Some(kv) => kv,
            None => continue,
        };
        if k.trim() != "entry" {
            continue;
        }
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    None
}

fn locate_wlift() -> Option<PathBuf> {
    // Prefer a `wlift` next to the hatch binary (the cargo install
    // / release-build layout puts both in the same dir), else walk
    // PATH for the executable. Avoids pulling in the `which` crate.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidate = parent.join("wlift");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    for entry in std::env::split_paths(&path) {
        let candidate = entry.join("wlift");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// test — discover and run *.spec.wren
// ---------------------------------------------------------------------------

fn cmd_test(dir: &Path) {
    if !dir.is_dir() {
        eprintln!("error: '{}' is not a directory", dir.display());
        process::exit(1);
    }
    let wlift = locate_wlift().unwrap_or_else(|| {
        eprintln!("error: cannot locate `wlift` binary");
        process::exit(1);
    });

    let mut specs = Vec::new();
    walk_specs(dir, &mut specs);
    if specs.is_empty() {
        eprintln!("hatch test: no *.spec.wren files found under {}", dir.display());
        return;
    }
    specs.sort();

    eprintln!("hatch test: {} spec file(s)", specs.len());
    let mut total_pass = 0u32;
    let mut total_fail = 0u32;
    let mut failed_files = Vec::new();
    for spec in &specs {
        eprintln!("  · {}", spec.display());
        let parent = spec.parent().unwrap_or(dir);
        let output = std::process::Command::new(&wlift)
            .arg("--mode")
            .arg("interpreter")
            .arg(spec.file_name().unwrap())
            .current_dir(parent)
            .output();
        let output = match output {
            Ok(o) => o,
            Err(e) => {
                eprintln!("    error spawning wlift: {}", e);
                failed_files.push(spec.clone());
                total_fail += 1;
                continue;
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Last "ok: N/M passed" line wins.
        let mut last_ok: Option<(u32, u32)> = None;
        for line in stdout.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("ok: ") {
                if let Some((p, m)) = rest.split_once('/') {
                    let m = m.trim_end_matches(" passed");
                    if let (Ok(p), Ok(m)) = (p.parse::<u32>(), m.parse::<u32>()) {
                        last_ok = Some((p, m));
                    }
                }
            }
        }
        match last_ok {
            Some((p, m)) => {
                total_pass += p;
                total_fail += m - p;
                if p < m {
                    failed_files.push(spec.clone());
                }
            }
            None if output.status.success() => {
                // No spec-style output but exited OK — treat as a pass.
                total_pass += 1;
            }
            None => {
                failed_files.push(spec.clone());
                total_fail += 1;
            }
        }
    }

    eprintln!(
        "\nhatch test: {} passed, {} failed across {} file(s)",
        total_pass,
        total_fail,
        specs.len()
    );
    if !failed_files.is_empty() {
        eprintln!("failures:");
        for f in failed_files {
            eprintln!("  - {}", f.display());
        }
        process::exit(1);
    }
}

fn walk_specs(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
        }
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if ft.is_dir() {
            walk_specs(&path, out);
        } else if ft.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".spec.wren") {
                    out.push(path);
                }
            }
        }
    }
}
