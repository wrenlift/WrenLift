//! Embedder-facing entry point for running Wren code that imports
//! `@hatch:*` packages.
//!
//! The CLI (`wlift` / `hatch run`) wires up hatch imports through a
//! mix of workspace discovery, catalog resolution, and network
//! fetch. Most embedders (games, tools, plugins) want something
//! simpler: "here are the packages I ship with, load them into the
//! VM, then run my script". `HatchRunner` is that surface.
//!
//! Quick tour
//! ----------
//!
//!     use wren_lift::hatch_runner::HatchRunner;
//!
//!     let mut runner = HatchRunner::new();
//!
//!     // Directories to scan for `.hatch` artifacts. Every
//!     // `<name>-<version>.hatch` we find gets registered; the
//!     // first matching name wins on conflict.
//!     runner.add_search_path("./packages")?;
//!     runner.add_search_path("./vendor/hatches")?;
//!
//!     // Directory to scan for plugin dylibs (plugin-backed
//!     // packages like @hatch:sqlite). Installed packages also
//!     // contribute their own native paths automatically.
//!     runner.add_native_path("./libs");
//!
//!     // Pre-install everything the script will import. Names
//!     // must match the `name` field in the hatchfile (no version
//!     // suffix — the runner picks up whatever it found on disk).
//!     runner.install("@hatch:math")?;
//!     runner.install("@hatch:fp")?;
//!
//!     // Now run application code that uses those imports.
//!     let src = r#"
//!         import "@hatch:math" for Vec3
//!         System.print(Vec3.new(1, 2, 3))
//!     "#;
//!     runner.run_source("main", src)?;
//!
//! The runner owns a `VM` for its lifetime. For more advanced
//! embedders (custom config, foreign class bindings, repeated
//! script execution), call [`HatchRunner::vm_mut`] to access the
//! underlying VM directly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::hatch_registry;
use crate::runtime::engine::InterpretResult;
use crate::runtime::vm::{VMConfig, VM};

/// Errors from the embedder-facing loader surface. Kept narrow on
/// purpose — the CLI's richer error machinery (with diagnostics
/// + spans) doesn't make sense for a library caller that just
///   wants "did it work, yes/no".
#[derive(Debug)]
pub enum RunnerError {
    /// A search path couldn't be read (missing directory, bad
    /// permissions). Contains the path + underlying I/O error.
    PathIo(PathBuf, std::io::Error),
    /// A `.hatch` file on disk couldn't be parsed.
    BadHatch(PathBuf, String),
    /// `install(name)` was called but no matching hatch was found
    /// in any search path and the name wasn't in the ambient cache
    /// (`$HOME/.hatch/cache` / `HATCH_CACHE_DIR`).
    NotFound(String),
    /// The VM rejected a hatch install (usually a compile error in
    /// the package's source, or a missing runtime-module import).
    Install(String),
    /// `run_source` failed (compile or runtime error). The captured
    /// output buffer is returned so callers can surface it to the
    /// end user; the `Display` impl already includes it.
    Run(String),
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunnerError::PathIo(p, e) => write!(f, "I/O on {}: {}", p.display(), e),
            RunnerError::BadHatch(p, e) => {
                write!(f, "invalid hatch at {}: {}", p.display(), e)
            }
            RunnerError::NotFound(n) => write!(f, "package '{}' not found", n),
            RunnerError::Install(m) => write!(f, "install failed: {}", m),
            RunnerError::Run(m) => write!(f, "run failed: {}", m),
        }
    }
}

impl std::error::Error for RunnerError {}

/// Entry point for embedders that want to load `@hatch:*` packages
/// from pre-built `.hatch` artifacts and run Wren code against them.
pub struct HatchRunner {
    vm: VM,
    /// Directories to scan for `.hatch` artifacts during `install`.
    /// Walked in registration order; first match wins.
    search_paths: Vec<PathBuf>,
    /// Map of `name` (e.g. `@hatch:math`) to the absolute path of
    /// the chosen `.hatch` artifact. Populated lazily on the first
    /// `install` call that needs a search-path scan.
    discovered: HashMap<String, PathBuf>,
    /// Tracks which names are already installed so repeat installs
    /// are cheap no-ops rather than re-parsing the same bytes.
    installed: Vec<String>,
}

impl HatchRunner {
    /// Create a runner wrapping a fresh VM (default [`VMConfig`]).
    pub fn new() -> Self {
        Self::with_config(VMConfig::default())
    }

    /// Wrap a VM built with a custom config (output capture, error
    /// callbacks, exec mode, etc.). The VM is owned by the runner
    /// after this; retrieve it with [`Self::into_vm`] if you want
    /// it back.
    pub fn with_config(config: VMConfig) -> Self {
        Self {
            vm: VM::new(config),
            search_paths: Vec::new(),
            discovered: HashMap::new(),
            installed: Vec::new(),
        }
    }

    /// Wrap an already-constructed VM. For embedders who built
    /// their VM through the C-API or some other path first.
    pub fn from_vm(vm: VM) -> Self {
        Self {
            vm,
            search_paths: Vec::new(),
            discovered: HashMap::new(),
            installed: Vec::new(),
        }
    }

    /// Access the underlying VM. Useful when the embedder wants to
    /// bind foreign methods, push initial values into slots, or
    /// invoke handles before `run_source`.
    pub fn vm_mut(&mut self) -> &mut VM {
        &mut self.vm
    }

    /// Immutable VM access.
    pub fn vm(&self) -> &VM {
        &self.vm
    }

    /// Recover the VM, dropping the runner. The installed packages
    /// stay loaded in the VM.
    pub fn into_vm(self) -> VM {
        self.vm
    }

    /// Register a directory to scan for `.hatch` artifacts. Re-
    /// registering the same path is a no-op. Previously-discovered
    /// names aren't forgotten — add paths early if you care about
    /// resolution order.
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) -> Result<(), RunnerError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(RunnerError::PathIo(
                path.clone(),
                std::io::Error::new(std::io::ErrorKind::NotFound, "search path does not exist"),
            ));
        }
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
        Ok(())
    }

    /// Add a directory the runtime should search when loading
    /// plugin dylibs (e.g. `libwlift_sqlite.dylib`). Paths the
    /// package's own manifest declares are handled automatically —
    /// this is for embedder-provided overrides.
    pub fn add_native_path<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref().to_path_buf();
        if !self.vm.native_search_paths.contains(&path) {
            self.vm.native_search_paths.push(path);
        }
    }

    /// Install a single `@hatch:name` package into the VM. Resolves
    /// in order:
    ///
    /// 1. The search paths added via [`Self::add_search_path`].
    /// 2. The ambient cache (`$HOME/.hatch/cache`, override via
    ///    `HATCH_CACHE_DIR`) — populated by a prior `hatch install`
    ///    or `hatch build`.
    ///
    /// Repeat calls with the same name are no-ops. To install
    /// several packages at once, use [`Self::install_many`].
    pub fn install(&mut self, name: &str) -> Result<(), RunnerError> {
        if self.installed.iter().any(|n| n == name) {
            return Ok(());
        }
        let path = self.locate(name)?;
        let bytes = std::fs::read(&path).map_err(|e| RunnerError::PathIo(path.clone(), e))?;
        self.install_bytes_tagged(name, &bytes)
    }

    /// Convenience: install a batch of packages. Stops on the first
    /// error; previously-installed packages stay installed.
    pub fn install_many<S: AsRef<str>>(&mut self, names: &[S]) -> Result<(), RunnerError> {
        for n in names {
            self.install(n.as_ref())?;
        }
        Ok(())
    }

    /// Install raw `.hatch` bytes without going through search-path
    /// resolution. Used for embedders that bake packages into the
    /// binary via `include_bytes!`.
    pub fn install_bytes(&mut self, bytes: &[u8]) -> Result<(), RunnerError> {
        match self.vm.install_hatch_modules(bytes) {
            InterpretResult::Success => Ok(()),
            InterpretResult::CompileError => {
                Err(RunnerError::Install("compile error in hatch".into()))
            }
            InterpretResult::RuntimeError => {
                Err(RunnerError::Install("runtime error in hatch".into()))
            }
        }
    }

    /// Install raw bytes but remember the name so future `install`
    /// calls for the same name no-op. Used internally; exposed in
    /// case an embedder wants to mirror that behaviour.
    pub fn install_bytes_tagged(&mut self, name: &str, bytes: &[u8]) -> Result<(), RunnerError> {
        self.install_bytes(bytes)?;
        self.installed.push(name.to_string());
        Ok(())
    }

    /// Install every `.hatch` file found in the search paths.
    /// Useful for "load everything in this dir" flows; individual
    /// `install(name)` calls are more selective.
    pub fn install_all_found(&mut self) -> Result<usize, RunnerError> {
        self.rescan_search_paths()?;
        let names: Vec<String> = self.discovered.keys().cloned().collect();
        let mut count = 0;
        for name in names {
            if !self.installed.iter().any(|n| n == &name) {
                self.install(&name)?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Did [`Self::install`] (or any of its siblings) successfully
    /// install this package?
    pub fn is_installed(&self, name: &str) -> bool {
        self.installed.iter().any(|n| n == name)
    }

    /// List every installed package name, in install order.
    pub fn installed_packages(&self) -> &[String] {
        &self.installed
    }

    /// Compile + execute Wren source in a named module. `module`
    /// is the identifier other code would `import` as — use
    /// `"main"` for one-shot execution.
    pub fn run_source(&mut self, module: &str, source: &str) -> Result<(), RunnerError> {
        match self.vm.interpret(module, source) {
            InterpretResult::Success => Ok(()),
            InterpretResult::CompileError => Err(RunnerError::Run("compile error".into())),
            InterpretResult::RuntimeError => Err(RunnerError::Run("runtime error".into())),
        }
    }

    /// Run the contents of a file as Wren source.
    pub fn run_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), RunnerError> {
        let path = path.as_ref().to_path_buf();
        let source =
            std::fs::read_to_string(&path).map_err(|e| RunnerError::PathIo(path.clone(), e))?;
        let module = path.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
        self.run_source(module, &source)
    }

    // -- Internal helpers ---------------------------------------------

    /// Walk all registered search paths and fill `discovered` with
    /// `name → path` entries. Cheap if no paths were added since
    /// the previous call — only new paths get walked.
    fn rescan_search_paths(&mut self) -> Result<(), RunnerError> {
        // Clone to avoid borrowing `self` while we populate the
        // discovery map.
        let paths = self.search_paths.clone();
        for dir in paths {
            self.scan_dir(&dir)?;
        }
        Ok(())
    }

    fn scan_dir(&mut self, dir: &Path) -> Result<(), RunnerError> {
        let entries =
            std::fs::read_dir(dir).map_err(|e| RunnerError::PathIo(dir.to_path_buf(), e))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("hatch") {
                continue;
            }
            // Filename shape: `@hatch:name-<version>.hatch` or
            // `name-<version>.hatch`. Strip the `-<version>` suffix
            // to recover the logical name.
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            let name = match stem.rsplit_once('-') {
                Some((n, _ver)) => n.to_string(),
                None => stem.to_string(),
            };
            self.discovered.entry(name).or_insert(path);
        }
        Ok(())
    }

    /// Resolve a package name to a path on disk — search paths
    /// first, then the ambient cache.
    fn locate(&mut self, name: &str) -> Result<PathBuf, RunnerError> {
        if let Some(p) = self.discovered.get(name) {
            return Ok(p.clone());
        }
        // Lazy rescan in case the caller just added a path.
        self.rescan_search_paths()?;
        if let Some(p) = self.discovered.get(name) {
            return Ok(p.clone());
        }
        // Fall back to the ambient cache. We don't know the version
        // offhand — scan whatever's sitting there for the newest
        // matching artifact. This matches the `hatch install`
        // behaviour where a version is resolved at install time and
        // parked in the cache as `name-<version>.hatch`.
        if let Ok(cache) = hatch_registry::cache_root() {
            if cache.exists() {
                if let Ok(found) = scan_cache_for(&cache, name) {
                    self.discovered.insert(name.to_string(), found.clone());
                    return Ok(found);
                }
            }
        }
        Err(RunnerError::NotFound(name.to_string()))
    }
}

impl Default for HatchRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Scan a directory for the newest `.hatch` matching `name`.
/// Version comparison is lexicographic — works for the usual
/// zero-padded semver most hatches ship with.
fn scan_cache_for(cache: &Path, name: &str) -> Result<PathBuf, RunnerError> {
    let prefix = format!("{}-", name);
    let entries =
        std::fs::read_dir(cache).map_err(|e| RunnerError::PathIo(cache.to_path_buf(), e))?;
    let mut best: Option<(String, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("hatch") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        if !stem.starts_with(&prefix) {
            continue;
        }
        let version = stem[prefix.len()..].to_string();
        match &best {
            None => best = Some((version, path)),
            Some((cur, _)) if &version > cur => best = Some((version, path)),
            _ => {}
        }
    }
    best.map(|(_, p)| p)
        .ok_or_else(|| RunnerError::NotFound(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_runner_has_no_installed_packages() {
        let runner = HatchRunner::new();
        assert!(runner.installed_packages().is_empty());
        assert!(!runner.is_installed("@hatch:math"));
    }

    #[test]
    fn install_bogus_name_returns_not_found() {
        let mut runner = HatchRunner::new();
        // Use a tmpdir to avoid picking up the ambient cache.
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HATCH_CACHE_DIR", tmp.path());
        let err = runner.install("@hatch:absolutely-not-real").unwrap_err();
        match err {
            RunnerError::NotFound(_) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn add_search_path_rejects_missing_dir() {
        let mut runner = HatchRunner::new();
        let err = runner
            .add_search_path("/nonexistent/hatch/path/xyz123")
            .unwrap_err();
        assert!(matches!(err, RunnerError::PathIo(_, _)));
    }

    #[test]
    fn run_source_with_no_imports() {
        let mut runner = HatchRunner::new();
        // Capture output through the VM's buffer — no equivalent
        // knob on VMConfig, so we set it post-construction.
        runner.vm_mut().output_buffer = Some(String::new());
        runner
            .run_source("main", "System.print(1 + 1)")
            .expect("should run");
        let out = runner.vm_mut().take_output();
        assert!(out.contains("2"), "output was {:?}", out);
    }
}
