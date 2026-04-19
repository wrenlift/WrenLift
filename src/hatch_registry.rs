//! Hatch package registry: fetch published hatches as release
//! artifacts from a remote registry and cache them locally. Build-time
//! dependency resolution then loads each pinned version straight out
//! of the cache, so no network I/O happens during a normal `hatch
//! build`.
//!
//! # Why release artifacts, not a source clone
//!
//! An earlier design tried a sparse/blobless git clone of a monorepo.
//! That still has to walk the upstream tree on every first install —
//! fine today, punishing as the catalog grows. Release artifacts
//! decouple the two: one small `.hatch` download per `hatch install`,
//! no cumulative metadata, easy to cache on a CDN.
//!
//! # URL layout
//!
//! ```text
//! {registry_base}/releases/download/{name}-{version}/{name}-{version}.hatch
//! ```
//!
//! where `{registry_base}` defaults to
//! `https://github.com/wrenlift/hatch`. The scheme matches GitHub
//! releases exactly — each package is published as its own tag
//! `{name}-{version}` with a single `{name}-{version}.hatch` asset.
//!
//! # Cache layout
//!
//! `{cache_root}/{name}-{version}.hatch` — one flat file per pinned
//! version. Versions never clobber each other; removing a cached
//! entry is `rm`. `HATCH_CACHE_DIR` overrides the root (used by
//! tests + CI); default is `$HOME/.hatch/cache`.
//!
//! # Transport
//!
//! Shells out to `curl`, matching how we already invoke `git`.
//! Preinstalled on macOS, Linux, and Windows 10+. `file://` URLs are
//! a first-class target of curl, which keeps tests hermetic — no HTTP
//! server needed, no new Rust deps.

use std::path::PathBuf;
use std::process::Command;

/// Default registry base URL. Overrideable via `HATCH_REGISTRY_URL`
/// so tests and private registries work without rebuilding.
pub const DEFAULT_REGISTRY_URL: &str = "https://github.com/wrenlift/hatch";

/// Errors from registry operations. Messages are actionable — the CLI
/// surfaces them verbatim.
#[derive(Debug)]
pub enum RegistryError {
    CurlFailed { code: Option<i32>, stderr: String },
    GitFailed { code: Option<i32>, stderr: String },
    Io(std::io::Error),
    NoHome,
    BadArtifact { path: PathBuf, reason: String },
    NoCurl,
    NoGit,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::CurlFailed { code, stderr } => {
                let c = code.map(|c| c.to_string()).unwrap_or_else(|| "?".into());
                write!(f, "curl exited with {}: {}", c, stderr.trim())
            }
            RegistryError::GitFailed { code, stderr } => {
                let c = code.map(|c| c.to_string()).unwrap_or_else(|| "?".into());
                write!(f, "git exited with {}: {}", c, stderr.trim())
            }
            RegistryError::Io(e) => write!(f, "io error: {}", e),
            RegistryError::NoHome => write!(
                f,
                "could not determine a home directory for the hatch cache"
            ),
            RegistryError::BadArtifact { path, reason } => write!(
                f,
                "downloaded artifact at {} is not a hatch: {}",
                path.display(),
                reason
            ),
            RegistryError::NoCurl => write!(
                f,
                "`curl` is required to fetch from the registry but wasn't found on PATH"
            ),
            RegistryError::NoGit => write!(
                f,
                "`git` is required to resolve git-hosted dependencies but wasn't found on PATH"
            ),
        }
    }
}

impl From<std::io::Error> for RegistryError {
    fn from(e: std::io::Error) -> Self {
        RegistryError::Io(e)
    }
}

/// Registry base URL in effect for this process.
pub fn registry_url() -> String {
    std::env::var("HATCH_REGISTRY_URL").unwrap_or_else(|_| DEFAULT_REGISTRY_URL.to_string())
}

/// Cache root directory — `$HOME/.hatch/cache` by default, or
/// whatever `HATCH_CACHE_DIR` points at.
pub fn cache_root() -> Result<PathBuf, RegistryError> {
    if let Ok(override_dir) = std::env::var("HATCH_CACHE_DIR") {
        return Ok(PathBuf::from(override_dir));
    }
    let home = std::env::var_os("HOME").ok_or(RegistryError::NoHome)?;
    Ok(PathBuf::from(home).join(".hatch").join("cache"))
}

/// Path this package+version would live at, regardless of whether
/// it's actually been downloaded yet. Uses the ambient cache root
/// (`HATCH_CACHE_DIR` or `$HOME/.hatch/cache`) — call
/// [`cached_artifact_path_in`] when you already have a cache dir.
pub fn cached_artifact_path(name: &str, version: &str) -> Result<PathBuf, RegistryError> {
    Ok(cache_root()?.join(artifact_filename(name, version)))
}

/// Same as [`cached_artifact_path`] but with an explicit cache root.
/// Used by the build-time resolver and by tests that don't want to
/// touch process-wide env vars.
pub fn cached_artifact_path_in(cache_dir: &std::path::Path, name: &str, version: &str) -> PathBuf {
    cache_dir.join(artifact_filename(name, version))
}

fn artifact_filename(name: &str, version: &str) -> String {
    format!("{}-{}.hatch", name, version)
}

/// Construct the release URL a `hatch install name@version` will hit.
/// Kept public so the CLI can print a human-readable hint when a
/// download fails.
pub fn release_url(registry_base: &str, name: &str, version: &str) -> String {
    format!(
        "{}/releases/download/{}-{}/{}-{}.hatch",
        registry_base.trim_end_matches('/'),
        name,
        version,
        name,
        version,
    )
}

/// Ensure `name@version` is present in the ambient cache (resolved
/// from `HATCH_CACHE_DIR` / `$HOME`). Convenience wrapper over
/// [`ensure_in_cache_dir`].
pub fn ensure_in_cache(
    registry_base: &str,
    name: &str,
    version: &str,
) -> Result<PathBuf, RegistryError> {
    let cache = cache_root()?;
    ensure_in_cache_dir(&cache, registry_base, name, version)
}

/// Ensure `name@version` is present in the given cache directory.
/// Cache hit skips the network entirely so `hatch build` never makes
/// outbound requests.
pub fn ensure_in_cache_dir(
    cache_dir: &std::path::Path,
    registry_base: &str,
    name: &str,
    version: &str,
) -> Result<PathBuf, RegistryError> {
    let dest = cached_artifact_path_in(cache_dir, name, version);
    if dest.exists() {
        validate_hatch_magic(&dest)?;
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let url = release_url(registry_base, name, version);
    download(&url, &dest)?;
    validate_hatch_magic(&dest)?;
    Ok(dest)
}

/// Parse the `hatch install` CLI argument: `<name>` or
/// `<name>@<version>`. Returns `(name, version_opt)` — no version
/// means "resolve against the hatchfile's `[dependencies]` entry".
pub fn split_name_version(spec: &str) -> (&str, Option<&str>) {
    // Scoped names start with `@` (e.g. `@hatch:assert`), so
    // splitting on the FIRST `@` mis-parses the scope prefix as a
    // version separator. Use the LAST `@` instead — `@hatch:assert`
    // → (spec, None), `@hatch:assert@0.1.0` → (name, version).
    match spec.rsplit_once('@') {
        Some((n, v)) if !n.is_empty() => (n, Some(v)),
        _ => (spec, None),
    }
}

// ---------------------------------------------------------------------------
// Git-hosted packages
// ---------------------------------------------------------------------------
//
// Separate code path from the release-artifact flow: each git
// dependency lives in its own repo, so there's no "don't clone the
// whole monorepo" concern. A plain shallow clone at the requested ref
// is fine and keeps each dependency one self-contained checkout.

/// Checkout path a git dep would land at inside the cache. Doesn't
/// check that it exists — callers do that after `ensure_git_checkout`.
/// Structured as `<cache>/git/<sanitized-url>/<kind>/<ref>/` so a
/// branch named `v1` and a tag named `v1` can coexist without
/// collision.
pub fn cached_git_checkout_path(
    cache_dir: &std::path::Path,
    git_url: &str,
    git_ref: crate::hatch::GitRef<'_>,
) -> PathBuf {
    cache_dir
        .join("git")
        .join(sanitize_url_for_path(git_url))
        .join(git_ref.kind())
        .join(sanitize_ref_for_path(git_ref.as_str()))
}

/// Shallow-clone (or re-fetch) `git_url` at `git_ref` into the git
/// cache. Tags and revs are immutable — a cached checkout is reused
/// verbatim. Branches refetch on every call so a subsequent `hatch
/// install` can pick up upstream updates; pinned deps don't pay that
/// cost.
pub fn ensure_git_checkout(
    cache_dir: &std::path::Path,
    git_url: &str,
    git_ref: crate::hatch::GitRef<'_>,
) -> Result<PathBuf, RegistryError> {
    let checkout = cached_git_checkout_path(cache_dir, git_url, git_ref);

    let is_branch = matches!(git_ref, crate::hatch::GitRef::Branch(_));
    let already_has_hatchfile = checkout.join("hatchfile").exists();
    if already_has_hatchfile && !is_branch {
        return Ok(checkout);
    }

    if let Some(parent) = checkout.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !checkout.exists() {
        std::fs::create_dir_all(&checkout)?;
        run_git(Some(&checkout), &["init", "-q"])?;
        run_git(Some(&checkout), &["remote", "add", "origin", git_url])?;
    }

    // `git fetch --depth 1 origin <ref>` works uniformly for tags,
    // branch heads, and (on servers that allow it — GitHub, GitLab)
    // specific commits. Falls back to a plain clone + checkout if the
    // one-liner errors out.
    let fetch = run_git(
        Some(&checkout),
        &["fetch", "--depth", "1", "origin", git_ref.as_str()],
    );
    if fetch.is_err() {
        // Retry without depth in case the remote rejected the
        // shallow fetch for this ref (e.g. servers without
        // allowAnySHA1InWant for raw revs).
        run_git(
            Some(&checkout),
            &["fetch", "origin", git_ref.as_str()],
        )?;
    }
    run_git(Some(&checkout), &["checkout", "-q", "FETCH_HEAD", "--"])?;

    if !checkout.join("hatchfile").exists() {
        return Err(RegistryError::BadArtifact {
            path: checkout,
            reason: format!(
                "git ref '{}' on '{}' contains no hatchfile",
                git_ref.as_str(),
                git_url
            ),
        });
    }
    Ok(checkout)
}

fn run_git(
    cwd: Option<&std::path::Path>,
    args: &[&str],
) -> Result<(), RegistryError> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(RegistryError::NoGit);
        }
        Err(e) => return Err(RegistryError::Io(e)),
    };
    if !output.status.success() {
        return Err(RegistryError::GitFailed {
            code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

/// Flatten a git URL into a single directory name. Different URLs
/// produce different sanitized strings in practice — separators
/// (`/`, `:`, `.`, `?`, etc.) collapse to `_`.
fn sanitize_url_for_path(url: &str) -> String {
    url.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

/// Refs can contain `/` (branch `release/1.x`) or other characters
/// we can't drop on disk unchanged; flatten them the same way.
fn sanitize_ref_for_path(git_ref: &str) -> String {
    sanitize_url_for_path(git_ref)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn download(url: &str, dest: &std::path::Path) -> Result<(), RegistryError> {
    let mut cmd = Command::new("curl");
    cmd.args([
        "-sSL",
        "--fail",
        "-o",
        dest.to_string_lossy().as_ref(),
        url,
    ]);
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(RegistryError::NoCurl);
        }
        Err(e) => return Err(RegistryError::Io(e)),
    };
    if !output.status.success() {
        // Remove partial file so the next attempt starts clean.
        let _ = std::fs::remove_file(dest);
        return Err(RegistryError::CurlFailed {
            code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

fn validate_hatch_magic(path: &std::path::Path) -> Result<(), RegistryError> {
    let bytes = std::fs::read(path)?;
    if !crate::hatch::looks_like_hatch(&bytes) {
        // Delete the bogus file so the caller gets a clean retry.
        let _ = std::fs::remove_file(path);
        return Err(RegistryError::BadArtifact {
            path: path.to_path_buf(),
            reason: "missing HATCH magic bytes".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_name_version_handles_both_shapes() {
        assert_eq!(split_name_version("json"), ("json", None));
        assert_eq!(split_name_version("json@1.0.0"), ("json", Some("1.0.0")));
        assert_eq!(
            split_name_version("counter@0.2.0-alpha"),
            ("counter", Some("0.2.0-alpha"))
        );
        // Scoped names own a leading `@` — it's part of the name,
        // not a version separator.
        assert_eq!(split_name_version("@hatch:assert"), ("@hatch:assert", None));
        assert_eq!(
            split_name_version("@hatch:assert@0.1.0"),
            ("@hatch:assert", Some("0.1.0"))
        );
    }

    #[test]
    fn release_url_has_expected_shape() {
        let url = release_url("https://example.com/reg", "json", "1.0.0");
        assert_eq!(
            url,
            "https://example.com/reg/releases/download/json-1.0.0/json-1.0.0.hatch"
        );
        // Trailing slash on the base must not double up.
        let url = release_url("https://example.com/reg/", "json", "1.0.0");
        assert_eq!(
            url,
            "https://example.com/reg/releases/download/json-1.0.0/json-1.0.0.hatch"
        );
    }

    /// Drive `ensure_in_cache` against a `file://` URL pointing at a
    /// real `.hatch` artifact. curl supports `file://` natively, so
    /// the test covers the full fetch path without needing an HTTP
    /// server. Also verifies cache-hit short-circuits the network.
    #[test]
    fn ensure_in_cache_fetches_and_is_idempotent() {
        // Guard: if curl isn't on PATH (stripped-down CI), skip.
        if Command::new("curl")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            return;
        }

        let scratch = tempfile::tempdir().expect("tempdir");

        // Produce a valid .hatch to serve as the "release asset". The
        // contents don't matter for this test — only the magic check
        // inside `ensure_in_cache` does.
        let hatch = crate::hatch::Hatch {
            manifest: crate::hatch::Manifest {
                name: "json".to_string(),
                version: "1.0.0".to_string(),
                entry: "json".to_string(),
                modules: vec!["json".to_string()],
                dependencies: std::collections::BTreeMap::new(),
                native_libs: std::collections::BTreeMap::new(),
                native_search_paths: Vec::new(),
            },
            sections: Vec::new(),
        };
        let bytes = crate::hatch::emit(&hatch).unwrap();

        // Lay out a fake "registry" directory matching the URL shape
        // `ensure_in_cache` expects: `.../releases/download/<tag>/<file>`.
        let reg_root = scratch.path().join("reg");
        let asset_dir = reg_root.join("releases/download/json-1.0.0");
        std::fs::create_dir_all(&asset_dir).unwrap();
        std::fs::write(asset_dir.join("json-1.0.0.hatch"), &bytes).unwrap();

        let cache_dir = scratch.path().join("cache");
        let url = format!("file://{}", reg_root.display());

        let cached = ensure_in_cache_dir(&cache_dir, &url, "json", "1.0.0").expect("fetch");
        assert!(cached.exists(), "artifact downloaded to cache");
        let reread = std::fs::read(&cached).unwrap();
        assert_eq!(reread, bytes, "bytes round-trip through cache");

        // Second call: remove the "upstream" file to prove we hit the
        // cache and don't re-download.
        std::fs::remove_file(asset_dir.join("json-1.0.0.hatch")).unwrap();
        let again = ensure_in_cache_dir(&cache_dir, &url, "json", "1.0.0").expect("cache hit");
        assert_eq!(again, cached);
    }

    /// Drive `ensure_git_checkout` against a local bare-ish git
    /// fixture. Verifies tags resolve via shallow fetch and that a
    /// second call is a cache hit (tags are immutable, so we don't
    /// refetch).
    #[test]
    fn ensure_git_checkout_clones_tag_and_is_cache_idempotent() {
        // Skip if git or curl are missing (we only need git here,
        // but consistent guard style).
        if Command::new("git")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            return;
        }

        let scratch = tempfile::tempdir().expect("tempdir");

        // Build a real git repo at scratch/upstream with a hatchfile
        // and a tagged commit.
        let upstream = scratch.path().join("upstream");
        std::fs::create_dir_all(&upstream).unwrap();
        std::fs::write(
            upstream.join("hatchfile"),
            "name = \"mylib\"\nversion = \"0.3.0\"\nentry = \"mylib\"\n",
        )
        .unwrap();
        std::fs::write(upstream.join("mylib.wren"), "1").unwrap();
        let _ = run_git(Some(&upstream), &["init", "-q", "-b", "main"]);
        let _ = run_git(
            Some(&upstream),
            &["-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        );
        let _ = run_git(
            Some(&upstream),
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "--no-gpg-sign",
                "-m",
                "first",
            ],
        );
        // Tag the first commit so the shallow fetch has something
        // stable to pull.
        let _ = run_git(Some(&upstream), &["tag", "v0.3.0"]);

        let cache_dir = scratch.path().join("cache");
        let url = upstream.to_string_lossy().into_owned();

        let ref_ = crate::hatch::GitRef::Tag("v0.3.0");
        let first = ensure_git_checkout(&cache_dir, &url, ref_).expect("checkout");
        assert!(first.join("hatchfile").exists());
        assert!(first.join("mylib.wren").exists());

        // Second call: must not re-fetch. Delete upstream to prove
        // we're hitting the cache (tags are immutable — no reason to
        // refresh).
        std::fs::remove_dir_all(&upstream).unwrap();
        let second = ensure_git_checkout(&cache_dir, &url, ref_).expect("cache hit");
        assert_eq!(first, second);
    }

    #[test]
    fn ensure_in_cache_rejects_non_hatch_payload() {
        if Command::new("curl")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            return;
        }

        let scratch = tempfile::tempdir().expect("tempdir");
        let reg_root = scratch.path().join("reg");
        let asset_dir = reg_root.join("releases/download/bad-0.0.1");
        std::fs::create_dir_all(&asset_dir).unwrap();
        // Garbage bytes, no HATCH magic.
        std::fs::write(asset_dir.join("bad-0.0.1.hatch"), b"not a hatch").unwrap();

        let cache_dir = scratch.path().join("cache");
        let url = format!("file://{}", reg_root.display());

        let result = ensure_in_cache_dir(&cache_dir, &url, "bad", "0.0.1");
        assert!(matches!(result, Err(RegistryError::BadArtifact { .. })));
        // And the corrupt file must not linger in the cache.
        assert!(!cached_artifact_path_in(&cache_dir, "bad", "0.0.1").exists());
    }
}
