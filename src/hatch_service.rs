//! Hatch registry service client: talks to the Supabase-backed
//! catalog that maps package names to their git locations. Writes
//! (publish) require a stored JWT from `hatch login`; reads
//! (find) use the project's anon key and work without auth.
//!
//! # What lives where
//!
//! * **Package bytes** — not here. They live in the git repos they
//!   were published from. Consumers still install + build via the
//!   existing [`crate::hatch_registry`] pipeline.
//! * **Catalog metadata** — `(name, git, description, owner, …)`
//!   rows stored in Supabase's `packages` table. Row-level security
//!   enforces "first publisher owns the name"; subsequent publishes
//!   to the same name require the same GitHub identity.
//!
//! # Transport
//!
//! All HTTP calls shell out to `curl` — same dependency we already
//! use for release artifact downloads. Keeps the build lean and
//! lines up with how a typical user's environment already has curl +
//! git on PATH.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use sha2::Digest;

/// Supabase project URL for the hatch catalog. Override via
/// `HATCH_SERVICE_URL` for tests, private mirrors, or forks.
pub const DEFAULT_SERVICE_URL: &str = "https://bncbgsfttafynyalrrbb.supabase.co";

/// Public anon key — safe to ship in source because Supabase gates
/// every request through row-level security policies. Write paths
/// additionally require the user's JWT from `hatch login`, so
/// leaking this key only exposes what's already public: reads of
/// the `packages` table. Override via `HATCH_SERVICE_ANON_KEY`.
pub const DEFAULT_SERVICE_ANON_KEY: &str = "sb_publishable_uTNCHuGnN5mzsWbuJI8CQg_0qFEBQRz";

/// Runtime configuration resolved from env vars (with defaults).
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub url: String,
    pub anon_key: String,
}

impl ServiceConfig {
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("HATCH_SERVICE_URL")
                .unwrap_or_else(|_| DEFAULT_SERVICE_URL.to_string()),
            anon_key: std::env::var("HATCH_SERVICE_ANON_KEY")
                .unwrap_or_else(|_| DEFAULT_SERVICE_ANON_KEY.to_string()),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.anon_key.is_empty()
    }
}

/// One row in the `packages` table — what `hatch find` returns and
/// what `hatch publish` sends. `(name, version)` is the compound
/// primary key: each published version lands as its own row so
/// consumers can pin exactly (`@hatch:assert-0.1.0` in the index
/// mirror, `@hatch:assert = "0.1.0"` in hatchfiles).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PackageRecord {
    pub name: String,
    pub version: String,
    pub git: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Present in server responses, omitted on submission (the
    /// server sets it from the auth'd JWT).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
}

/// Persisted auth state. Lives at `~/.hatch/credentials` with 0600
/// perms. The refresh token lets the CLI quietly renew the access
/// token when Supabase's short-lived JWT expires; `expires_at`
/// (Unix seconds) tells `ensure_fresh_credentials` when to bother.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Credentials {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub service_url: Option<String>,
    /// Unix seconds at which the access_token stops being valid.
    /// Missing for `--token` logins where we can't infer the expiry.
    #[serde(default)]
    pub expires_at: Option<i64>,
}

#[derive(Debug)]
pub enum ServiceError {
    NotConfigured,
    NoCurl,
    NoHome,
    Io(std::io::Error),
    CurlFailed { code: Option<i32>, stderr: String },
    Http { status: u16, body: String },
    Decode(String),
    PackageNotFound(String),
    NotLoggedIn,
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceError::NotConfigured => write!(
                f,
                "hatch service is not configured — set HATCH_SERVICE_URL / HATCH_SERVICE_ANON_KEY"
            ),
            ServiceError::NoCurl => write!(f, "`curl` is required but wasn't found on PATH"),
            ServiceError::NoHome => write!(f, "could not determine a home directory"),
            ServiceError::Io(e) => write!(f, "io error: {}", e),
            ServiceError::CurlFailed { code, stderr } => {
                let c = code.map(|c| c.to_string()).unwrap_or_else(|| "?".into());
                write!(f, "curl exited with {}: {}", c, stderr.trim())
            }
            ServiceError::Http { status, body } => {
                write!(f, "HTTP {} from service: {}", status, body.trim())
            }
            ServiceError::Decode(msg) => write!(f, "could not decode service response: {}", msg),
            ServiceError::PackageNotFound(name) => {
                write!(f, "no package '{}' in the catalog", name)
            }
            ServiceError::NotLoggedIn => write!(
                f,
                "not logged in — run `hatch login` (or `hatch login --token <JWT>`)"
            ),
        }
    }
}

impl From<std::io::Error> for ServiceError {
    fn from(e: std::io::Error) -> Self {
        ServiceError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Credentials storage
// ---------------------------------------------------------------------------

/// Canonical path where `hatch login` stores the JWT. Overridable
/// via `HATCH_CREDENTIALS_FILE` so tests don't touch the real home
/// directory.
pub fn credentials_path() -> Result<PathBuf, ServiceError> {
    if let Ok(p) = std::env::var("HATCH_CREDENTIALS_FILE") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME").ok_or(ServiceError::NoHome)?;
    Ok(PathBuf::from(home).join(".hatch").join("credentials"))
}

pub fn load_credentials() -> Result<Option<Credentials>, ServiceError> {
    let path = credentials_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    let creds: Credentials =
        toml::from_str(&text).map_err(|e| ServiceError::Decode(format!("credentials: {}", e)))?;
    Ok(Some(creds))
}

pub fn save_credentials(creds: &Credentials) -> Result<PathBuf, ServiceError> {
    let path = credentials_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(creds)
        .map_err(|e| ServiceError::Decode(format!("credentials: {}", e)))?;
    std::fs::write(&path, text)?;
    // 0600 so tokens aren't world-readable on shared systems.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&path)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&path, perms)?;
    }
    Ok(path)
}

pub fn clear_credentials() -> Result<bool, ServiceError> {
    let path = credentials_path()?;
    if !path.exists() {
        return Ok(false);
    }
    std::fs::remove_file(&path)?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Catalog reads: `hatch find`
// ---------------------------------------------------------------------------

/// Look up every published version of a package by name. Returns
/// an empty list if the catalog has nothing under the name —
/// distinct from a transport error, which returns `Err`.
///
/// The rows arrive sorted newest-first by `version`-desc so
/// `find_versions(...)?.first()` is always the latest entry.
pub fn find_versions(
    config: &ServiceConfig,
    name: &str,
) -> Result<Vec<PackageRecord>, ServiceError> {
    if !config.is_configured() {
        return Err(ServiceError::NotConfigured);
    }
    let url = format!(
        "{}/rest/v1/packages?select=*&name=eq.{}&order=version.desc",
        config.url.trim_end_matches('/'),
        urlencode(name),
    );
    let body = curl_get(&url, &[&format!("apikey: {}", config.anon_key)])?;
    let rows: Vec<PackageRecord> = serde_json::from_str(&body)
        .map_err(|e| ServiceError::Decode(format!("find_versions: {}", e)))?;
    Ok(rows)
}

/// Look up one specific `(name, version)` row. Used by the build-
/// time resolver before it reaches for the cached artifact.
pub fn find_exact(
    config: &ServiceConfig,
    name: &str,
    version: &str,
) -> Result<Option<PackageRecord>, ServiceError> {
    if !config.is_configured() {
        return Err(ServiceError::NotConfigured);
    }
    let url = format!(
        "{}/rest/v1/packages?select=*&name=eq.{}&version=eq.{}",
        config.url.trim_end_matches('/'),
        urlencode(name),
        urlencode(version),
    );
    let body = curl_get(&url, &[&format!("apikey: {}", config.anon_key)])?;
    let rows: Vec<PackageRecord> = serde_json::from_str(&body)
        .map_err(|e| ServiceError::Decode(format!("find_exact: {}", e)))?;
    Ok(rows.into_iter().next())
}

/// Backwards-compatible shim for `find_versions(...)?.first()` —
/// kept so existing callers continue to work while we roll out
/// version-aware clients.
pub fn find_package(
    config: &ServiceConfig,
    name: &str,
) -> Result<Option<PackageRecord>, ServiceError> {
    Ok(find_versions(config, name)?.into_iter().next())
}

// ---------------------------------------------------------------------------
// Catalog writes: `hatch publish`
// ---------------------------------------------------------------------------

/// Submit a package record to the catalog. Two transport paths:
///
/// 1. **Default** — PostgREST at `<service_url>/rest/v1/packages`.
///    Supabase's RLS policies enforce name ownership: the first
///    insert claims the name for the caller's GitHub identity,
///    subsequent updates require that same JWT-carried identity.
///
/// 2. **`HATCH_BOT_URL` override** — POST to a custom endpoint
///    (the hatch-bot Edge Function is the canonical one). The
///    function owns its own auth via a shared-secret bearer, and
///    writes to the same `packages` table from inside Supabase's
///    trust domain. Used by CI pipelines where the legacy JWT
///    flow isn't viable (e.g. projects migrated to asymmetric
///    JWT signing keys where no HS256-signed CI tokens exist).
///
/// When `HATCH_BOT_URL` is set, the `access_token` is sent as
/// `Authorization: Bearer <token>` directly — no PostgREST-
/// specific `apikey` / `Prefer` headers. The body is still the
/// same `PackageRecord` JSON.
pub fn publish_package(
    config: &ServiceConfig,
    creds: &Credentials,
    record: &PackageRecord,
) -> Result<(), ServiceError> {
    if !config.is_configured() {
        return Err(ServiceError::NotConfigured);
    }
    let body = serde_json::to_string(&record)
        .map_err(|e| ServiceError::Decode(format!("publish payload: {}", e)))?;

    if let Ok(bot_url) = std::env::var("HATCH_BOT_URL") {
        let url = bot_url.trim_end_matches('/').to_string();
        let headers = [
            format!("Authorization: Bearer {}", creds.access_token),
            "Content-Type: application/json".to_string(),
        ];
        let header_refs: Vec<&str> = headers.iter().map(String::as_str).collect();
        let _ = curl_post(&url, &header_refs, &body)?;
        return Ok(());
    }

    // Default: direct PostgREST upsert.
    let url = format!(
        "{}/rest/v1/packages?on_conflict=name,version",
        config.url.trim_end_matches('/')
    );
    let headers = [
        format!("apikey: {}", config.anon_key),
        format!("Authorization: Bearer {}", creds.access_token),
        "Content-Type: application/json".to_string(),
        // Upsert — so re-publishing updates the existing row rather
        // than failing on the primary key.
        "Prefer: resolution=merge-duplicates".to_string(),
    ];
    let header_refs: Vec<&str> = headers.iter().map(String::as_str).collect();
    let _ = curl_post(&url, &header_refs, &body)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive login (GitHub OAuth via Supabase + PKCE)
// ---------------------------------------------------------------------------
//
// CLI OAuth flow: CLI starts a one-shot HTTP listener on localhost,
// opens a browser against Supabase's authorize endpoint with a PKCE
// challenge, waits for Supabase to redirect back with an auth code,
// then exchanges the code for access/refresh tokens. Entirely
// self-contained — no `gh`, no pre-minted tokens, no manual copy-
// paste. A timeout short-circuits stuck browsers so an abandoned
// `hatch login` doesn't hang indefinitely.

/// Default maximum time we'll wait for the browser redirect before
/// giving up. User can Ctrl-C sooner.
const LOGIN_TIMEOUT_SECS: u64 = 180;

/// Run the full interactive login. Blocks until the user completes
/// (or abandons) the browser flow. Returns the stored credentials on
/// success. The listener port is chosen by the OS — whatever it
/// picks gets plugged into `redirect_to` and the Supabase GitHub
/// OAuth app must have `http://localhost:*` on its allowed redirect
/// list (Supabase treats `http://localhost` loopback specially and
/// permits any port).
pub fn interactive_login(config: &ServiceConfig) -> Result<Credentials, ServiceError> {
    if !config.is_configured() {
        return Err(ServiceError::NotConfigured);
    }

    let verifier = gen_pkce_verifier();
    let challenge = pkce_challenge(&verifier);

    let listener = TcpListener::bind("127.0.0.1:0").map_err(ServiceError::Io)?;
    let port = listener.local_addr().map_err(ServiceError::Io)?.port();
    let redirect = format!("http://localhost:{}/callback", port);

    let auth_url = format!(
        "{}/auth/v1/authorize?provider=github&flow_type=pkce&redirect_to={}&code_challenge={}&code_challenge_method=s256",
        config.url.trim_end_matches('/'),
        urlencode(&redirect),
        challenge,
    );

    eprintln!("opening browser for GitHub authorization…");
    eprintln!("if it doesn't open, visit:\n  {}", auth_url);
    let _ = open_in_browser(&auth_url);

    let code = accept_callback(&listener, Duration::from_secs(LOGIN_TIMEOUT_SECS))?;

    // Exchange the auth code for tokens.
    let body = format!(
        "{{\"auth_code\":\"{}\",\"code_verifier\":\"{}\"}}",
        code, verifier
    );
    let url = format!(
        "{}/auth/v1/token?grant_type=pkce",
        config.url.trim_end_matches('/')
    );
    let resp = curl_post(
        &url,
        &[
            &format!("apikey: {}", config.anon_key),
            "Content-Type: application/json",
        ],
        &body,
    )?;

    let tokens: TokenResponse = serde_json::from_str(&resp)
        .map_err(|e| ServiceError::Decode(format!("token response: {}", e)))?;
    Ok(tokens_to_credentials(config, tokens))
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    /// Seconds until the access token expires. Supabase sets this
    /// on every issuance; missing only on non-standard servers.
    #[serde(default)]
    expires_in: Option<i64>,
    /// Unix seconds of expiry. Some Supabase versions include this
    /// directly; when absent we derive it from `expires_in`.
    #[serde(default)]
    expires_at: Option<i64>,
}

/// Pull `access_token`, `refresh_token`, and a computed expiry out
/// of a Supabase token response. Centralized so the same logic
/// runs for login and refresh; if Supabase ever sends `expires_at`
/// directly we respect it verbatim instead of re-deriving.
fn tokens_to_credentials(config: &ServiceConfig, tokens: TokenResponse) -> Credentials {
    let expires_at = tokens
        .expires_at
        .or_else(|| tokens.expires_in.map(|dur| now_unix() + dur));
    Credentials {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        service_url: Some(config.url.clone()),
        expires_at,
    }
}

/// Trade the stored refresh token for a fresh access/refresh pair.
/// Returns an error if the credentials have no refresh token — the
/// caller should push the user toward `hatch login` in that case.
pub fn refresh_access_token(
    config: &ServiceConfig,
    creds: &Credentials,
) -> Result<Credentials, ServiceError> {
    if !config.is_configured() {
        return Err(ServiceError::NotConfigured);
    }
    let Some(refresh) = creds.refresh_token.as_deref() else {
        return Err(ServiceError::NotLoggedIn);
    };
    let body = format!("{{\"refresh_token\":\"{}\"}}", refresh);
    let url = format!(
        "{}/auth/v1/token?grant_type=refresh_token",
        config.url.trim_end_matches('/')
    );
    let resp = curl_post(
        &url,
        &[
            &format!("apikey: {}", config.anon_key),
            "Content-Type: application/json",
        ],
        &body,
    )?;
    let tokens: TokenResponse = serde_json::from_str(&resp)
        .map_err(|e| ServiceError::Decode(format!("refresh response: {}", e)))?;
    Ok(tokens_to_credentials(config, tokens))
}

/// Seconds of slack we keep before the nominal expiry. A request
/// that takes 30s to fly to Supabase + come back shouldn't expire
/// mid-flight; 60s cushion covers that plus clock skew.
const EXPIRY_SKEW_SECS: i64 = 60;

/// Refresh the access token if it's close to (or past) expiry, and
/// persist the new credentials to disk. Returns the current-good
/// credentials — possibly the original, possibly freshly minted.
///
/// Used before every authenticated request so long-running sessions
/// don't break when Supabase's hour-ish JWT times out.
pub fn ensure_fresh_credentials(
    config: &ServiceConfig,
    creds: Credentials,
) -> Result<Credentials, ServiceError> {
    let needs_refresh = match creds.expires_at {
        Some(t) => now_unix() + EXPIRY_SKEW_SECS >= t,
        // Unknown expiry — can't tell whether to refresh. If we
        // have a refresh token we could optimistically refresh
        // every call, but that hammers the auth endpoint for
        // legacy --token logins where refresh isn't possible
        // anyway. Leave it alone and let the request itself 401.
        None => false,
    };
    if !needs_refresh {
        return Ok(creds);
    }
    if creds.refresh_token.is_none() {
        // Expired but no refresh token — the user has to log in
        // again manually.
        return Err(ServiceError::NotLoggedIn);
    }
    let refreshed = refresh_access_token(config, &creds)?;
    let _ = save_credentials(&refreshed)?;
    Ok(refreshed)
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Generate a PKCE code verifier — 96 random bytes encoded
/// base64url, yielding a 128-char string. Well above the 43-char
/// RFC 7636 minimum.
fn gen_pkce_verifier() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 96];
    rand::rng().fill_bytes(&mut bytes);
    base64url_encode(&bytes)
}

/// PKCE S256 challenge: SHA-256 of the verifier, base64url-encoded
/// without padding (RFC 7636 §4.2).
pub(crate) fn pkce_challenge(verifier: &str) -> String {
    let hash = sha2::Sha256::digest(verifier.as_bytes());
    base64url_encode(&hash)
}

/// Base64url without padding — RFC 7636 / 4648. Tiny hand-rolled
/// impl to avoid pulling in the `base64` crate for 20 lines of
/// output.
pub(crate) fn base64url_encode(input: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHA[(n & 0x3F) as usize] as char);
        }
    }
    out
}

/// Accept one HTTP request on `listener`, extract `?code=<CODE>`
/// from the request line, respond with a friendly "you can close
/// this tab" page, and return the code. Bounded by `timeout` so a
/// user who never completes the browser flow doesn't leave the CLI
/// wedged.
fn accept_callback(listener: &TcpListener, timeout: Duration) -> Result<String, ServiceError> {
    listener.set_nonblocking(true).map_err(ServiceError::Io)?;
    let deadline = std::time::Instant::now() + timeout;

    let (mut stream, _) = loop {
        match listener.accept() {
            Ok(s) => break s,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return Err(ServiceError::Decode(
                        "timed out waiting for browser redirect".to_string(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(ServiceError::Io(e)),
        }
    };
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(ServiceError::Io)?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(ServiceError::Io)?;

    // Read until we see the end of headers. A 2KB cap covers any
    // reasonable GET + a few hundred bytes of headers; OAuth
    // redirects don't carry bodies.
    let mut buf = [0u8; 4096];
    let n = stream.read(&mut buf).map_err(ServiceError::Io)?;
    let req = String::from_utf8_lossy(&buf[..n]).into_owned();

    let code = extract_code_from_request(&req)
        .ok_or_else(|| ServiceError::Decode("callback missing `code` query param".to_string()))?;

    let html = b"<!doctype html><html><head><meta charset=utf-8><title>hatch login</title></head>\
        <body style='font-family:system-ui;text-align:center;padding-top:4rem'>\
        <h2>You're signed in to hatch.</h2>\
        <p>Return to your terminal - you can close this tab.</p>\
        </body></html>";
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        html.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.write_all(html);
    let _ = stream.flush();

    Ok(code)
}

/// Pull the `code` query param out of a raw HTTP request. Exposed
/// for unit tests.
pub(crate) fn extract_code_from_request(req: &str) -> Option<String> {
    let first = req.lines().next()?;
    let path = first.split_whitespace().nth(1)?;
    let (_, query) = path.split_once('?')?;
    for kv in query.split('&') {
        if let Some((k, v)) = kv.split_once('=') {
            if k == "code" {
                return Some(urldecode(v));
            }
        }
    }
    None
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_digit(bytes[i + 1]), hex_digit(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn open_in_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };
    #[cfg(windows)]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/c", "start", "", url]);
        c
    };
    cmd.spawn()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// curl plumbing
// ---------------------------------------------------------------------------

fn curl_get(url: &str, headers: &[&str]) -> Result<String, ServiceError> {
    let mut cmd = Command::new("curl");
    cmd.arg("-sS").arg("--fail-with-body");
    for h in headers {
        cmd.arg("-H").arg(h);
    }
    cmd.arg(url);
    run_curl(cmd)
}

fn curl_post(url: &str, headers: &[&str], body: &str) -> Result<String, ServiceError> {
    let mut cmd = Command::new("curl");
    cmd.arg("-sS").arg("--fail-with-body").arg("-X").arg("POST");
    for h in headers {
        cmd.arg("-H").arg(h);
    }
    cmd.arg("--data-binary").arg(body);
    cmd.arg(url);
    run_curl(cmd)
}

fn run_curl(mut cmd: Command) -> Result<String, ServiceError> {
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(ServiceError::NoCurl),
        Err(e) => return Err(ServiceError::Io(e)),
    };
    if !output.status.success() {
        // --fail-with-body writes body to stdout on HTTP errors and
        // non-zero-exits curl; surface both so callers can show the
        // user what the service actually said.
        let body = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(ServiceError::CurlFailed {
            code: output.status.code(),
            stderr: if body.is_empty() { stderr } else { body },
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn urlencode(input: &str) -> String {
    // Minimal percent-encoding for the handful of characters a
    // package name could plausibly contain. Supabase / PostgREST
    // accept ASCII alnum plus `-` and `_` bare; everything else we
    // replace so we never emit an invalid URL for a weird name.
    let mut out = String::with_capacity(input.len());
    for b in input.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencode_passes_alnum_through() {
        assert_eq!(urlencode("json"), "json");
        assert_eq!(urlencode("alice-mylib_2"), "alice-mylib_2");
        assert_eq!(urlencode("a/b"), "a%2Fb");
        assert_eq!(urlencode("space here"), "space%20here");
    }

    #[test]
    fn service_config_detects_unset_anon_key() {
        // Guard against the default empty anon key so a fresh
        // install fails loudly instead of hitting a real URL.
        let c = ServiceConfig {
            url: "https://anywhere".to_string(),
            anon_key: String::new(),
        };
        assert!(!c.is_configured());
        let c2 = ServiceConfig {
            url: "https://anywhere".to_string(),
            anon_key: "abc".to_string(),
        };
        assert!(c2.is_configured());
    }

    #[test]
    fn credentials_round_trip_through_disk() {
        // Point at a scratch file so the test never touches $HOME.
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("creds");
        std::env::set_var("HATCH_CREDENTIALS_FILE", &path);

        let before = Credentials {
            access_token: "jwt-abc".to_string(),
            refresh_token: Some("refresh-xyz".to_string()),
            service_url: Some("https://hatch.supabase.co".to_string()),
            expires_at: Some(1_700_000_000),
        };
        save_credentials(&before).expect("save");
        assert!(path.exists());

        let after = load_credentials().expect("load").expect("some");
        assert_eq!(after.access_token, "jwt-abc");
        assert_eq!(after.refresh_token.as_deref(), Some("refresh-xyz"));
        assert_eq!(after.expires_at, Some(1_700_000_000));

        // 0600 perms on unix.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        assert!(clear_credentials().expect("clear"));
        assert!(load_credentials().expect("reload").is_none());

        std::env::remove_var("HATCH_CREDENTIALS_FILE");
    }

    #[test]
    fn package_record_omits_owner_on_submit() {
        // Submissions shouldn't include the owner field — the server
        // sets it from the auth'd JWT. We rely on skip_serializing_if
        // to keep the payload clean. Version IS included so the
        // compound primary key gets populated.
        let r = PackageRecord {
            name: "json".into(),
            version: "1.0.0".into(),
            git: "https://github.com/wrenlift/hatch.git".into(),
            description: Some("JSON parsing".into()),
            owner: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            !json.contains("owner"),
            "owner leaked into submit body: {}",
            json
        );
        assert!(json.contains("json"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn package_record_reads_owner_from_service() {
        // And responses carry owner back — the CLI uses it to show
        // who holds the name in `hatch find`.
        let json = r#"{
            "name": "json",
            "version": "1.0.0",
            "git": "https://github.com/wrenlift/hatch.git",
            "description": "JSON parsing",
            "owner": "01234567-89ab-cdef-0123-456789abcdef"
        }"#;
        let r: PackageRecord = serde_json::from_str(json).unwrap();
        assert_eq!(
            r.owner.as_deref(),
            Some("01234567-89ab-cdef-0123-456789abcdef")
        );
        assert_eq!(r.version, "1.0.0");
    }

    #[test]
    fn base64url_encodes_rfc_vectors() {
        // RFC 4648 test vectors (no padding).
        assert_eq!(base64url_encode(b""), "");
        assert_eq!(base64url_encode(b"f"), "Zg");
        assert_eq!(base64url_encode(b"fo"), "Zm8");
        assert_eq!(base64url_encode(b"foo"), "Zm9v");
        assert_eq!(base64url_encode(b"foob"), "Zm9vYg");
        assert_eq!(base64url_encode(b"fooba"), "Zm9vYmE");
        assert_eq!(base64url_encode(b"foobar"), "Zm9vYmFy");
        // `-` and `_` instead of `+` and `/`.
        let with_plus_slash = [0xfb, 0xff, 0xbf];
        let out = base64url_encode(&with_plus_slash);
        assert!(!out.contains('+'));
        assert!(!out.contains('/'));
        assert!(!out.contains('='));
    }

    #[test]
    fn pkce_challenge_matches_rfc_7636_appendix_b() {
        // From RFC 7636 Appendix B — canonical PKCE vector.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(
            pkce_challenge(verifier),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn extract_code_handles_happy_path_and_reorderings() {
        // Standard callback from Supabase.
        let req = "GET /callback?code=abc123&state=xyz HTTP/1.1\r\nHost: localhost:51234\r\n\r\n";
        assert_eq!(extract_code_from_request(req).as_deref(), Some("abc123"));
        // Order-independent.
        let req = "GET /callback?state=xyz&code=abc123 HTTP/1.1\r\n\r\n";
        assert_eq!(extract_code_from_request(req).as_deref(), Some("abc123"));
        // URL-encoded payload decodes.
        let req = "GET /callback?code=a%20b HTTP/1.1\r\n\r\n";
        assert_eq!(extract_code_from_request(req).as_deref(), Some("a b"));
    }

    #[test]
    fn extract_code_none_when_missing() {
        let req = "GET /callback?state=xyz HTTP/1.1\r\n\r\n";
        assert_eq!(extract_code_from_request(req), None);
        // Not a callback path at all.
        let req = "GET /favicon.ico HTTP/1.1\r\n\r\n";
        assert_eq!(extract_code_from_request(req), None);
    }

    #[test]
    fn tokens_to_credentials_derives_expiry_from_expires_in() {
        // Supabase always sends `expires_in`; we turn it into an
        // absolute `expires_at` so later refresh decisions don't
        // have to remember when the token was issued.
        let cfg = ServiceConfig {
            url: "https://x".into(),
            anon_key: "k".into(),
        };
        let tokens = TokenResponse {
            access_token: "jwt".to_string(),
            refresh_token: Some("rt".to_string()),
            expires_in: Some(3600),
            expires_at: None,
        };
        let before = now_unix();
        let creds = tokens_to_credentials(&cfg, tokens);
        let after = now_unix();
        let got = creds.expires_at.expect("expiry derived");
        assert!(got >= before + 3600 && got <= after + 3600);
    }

    #[test]
    fn tokens_to_credentials_honours_explicit_expires_at() {
        // If Supabase sends `expires_at` directly we trust it
        // verbatim instead of re-deriving.
        let cfg = ServiceConfig {
            url: "https://x".into(),
            anon_key: "k".into(),
        };
        let tokens = TokenResponse {
            access_token: "jwt".to_string(),
            refresh_token: None,
            expires_in: Some(3600),
            expires_at: Some(2_000_000_000),
        };
        let creds = tokens_to_credentials(&cfg, tokens);
        assert_eq!(creds.expires_at, Some(2_000_000_000));
    }

    #[test]
    fn ensure_fresh_credentials_is_noop_when_token_is_valid() {
        // Valid for another hour → pass through untouched. Reaches
        // the network only when the token is near expiry, and we
        // aren't near expiry here.
        let cfg = ServiceConfig {
            url: "https://x".into(),
            anon_key: "k".into(),
        };
        let creds = Credentials {
            access_token: "still-good".to_string(),
            refresh_token: Some("rt".to_string()),
            service_url: Some(cfg.url.clone()),
            expires_at: Some(now_unix() + 3600),
        };
        let out = ensure_fresh_credentials(&cfg, creds.clone()).expect("no refresh needed");
        assert_eq!(out.access_token, "still-good");
    }

    #[test]
    fn ensure_fresh_credentials_errors_when_expired_without_refresh_token() {
        // --token logins don't carry a refresh token; once the
        // token expires the user has to re-auth manually.
        let cfg = ServiceConfig {
            url: "https://x".into(),
            anon_key: "k".into(),
        };
        let creds = Credentials {
            access_token: "expired".to_string(),
            refresh_token: None,
            service_url: Some(cfg.url.clone()),
            expires_at: Some(now_unix() - 60), // already past expiry
        };
        let err = ensure_fresh_credentials(&cfg, creds).expect_err("should require re-login");
        assert!(matches!(err, ServiceError::NotLoggedIn));
    }

    #[test]
    fn ensure_fresh_credentials_passes_through_when_expiry_unknown() {
        // Legacy credentials without expires_at (older CLI, or
        // manual --token) must not constantly hammer refresh —
        // we pass them through and let the actual request decide
        // whether they're still good.
        let cfg = ServiceConfig {
            url: "https://x".into(),
            anon_key: "k".into(),
        };
        let creds = Credentials {
            access_token: "unknown".to_string(),
            refresh_token: Some("rt".to_string()),
            service_url: Some(cfg.url.clone()),
            expires_at: None,
        };
        let out = ensure_fresh_credentials(&cfg, creds).expect("pass through");
        assert_eq!(out.access_token, "unknown");
    }

    #[test]
    fn find_errors_when_not_configured() {
        let cfg = ServiceConfig {
            url: DEFAULT_SERVICE_URL.into(),
            anon_key: String::new(),
        };
        let err = find_package(&cfg, "json").expect_err("should fail unconfigured");
        assert!(matches!(err, ServiceError::NotConfigured));
    }
}
