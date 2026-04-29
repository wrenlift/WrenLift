// Façade over the wlift_wasm wasm-bindgen module.
//
// Two execution modes share one async API:
//
//   import { createWlift } from "./wlift.js";
//   const wlift = await createWlift();              // worker (default)
//   const wlift = await createWlift({ mode: "main" }); // main thread
//
//   wlift.version           // string
//   await wlift.run(source) // { output, ok, errorKind, elapsedMs }
//   wlift.memory            // WebAssembly.Memory | null
//   wlift.terminate()
//
// `mode: "worker"` (default for compute-only workloads) —
// wlift_wasm runs inside `worker.js`. The page never sees the
// wasm linear memory directly; everything crosses the boundary
// as structured-cloned messages. UI thread stays responsive
// during heavy compute.
//
// **Async-bridge caveat:** today's worker mode does NOT have a
// top-level await scheduler. Foreign methods that resolve
// asynchronously — `Browser.fetch`, `Browser.setTimeout`,
// `Browser.connect`, and worker-mode `Dom.*` — settle their
// future *after* the awaiting fiber's `Fiber.yield` returned to
// the script's outer fiber, and nothing wakes that outer fiber
// back up. The first `await` returns `null`. Sync compute, plus
// any foreign method whose JS shim resolves its future inline,
// still works fine. For now: pick `mode: "main"` to demo the
// async bridges (the JS shims are inline-resolving in main mode
// for sync ops; truly-async ones still need the scheduler).
// The scheduler lands in a future phase.
//
// `mode: "main"` (opt-in) — wlift_wasm runs inline on the same
// thread as the page. `wlift.memory` returns the live
// `WebAssembly.Memory` so the page can read Wren-heap bytes
// directly (`new Uint8Array(wlift.memory.buffer, ptr, len)`)
// without the postMessage marshal cost. Tradeoff: long-running
// scripts freeze the UI; this mode is for pages that prioritise
// memory-throughput over responsiveness — e.g. an in-page tool
// streaming many KB/frame to a canvas, or a debugger that
// inspects Wren strings without copying them out.
//
// `mode: "worker", shared: true` (opt-in, gated on environment) —
// when `crossOriginIsolated` is true and SAB is available, asks
// the worker to expose its `WebAssembly.Memory` (now backed by
// `SharedArrayBuffer`) to the page via a one-shot
// `postMessage({cmd: "memory"})`. The page can then read worker-
// side wasm bytes directly with no marshal cost — useful for
// streaming large outputs. Falls back to plain worker mode if
// the page isn't crossOriginIsolated. Page must be served with
// COOP/COEP headers; `serve.js` next to this file sets both.
//
// Threaded wasm-pack build (required for true SAB-backed wasm):
//   RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
//     wasm-pack build wasm --target web --release \
//                          --no-default-features -- -Z build-std=std,panic_abort
// Needs nightly Rust. Without the threaded build, the wasm
// module's memory isn't shared even if SAB is allowed; the
// `shared: true` flag falls through to plain worker mode.

export async function createWlift(opts = {}) {
  const { mode = "worker", shared = false } = opts;
  if (mode === "worker") return new WorkerWlift({ shared }).init();
  if (mode === "main")   return new MainWlift().init();
  throw new Error(`createWlift: unknown mode '${mode}', expected 'worker' or 'main'`);
}

// Returns true iff SAB is available *and* the page is
// crossOriginIsolated (i.e. served with COOP/COEP). The wasm
// module still needs to be built with threading for the wasm
// memory itself to be SAB-backed; `shared: true` opts the page
// in to *requesting* it from the worker.
export function sharedMemorySupported() {
  return typeof SharedArrayBuffer !== "undefined"
      && typeof crossOriginIsolated !== "undefined"
      && crossOriginIsolated === true;
}

// Scan a Wren source string for `import "..."` lines whose
// module name is a *scoped* package — i.e. one containing any
// of `:`, `@`, or `/`. This matches the host-side build's
// `is_scoped` rule (cf. `hatch::rename_entry_to_manifest_name`),
// which is what tells the build to rename the entry module to
// the package name so consumer imports resolve.
//
// Returns unique package names in source order. The playground
// uses this to drive the prefetch step before
// `runWithHatches`. The regex tolerates either quote style,
// whitespace between `import` and the quoted name, and any
// non-quote bytes inside the package name (so `@user:lib`,
// `@scope/pkg`, `@hatch:assert`, registry URLs, etc. all
// match — first-party `@hatch:*` packages aren't privileged).
//
// Bare names like `import "util"` (a relative module inside
// the same project) are intentionally *not* matched: those
// resolve through the in-project loader, not through a
// dependency hatch.
export function scanScopedImports(source) {
  const re = /import\s+["']([^"']*[:@/][^"']*)["']/g;
  const names = [];
  const seen = new Set();
  for (const m of source.matchAll(re)) {
    const name = m[1];
    if (!seen.has(name)) {
      seen.add(name);
      names.push(name);
    }
  }
  return names;
}

// Backwards-compat alias kept around in case callers depend on
// the original name. New code should use `scanScopedImports`.
export const scanHatchImports = scanScopedImports;

// Default registry base — first-party `@hatch:*` packages are
// published here as GitHub release artifacts. The publish-pkg.yml
// workflow attaches two assets per `publish/<dir>@<ver>` tag:
//
//   <base>/releases/download/publish/<dir>%40<ver>/<dir>-<ver>.hatch
//   <base>/releases/download/publish/<dir>%40<ver>/<dir>-<ver>-wasm32.hatch
//
// where `<dir>` is the package's source directory name (e.g.
// `hatch-assert`). The `@hatch:foo` import name maps to `hatch-foo`
// via `scopedNameToDir` below — strip the `@`, swap `:` / `/` for
// `-`. Override at the call site if you're hosting a private mirror.
export const DEFAULT_HATCH_REGISTRY = "https://github.com/wrenlift/hatch";

// Map a scoped package name (`@hatch:assert`, `@user:lib`,
// `@scope/pkg`) to its release-asset directory basename
// (`hatch-assert`, `user-lib`, `scope-pkg`). Bare names are
// returned verbatim. Mirrors the workflow's own naming: it
// publishes under `packages/<dir>/`, so `<dir>` is the
// canonical artifact stem.
export function scopedNameToDir(name) {
  return name.replace(/^@/, "").replace(/[:/]/g, "-");
}

// Lazy import so `parseHatchfileToml` / `peekManifest` callers
// don't pay the wasm-init cost twice (worker mode already has
// its own copy; this is for main-thread sync helpers).
let _modPromise = null;
async function getMod() {
  if (!_modPromise) {
    _modPromise = (async () => {
      const m = await import("../pkg/wlift_wasm.js");
      await m.default();
      return m;
    })();
  }
  return _modPromise;
}

/// Parse hatchfile TOML → JS object. Same shape as
/// `crate::hatch::Manifest`. Throws on malformed TOML or
/// missing required fields.
export async function parseHatchfileToml(text) {
  const m = await getMod();
  return m.parse_hatchfile_toml(text);
}

/// Peek the manifest from already-fetched `.hatch` bytes
/// without installing the bundle. Used by the dep walker to
/// discover transitive `[dependencies]`.
export async function peekManifest(bytes) {
  const m = await getMod();
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return m.peek_manifest(buf);
}

// Map a parsed git remote URL to the host's release-asset URL
// pattern. Each forge has its own; this is intentionally a
// short list — anything self-hosted should use an explicit
// `url = "..."` override on the dep entry instead of relying
// on us guessing.
//
// Returns `null` on unrecognised hosts; the caller turns that
// into an actionable error pointing at the override.
function gitForgeReleaseUrl(gitUrl, rev, name) {
  // Strip `.git` suffix and any trailing slash so `${base}/...`
  // composes cleanly regardless of how the publisher wrote it.
  const base = gitUrl.replace(/\.git\/?$/, "").replace(/\/$/, "");
  // GitHub: `<repo>/releases/download/<tag>/<asset>`.
  if (/(^|\/\/)([^/]+\.)?github\.com\//.test(base)) {
    return `${base}/releases/download/${rev}/${name}.hatch`;
  }
  // GitLab (and self-managed CE/EE installs sharing the same
  // route): `<repo>/-/releases/<tag>/downloads/<asset>`.
  if (/(^|\/\/)([^/]+\.)?gitlab(\.[^/]+)?\//.test(base)) {
    return `${base}/-/releases/${rev}/downloads/${name}.hatch`;
  }
  // Codeberg / Gitea: `<repo>/releases/download/<tag>/<asset>`
  // (same shape as GitHub since Gitea modeled its API after
  // it). Match codeberg.org by name; other Gitea installs
  // should set `url` explicitly.
  if (/(^|\/\/)codeberg\.org\//.test(base)) {
    return `${base}/releases/download/${rev}/${name}.hatch`;
  }
  return null;
}

// Resolve a single `[dependencies]` entry into a fetch URL.
// Mirrors the host's `Dependency` enum:
//
//   "@hatch:assert" = "0.2.0"             // version (registry CDN)
//   "@user:lib"     = { git = "...", rev = "abc" }
//   "@user:lib"     = { git = "...", tag = "v1" }
//   "@user:lib"     = { git = "...", rev = "abc", url = "https://..." }   // explicit override
//   "@user:lib"     = { url = "https://..." }                             // direct, no git
//   "local"         = { path = "../local" }
//
// `target` selects which release-asset variant to fetch for
// version-pinned deps:
//
//   "wasm32" (default in this file — we're the wasm runtime) →
//     `<dir>-<ver>-wasm32.hatch` (uncompressed, no zstd, target
//     stamp matches the wasm runtime's family check).
//   "host" → `<dir>-<ver>.hatch` (zstd-compressed, host-loadable).
//
// Browser limitations:
//
//   * Path deps fail with a clear error — they're build-time
//     references on the host CLI; nothing to fetch.
//   * Git deps without an explicit `url` need a known forge
//     (github / gitlab / codeberg). Self-hosted forges
//     (Forgejo, Gitea installs, custom CDN routes) should set
//     `url = "..."` on the dep entry, which wins over the
//     forge-pattern guess.
//
// Returns `{ url }` on success; throws on unsupported entries.
export function depEntryToUrl(
  name,
  dep,
  registryBase = DEFAULT_HATCH_REGISTRY,
  target = "wasm32",
) {
  if (typeof dep === "string") {
    return { url: registryReleaseUrl(registryBase, name, dep, target) };
  }
  if (dep && typeof dep === "object") {
    if (typeof dep.path === "string") {
      throw new Error(
        `dep '${name}' is a path reference (${dep.path}); path deps only resolve at build time. ` +
        `Build the parent project on host with \`wlift project/ --bundle out.hatch --bundle-target wasm32-*\` and load the bundle directly.`,
      );
    }
    // Direct-URL entry (`{ url = "..." }` with no git) — wasm-
    // runtime-only, fetched verbatim. Host CLI rejects this
    // shape since it has no rev / version to lock against.
    if (typeof dep.url === "string" && typeof dep.git !== "string") {
      return { url: dep.url };
    }
    if (typeof dep.git === "string") {
      const rev = dep.rev ?? dep.tag;
      if (!rev) {
        throw new Error(`dep '${name}' has 'git' but no 'rev' or 'tag' — can't pin a specific release.`);
      }
      // Explicit `url` overrides any forge-pattern guess so
      // self-hosted installs (Forgejo, custom CDN, etc.) work
      // without us hardcoding their route shape.
      if (typeof dep.url === "string") {
        return { url: dep.url };
      }
      const forgeUrl = gitForgeReleaseUrl(dep.git, rev, name);
      if (forgeUrl) return { url: forgeUrl };
      throw new Error(
        `dep '${name}' is hosted at '${dep.git}' — that forge isn't in the known release-URL list ` +
        `(github / gitlab / codeberg). Add an explicit \`url = "..."\` to the dep entry pointing at ` +
        `the .hatch download URL on this host.`,
      );
    }
    // Bare table — try a `version` key as a fallback so
    // `{ version = "1.0.0" }` works the same as the bare
    // string form.
    if (typeof dep.version === "string") {
      return { url: registryReleaseUrl(registryBase, name, dep.version, target) };
    }
  }
  throw new Error(`dep '${name}' has an unsupported entry shape: ${JSON.stringify(dep)}`);
}

// Build the URL for a version-pinned dep on the first-party
// registry. Matches the asset layout `publish-pkg.yml` produces:
//
//   <base>/releases/download/publish/<dir>%40<ver>/<dir>-<ver>[-wasm32].hatch
//
// The tag itself (`publish/<dir>@<ver>`) is part of the URL path,
// which is why `@` is encoded as `%40` — GitHub keeps the tag
// verbatim in the asset URL it serves.
function registryReleaseUrl(registryBase, name, version, target) {
  const base = registryBase.replace(/\/$/, "");
  const dir = scopedNameToDir(name);
  const variant = target === "host" ? "" : "-wasm32";
  return `${base}/releases/download/publish/${dir}%40${version}/${dir}-${version}${variant}.hatch`;
}

// Recursively fetch every dep declared by `manifest`'s
// `[dependencies]` table, peek inside each fetched bundle for
// transitive `[dependencies]`, and return the byte streams in
// install order (deps before consumers — depth-first post-
// order over the dep graph). `peekManifest` runs synchronously
// off the already-loaded wasm module so transitive resolution
// doesn't pay an extra worker round-trip.
//
// Pass `registryBase` to override the first-party CDN.
// `fetcher` defaults to browser `fetch()`; tests / private
// mirrors / IndexedDB caches can substitute.
export async function resolveDepsFromManifest(
  manifest,
  {
    registryBase = DEFAULT_HATCH_REGISTRY,
    fetcher = defaultFetcher,
    target = "wasm32",
  } = {},
) {
  const order = [];                  // Uint8Array[] in install order
  const seen = new Set();            // package names already queued
  await visit(manifest);
  return order;

  async function visit(m) {
    const deps = (m && m.dependencies) || {};
    for (const [name, entry] of Object.entries(deps)) {
      if (seen.has(name)) continue;
      seen.add(name);
      const { url } = depEntryToUrl(name, entry, registryBase, target);
      const bytes = await fetcher(url, name);
      // Peek before installing so transitive deps queue ahead
      // of this one (post-order: deps install before consumers).
      const childManifest = await peekManifest(bytes);
      await visit(childManifest);
      order.push(bytes);
    }
  }
}

async function defaultFetcher(url, name) {
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`fetch dep '${name}' from ${url} failed: ${r.status} ${r.statusText}`);
  }
  return new Uint8Array(await r.arrayBuffer());
}

/// One-call convenience: given the user's source + their
/// hatchfile (TOML text), parse the manifest, walk the dep
/// graph, fetch each declared dependency from its hatchfile-
/// metadata-derived URL, then run the user source against the
/// installed deps. Returns the same `RunResult` shape as
/// `wlift.run` / `wlift.runWithHatches`.
///
///   const result = await runProject(wlift, source, hatchfileText);
///   // result.output / .ok / .errorKind / .elapsedMs
///
/// Errors from any stage (parse, fetch, peek, run) surface as
/// `result.ok === false` with the error message in
/// `result.output`.
export async function runProject(wlift, source, hatchfileText, opts = {}) {
  const t0 = performance.now();
  let manifest;
  try {
    manifest = await parseHatchfileToml(hatchfileText);
  } catch (err) {
    return { output: `hatchfile parse failed: ${err}`, ok: false, errorKind: -1, elapsedMs: performance.now() - t0 };
  }
  let deps;
  try {
    deps = await resolveDepsFromManifest(manifest, opts);
  } catch (err) {
    return { output: `dep resolution failed: ${err}`, ok: false, errorKind: -1, elapsedMs: performance.now() - t0 };
  }
  return wlift.runWithHatches(source, deps);
}

// ---------------------------------------------------------------------------
// Worker mode (default)
// ---------------------------------------------------------------------------

class WorkerWlift {
  constructor({ shared = false } = {}) {
    this.mode    = "worker";
    this.version = null;
    // SAB requires both crossOriginIsolated AND a SAB-aware
    // wasm build. We probe the page side (cheap); the worker
    // side resolves on `cmd: "memory"` reply — if the wasm
    // module isn't SAB-built, the memory comes back as a
    // regular ArrayBuffer-backed `WebAssembly.Memory` which
    // can't cross worker → page, so the request is a no-op.
    this._sharedRequested = shared && sharedMemorySupported();
    this._memory = null;
    this.worker  = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId  = 1;
    this._ready  = new Promise((resolve) => { this._resolveReady = resolve; });

    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.cmd === "ready") {
        this.version = m.version;
        if (this._sharedRequested) {
          this.worker.postMessage({ cmd: "memory" });
        } else {
          this._resolveReady();
        }
      } else if (m.cmd === "memory") {
        // Worker handed back its `WebAssembly.Memory` (only
        // structurally cloneable when its buffer is a SAB).
        // If the worker's wasm isn't SAB-built we get null
        // here; surface as `wlift.memory === null` (same as
        // plain worker mode).
        this._memory = m.memory ?? null;
        this._resolveReady();
      } else if (m.cmd === "result") {
        const resolve = this.pending.get(m.id);
        if (resolve) {
          this.pending.delete(m.id);
          resolve(m);
        }
      } else if (m.cmd === "dom-op") {
        // Worker can't reach `document`; we run the op on the
        // page's behalf and post the answer back. Each `op` is
        // matched by a `_wlift_dom_*` shim in `worker.js`.
        try {
          const value = dispatchDomOp(m.op, m.args);
          this.worker.postMessage({ cmd: "dom-result", handle: m.handle, value });
        } catch (err) {
          this.worker.postMessage({ cmd: "dom-error", handle: m.handle, error: String(err) });
        }
      }
    });
  }

  async init() {
    await this._ready;
    return this;
  }

  run(source) {
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve);
      this.worker.postMessage({ cmd: "run", id, source });
    });
  }

  // Run a `.hatch` byte stream — multi-module bundle produced by
  // `wlift src/ --bundle out.hatch --bundle-target wasm32-unknown-unknown`.
  // The bundle's `target` manifest field is checked against the
  // wasm runtime's family marker; cross-family bundles fail
  // load with a `WrongTarget` message in the result.
  runHatch(bytes) {
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve);
      // Transfer the underlying buffer so we don't pay a copy on
      // postMessage. `bytes` should be a Uint8Array.
      const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      this.worker.postMessage(
        { cmd: "run-hatch", id, bytes: buf },
        [buf.buffer],
      );
    });
  }

  // Run user source with a list of pre-fetched `@hatch:*`
  // dependency bundles installed first. `deps` is an array of
  // `Uint8Array` (one `.hatch` per dep). Use `scanHatchImports` +
  // `fetchHatchPackages` together to populate it:
  //
  //   const names = scanHatchImports(source);
  //   const deps  = await fetchHatchPackages(names, fetcher);
  //   const r     = await wlift.runWithHatches(source, deps);
  runWithHatches(source, deps) {
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve);
      // `slice()` clones each dep onto a fresh ArrayBuffer
      // *before* the transfer list detaches it. Without the
      // clone, callers that hold their dep array across
      // multiple runs (the playground's dep cache, mode
      // switches that re-feed the same bundles to a new
      // worker) hit `DataCloneError: ArrayBuffer is already
      // detached` on the second send. The clone is one memcpy
      // per dep — negligible vs the wasm install — and keeps
      // the per-call transfer optimisation intact.
      const bufs = (deps ?? []).map((b) => {
        const u8 = b instanceof Uint8Array ? b : new Uint8Array(b);
        return u8.slice();
      });
      this.worker.postMessage(
        { cmd: "run-with-hatches", id, source, deps: bufs },
        bufs.map((b) => b.buffer),
      );
    });
  }

  // SAB-mode: the page can read worker-side wasm bytes directly
  // through `memory.buffer`. Plain worker mode: returns null
  // (the boundary is the point). Feature-probe with `if (wlift.memory)`.
  get memory() { return this._memory; }

  terminate() { this.worker.terminate(); }
}

// ---------------------------------------------------------------------------
// Main-thread mode (opt-in)
// ---------------------------------------------------------------------------

class MainWlift {
  constructor() {
    this.mode    = "main";
    this.version = null;
    this._mod    = null;
    this._memory = null;
  }

  async init() {
    const mod = await import("../pkg/wlift_wasm.js");

    // Same async-bridge shims `worker.js` installs, but on the
    // page's `globalThis` instead. The bridges are workload-side
    // (setTimeout / fetch / WebSocket exist on both window and
    // WorkerGlobalScope), so the install is a verbatim copy.
    globalThis._wlift_set_timeout = (handle, ms) => {
      setTimeout(() => mod.resolve_future(handle, ""), ms);
    };
    globalThis._wlift_fetch = (handle, url) => {
      fetch(url)
        .then((r) => r.text())
        .then((t) => mod.resolve_future(handle, t))
        .catch((e) => mod.reject_future(handle, String(e)));
    };
    globalThis.__wlift_sockets ??= new Map();
    globalThis._wlift_ws_open = (handle, url) => {
      const ws = new WebSocket(url);
      ws.addEventListener("open",    () => mod.ws_open(handle));
      ws.addEventListener("message", (e) => mod.ws_message(handle, String(e.data)));
      ws.addEventListener("close",   () => mod.ws_close(handle));
      ws.addEventListener("error",   () => mod.ws_close(handle));
      globalThis.__wlift_sockets.set(handle, ws);
    };
    globalThis._wlift_ws_send = (handle, text) => {
      globalThis.__wlift_sockets.get(handle)?.send(text);
    };
    globalThis._wlift_ws_close = (handle) => {
      const ws = globalThis.__wlift_sockets.get(handle);
      if (ws) {
        ws.close();
        globalThis.__wlift_sockets.delete(handle);
      }
    };

    // DOM shims — main mode runs `document.*` directly (we're on
    // the page thread; no postMessage hop). The worker mode's
    // round-trip uses the same `dispatchDomOp` table below, so
    // both paths share the implementation of each op.
    // Each main-mode shim is a thin wrapper around the same
    // `dispatchDomOp` table the worker mode uses on the page
    // side. One implementation, two callers.
    const settleSync = (handle, op, args) => {
      try {
        mod.resolve_future(handle, dispatchDomOp(op, args));
      } catch (err) {
        mod.reject_future(handle, String(err));
      }
    };
    globalThis._wlift_dom_text          = (h, s)        => settleSync(h, "text", [s]);
    globalThis._wlift_dom_set_text      = (h, s, v)     => settleSync(h, "setText", [s, v]);
    globalThis._wlift_dom_get_attribute = (h, s, n)     => settleSync(h, "getAttribute", [s, n]);
    globalThis._wlift_dom_set_attribute = (h, s, n, v)  => settleSync(h, "setAttribute", [s, n, v]);
    globalThis._wlift_dom_add_class     = (h, s, n)     => settleSync(h, "addClass", [s, n]);
    globalThis._wlift_dom_remove_class  = (h, s, n)     => settleSync(h, "removeClass", [s, n]);
    globalThis._wlift_dom_query_all     = (h, s)        => settleSync(h, "queryAll", [s]);

    // Storage shims — same install in both modes since
    // localStorage/sessionStorage are sync and worker-safe.
    const pickStorage = (scope) => (scope === "session" ? sessionStorage : localStorage);
    globalThis._wlift_storage_get = (handle, scope, key) => {
      try {
        const v = pickStorage(scope).getItem(key);
        mod.resolve_future(handle, v ?? "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };
    globalThis._wlift_storage_set = (handle, scope, key, value) => {
      try {
        pickStorage(scope).setItem(key, value);
        mod.resolve_future(handle, "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };
    globalThis._wlift_storage_remove = (handle, scope, key) => {
      try {
        pickStorage(scope).removeItem(key);
        mod.resolve_future(handle, "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };
    globalThis._wlift_storage_clear = (handle, scope) => {
      try {
        pickStorage(scope).clear();
        mod.resolve_future(handle, "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };

    // Scheduler yield helper — see `worker.js` for rationale.
    globalThis._wlift_yield_to_event_loop = () =>
      new Promise((resolve) => setTimeout(resolve, 0));

    // Perf logger — gated by `globalThis.__wlift_perf_enabled`.
    // Same install in worker.js so worker-mode runs are
    // diagnosable too.
    globalThis._wlift_perf_log = (label, ms) => {
      if (globalThis.__wlift_perf_enabled) {
        console.log(`[wlift perf] ${label}: ${ms.toFixed(2)} ms`);
      }
    };

    // wasm-bindgen `--target web` returns the wasm exports object
    // from the default `init()`. `exports.memory` is the live
    // `WebAssembly.Memory` instance — that's what main-mode
    // callers want for direct buffer access.
    const wasm = await mod.default();
    this._mod    = mod;
    this._memory = wasm.memory;
    this.version = mod.version();

    // Expose the module namespace to `globalThis` so the dev
    // console can poke at JIT counters / debug exports in main
    // mode without an extra `import` step. Worker mode stashes
    // its own copy on the worker's globalThis (see worker.js)
    // — the page-side `wlift_wasm` only reflects main-mode
    // state. Two separate instances; that's working as
    // designed.
    globalThis.wlift_wasm = mod;

    // ----------------------------------------------------------
    // On-demand wasm plugin loader (main-mode)
    // ----------------------------------------------------------
    // A `.hatch` bundle whose [native_libs] declares a `wasm = …`
    // entry carries the plugin's wasm bytes alongside its Wren
    // source. At install time we extract each, instantiate it as a
    // *separate* WebAssembly.Instance (so wgpu / winit-style plugins
    // can keep their own wasm-bindgen JS heap if they ever need
    // one), wire up the host-side slot bridges as imports, and
    // register every `wlift_*` export in the host's foreign-method
    // registry via `wlift_register_plugin_dynamic_export`. The
    // index that returns becomes the key in the dispatch table,
    // and the same index ends up in `Method::ForeignCDynamic(idx)`
    // for the Wren side. Run-time foreign-method calls flow:
    //
    //   Wren -> dispatch_dynamic(idx) -> JS bridge import
    //         -> globalThis.wliftDynamicPluginDispatch(idx, vm)
    //         -> instance.exports[name](vm)
    //
    // Plugin imports look the other direction:
    //
    //   plugin.exports.* -> wlift_get_slot_str / wlift_set_slot_str
    //                    -> host's wrenGetSlotString / wrenSetSlotString
    //                    (with explicit memcpy through wlift_host_alloc).
    const pluginInstances = new Map(); // libName -> WebAssembly.Instance
    const dispatchTable   = [];        // idx -> {instance, exportName}

    globalThis.wliftDynamicPluginDispatch = (idx, vm) => {
      const entry = dispatchTable[idx];
      if (!entry) {
        console.warn(`[wlift plugin] dispatch ${idx}: no entry`);
        return;
      }
      try {
        entry.instance.exports[entry.exportName](vm);
      } catch (err) {
        console.error(`[wlift plugin] ${entry.libName}.${entry.exportName} threw:`, err);
      }
    };

    function makePluginImports(libName, getInstance) {
      // `getInstance` is a getter because the plugin's
      // `instance.exports.memory` isn't available until *after*
      // `WebAssembly.instantiate` resolves — the imports object
      // we hand it has to be built first, and the bridges that
      // reference plugin memory close over a deferred lookup.
      //
      // The host side here uses `wasm` (the exports object
      // returned by `mod.default()`) for raw `wrenGetSlotString`
      // / `wrenSetSlotString` / `wlift_host_alloc` / memory
      // access. `mod`'s wasm-bindgen-wrapped exports go through
      // string-marshalling wrappers that aren't what the bridge
      // contract wants — the bridge already has raw C-string
      // pointers in hand.
      return {
        env: {
          // Plugin asks for `wrenGetSlotString(vm, slot)` →
          // returns a host *const char (pointer into HOST memory).
          // We read that C-string out of host memory and copy the
          // bytes into the plugin's memory at `out_ptr`, capped
          // at `max`. Returns the number of bytes written, or -1
          // on a NULL host pointer.
          wlift_get_slot_str: (vm, slot, out_ptr, max) => {
            const host_ptr = wasm.wrenGetSlotString(vm, slot);
            if (!host_ptr) return -1;
            const host_view = new Uint8Array(wasm.memory.buffer);
            let len = 0;
            while (len < max && host_view[host_ptr + len] !== 0) len++;
            const inst = getInstance();
            const plugin_view = new Uint8Array(inst.exports.memory.buffer, out_ptr, max);
            plugin_view.set(host_view.subarray(host_ptr, host_ptr + len));
            return len;
          },

          // Plugin built a string in its own memory (`ptr`/`len`)
          // and wants it written to slot `slot`. We allocate a
          // null-terminated buffer in *host* memory, copy the
          // bytes across, call `wrenSetSlotString` (which copies
          // again into a Wren-managed string), then free.
          wlift_set_slot_str: (vm, slot, ptr, len) => {
            const inst = getInstance();
            const plugin_view = new Uint8Array(inst.exports.memory.buffer, ptr, len);
            const host_ptr = wasm.wlift_host_alloc(len + 1);
            const host_view = new Uint8Array(wasm.memory.buffer, host_ptr, len + 1);
            host_view.set(plugin_view);
            host_view[len] = 0;
            wasm.wrenSetSlotString(vm, slot, host_ptr);
            wasm.wlift_host_free(host_ptr, len + 1);
          },
        },
      };
    }

    async function installWasmPlugin(libName, wasmBytes) {
      if (pluginInstances.has(libName)) return; // idempotent on repeat installs
      let instance;
      const imports = makePluginImports(libName, () => instance);
      const result = await WebAssembly.instantiate(wasmBytes, imports);
      instance = result.instance;
      pluginInstances.set(libName, instance);

      // Walk the plugin's exports and register each `wlift_*`
      // function in the host's foreign-method registry. The host
      // returns a u32 index per registration; we store
      // `(instance, exportName)` keyed by that index for the
      // dispatcher.
      for (const exportName of Object.keys(instance.exports)) {
        if (!exportName.startsWith("wlift_")) continue;
        if (typeof instance.exports[exportName] !== "function") continue;
        const idx = mod.wlift_register_plugin_dynamic_export(libName, exportName);
        dispatchTable[idx] = { instance, exportName, libName };
      }
    }

    // Expose for the runWithHatches override below.
    this._installWasmPlugin = installWasmPlugin;
    this._wliftExtractWasmPlugins = mod.wlift_extract_wasm_plugins;

    // Tier-up Phase 2b — JIT instantiate + call shims. The Rust
    // side calls `wlift_wasm::tier_up::emit_*` to produce wasm
    // bytes; these shims compile + instantiate them with the
    // `wren_*` runtime helpers as imports, store the instance
    // in a per-page table, and expose `call_0` / `call_1` /
    // `call_2` arities for invocation. Must run AFTER
    // `await mod.default()` so the `wasm` exports object is
    // populated for the import bindings. Phase 4 generalises
    // the call shape; for now arity caps at 2 (fib's signature).
    const __wliftJitInstances = [];
    // Phase 5 — shared funcref table. Each JIT'd module imports
    // `wren.__wlift_jit_table` and emits `call_indirect` through
    // it for inter-fn calls (skipping the JS hop). We grow the
    // table by one entry per `_wlift_jit_instantiate` and
    // `set(slot, fn)` so all modules see every JIT'd function at
    // its host-side slot. No upper bound declared on the table
    // type, so growth is unbounded by the wasm spec — the runtime
    // caps via `wasm_jit_slot()` lookup, not the table.
    const __wliftJitTable = new WebAssembly.Table({ initial: 0, element: "anyfunc" });
    const __wliftWrenImports = {
      wren_num_add:  wasm.wren_num_add,
      wren_num_sub:  wasm.wren_num_sub,
      wren_num_mul:  wasm.wren_num_mul,
      wren_num_div:  wasm.wren_num_div,
      wren_num_mod:  wasm.wren_num_mod,
      wren_num_neg:  wasm.wren_num_neg,
      wren_cmp_lt:   wasm.wren_cmp_lt,
      wren_cmp_gt:   wasm.wren_cmp_gt,
      wren_cmp_le:   wasm.wren_cmp_le,
      wren_cmp_ge:   wasm.wren_cmp_ge,
      wren_cmp_eq:   wasm.wren_cmp_eq,
      wren_cmp_ne:   wasm.wren_cmp_ne,
      wren_not:      wasm.wren_not,
      wren_is_truthy:wasm.wren_is_truthy,
      wren_call_1_slow: wasm.wren_call_1,
      wren_jit_slot_plus_one: wasm.wren_jit_slot_plus_one,
      wren_jit_slot_for_module_var: wasm.wren_jit_slot_for_module_var,
      wren_get_module_var: wasm.wren_get_module_var,
      __wlift_jit_table: __wliftJitTable,
    };
    globalThis._wlift_jit_instantiate = (bytes) => {
      const module = new WebAssembly.Module(bytes);
      const instance = new WebAssembly.Instance(module, {
        wren: __wliftWrenImports,
      });
      // `emit_mir` exports exactly one `fn_<symbol_index>` per
      // module — find it once and stash the function reference
      // directly so subsequent calls don't pay name-lookup cost.
      let fn = null;
      for (const key of Object.keys(instance.exports)) {
        if (key.startsWith("fn_")) { fn = instance.exports[key]; break; }
      }
      const slot = __wliftJitInstances.length;
      __wliftJitInstances.push(fn);
      // Phase 5 — publish the funcref so other JIT'd modules can
      // `call_indirect` to us. Order matters: the slot returned
      // here must match `wasm_jit_slot()` on the Rust side, since
      // `wren_jit_slot_plus_one` resolves callees against that
      // index.
      __wliftJitTable.grow(1);
      __wliftJitTable.set(slot, fn);
      return slot;
    };
    // Defensive shim — if the wasm function returns undefined
    // (mismatched signature, unbound import filled with stub,
    // wrong slot), surface the failure as `null`-bits rather
    // than letting wasm-bindgen throw "Cannot convert undefined
    // to BigInt" deep in the dispatch path. The console.warn
    // makes the misuse visible without crashing the whole run.
    const __wliftSafeCall = (slot, fn, label) => {
      const target = __wliftJitInstances[slot];
      if (typeof target !== "function") {
        console.warn(`[wlift jit] ${label}: slot ${slot} is not a function (${typeof target})`);
        return 0x7FFC000000000000n; // Value::null() — TAG_NULL = QNAN
      }
      const r = fn(target);
      if (typeof r !== "bigint") {
        console.warn(`[wlift jit] ${label}: slot ${slot} returned non-BigInt (${typeof r}, ${r}); falling back to null`);
        return 0x7FFC000000000000n;
      }
      return r;
    };
    globalThis._wlift_jit_call_0 = (slot)       => __wliftSafeCall(slot, (fn) => fn(),   "call_0");
    globalThis._wlift_jit_call_1 = (slot, a)    => __wliftSafeCall(slot, (fn) => fn(a),  "call_1");
    globalThis._wlift_jit_call_2 = (slot, a, b) => __wliftSafeCall(slot, (fn) => fn(a, b), "call_2");

    // Per-run reset — wlift_wasm calls this at the top of each
    // `run()` so JIT'd modules from prior VMs become
    // garbage-collectable. We can't shrink the funcref Table
    // (no API), so we null-out every entry and reset the
    // companion array. The JIT'd wasm modules themselves drop
    // out of scope once the table releases its references.
    globalThis._wlift_jit_reset = () => {
      for (let i = 0; i < __wliftJitInstances.length; i++) {
        try { __wliftJitTable.set(i, null); } catch (_) { /* table ref types vary by browser */ }
      }
      __wliftJitInstances.length = 0;
    };

    return this;
  }

  async runHatch(bytes) {
    // Same shape as `run`, but the user code is a `.hatch`
    // bundle. `_mod.run_hatch` is the wasm-bindgen export added
    // alongside `run` — both go through the same VM-setup +
    // scheduler-loop machinery in `wlift_wasm::lib::run_inner`.
    const t0 = performance.now();
    let result;
    try {
      const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      result = await this._mod.run_hatch(buf);
    } catch (err) {
      return {
        cmd: "result",
        output: String(err),
        ok: false,
        errorKind: -1,
        elapsedMs: performance.now() - t0,
      };
    }
    return {
      cmd: "result",
      output: result.output,
      ok: result.ok,
      errorKind: result.errorKind,
      elapsedMs: performance.now() - t0,
    };
  }

  async runWithHatches(source, deps) {
    // `_mod.run_with_hatches` takes a `js_sys::Array` whose
    // elements are `Uint8Array`s. wasm-bindgen marshals the
    // outer array; the inner arrays cross by reference + the
    // Rust side `to_vec()`s them once. Errors during dep install
    // are surfaced through `result.output` with `ok: false`.
    const t0 = performance.now();
    let result;
    try {
      const bufs = (deps ?? []).map((b) =>
        b instanceof Uint8Array ? b : new Uint8Array(b),
      );
      // Phase 1d: install any wasm-targeted NativeLib plugins
      // each dep declares *before* handing the dep bytes to
      // `run_with_hatches`. Foreign-class declarations in the
      // bundle's Wlbc resolve their `#!native` / `#!symbol`
      // bindings at install time, so the plugin's symbols need
      // to already be in the registry. `wlift_extract_wasm_plugins`
      // returns `[{lib, bytes}]` per dep (empty for pure-Wren
      // deps).
      if (this._installWasmPlugin) {
        for (const buf of bufs) {
          const plugins = this._wliftExtractWasmPlugins(buf);
          for (let i = 0; i < plugins.length; i++) {
            const p = plugins[i];
            await this._installWasmPlugin(p.lib, p.bytes);
          }
        }
      }
      result = await this._mod.run_with_hatches(source, bufs);
    } catch (err) {
      return {
        cmd: "result",
        output: String(err),
        ok: false,
        errorKind: -1,
        elapsedMs: performance.now() - t0,
      };
    }
    return {
      cmd: "result",
      output: result.output,
      ok: result.ok,
      errorKind: result.errorKind,
      elapsedMs: performance.now() - t0,
    };
  }

  async run(source) {
    // `_mod.run` is `pub async fn` on the Rust side, so it
    // returns a Promise. `await` it so the scheduler loop
    // (which drains parked fibers across `await` points) gets
    // to drive async bridges to completion before we hand back
    // a result.
    const t0 = performance.now();
    let result;
    try {
      result = await this._mod.run(source);
    } catch (err) {
      return {
        cmd: "result",
        output: String(err),
        ok: false,
        errorKind: -1,
        elapsedMs: performance.now() - t0,
      };
    }
    // Same shape as the worker reply so the UI doesn't have to
    // care which mode it's talking to.
    return {
      cmd: "result",
      output: result.output,
      ok: result.ok,
      errorKind: result.errorKind,
      elapsedMs: performance.now() - t0,
    };
  }

  get memory() { return this._memory; }

  terminate() {
    // No worker to kill; nothing to do. Provided for API parity.
  }
}

// ---------------------------------------------------------------------------
// DOM op dispatch table — shared by worker mode (main runs the
// op on the worker's behalf) and main mode (we run inline).
// Each entry returns a String (or throws). Adding a DOM op needs
// three pieces:
//
//   1. A `dom_*` foreign method in `wasm/src/dom.rs`.
//   2. A `_wlift_dom_*` shim in `worker.js` (posts a `dom-op`)
//      and another in `MainWlift.init()` (calls dispatch directly).
//   3. A `case` here.
//
// Returning a string keeps the protocol JSON-serialisable
// without bringing in a richer wire format yet — the existing
// future store flows everything as `String`s anyway.
// ---------------------------------------------------------------------------

function dispatchDomOp(op, args) {
  switch (op) {
    case "text": {
      const [selector] = args;
      const el = document.querySelector(selector);
      return el ? el.textContent ?? "" : "";
    }
    case "setText": {
      const [selector, value] = args;
      const el = document.querySelector(selector);
      if (el) el.textContent = value;
      return "";
    }
    case "getAttribute": {
      const [selector, name] = args;
      const el = document.querySelector(selector);
      return el ? (el.getAttribute(name) ?? "") : "";
    }
    case "setAttribute": {
      const [selector, name, value] = args;
      const el = document.querySelector(selector);
      if (el) el.setAttribute(name, value);
      return "";
    }
    case "addClass": {
      const [selector, name] = args;
      const el = document.querySelector(selector);
      if (el) el.classList.add(name);
      return "";
    }
    case "removeClass": {
      const [selector, name] = args;
      const el = document.querySelector(selector);
      if (el) el.classList.remove(name);
      return "";
    }
    case "queryAll": {
      const [selector] = args;
      // Encode count as a decimal string so it rides the
      // `Future<String>` transport. Future variant: return a
      // JSON-encoded list of element ids/handles.
      return String(document.querySelectorAll(selector).length);
    }
    default:
      throw new Error(`unknown DOM op '${op}'`);
  }
}
