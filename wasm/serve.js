#!/usr/bin/env node
// Minimal static dev server for the wlift playground that sets
// the COOP/COEP headers `SharedArrayBuffer` and threaded wasm
// require:
//
//   Cross-Origin-Opener-Policy: same-origin
//   Cross-Origin-Embedder-Policy: require-corp
//
// Without these, `new WebAssembly.Memory({ shared: true })` fails
// and SAB is undefined on `window`. Plain `python -m http.server`
// won't set them either.
//
// Usage (from `wasm/`):
//
//   node serve.js [port]
//
// Default port: 8080. Hit Ctrl-C to stop. Serves the
// playground at `/` (which redirects to `web/index.html`),
// the wasm-pack output under `/pkg/`, and arbitrary files
// under `/web/`.

import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { stat, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// Script lives at `wasm/serve.js`, so ROOT *is* `wasm/`.
const ROOT = resolve(fileURLToPath(import.meta.url), "..");
const PORT = Number(process.argv[2] ?? 8080);

// Hosts the dev-server proxy will forward to. The page is COEP
// `require-corp`, which means cross-origin assets it loads must
// carry `Cross-Origin-Resource-Policy: cross-origin` — GitHub's
// release-download responses don't, so a direct `fetch()` from
// the playground hits a CORS / COEP wall. Routing through
// `/proxy?url=...` rewrites the response with the right CORP
// header, while the allowlist keeps the proxy from being a
// generic SSRF gadget. Override at run time with
// `HATCH_PROXY_ALLOW="https://my.host/,https://other/"`.
const DEFAULT_PROXY_HOSTS = [
  "https://github.com/",
  "https://objects.githubusercontent.com/",
  "https://gitlab.com/",
  "https://codeberg.org/",
];
const PROXY_ALLOW = (process.env.HATCH_PROXY_ALLOW
  ? process.env.HATCH_PROXY_ALLOW.split(",").map((s) => s.trim()).filter(Boolean)
  : DEFAULT_PROXY_HOSTS);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js":   "application/javascript; charset=utf-8",
  ".mjs":  "application/javascript; charset=utf-8",
  ".css":  "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json; charset=utf-8",
  ".svg":  "image/svg+xml",
  ".png":  "image/png",
  ".ico":  "image/x-icon",
  ".map":  "application/json; charset=utf-8",
};

function setSecurityHeaders(res) {
  // Required for SharedArrayBuffer + threaded wasm. Setting these
  // unconditionally on every response — they're cheap and avoid
  // sub-resource bypass.
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  // No caching during dev so edits show up on hard-refresh.
  res.setHeader("Cache-Control", "no-store");
}

function resolveRequest(urlPath) {
  // `/` and `/index.html` map to `web/index.html`. Everything
  // else routes by prefix: `/pkg/...` → `wasm/pkg/...`,
  // anything else → `wasm/web/...`. The `..` traversal guard
  // is the prefix check after `normalize`.
  if (urlPath === "/" || urlPath === "/index.html") {
    return join(ROOT, "web", "index.html");
  }
  if (urlPath.startsWith("/pkg/")) {
    return normalize(join(ROOT, urlPath.slice(1)));
  }
  if (urlPath.startsWith("/web/")) {
    return normalize(join(ROOT, urlPath.slice(1)));
  }
  // Default: treat as relative to `web/` so `<script src="worker.js">`
  // and `<link href="style.css">` work without a leading `/web/`.
  return normalize(join(ROOT, "web", urlPath));
}

async function handleProxy(req, res) {
  // `/proxy?url=<encoded-upstream>` — fetch the upstream asset
  // server-side and stream it back with the COEP-friendly CORP
  // header. Allowlist enforced so the proxy can't be aimed at
  // arbitrary internal targets.
  const url = new URL(req.url ?? "", "http://localhost").searchParams.get("url");
  if (!url) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end("400 Bad Request: missing ?url=\n");
    return;
  }
  if (!PROXY_ALLOW.some((prefix) => url.startsWith(prefix))) {
    res.statusCode = 403;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(`403 Forbidden: ${url} is not in the proxy allowlist\n`);
    return;
  }
  let upstream;
  try {
    upstream = await fetch(url, { redirect: "follow" });
  } catch (err) {
    res.statusCode = 502;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(`502 Bad Gateway: ${err}\n`);
    return;
  }
  res.statusCode = upstream.status;
  res.setHeader(
    "Content-Type",
    upstream.headers.get("content-type") ?? "application/octet-stream",
  );
  // Page is COEP require-corp; without CORP cross-origin the
  // browser drops the response. Setting CORP here is what makes
  // the proxy useful at all — the page can then consume the
  // bytes.
  res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
  // Permissive CORS so a `fetch('/proxy?url=...')` in any mode
  // works; the allowlist above is the actual security boundary.
  res.setHeader("Access-Control-Allow-Origin", "*");
  const buf = Buffer.from(await upstream.arrayBuffer());
  res.setHeader("Content-Length", buf.length);
  res.end(buf);
}

const server = createServer(async (req, res) => {
  setSecurityHeaders(res);

  const urlPath = decodeURIComponent((req.url ?? "/").split("?")[0]);

  if (urlPath === "/proxy") {
    await handleProxy(req, res);
    return;
  }

  let requested = resolveRequest(urlPath);

  // Sanity: ensure the resolved path is still under ROOT.
  if (!requested.startsWith(ROOT + "/")) {
    res.statusCode = 403;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(`403 Forbidden\n${requested}\n`);
    return;
  }

  try {
    const info = await stat(requested);
    if (info.isDirectory()) {
      requested = join(requested, "index.html");
    }
    const body = await readFile(requested);
    const mime = MIME[extname(requested).toLowerCase()] ?? "application/octet-stream";
    res.setHeader("Content-Type", mime);
    res.setHeader("Content-Length", body.length);
    res.end(body);
  } catch (err) {
    res.statusCode = err.code === "ENOENT" ? 404 : 500;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(`${res.statusCode} ${err.code === "ENOENT" ? "Not Found" : "Internal Error"}\n${requested}\n`);
  }
});

server.listen(PORT, () => {
  console.log(`wlift dev server: http://localhost:${PORT}/`);
  console.log(`COOP/COEP headers set — SharedArrayBuffer / threaded wasm enabled.`);
});
