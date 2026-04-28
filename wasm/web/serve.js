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
// Usage:
//   node serve.js [port]
//
// Default port: 8080. Hit Ctrl-C to stop.

import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { stat, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(import.meta.url), "..", "..");
const PORT = Number(process.argv[2] ?? 8080);

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

const server = createServer(async (req, res) => {
  setSecurityHeaders(res);

  // Strip query string and decode percent-encoded segments.
  const urlPath = decodeURIComponent((req.url ?? "/").split("?")[0]);

  // `..` traversal guard via `normalize` and the prefix check.
  let requested = normalize(join(ROOT, "web", urlPath));
  if (urlPath === "/" || urlPath.endsWith("/")) {
    requested = join(requested, "index.html");
  }
  // The wasm-pack output sits a level up (`wasm/pkg`); the
  // `wlift.js` import path uses `../pkg/...`. Allow access to
  // either `wasm/web/...` or `wasm/pkg/...`.
  if (!requested.startsWith(ROOT + "/web") && !requested.startsWith(ROOT + "/pkg")) {
    requested = join(ROOT, "web", urlPath);
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
