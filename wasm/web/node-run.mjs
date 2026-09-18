// Run a Wren program through the browser bundle under Node, on the
// main-thread path, so the wasm tier-up is exercised without a
// browser: `node wasm/web/node-run.mjs prog.wren`. Prints the
// program's output; exits non-zero when the run fails. The tier
// counters go to stderr.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const realFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
  const u = String(url);
  if (u.startsWith("file:")) {
    const bytes = await readFile(fileURLToPath(u));
    return new Response(bytes, { headers: { "content-type": "application/wasm" } });
  }
  return realFetch(url, opts);
};
globalThis.requestAnimationFrame = (f) => setTimeout(f, 0);

const { createWlift } = await import("./wlift.js");
const wlift = await createWlift({
  mode: "main",
  wasm: new URL("./wlift_wasm_bg.wasm", import.meta.url),
});
const source = await readFile(process.argv[2], "utf8");
const t0 = performance.now();
const result = await wlift.run(source);
const ms = performance.now() - t0;
process.stdout.write(result.output ?? "");
const m = globalThis.wlift_wasm;
console.error(
  `ok=${result.ok} ${ms.toFixed(0)}ms compiled=${m.jit_compile_count()} ` +
    `rejected=${m.jit_compile_reject_count()} bc_dispatch=${m.jit_dispatch_from_bc_count()} ` +
    `fast=${m.jit_dispatch_fast_path_count()}`,
);
process.exit(result.ok ? 0 : 1);
