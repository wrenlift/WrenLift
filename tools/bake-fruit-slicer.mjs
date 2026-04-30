// Bake the Fruit Ninja sprite atlas. Pulls a curated subset of
// the 60 source fruit PNGs in `site/playground/assets/fruit1/`,
// downsamples to 128×128, packs into a single 768×256 atlas
// (3 cols × 2 rows), and emits three artifacts:
//
//   wasm/web/assets/fruit-ninja/atlas.rgba8   raw bytes for direct
//                                              `device.writeTexture`
//   wasm/web/assets/fruit-ninja/atlas.png     same image, for human
//                                              inspection / re-bake
//                                              source-of-truth
//   wasm/web/assets/fruit-ninja/atlas.json    sprite name → rect+UV
//
// Run with:  node tools/bake-fruit-ninja.mjs
//
// Re-curate by editing the `FRUITS` list below — `id` matches the
// `T_fruit_NN.png` filename, `name` is what the Wren game looks
// up in the manifest. The artifacts are committed so the
// playground works without anyone needing to install Node.
//
// Atlas layout (3×2 grid of 128×128 cells):
//
//   | apple    | orange   | banana     |
//   | grape    | melon    | strawberry |

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const SRC_DIR   = path.join(REPO_ROOT, "site/playground/assets/fruit1");
const OUT_DIR   = path.join(REPO_ROOT, "wasm/web/assets/fruit-ninja");

// Curated fruit subset. Edit `id` to swap; `name` is the Wren
// lookup key in `manifest.sprites`. Order is left-to-right,
// top-to-bottom in the atlas. Names default to the source ID
// since the source pack is anonymous; rename in this list to
// drive the manifest if you want semantic Wren-side keys.
const FRUITS = [
  { name: "fruit_45", id: "T_fruit_45" },
  { name: "fruit_46", id: "T_fruit_46" },
  { name: "fruit_48", id: "T_fruit_48" },
  { name: "fruit_49", id: "T_fruit_49" },
  { name: "fruit_57", id: "T_fruit_57" },
  { name: "fruit_60", id: "T_fruit_60" },
];

const COLS = 3;
const ROWS = Math.ceil(FRUITS.length / COLS);
const CELL = 128;
const W = COLS * CELL;
const H = ROWS * CELL;

fs.mkdirSync(OUT_DIR, { recursive: true });

// --- Step 1: resize each source PNG to CELL×CELL --------------------
// Going through ImageMagick's `magick` CLI rather than a Node
// image lib keeps zero npm install. Each input is 256×256 RGBA;
// `-resize CELLxCELL` produces a CELL×CELL transparent-bg PNG.
const TMP = fs.mkdtempSync(path.join(fs.realpathSync("/tmp"), "fn-bake-"));
const tmpPaths = [];
for (const fr of FRUITS) {
  const src = path.join(SRC_DIR, `${fr.id}.png`);
  if (!fs.existsSync(src)) throw new Error(`source not found: ${src}`);
  const dst = path.join(TMP, `${fr.name}.png`);
  execFileSync("magick", [src, "-resize", `${CELL}x${CELL}`, dst]);
  tmpPaths.push(dst);
}

// --- Step 2: composite tiles into a single atlas PNG ----------------
// `magick montage` is the natural fit, but it tries to render
// labels even when none are requested and bombs out on
// font-less ImageMagick installs. Per-tile composite over a
// transparent canvas does the same job without touching the
// text-rendering codepath.
const atlasPng = path.join(OUT_DIR, "atlas.png");
const compositeArgs = ["-size", `${W}x${H}`, "xc:transparent"];
tmpPaths.forEach((p, idx) => {
  const col = idx % COLS;
  const row = Math.floor(idx / COLS);
  compositeArgs.push(p, "-geometry", `+${col * CELL}+${row * CELL}`, "-composite");
});
compositeArgs.push(atlasPng);
execFileSync("magick", compositeArgs);

// --- Step 3: convert atlas.png → raw RGBA8 bytes -------------------
// `RGBA:filename` is ImageMagick's prefix for raw output. Saves
// the runtime an entire PNG decoder dependency.
const atlasRaw = path.join(OUT_DIR, "atlas.rgba8");
execFileSync("magick", [atlasPng, "-depth", "8", `RGBA:${atlasRaw}`]);

// --- Step 4: emit JSON manifest ------------------------------------
// `rect` is pixel-space [x, y, w, h]; `uv` is [0..1] for direct
// `Sprite.uv(u0, v0, u1, v1)` consumption. Halves split each cell
// vertically — handy for the slice-into-two animation; the game
// can ignore them if it animates whole sprites instead.
const manifest = {
  size: [W, H],
  cell: [CELL, CELL],
  sprites: {},
  halves:  {},
};
FRUITS.forEach((fr, idx) => {
  const col = idx % COLS;
  const row = Math.floor(idx / COLS);
  const x = col * CELL, y = row * CELL;
  const u0 = x / W,         v0 = y / H;
  const u1 = (x + CELL) / W, v1 = (y + CELL) / H;
  manifest.sprites[fr.name] = {
    source: `${fr.id}.png`,
    rect:   [x, y, CELL, CELL],
    uv:     [u0, v0, u1, v1],
  };
  const uMid = (u0 + u1) / 2;
  manifest.halves[`${fr.name}_left`]  = {
    rect: [x, y, CELL/2, CELL],
    uv:   [u0, v0, uMid, v1],
  };
  manifest.halves[`${fr.name}_right`] = {
    rect: [x + CELL/2, y, CELL/2, CELL],
    uv:   [uMid, v0, u1, v1],
  };
});
fs.writeFileSync(
  path.join(OUT_DIR, "atlas.json"),
  JSON.stringify(manifest, null, 2) + "\n",
);

// Cleanup tmpdir.
for (const p of tmpPaths) fs.unlinkSync(p);
fs.rmdirSync(TMP);

const rawSize = fs.statSync(atlasRaw).size;
const pngSize = fs.statSync(atlasPng).size;
console.log(`baked ${W}×${H} atlas (${FRUITS.length} fruits)`);
console.log(`  ${atlasPng}  ${pngSize.toLocaleString()} bytes (PNG)`);
console.log(`  ${atlasRaw}  ${rawSize.toLocaleString()} bytes (raw RGBA8)`);
console.log(`  ${OUT_DIR}/atlas.json`);
