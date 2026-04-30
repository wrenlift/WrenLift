// Bake the Fruit Slicer asset atlas. Runs ImageMagick to
// down-sample the source fruit PNGs to 128×128, reads them back
// as raw RGBA, then composes the full 384×384 atlas in Node so
// HUD glyphs (digits, heart, streak glow) can be painted into
// the same buffer alongside the fruits. Outputs:
//
//   wasm/web/assets/fruit-slicer/atlas.rgba8   raw bytes for
//                                              `device.writeTexture`
//   wasm/web/assets/fruit-slicer/atlas.png     human-inspect
//   wasm/web/assets/fruit-slicer/atlas.json    sprite manifest
//                                              (rect + UV per name)
//
// Run with:  node tools/bake-fruit-slicer.mjs
//
// Re-curate by editing the FRUITS list at the top — `id` is the
// source filename, `name` is the manifest key the Wren game
// looks up. The artifacts are committed so the playground works
// without anyone needing Node to bake.
//
// Atlas layout (384×384):
//
//   y=  0..127   3 fruits  (cells x=0,128,256, w/h=128)
//   y=128..255   3 fruits  (same column placement)
//   y=256..287   10 digits at 32×32 (x=0..320), heart at (320,256,32×32)
//   y=288..351   streak glow at 64×64 (x=0..64), spare to right
//   y=352..383   "GAME OVER" letters at 32×32 (x=0..224, 7 cells)

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");
const SRC_DIR   = path.join(REPO_ROOT, "site/playground/assets/fruit1");
const OUT_DIR   = path.join(REPO_ROOT, "wasm/web/assets/fruit-slicer");

// Curated fruit subset. Edit `id` to swap; `name` is the Wren
// lookup key. Order is left-to-right, top-to-bottom in the atlas.
const FRUITS = [
  { name: "fruit_45", id: "T_fruit_45" },
  { name: "fruit_46", id: "T_fruit_46" },
  { name: "fruit_48", id: "T_fruit_48" },
  { name: "fruit_49", id: "T_fruit_49" },
  { name: "fruit_57", id: "T_fruit_57" },
  { name: "fruit_60", id: "T_fruit_60" },
];

const FRUIT_COLS = 3;
const FRUIT_CELL = 128;
const W = FRUIT_COLS * FRUIT_CELL;     // 384
const H = 384;
const DIGIT_CELL = 32;
const DIGIT_ROW_Y = 256;
const HEART_CELL = 32;
const HEART_X = 320, HEART_Y = 256;
const STREAK_SIZE = 64;
const STREAK_X = 0, STREAK_Y = 288;
const LETTER_CELL = 32;
// Two-row letter strip — first row at y=352 holds the
// "GAME OVER" headline; second row at y=320 (next to the streak
// glow which only occupies x=0..63) carries the lower-row
// "CLICK TO RESUME" hint glyphs. Together these cover every
// unique letter in both strings.
const LETTER_ROW_Y_HEAD = 352;
const LETTER_ROW_Y_HINT = 320;
const LETTERS_HEAD = ["G", "A", "M", "E", "O", "V", "R"];
const LETTERS_HINT = ["C", "L", "I", "K", "T", "S", "U"];
const LETTERS_HINT_X = 64; // skip the streak glow cell at x=0..63

fs.mkdirSync(OUT_DIR, { recursive: true });

// Atlas RGBA buffer — one big Uint8Array (384*384*4 = 589 824 bytes).
// All-zero = fully-transparent black = the natural blank state.
const atlas = new Uint8Array(W * H * 4);

// --- Pixel helpers ------------------------------------------------
function setPx(x, y, r, g, b, a) {
  if (x < 0 || x >= W || y < 0 || y >= H) return;
  const i = (y * W + x) * 4;
  atlas[i] = r; atlas[i+1] = g; atlas[i+2] = b; atlas[i+3] = a;
}

// Copy a `srcW`×`srcH` RGBA buffer onto the atlas at (dstX, dstY).
// Source must be tightly packed (no row-stride padding).
function blit(src, srcW, srcH, dstX, dstY) {
  for (let y = 0; y < srcH; y++) {
    const srcRow = y * srcW * 4;
    const dstRow = ((dstY + y) * W + dstX) * 4;
    atlas.set(src.subarray(srcRow, srcRow + srcW * 4), dstRow);
  }
}

// --- 1: fruit cells -----------------------------------------------
// Resize each source PNG to 128×128 via ImageMagick, read back as
// raw RGBA, blit into the atlas at the right grid cell.
const TMP = fs.mkdtempSync("/tmp/fn-bake-");
FRUITS.forEach((fr, idx) => {
  const src = path.join(SRC_DIR, `${fr.id}.png`);
  if (!fs.existsSync(src)) throw new Error(`source not found: ${src}`);
  const tmpRgba = path.join(TMP, `${fr.name}.rgba8`);
  execFileSync("magick", [
    src, "-resize", `${FRUIT_CELL}x${FRUIT_CELL}`,
    "-depth", "8",
    `RGBA:${tmpRgba}`,
  ]);
  const fruitBytes = fs.readFileSync(tmpRgba);
  const col = idx % FRUIT_COLS;
  const row = Math.floor(idx / FRUIT_COLS);
  blit(fruitBytes, FRUIT_CELL, FRUIT_CELL, col * FRUIT_CELL, row * FRUIT_CELL);
  fs.unlinkSync(tmpRgba);
});
fs.rmdirSync(TMP);

// --- 2: digits 0-9 -----------------------------------------------
// Hardcoded 5×7 ASCII digit patterns (well-known shapes). `#`
// pixels paint white, anything else stays transparent. Scaled
// 4× nearest-neighbour to a 20×28 logical glyph, then centred
// in a 32×32 atlas cell with 6-px padding on each side.
const DIGIT_GLYPHS_5x7 = {
  "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
  "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
  "2": [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
  "3": [" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "],
  "4": ["#   #", "#   #", "#   #", "#####", "    #", "    #", "    #"],
  "5": ["#####", "#    ", "#    ", "#### ", "    #", "#   #", " ### "],
  "6": [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
  "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
  "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
  "9": [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
};

const DIGIT_SCALE = 4;             // 5×7 → 20×28 logical pixels
const DIGIT_PAD_X = (DIGIT_CELL - 5 * DIGIT_SCALE) / 2;  // 6
const DIGIT_PAD_Y = (DIGIT_CELL - 7 * DIGIT_SCALE) / 2;  // 2

function paintDigit(ch, dstX, dstY) {
  const rows = DIGIT_GLYPHS_5x7[ch];
  for (let r = 0; r < 7; r++) {
    const row = rows[r];
    for (let c = 0; c < 5; c++) {
      if (row[c] !== "#") continue;
      // Paint a DIGIT_SCALE × DIGIT_SCALE block of white.
      for (let dy = 0; dy < DIGIT_SCALE; dy++) {
        for (let dx = 0; dx < DIGIT_SCALE; dx++) {
          setPx(
            dstX + DIGIT_PAD_X + c * DIGIT_SCALE + dx,
            dstY + DIGIT_PAD_Y + r * DIGIT_SCALE + dy,
            255, 255, 255, 255,
          );
        }
      }
    }
  }
}

for (let d = 0; d < 10; d++) {
  paintDigit(String(d), d * DIGIT_CELL, DIGIT_ROW_Y);
}

// --- 2b: letters for "GAME OVER" ----------------------------------
// Same 5×7 pixel font as digits, scaled 4× into 32×32 cells. Only
// the seven unique characters in "GAME OVER" — keeps the atlas
// small. Add more later if more text is needed.
const LETTER_GLYPHS_5x7 = {
  "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "],
  "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
  "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
  "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
  "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
  "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
  "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
  "C": [" ####", "#    ", "#    ", "#    ", "#    ", "#    ", " ####"],
  "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
  "I": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"],
  "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
  "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
  "S": [" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "],
  "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
};

function paintLetter(ch, dstX, dstY) {
  const rows = LETTER_GLYPHS_5x7[ch];
  if (!rows) return;
  for (let r = 0; r < 7; r++) {
    const row = rows[r];
    for (let c = 0; c < 5; c++) {
      if (row[c] !== "#") continue;
      for (let dy = 0; dy < DIGIT_SCALE; dy++) {
        for (let dx = 0; dx < DIGIT_SCALE; dx++) {
          setPx(
            dstX + DIGIT_PAD_X + c * DIGIT_SCALE + dx,
            dstY + DIGIT_PAD_Y + r * DIGIT_SCALE + dy,
            255, 255, 255, 255,
          );
        }
      }
    }
  }
}

LETTERS_HEAD.forEach((ch, idx) => {
  paintLetter(ch, idx * LETTER_CELL, LETTER_ROW_Y_HEAD);
});
LETTERS_HINT.forEach((ch, idx) => {
  paintLetter(ch, LETTERS_HINT_X + idx * LETTER_CELL, LETTER_ROW_Y_HINT);
});

// --- 3: heart icon ------------------------------------------------
// Real heart sprite from `wasm/web/assets/Icon_Small_HeartFull.png`
// (170×150 RGBA). ImageMagick resizes-and-pads it to a square
// 32×32 cell with transparent margin, preserving aspect; the
// `-extent` step pads to the cell size so the blit always
// receives a full 32×32 buffer. Used for the lives HUD —
// rendered N times stacked horizontally.
const HEART_SRC = path.join(REPO_ROOT, "wasm/web/assets/Icon_Small_HeartFull.png");
function bakeIconCell(srcPath, cellW, cellH) {
  const tmp = fs.mkdtempSync("/tmp/fn-icon-");
  const dst = path.join(tmp, "icon.rgba8");
  execFileSync("magick", [
    srcPath,
    "-resize",     `${cellW}x${cellH}`,
    "-background", "none",
    "-gravity",    "center",
    "-extent",     `${cellW}x${cellH}`,
    "-depth",      "8",
    `RGBA:${dst}`,
  ]);
  const bytes = fs.readFileSync(dst);
  fs.unlinkSync(dst);
  fs.rmdirSync(tmp);
  return bytes;
}
const heartBytes = bakeIconCell(HEART_SRC, HEART_CELL, HEART_CELL);
blit(heartBytes, HEART_CELL, HEART_CELL, HEART_X, HEART_Y);

// --- 4: streak glow -----------------------------------------------
// 64×64 radial alpha gradient: bright white centre, fully
// transparent edges. Stamping this sprite repeatedly along the
// mouse trail produces the swipe streak effect.
function paintStreakGlow() {
  const r = STREAK_SIZE / 2;
  for (let py = 0; py < STREAK_SIZE; py++) {
    for (let px = 0; px < STREAK_SIZE; px++) {
      const dx = px - STREAK_SIZE / 2;
      const dy = py - STREAK_SIZE / 2;
      const d = Math.sqrt(dx*dx + dy*dy);
      if (d > r) continue;
      // Squared falloff — hot at centre, gentle tail.
      const t = 1 - d / r;
      const alpha = Math.round(255 * t * t);
      setPx(STREAK_X + px, STREAK_Y + py, 255, 255, 240, alpha);
    }
  }
}
paintStreakGlow();

// --- Write artifacts ----------------------------------------------
const rgbaPath = path.join(OUT_DIR, "atlas.rgba8");
const pngPath  = path.join(OUT_DIR, "atlas.png");
const jsonPath = path.join(OUT_DIR, "atlas.json");

fs.writeFileSync(rgbaPath, atlas);

// Convert raw RGBA → PNG via ImageMagick. `RGBA:` prefix with
// `-size W×H -depth 8` tells ImageMagick how to interpret the
// raw bytes. Output is just for human inspection / commit
// review — runtime reads the .rgba8 directly.
execFileSync("magick", [
  "-size", `${W}x${H}`,
  "-depth", "8",
  `RGBA:${rgbaPath}`,
  pngPath,
]);

// --- Manifest -----------------------------------------------------
// `rect` is pixel-space [x, y, w, h]; `uv` is [0..1] for direct
// `Sprite.uv(u0, v0, u1, v1)` consumption. Halves are the
// vertical mid-line split per fruit cell — useful for the
// slice-into-two animation.
const manifest = {
  size: [W, H],
  cell: [FRUIT_CELL, FRUIT_CELL],
  sprites: {},
  halves: {},
  hud: {
    digits: {},
    letters: {},
  },
};

const cellRect = (x, y, w, h) => ({
  rect: [x, y, w, h],
  uv: [x / W, y / H, (x + w) / W, (y + h) / H],
});

FRUITS.forEach((fr, idx) => {
  const col = idx % FRUIT_COLS;
  const row = Math.floor(idx / FRUIT_COLS);
  const x = col * FRUIT_CELL, y = row * FRUIT_CELL;
  manifest.sprites[fr.name] = {
    source: `${fr.id}.png`,
    ...cellRect(x, y, FRUIT_CELL, FRUIT_CELL),
  };
  manifest.halves[`${fr.name}_left`]  = cellRect(x, y, FRUIT_CELL/2, FRUIT_CELL);
  manifest.halves[`${fr.name}_right`] = cellRect(x + FRUIT_CELL/2, y, FRUIT_CELL/2, FRUIT_CELL);
});

for (let d = 0; d < 10; d++) {
  manifest.hud.digits[String(d)] = cellRect(d * DIGIT_CELL, DIGIT_ROW_Y, DIGIT_CELL, DIGIT_CELL);
}
LETTERS_HEAD.forEach((ch, idx) => {
  manifest.hud.letters[ch] = cellRect(idx * LETTER_CELL, LETTER_ROW_Y_HEAD, LETTER_CELL, LETTER_CELL);
});
LETTERS_HINT.forEach((ch, idx) => {
  manifest.hud.letters[ch] = cellRect(LETTERS_HINT_X + idx * LETTER_CELL, LETTER_ROW_Y_HINT, LETTER_CELL, LETTER_CELL);
});
manifest.hud.heart  = cellRect(HEART_X, HEART_Y, HEART_CELL, HEART_CELL);
manifest.hud.streak = cellRect(STREAK_X, STREAK_Y, STREAK_SIZE, STREAK_SIZE);

fs.writeFileSync(jsonPath, JSON.stringify(manifest, null, 2) + "\n");

console.log(`baked ${W}×${H} atlas (${FRUITS.length} fruits + 10 digits + heart + streak)`);
console.log(`  ${pngPath}   ${fs.statSync(pngPath).size.toLocaleString()} bytes (PNG)`);
console.log(`  ${rgbaPath}  ${fs.statSync(rgbaPath).size.toLocaleString()} bytes (raw RGBA8)`);
console.log(`  ${jsonPath}`);
