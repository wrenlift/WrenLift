#!/usr/bin/env node
//! migrate-doc-comments.mjs
//
// Mechanical migration helper for the comment-format-spec
// (docs/comment-format-spec.md):
//
//   * Top-of-file `//` blocks → `//!` (module-level docs).
//   * In-body `// ...` immediately above a `class` / `static foo` /
//     `foo(args)` / `getter`-shaped declaration → `///` (decl docs).
//
// Other `//` comments (in-method, after a statement, anywhere
// inside a function body) stay as plain code annotations. The
// rewrite is deliberately conservative — false positives ship
// noise into the generated docs, which is worse than missing a
// few decl-level migrations.
//
// Usage:
//
//   node tools/migrate-doc-comments.mjs path/to/package.wren  # in-place
//   node tools/migrate-doc-comments.mjs path/to/package/      # walks dir
//   node tools/migrate-doc-comments.mjs --dry path/to/...      # preview only
//
// The script never overwrites a file that already contains any
// `///` or `//!` markers — assume the author has already started
// the migration by hand and we'd just fight them.

import fs from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const dry = args.includes("--dry");
const targets = args.filter((a) => !a.startsWith("--"));
if (targets.length === 0) {
  console.error("usage: migrate-doc-comments.mjs [--dry] <path> [<path>...]");
  process.exit(2);
}

let total = 0;
let changed = 0;
// Defer the walk until every helper + const is initialised
// (STATEMENT_KEYWORDS is a `const` defined further down).
// Top-level `for` would run before that, hitting the TDZ.
function main() {
  for (const t of targets) {
    walk(t);
  }
  console.log(`scanned ${total} .wren file(s); ${changed} updated`);
}

function walk(p) {
  const stat = fs.statSync(p);
  if (stat.isDirectory()) {
    for (const child of fs.readdirSync(p)) {
      if (child.startsWith(".") || child === "target" || child === "node_modules") continue;
      walk(path.join(p, child));
    }
    return;
  }
  if (!p.endsWith(".wren")) return;
  total += 1;
  const text = fs.readFileSync(p, "utf8");
  if (text.includes("//!") || /^\s*\/\/\//m.test(text)) {
    // Author's already begun migrating this file — leave it.
    return;
  }
  const out = migrate(text);
  if (out === text) return;
  changed += 1;
  if (dry) {
    console.log(`would rewrite: ${p}`);
  } else {
    fs.writeFileSync(p, out);
    console.log(`rewrote: ${p}`);
  }
}

// Two-pass rewrite. The migration only touches comments — never
// code — so we walk line by line.
function migrate(src) {
  const lines = src.split("\n");

  // Pass 1: leading `//` block (line 0 onwards, until the first
  // non-comment, non-blank line) → `//!`. Stop the moment we
  // see a code line or end of file.
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === "") { i++; continue; }
    if (!isPlainLineComment(line)) break;
    lines[i] = rewriteLineComment(line, "//!");
    i++;
  }

  // Pass 2: walk the rest. For each non-comment, non-blank line
  // that looks like a *class-body* declaration (top-level `class`
  // or member at exactly the class-body indent), look back at the
  // immediately-preceding `//` block (no blank-line gap, same
  // indentation) and convert it to `///`. Comments inside method
  // bodies are off-limits — they'd never be picked up by the doc
  // collector anyway and the migration would just add noise.
  for (let j = lines.length - 1; j >= 0; j--) {
    if (!isDeclLine(lines[j])) continue;
    const declIndent = leadingSpace(lines[j]);
    let k = j - 1;
    // Walk back through `//` lines that share the decl's indent.
    let block = [];
    while (k >= 0 && isPlainLineComment(lines[k]) && leadingSpace(lines[k]) === declIndent) {
      block.unshift(k);
      k--;
    }
    if (block.length === 0) continue;
    if (k >= 0 && lines[k].trim() === "") continue; // blank line gap
    if (block[0] === 0) continue;                    // top-of-file owns this run
    for (const idx of block) {
      lines[idx] = rewriteLineComment(lines[idx], "///");
    }
  }

  return lines.join("\n");
}

function leadingSpace(line) {
  const m = line.match(/^(\s*)/);
  return m ? m[1] : "";
}

function isPlainLineComment(line) {
  // Line whose first non-whitespace token is `//` and not `///`
  // or `//!` (we never re-mark already-marked lines).
  const t = line.trimStart();
  if (!t.startsWith("//")) return false;
  if (t.startsWith("///") || t.startsWith("//!")) return false;
  return true;
}

function rewriteLineComment(line, prefix) {
  // Preserve indentation, replace the leading `//` with the new
  // marker. Keep the existing space-before-body if there was one.
  return line.replace(/^(\s*)\/\/(\s?)/, (_, indent, sp) => `${indent}${prefix}${sp}`);
}

// Keywords that begin statements / control flow. The decl
// detector matches by shape (`name(args) {`, `name { ... }`,
// `name=(arg)`), so anything that starts with one of these
// words isn't a class-body declaration even if it follows the
// same shape.
const STATEMENT_KEYWORDS = new Set([
  "if", "else", "while", "for", "return", "var", "break", "continue",
  "import", "is", "this", "super", "true", "false", "null", "in",
  "as", "and", "or", "not",
]);

function isDeclLine(line) {
  // A class-body declaration: class header, method, getter,
  // setter, constructor, operator overload. The shape regex is
  // permissive — we follow it with a keyword check to reject
  // statement starts that happen to look identifier-ish.
  const t = line.trim();
  if (!t) return false;
  if (/^(?:foreign\s+)?class\b/.test(t)) return true;
  // Operator overloads with symbols.
  if (/^(?:foreign\s+)?(?:static\s+)?[+\-*/%<>=!&|^~](?:[+\-*/%<>=!&|^~])?\s*(?:\(|\{)/.test(t)) return true;
  // Identifier-led shapes. Capture the leading word, reject
  // statement keywords.
  const m = t.match(/^(?:foreign\s+)?(?:static\s+)?(?:construct\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(\(|=\(|\{)/);
  if (!m) return false;
  if (STATEMENT_KEYWORDS.has(m[1])) return false;
  return true;
}
main();
