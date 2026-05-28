// codemirror-wren.js
//
// Exposes `window.mountWrenCode(root)` — finds every
// `<pre><code class="language-wren">` (or `language-bash` / `language-sh`)
// node under `root` and replaces it with a read-only CodeMirror 6 view
// using the same Wren stream parser the playground uses (sans the
// line-number gutter).
//
// Why a module: CodeMirror 6 ships ESM-only, so esm.sh + a `<script
// type="module">` is the lowest-friction integration. The page-side
// script that triggers this lives in the README htmx fragment — after
// marked.parse() rebuilds the DOM, it calls `window.mountWrenCode(host)`
// to upgrade the freshly-parsed code blocks in place.

import { EditorState } from "https://esm.sh/@codemirror/state@6";
import { EditorView, drawSelection } from "https://esm.sh/@codemirror/view@6";
import { StreamLanguage, syntaxHighlighting,
         HighlightStyle, bracketMatching } from "https://esm.sh/@codemirror/language@6";
import { tags } from "https://esm.sh/@lezer/highlight@1";

// ── Wren stream parser. Lifted verbatim from the playground; the
//    keyword set + token classifier match Wren 0.4 + WrenLift's
//    extensions (`construct`, `foreign`, `import`, `is`, `static`).
const wrenKeywords = new Set([
  "as","break","class","construct","continue","else","false",
  "for","foreign","if","import","in","is","null","return",
  "static","super","this","true","var","while",
]);
const wrenStreamParser = {
  name: "wren",
  startState() { return { inBlockComment: false }; },
  token(stream, state) {
    if (state.inBlockComment) {
      if (stream.match(/.*?\*\//)) { state.inBlockComment = false; return "comment"; }
      stream.skipToEnd();
      return "comment";
    }
    if (stream.eatSpace()) return null;
    if (stream.match("//")) { stream.skipToEnd(); return "lineComment"; }
    if (stream.match("/*")) { state.inBlockComment = true; return "blockComment"; }
    if (stream.match(/^"(?:\\.|[^"\\])*"/)) return "string";
    if (stream.match(/^[0-9]+(?:\.[0-9]+)?/)) return "number";
    if (stream.match(/^[A-Za-z_][A-Za-z0-9_]*/)) {
      const word = stream.current();
      if (wrenKeywords.has(word)) return "keyword";
      if (/^[A-Z]/.test(word)) return "typeName";
      if (stream.peek() === "(") return "function";
      return "variableName";
    }
    if (stream.match(/^[+\-*/=<>!&|%^~?:]+/)) return "operator";
    stream.next();
    return null;
  },
  languageData: { commentTokens: { line: "//", block: { open: "/*", close: "*/" } } },
};

// Token-color map. Reads the same `--syn-*` CSS variables the
// playground exports — keeps the docs dark code-blocks visually
// identical to the editor pane.
const wrenHighlight = HighlightStyle.define([
  { tag: tags.comment,        color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: tags.lineComment,    color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: tags.blockComment,   color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: tags.keyword,        color: "var(--syn-keyword)" },
  { tag: tags.string,         color: "var(--syn-string)" },
  { tag: tags.number,         color: "var(--syn-number)" },
  { tag: tags.bool,           color: "var(--syn-number)" },
  { tag: tags.atom,           color: "var(--syn-number)" },
  { tag: tags.typeName,       color: "var(--syn-type)", fontWeight: "600" },
  { tag: tags.variableName,   color: "var(--syn-type)" },
  { tag: tags.function(tags.variableName), color: "var(--syn-fn)" },
  { tag: tags.propertyName,   color: "var(--syn-fn)" },
  { tag: tags.operator,       color: "var(--syn-text)" },
]);

// Editor theme: matches the playground's dark code panel. No
// `lineNumbers()` extension — per the docs design, the README
// code blocks stay borderless / gutter-less.
const docsCodeTheme = EditorView.theme({
  "&": {
    backgroundColor: "var(--syn-panel)",
    color: "var(--syn-text)",
    fontFamily: "var(--mono)",
    fontSize: "13.5px",
    lineHeight: "1.55",
    border: "2px solid var(--ink)",
    borderRadius: "12px",
    overflow: "hidden",
  },
  ".cm-content": {
    padding: "16px 20px",
    caretColor: "var(--syn-keyword)",
  },
  "&.cm-editor.cm-focused": { outline: "none" },
  ".cm-scroller": { overflowX: "auto" },
  ".cm-selectionBackground, ::selection": {
    backgroundColor: "rgba(244, 194, 74, 0.25)",
  },
}, { dark: true });

const wrenLanguage = StreamLanguage.define(wrenStreamParser);

function mountOne(codeNode) {
  // The `<code>` lives inside a `<pre>` — replace the whole `<pre>`
  // so we don't keep the surrounding `<pre>` styles fighting the
  // CodeMirror frame.
  const pre = codeNode.parentNode;
  if (!pre || pre.tagName !== "PRE" || pre.dataset.cmMounted === "1") return;
  const text = codeNode.textContent.replace(/\s+$/, "");
  const host = document.createElement("div");
  host.className = "doc-code";
  pre.replaceWith(host);

  // eslint-disable-next-line no-new
  new EditorView({
    parent: host,
    state: EditorState.create({
      doc: text,
      extensions: [
        EditorView.editable.of(false),
        EditorView.lineWrapping,
        drawSelection(),
        bracketMatching(),
        wrenLanguage,
        syntaxHighlighting(wrenHighlight),
        docsCodeTheme,
      ],
    }),
  });
  host.dataset.cmMounted = "1";
}

window.mountWrenCode = function (root) {
  if (!root) return;
  const blocks = root.querySelectorAll("pre > code.language-wren");
  blocks.forEach(mountOne);
};

// First-paint: mount any wren code blocks that already shipped in
// the initial HTML (e.g. server-rendered docs, not just htmx-loaded
// READMEs).
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => window.mountWrenCode(document));
} else {
  window.mountWrenCode(document);
}
