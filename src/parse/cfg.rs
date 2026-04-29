//! Pre-parse pass that strips target-conditional declarations
//! from `.wren` source. Lives next to the lexer + parser
//! because it's the same pipeline — one stage earlier — even
//! though it's line-based string surgery rather than full Wren
//! tokenisation.
//!
//! Wren authors mark target-specific top-level declarations with
//! a bare attribute on the line before:
//!
//! ```wren,no_run
//! #!wasm
//! import "@hatch:gpu-web" for Gpu
//!
//! #!native
//! import "@hatch:gpu-native" for Gpu
//! ```
//!
//! When the bundler is packing a hatch for a particular target,
//! it asks this module to strip declarations whose attribute
//! doesn't match. The output is plain Wren that the regular
//! compiler accepts unmodified.
//!
//! Recognised attributes:
//!
//! * `#!wasm`   — keep only when bundle target is `wasm32-*`.
//! * `#!native` — keep only when bundle target is anything else
//!                 (host build, no `--bundle-target` flag).
//!
//! Adding new ones (`#!macos`, `#!arm64`, ...) is a matter of
//! extending [`AttrKind`]; the rest of the pipeline composes.
//!
//! Multiple attributes on the same declaration AND together —
//! `#!macos` + `#!arm64` keeps the declaration only on macOS
//! arm64. The whole stack lives on consecutive lines above the
//! gated declaration; a blank line breaks the chain.
//!
//! Implementation note: this is a line-level scanner, not a
//! parser. It identifies attribute lines by their leading `#!`
//! prefix and the bare-token shape; everything else is
//! pass-through. The contract is "drop attribute lines that
//! we recognise, drop the *next non-blank line* if any
//! recognised attribute fails." Comments, blank lines, and
//! source the scanner doesn't recognise survive untouched —
//! the regular Wren parser sees them as it always did.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrKind {
    /// Keep only when bundle target is wasm32-*.
    Wasm,
    /// Keep only when bundle target is non-wasm (host).
    Native,
}

impl AttrKind {
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "wasm" => Some(AttrKind::Wasm),
            "native" => Some(AttrKind::Native),
            _ => None,
        }
    }

    /// Does this attribute accept the current bundle target?
    /// `target` is `None` for host builds.
    fn matches(self, target: Option<&str>) -> bool {
        let is_wasm = matches!(target, Some(t) if t == "wasm32" || t.starts_with("wasm32-"));
        match self {
            AttrKind::Wasm => is_wasm,
            AttrKind::Native => !is_wasm,
        }
    }
}

/// Apply the build-time attribute filter to a `.wren` source
/// string. Returns the rewritten source with declarations whose
/// attributes don't match the target elided. Unrecognised
/// attribute lines pass through untouched (compiled Wren keeps
/// using `#!native = "..."` / `#!symbol = "..."` for
/// foreign-method binding — those have a `=` and aren't bare
/// tokens, so the scanner ignores them).
///
/// Multi-line declarations are handled by tracking `{` / `}`
/// depth on the line that follows a failed gate. The first
/// non-blank line after a failed cfg attribute opens the
/// "drop region"; subsequent lines stay in drop mode until
/// brace depth returns to zero. Strings containing literal
/// braces could in principle confuse this, but Wren's idiomatic
/// top-level shape (imports, classes, vars) doesn't run into
/// it; if it ever does, the source can lift the gated block
/// into its own module and gate the `import` instead.
pub fn apply(source: &str, target: Option<&str>) -> String {
    let mut out = String::with_capacity(source.len());
    let mut pending: Vec<AttrKind> = Vec::new();
    let mut chain_keep = true;
    // When we're elidng a multi-line block, this counts the
    // unmatched `{`s we've seen so far. Drop mode lasts until
    // the count returns to zero.
    let mut drop_depth: i32 = 0;

    for raw_line in source.split_inclusive('\n') {
        // `split_inclusive` keeps the trailing newline so we can
        // round-trip line endings (CRLF or LF) verbatim.

        // If we're already inside an elided multi-line block,
        // keep dropping until the brace depth returns to zero.
        if drop_depth > 0 {
            drop_depth += brace_delta(raw_line);
            if drop_depth <= 0 {
                drop_depth = 0;
                pending.clear();
                chain_keep = true;
            }
            continue;
        }

        let trimmed = raw_line.trim_start();

        // Recognised cfg attribute? Stash it and don't emit the
        // line — these are bundler-only metadata.
        if let Some(rest) = trimmed.strip_prefix("#!") {
            let attr_body = rest.trim_end();
            if let Some(kind) = parse_bare_attr(attr_body) {
                pending.push(kind);
                if !kind.matches(target) {
                    chain_keep = false;
                }
                continue;
            }
            // Falls through — `#!native = "..."` etc. emit as-is.
        }

        // Blank line breaks any pending attribute chain — the
        // attributes attach to the next *real* declaration.
        if trimmed.trim().is_empty() {
            pending.clear();
            chain_keep = true;
            out.push_str(raw_line);
            continue;
        }

        // Non-attribute, non-blank: the gated declaration begins
        // here. If it failed the cfg check, drop the line and
        // — if the line opens a brace block — keep dropping
        // subsequent lines until the matching close.
        if !chain_keep {
            let delta = brace_delta(raw_line);
            if delta > 0 {
                drop_depth = delta;
            } else {
                // Single-line declaration (import, var, etc.).
                pending.clear();
                chain_keep = true;
            }
            continue;
        }

        out.push_str(raw_line);
        pending.clear();
        chain_keep = true;
    }

    out
}

/// Net change in brace depth contributed by `line`. Counts `{`
/// minus `}`. Doesn't try to be string-literal-aware; Wren's
/// top-level forms (class headers, var inits, imports) don't
/// hit pathological cases in practice.
fn brace_delta(line: &str) -> i32 {
    let mut delta: i32 = 0;
    for b in line.bytes() {
        match b {
            b'{' => delta += 1,
            b'}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

/// Parse a bare-token attribute body. Returns `Some(kind)` for
/// recognised tokens; `None` for everything else (including the
/// `name = "value"` form used by foreign-method binding).
fn parse_bare_attr(body: &str) -> Option<AttrKind> {
    // Reject the `name = value` form outright — that's
    // foreign-method-binding metadata, not a target gate.
    if body.contains('=') {
        return None;
    }
    AttrKind::from_token(body.trim())
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run(source: &str, target: Option<&str>) -> String {
        apply(source, target)
    }

    #[test]
    fn unattributed_source_passes_through() {
        let src = "class Foo {\n  bar() { 1 }\n}\n";
        assert_eq!(run(src, None), src);
        assert_eq!(run(src, Some("wasm32-unknown-unknown")), src);
    }

    #[test]
    fn foreign_method_binding_attrs_are_ignored() {
        // `#!native = "..."` and `#!symbol = "..."` are *not*
        // target gates; they must survive verbatim.
        let src = "#!native = \"libfoo\"\nforeign class Foo {\n  #!symbol = \"foo_bar\"\n  foreign static bar()\n}\n";
        assert_eq!(run(src, None), src);
        assert_eq!(run(src, Some("wasm32-unknown-unknown")), src);
    }

    #[test]
    fn wasm_attr_keeps_only_on_wasm_target() {
        let src = "\
#!wasm
import \"@hatch:gpu-web\" for Gpu

#!native
import \"@hatch:gpu-native\" for Gpu

System.print(\"loaded\")
";
        let wasm_out = run(src, Some("wasm32-unknown-unknown"));
        assert!(wasm_out.contains("@hatch:gpu-web"));
        assert!(!wasm_out.contains("@hatch:gpu-native"));
        assert!(wasm_out.contains("System.print"));

        let native_out = run(src, None);
        assert!(!native_out.contains("@hatch:gpu-web"));
        assert!(native_out.contains("@hatch:gpu-native"));
        assert!(native_out.contains("System.print"));
    }

    #[test]
    fn attribute_lines_themselves_never_emitted() {
        // The scanner consumes recognised cfg lines wholesale —
        // they don't belong in the compiled Wren the parser sees.
        let src = "#!wasm\nimport \"x\" for Y\n";
        let out = run(src, Some("wasm32-unknown-unknown"));
        assert!(!out.contains("#!wasm"));
        assert!(out.contains("import \"x\""));
    }

    #[test]
    fn stacked_attrs_and_together() {
        // Two cfg attributes on the same declaration: both must
        // match for the declaration to survive. (Today the only
        // attrs are wasm/native, which are mutually exclusive,
        // so any stack of both always elides — the test still
        // pins the AND semantics for when more attrs land.)
        let src = "#!wasm\n#!native\nimport \"x\" for Y\n";
        assert!(!run(src, Some("wasm32-unknown-unknown")).contains("import"));
        assert!(!run(src, None).contains("import"));
    }

    #[test]
    fn blank_line_breaks_attr_chain() {
        // Blank line between `#!wasm` and the import drops the
        // attribute chain — the import is unconditional.
        let src = "#!wasm\n\nimport \"x\" for Y\n";
        let out = run(src, None);
        assert!(out.contains("import"));
    }

    #[test]
    fn unknown_bare_attrs_pass_through() {
        // A `#!something_we_dont_know` line stays in the source
        // so future Wren attributes don't get silently eaten.
        let src = "#!coolfeature\nclass Foo {}\n";
        let out = run(src, None);
        assert!(out.contains("#!coolfeature"));
        assert!(out.contains("class Foo"));
    }

    #[test]
    fn multi_line_class_drops_through_matching_brace() {
        // The header line opens a brace; drop mode persists
        // until the matching close brace.
        let src = "\
class Greeter {
  static hi() { System.print(\"hi\") }
}

#!native
class NativeOnly {
  static greet() { System.print(\"native\") }
}

class AlwaysHere {}
";
        let out = run(src, Some("wasm32-unknown-unknown"));
        assert!(out.contains("class Greeter"));
        assert!(out.contains("class AlwaysHere"));
        assert!(!out.contains("NativeOnly"));
        assert!(!out.contains("\"native\""));
    }

    #[test]
    fn wasm_target_prefix_match() {
        // Any wasm32-* triple counts.
        let src = "#!wasm\nimport \"x\" for Y\n";
        assert!(run(src, Some("wasm32")).contains("import"));
        assert!(run(src, Some("wasm32-unknown-unknown")).contains("import"));
        assert!(run(src, Some("wasm32-wasip1")).contains("import"));
        assert!(!run(src, Some("x86_64-apple-darwin")).contains("import"));
    }
}
