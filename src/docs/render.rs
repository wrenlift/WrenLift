//! Serialize a [`ModuleDoc`] to JSON. The wrenlift.com docs viewer
//! consumes this verbatim; styling lives there.

use serde::Serialize;

use super::model::ModuleDoc;

/// Render a single module's docs as a JSON document. Pretty-printed
/// (2-space indent) so the on-disk diff stays readable.
pub fn render_module_json(module: &ModuleDoc) -> String {
    serde_json::to_string_pretty(module).expect("ModuleDoc JSON serialize")
}

/// Top-level entry the docs site loads first. Lists every module
/// in a package along with its first-paragraph summary so the
/// sidebar / search index can populate without fetching every
/// module page.
#[derive(Debug, Clone, Serialize)]
pub struct PackageManifest {
    /// `name` from the consumer's hatchfile, when known. The CLI
    /// passes the directory basename when no hatchfile is around.
    pub name: String,
    pub modules: Vec<ModuleEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModuleEntry {
    pub name: String,
    /// First-paragraph summary, plain Markdown. Empty when the
    /// module has no `//!` block.
    pub summary: String,
    /// Number of public classes in the module — lets the sidebar
    /// hint at the surface area without loading the full doc.
    pub class_count: usize,
}

/// Schema version embedded in the docs.json wrapper. Bumped
/// whenever the wire shape changes; consumers fall back to the
/// legacy array (treated as v1) when this field is absent.
pub const DOCS_SCHEMA_VERSION: u32 = 2;

/// Top-level docs.json wrapper. The bundler / `hatch docs` /
/// `upload_workspace_docs` all serialise this; the site, the
/// `hatch docs` CLI, and `register_hatch_docs` all deserialise
/// it. Sibling fields beyond `modules` (changelog, readme) let
/// the same payload carry the package's narrative markdown
/// without a separate fetch — the site renders them inline
/// when the registry URL 404s during dev.
#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub struct DocsBundle {
    pub schema_version: u32,
    pub modules: Vec<ModuleDoc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub changelog: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readme: Option<String>,
}

/// Serialise the docs payload as the v2 wrapper object. Compact
/// (no whitespace) because the bundle / Storage object is
/// gzip/zstd-compressed in transit; readable spelling lives in
/// [`render_module_json`].
///
/// Returns the encoder error verbatim so callers can decide
/// whether to fail the publish or skip the docs section.
pub fn render_module_docs_json(modules: &[ModuleDoc]) -> Result<String, serde_json::Error> {
    render_docs_bundle(&DocsBundle {
        schema_version: DOCS_SCHEMA_VERSION,
        modules: modules.to_vec(),
        changelog: None,
        readme: None,
    })
}

/// Serialise an explicit [`DocsBundle`] (modules + changelog +
/// readme). The producer in `collect_workspace_docs` builds the
/// bundle with the narrative fields populated; callers that
/// only have modules in hand can use the thinner
/// [`render_module_docs_json`] above.
pub fn render_docs_bundle(bundle: &DocsBundle) -> Result<String, serde_json::Error> {
    serde_json::to_string(bundle)
}

/// Round-trip the publish-time encoding back into a
/// `Vec<ModuleDoc>`. Used by the runtime side
/// (`register_hatch_docs`) when a bundle ships a `Docs`
/// section so we don't have to re-parse the source.
///
/// Shape-tolerant: accepts both the v2 wrapper object and the
/// legacy v1 top-level array. Older `.hatch` bundles that
/// shipped a `SectionKind::Docs` payload pre-wrapper still
/// decode cleanly — only the modules list survives, narrative
/// fields are silently dropped (they were never there).
pub fn parse_module_docs_json(bytes: &[u8]) -> Result<Vec<ModuleDoc>, serde_json::Error> {
    Ok(parse_docs_bundle(bytes)?.modules)
}

/// Deserialise the full docs payload (modules + narrative).
/// Tries the v2 wrapper first; on shape mismatch falls back to
/// the legacy v1 top-level array and wraps it with empty
/// narrative fields.
pub fn parse_docs_bundle(bytes: &[u8]) -> Result<DocsBundle, serde_json::Error> {
    // Cheap shape sniff: the v2 wrapper starts with `{`, the v1
    // legacy array with `[`. Skip leading ASCII whitespace.
    let first = bytes
        .iter()
        .copied()
        .find(|b| !matches!(*b, b' ' | b'\t' | b'\r' | b'\n'));
    if matches!(first, Some(b'[')) {
        let modules: Vec<ModuleDoc> = serde_json::from_slice(bytes)?;
        return Ok(DocsBundle {
            schema_version: 1,
            modules,
            changelog: None,
            readme: None,
        });
    }
    serde_json::from_slice(bytes)
}

pub fn render_manifest(name: impl Into<String>, modules: &[ModuleDoc]) -> String {
    let pkg = PackageManifest {
        name: name.into(),
        modules: modules
            .iter()
            .map(|m| ModuleEntry {
                name: m.name.clone(),
                summary: m.summary().to_string(),
                class_count: m.classes.len(),
            })
            .collect(),
    };
    serde_json::to_string_pretty(&pkg).expect("PackageManifest JSON serialize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docs::collect_module;

    #[test]
    fn module_docs_round_trip() {
        // Bundle's `Docs` section is a JSON encoding produced
        // by `render_module_docs_json` and parsed at runtime by
        // `parse_module_docs_json`. Round-tripping through that
        // pair must preserve every field (signatures, JSDoc
        // splices, spans).
        let src = r#"
/// Toy class for the round-trip test.
class Toy {
  /// Greet someone by name.
  ///
  /// @param {String} who
  /// @returns {String}
  greet(who) { "hello, " + who }
}
"#;
        let pr = crate::parse::parser::parse(src);
        assert!(pr.errors.is_empty(), "parse errors: {:?}", pr.errors);
        let m = collect_module("toy", src, &pr.module, &pr.docs, &pr.interner);
        let json = render_module_docs_json(std::slice::from_ref(&m)).expect("encode");
        let back = parse_module_docs_json(json.as_bytes()).expect("decode");
        assert_eq!(back.len(), 1);
        let r = &back[0];
        assert_eq!(r.name, "toy");
        assert_eq!(r.classes.len(), 1);
        assert_eq!(r.classes[0].name, "Toy");
        let g = r.classes[0]
            .members
            .iter()
            .find(|mb| mb.name == "greet")
            .unwrap();
        assert_eq!(g.signature, "greet(who: String) → String");
        assert_eq!(g.return_type.as_deref(), Some("String"));
        assert!(!g.doc.is_empty());
    }

    #[test]
    fn docs_bundle_round_trip_with_narrative() {
        // The v2 wrapper carries changelog + readme alongside
        // the module list. Round-trip both through
        // render_docs_bundle / parse_docs_bundle and check the
        // schema_version stamp survives.
        let bundle = DocsBundle {
            schema_version: DOCS_SCHEMA_VERSION,
            modules: Vec::new(),
            changelog: Some("# Changelog\n\n## 0.1.0\n- initial\n".to_string()),
            readme: Some("# Demo\n\nA demo package.\n".to_string()),
        };
        let json = render_docs_bundle(&bundle).expect("encode");
        let back = parse_docs_bundle(json.as_bytes()).expect("decode");
        assert_eq!(back.schema_version, DOCS_SCHEMA_VERSION);
        assert_eq!(
            back.changelog.as_deref(),
            Some("# Changelog\n\n## 0.1.0\n- initial\n")
        );
        assert_eq!(back.readme.as_deref(), Some("# Demo\n\nA demo package.\n"));
    }

    #[test]
    fn parse_docs_bundle_accepts_legacy_array() {
        // Older `.hatch` bundles ship the raw `Vec<ModuleDoc>`
        // array as their Docs section payload. The new
        // parse_docs_bundle must still decode that into a
        // schema_version=1 wrapper with empty narrative fields
        // so register_hatch_docs keeps working.
        let legacy = b"[]";
        let back = parse_docs_bundle(legacy).expect("decode legacy");
        assert_eq!(back.schema_version, 1);
        assert!(back.modules.is_empty());
        assert!(back.changelog.is_none());
        assert!(back.readme.is_none());
    }
}
