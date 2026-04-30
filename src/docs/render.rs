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

/// Serialise every collected [`ModuleDoc`] as a single
/// compact JSON array — the on-wire shape the bundle's
/// `Docs` section carries. Compact (no whitespace) because
/// the bundle is zstd-compressed; readable spelling lives in
/// [`render_module_json`].
///
/// Returns the encoder error verbatim so callers can decide
/// whether to fail the publish or skip the docs section.
pub fn render_module_docs_json(modules: &[ModuleDoc]) -> Result<String, serde_json::Error> {
    serde_json::to_string(modules)
}

/// Round-trip the publish-time encoding back into a
/// `Vec<ModuleDoc>`. Used by the runtime side
/// (`register_hatch_docs`) when a bundle ships a `Docs`
/// section so we don't have to re-parse the source.
pub fn parse_module_docs_json(bytes: &[u8]) -> Result<Vec<ModuleDoc>, serde_json::Error> {
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
        let json = render_module_docs_json(&[m.clone()]).expect("encode");
        let back = parse_module_docs_json(json.as_bytes()).expect("decode");
        assert_eq!(back.len(), 1);
        let r = &back[0];
        assert_eq!(r.name, "toy");
        assert_eq!(r.classes.len(), 1);
        assert_eq!(r.classes[0].name, "Toy");
        let g = r.classes[0].members.iter().find(|mb| mb.name == "greet").unwrap();
        assert_eq!(g.signature, "greet(who: String) → String");
        assert_eq!(g.return_type.as_deref(), Some("String"));
        assert!(!g.doc.is_empty());
    }
}
