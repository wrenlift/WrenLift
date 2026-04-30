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
