//! wlift::docs — doc generator for hatch packages.
//!
//! Pipeline (top of the
//! [docs plan](https://github.com/wrenlift/WrenLift/blob/main/docs/hatch-docs-generator-plan.md)):
//!
//! ```text
//! source files ─► parse + sema ─► collector ─► doc model ─► renderer
//!                                                    │
//!                                                    └─► JSON / HTML
//! ```
//!
//! Lives inside `wren_lift` (rather than a sibling crate) so the CLI
//! can call it without inducing a cargo cycle through `wren_lift`'s
//! parser. Both the `wlift --docs` subcommand and the future LSP
//! hover handler pull from here.

pub mod collect;
pub mod model;
pub mod render;

pub use collect::collect_module;
pub use model::{ClassDoc, MemberDoc, MemberKind, ModuleDoc};
pub use render::render_module_html;
