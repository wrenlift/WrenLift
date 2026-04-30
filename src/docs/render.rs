//! Render a [`ModuleDoc`] to a self-contained HTML page.
//!
//! v0 keeps styling minimal — same parchment palette as
//! `wrenlift.com`, no sidebar nav, no cross-package linking. The
//! goal is a readable artefact you can open in a browser today;
//! polish + indices land in v1.

use pulldown_cmark::{html, Options, Parser};

use super::model::{ClassDoc, MemberDoc, MemberKind, ModuleDoc};

const CSS: &str = r#"
body {
  font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
  max-width: 760px; margin: 40px auto; padding: 0 24px;
  background: #f1e3cc; color: #2a1f12;
}
h1 { font: 800 26px ui-sans-serif, system-ui, sans-serif; margin: 0 0 18px; letter-spacing: -0.01em; }
h2 { font: 700 20px ui-sans-serif, system-ui, sans-serif; margin: 32px 0 6px; letter-spacing: -0.005em; color: #1d1812; }
h3 { font: 600 14px ui-monospace, monospace; margin: 18px 0 4px; color: #1d1812; }
p { margin: 6px 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
code { background: rgba(58,42,24,0.07); padding: 1.5px 6px; border-radius: 4px; }
pre {
  background: #1d1812; color: #e8dcc8;
  border-radius: 10px; padding: 14px 18px; overflow-x: auto;
}
pre code { background: transparent; padding: 0; }
a { color: #c9a878; }
.member { border-top: 1px dashed rgba(58,42,24,0.25); padding: 10px 0 4px; }
.member .sig {
  display: inline-block; background: #1d1812; color: #e8dcc8;
  padding: 4px 10px; border-radius: 6px; font-size: 0.92em;
}
.kind {
  display: inline-block; font-size: 10px; letter-spacing: 0.04em;
  text-transform: uppercase; color: #6d6354; margin-left: 8px;
}
.summary { color: #4a3a26; }
.module-doc { background: rgba(255,255,255,0.4); padding: 14px 18px; border-radius: 10px; }
"#;

pub fn render_module_html(module: &ModuleDoc) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("<!doctype html>\n<html><head><meta charset=\"utf-8\">");
    out.push_str("<title>");
    push_escaped(&mut out, &module.name);
    out.push_str("</title><style>");
    out.push_str(CSS);
    out.push_str("</style></head><body>");

    out.push_str("<h1>");
    push_escaped(&mut out, &module.name);
    out.push_str("</h1>");

    if !module.doc.is_empty() {
        out.push_str("<div class=\"module-doc\">");
        push_markdown(&mut out, &module.doc);
        out.push_str("</div>");
    }

    for class in &module.classes {
        render_class(&mut out, class);
    }

    out.push_str("</body></html>");
    out
}

fn render_class(out: &mut String, class: &ClassDoc) {
    out.push_str("<h2 id=\"class-");
    push_escaped(out, &class.name);
    out.push_str("\">class ");
    push_escaped(out, &class.name);
    out.push_str("</h2>");

    if !class.doc.is_empty() {
        push_markdown(out, &class.doc);
    } else {
        out.push_str("<p class=\"summary\"><em>No class-level docs.</em></p>");
    }

    for member in &class.members {
        render_member(out, &class.name, member);
    }
}

fn render_member(out: &mut String, class_name: &str, member: &MemberDoc) {
    out.push_str("<div class=\"member\">");
    out.push_str("<h3 id=\"");
    push_escaped(out, class_name);
    out.push('-');
    push_escaped(out, &member.name);
    out.push_str("\"><span class=\"sig\">");
    push_escaped(out, &member.signature);
    out.push_str("</span><span class=\"kind\">");
    out.push_str(kind_label(member.kind));
    out.push_str("</span></h3>");
    if !member.doc.is_empty() {
        push_markdown(out, &member.doc);
    } else {
        out.push_str("<p class=\"summary\"><em>No docs.</em></p>");
    }
    out.push_str("</div>");
}

fn kind_label(kind: MemberKind) -> &'static str {
    match kind {
        MemberKind::Method => "method",
        MemberKind::StaticMethod => "static method",
        MemberKind::Getter => "getter",
        MemberKind::Setter => "setter",
        MemberKind::Constructor => "constructor",
        MemberKind::Field => "field",
    }
}

fn push_markdown(out: &mut String, src: &str) {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_FOOTNOTES);
    let parser = Parser::new_ext(src, opts);
    html::push_html(out, parser);
}

fn push_escaped(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
}
