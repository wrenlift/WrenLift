//! `#export`: the signature a member exposes to other languages, and the
//! types the compiler takes from it.
//!
//! ```wren
//! #export = "add(n: Num) -> Num"
//! add(n) { _score = _score + n }
//! ```
//!
//! The value names the member as other languages see it (it may differ
//! from the Wren name), gives one entry per Wren parameter, `name`,
//! `name: Type` or `_: Type` (`_` keeps the source's name), and `-> Type`
//! for the result. A getter is `name -> Type`, a setter `name=(v: Type)`.
//! Types are `Num`, `Bool`, `String`, `Null`, `List`, `Fn`, a function
//! shape `Fn(Num, Hud) -> Bool`, or a class name; anything else is
//! dynamic. Parameters match by position, so a running class, which has
//! no parameter names, reads the attribute the way the source does.
//!
//! A declared `Num`, `Bool` or `String` is a contract: the member checks
//! it on entry and on return, raising when it does not hold, and the
//! compiler uses what it then knows. The other types describe values
//! another language may hand over in its own form, so they are not
//! checked.

use crate::ast::{Attribute, AttributeBody, AttributeLiteral, MethodSig};
use crate::intern::Interner;
use crate::mir::{AttrEntry, AttrValue};

/// The attribute carrying the exported signature.
pub const EXPORT: &str = "export";

/// One parameter of an exported signature: its name, unless `_`, and its
/// type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportParam {
    pub name: Option<String>,
    pub ty: Option<String>,
}

/// An exported signature, parsed.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub params: Vec<ExportParam>,
    /// Whether the signature has a parameter list at all: `count` against
    /// `count()`.
    pub has_params: bool,
    /// Whether it is a setter, `name=(v)`.
    pub is_setter: bool,
    pub ret: Option<String>,
}

/// A declared type the member checks and the compiler relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Checked {
    Num,
    Bool,
    String,
}

impl Checked {
    /// The checked type a type name declares, if any.
    pub fn of(name: &str) -> Option<Checked> {
        match name.trim() {
            "Num" => Some(Checked::Num),
            "Bool" => Some(Checked::Bool),
            "String" => Some(Checked::String),
            _ => None,
        }
    }

    /// The class `is` tests for it.
    pub fn class_name(self) -> &'static str {
        match self {
            Checked::Num => "Num",
            Checked::Bool => "Bool",
            Checked::String => "String",
        }
    }
}

/// The index just past the `)` matching the `(` at `open`, if any.
fn close_of(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in text[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// `text` split at the commas outside parentheses.
fn split_top(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl Export {
    /// Parse `name(a: T, b) -> R`, `name -> R` or `name=(v: T)`. A type
    /// may be a function's, `Fn(T, U) -> R`, so the parameter list is the
    /// first balanced one and the result what follows its `->`.
    pub fn parse(text: &str) -> Result<Export, String> {
        let text = text.trim();
        let (name, params, has_params, is_setter, rest) = if let Some(i) = text.find("=(") {
            let close = close_of(text, i + 1)
                .ok_or_else(|| format!("`{text}`: unclosed parameter list"))?;
            (
                &text[..i],
                &text[i + 2..close - 1],
                true,
                true,
                &text[close..],
            )
        } else if let Some(i) = text.find('(')
            && text.find("->").is_none_or(|arrow| i < arrow)
        {
            let close =
                close_of(text, i).ok_or_else(|| format!("`{text}`: unclosed parameter list"))?;
            (
                &text[..i],
                &text[i + 1..close - 1],
                true,
                false,
                &text[close..],
            )
        } else {
            match text.find("->") {
                Some(arrow) => (&text[..arrow], "", false, false, &text[arrow..]),
                None => (text, "", false, false, ""),
            }
        };
        let rest = rest.trim();
        let ret = match rest.strip_prefix("->") {
            Some(r) => Some(r.trim()),
            None if rest.is_empty() => None,
            None => return Err(format!("`{text}`: `{rest}` after the parameter list")),
        };
        if ret == Some("") {
            return Err(format!("`{text}`: nothing after `->`"));
        }
        let name = name.trim();
        if !is_identifier(name) {
            return Err(format!("`{text}`: `{name}` is not a name"));
        }
        let mut out = Vec::new();
        if !params.trim().is_empty() {
            for p in split_top(params) {
                let (pname, ty) = match p.split_once(':') {
                    Some((n, t)) => (n.trim(), Some(t.trim())),
                    None => (p.trim(), None),
                };
                if pname != "_" && !is_identifier(pname) {
                    return Err(format!("`{text}`: `{pname}` is not a parameter name"));
                }
                if ty == Some("") {
                    return Err(format!("`{text}`: `{pname}` has no type after `:`"));
                }
                out.push(ExportParam {
                    name: (pname != "_").then(|| pname.to_owned()),
                    ty: ty.map(str::to_owned),
                });
            }
        }
        if is_setter && out.len() != 1 {
            return Err(format!("`{text}`: a setter takes one parameter"));
        }
        Ok(Export {
            name: name.to_owned(),
            params: out,
            has_params,
            is_setter,
            ret: ret.map(str::to_owned),
        })
    }

    /// The export among the entries the VM keeps for a method.
    pub fn from_entries(entries: &[AttrEntry]) -> Result<Option<Export>, String> {
        for e in entries {
            if e.group.is_none() && e.key == EXPORT {
                return match &e.value {
                    Some(AttrValue::Str(s)) => Export::parse(s).map(Some),
                    _ => Err(format!("`#{EXPORT}` takes a signature string")),
                };
            }
        }
        Ok(None)
    }

    /// The export among a method's attributes in source, with the span
    /// of the attribute for a diagnostic.
    pub fn from_ast(
        attrs: &[Attribute],
        interner: &Interner,
    ) -> Option<(Result<Export, String>, crate::ast::Span)> {
        let a = attrs
            .iter()
            .find(|a| a.is_runtime && interner.resolve(a.name.0) == EXPORT)?;
        let parsed = match &a.body {
            AttributeBody::Value((AttributeLiteral::Str(s), _)) => Export::parse(s),
            _ => Err(format!("`#{EXPORT}` takes a signature string")),
        };
        Some((parsed, a.span.clone()))
    }

    /// Whether the signature fits the member it is on: the same shape
    /// and one entry per parameter. Operators and subscripts are not
    /// exported.
    pub fn check_member(&self, sig: &MethodSig) -> Result<(), String> {
        let (shape, arity) = match sig {
            MethodSig::Named { params, .. } | MethodSig::Construct { params, .. } => {
                ("a method", params.len())
            }
            MethodSig::Getter(_) => ("a getter", 0),
            MethodSig::Setter { .. } => ("a setter", 1),
            MethodSig::Operator { .. }
            | MethodSig::Subscript { .. }
            | MethodSig::SubscriptSetter { .. } => {
                return Err("operators and subscripts are not exported".to_string());
            }
        };
        let declared = if self.is_setter {
            "a setter"
        } else if self.has_params {
            "a method"
        } else {
            "a getter"
        };
        if declared != shape {
            return Err(format!(
                "`{}` declares {declared}, but the member is {shape}",
                self.name
            ));
        }
        if self.params.len() != arity {
            return Err(format!(
                "`{}` declares {} parameter(s), but the member takes {arity}",
                self.name,
                self.params.len()
            ));
        }
        Ok(())
    }

    /// The checked type of the parameter at `index`.
    pub fn checked_param(&self, index: usize) -> Option<Checked> {
        self.params
            .get(index)
            .and_then(|p| p.ty.as_deref())
            .and_then(Checked::of)
    }

    /// The checked type of the result.
    pub fn checked_ret(&self) -> Option<Checked> {
        self.ret.as_deref().and_then(Checked::of)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signatures_parse_into_name_parameters_and_result() {
        let e = Export::parse("hit(n: Num, other: Hud) -> Bool").unwrap();
        assert_eq!(e.name, "hit");
        assert_eq!(e.params.len(), 2);
        assert_eq!(e.params[0].name.as_deref(), Some("n"));
        assert_eq!(e.params[1].ty.as_deref(), Some("Hud"));
        assert_eq!(e.ret.as_deref(), Some("Bool"));
        assert!(e.has_params && !e.is_setter);
        assert_eq!(e.checked_param(0), Some(Checked::Num));
        assert_eq!(e.checked_param(1), None);
        assert_eq!(e.checked_ret(), Some(Checked::Bool));

        let e = Export::parse("score -> Num").unwrap();
        assert!(!e.has_params && e.params.is_empty());
        let e = Export::parse("adder() -> Fn(Num) -> Num").unwrap();
        assert!(e.has_params && e.params.is_empty());
        assert_eq!(e.ret.as_deref(), Some("Fn(Num) -> Num"));
        assert_eq!(e.checked_ret(), None);
        let e = Export::parse("each(f: Fn(Hud, Num), n: Num)").unwrap();
        assert_eq!(e.params.len(), 2);
        assert_eq!(e.checked_param(1), Some(Checked::Num));
        let e = Export::parse("score=(v: Num)").unwrap();
        assert!(e.is_setter && e.params[0].ty.as_deref() == Some("Num"));
        let e = Export::parse("f(_: Num, b)").unwrap();
        assert_eq!(e.params[0].name, None);
        assert_eq!(
            e.params[1],
            ExportParam {
                name: Some("b".into()),
                ty: None
            }
        );
        assert_eq!(Export::parse("count()").unwrap().params.len(), 0);

        for bad in ["", "1abc()", "f(a:)", "f(a", "s=(a, b)", "f() ->"] {
            assert!(Export::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_signature_must_fit_its_member() {
        let parsed = crate::parse::parser::parse(
            "class C {\n  a(x, y) { x }\n  b { 1 }\n  c=(v) { v }\n  +(o) { o }\n}\n",
        );
        let crate::ast::Stmt::Class(class) = &parsed.module[0].0 else {
            panic!("a class");
        };
        let sig = |i: usize| &class.methods[i].0.signature;
        assert!(
            Export::parse("add(x: Num, y)")
                .unwrap()
                .check_member(sig(0))
                .is_ok()
        );
        assert!(
            Export::parse("add(x)")
                .unwrap()
                .check_member(sig(0))
                .is_err()
        );
        assert!(
            Export::parse("b -> Num")
                .unwrap()
                .check_member(sig(1))
                .is_ok()
        );
        assert!(
            Export::parse("b() -> Num")
                .unwrap()
                .check_member(sig(1))
                .is_err()
        );
        assert!(
            Export::parse("c=(v: Num)")
                .unwrap()
                .check_member(sig(2))
                .is_ok()
        );
        assert!(Export::parse("c(v)").unwrap().check_member(sig(2)).is_err());
        assert!(
            Export::parse("plus(o)")
                .unwrap()
                .check_member(sig(3))
                .is_err()
        );
    }

    #[test]
    fn the_attribute_is_read_from_source_and_from_the_runtime_entries() {
        let source = "class Hud {\n  #export = \"add(n: Num) -> Num\"\n  hit(n) { n }\n}\n";
        let parsed = crate::parse::parser::parse(source);
        let crate::ast::Stmt::Class(class) = &parsed.module[0].0 else {
            panic!("a class");
        };
        let (e, _) =
            Export::from_ast(&class.methods[0].0.attributes, &parsed.interner).expect("exported");
        let e = e.unwrap();
        assert_eq!(e.name, "add");
        let entries = vec![AttrEntry {
            group: None,
            key: EXPORT.to_owned(),
            value: Some(AttrValue::Str("add(n: Num) -> Num".to_owned())),
        }];
        assert_eq!(Export::from_entries(&entries).unwrap(), Some(e));
        assert_eq!(Export::from_entries(&[]).unwrap(), None);
    }
}
