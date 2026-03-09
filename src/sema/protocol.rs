/// Protocol system for WrenLift metaclasses.
///
/// Protocols are compile-time contracts that describe a set of required methods.
/// Classes that conform to a protocol gain its provided (default) methods and
/// unlock compiler optimizations like devirtualization and loop specialization.
///
/// This is analogous to Rust's trait system:
/// - `ProtocolDef` ≈ `trait` definition (required + provided methods)
/// - `ProtocolSet` ≈ trait bounds (what a type is known to implement)
/// - Conformance checking ≈ `impl Trait for Type` verification
/// - Auto-derive ≈ `#[derive(Trait)]`
///
/// Both built-in (Sequence, Comparable, Hashable) and user-defined classes
/// participate equally. The compiler uses protocol knowledge for:
/// - Devirtualization: replace dynamic dispatch with direct calls
/// - Loop specialization: inline iterate protocol for known Sequence types
/// - Type folding: `x is Sequence` → true when x's class conforms

// ---------------------------------------------------------------------------
// Protocol identity
// ---------------------------------------------------------------------------

/// A unique identifier for a protocol. Fits in a u8 (max 32 protocols for bitset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProtocolId(pub u8);

// ---------------------------------------------------------------------------
// Built-in protocol IDs
// ---------------------------------------------------------------------------

pub const SEQUENCE: ProtocolId = ProtocolId(0);
pub const COMPARABLE: ProtocolId = ProtocolId(1);
pub const HASHABLE: ProtocolId = ProtocolId(2);
pub const STRINGABLE: ProtocolId = ProtocolId(3);

// ---------------------------------------------------------------------------
// Protocol definition
// ---------------------------------------------------------------------------

/// Defines a protocol: its required methods, default (provided) methods,
/// and what optimizations the compiler may apply when conformance is known.
#[derive(Debug, Clone)]
pub struct ProtocolDef {
    pub id: ProtocolId,
    pub name: &'static str,
    /// Method signatures that a conforming class MUST implement.
    pub required: &'static [&'static str],
    /// Method signatures provided by default (inherited from the protocol).
    /// A conforming class may override these.
    pub provided: &'static [&'static str],
    /// Compiler optimization hints enabled by this protocol.
    pub opts: ProtocolOpts,
}

/// Optimization hints for a protocol.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProtocolOpts {
    /// The iterate loop can be lowered to a counted loop when the
    /// concrete Sequence type has a known length (List, Range, String).
    pub devirtualize_iterate: bool,
    /// Comparison calls can be inlined when both operands conform.
    pub specialize_compare: bool,
    /// Hash computation can be inlined (NaN-box bit hash for primitives).
    pub inline_hash: bool,
    /// toString can be inlined for known types.
    pub inline_to_string: bool,
}

// ---------------------------------------------------------------------------
// Built-in protocol definitions
// ---------------------------------------------------------------------------

pub const SEQUENCE_DEF: ProtocolDef = ProtocolDef {
    id: SEQUENCE,
    name: "Sequence",
    required: &["iterate(_)", "iteratorValue(_)"],
    provided: &[
        "all(_)", "any(_)", "contains(_)", "count", "count(_)",
        "each(_)", "isEmpty", "join()", "join(_)", "toList",
        "reduce(_)", "reduce(_,_)",
    ],
    opts: ProtocolOpts {
        devirtualize_iterate: true,
        specialize_compare: false,
        inline_hash: false,
        inline_to_string: false,
    },
};

pub const COMPARABLE_DEF: ProtocolDef = ProtocolDef {
    id: COMPARABLE,
    name: "Comparable",
    required: &["<(_)", ">(_)", "<=(_)", ">=(_)"],
    provided: &["clamp(_,_)"],
    opts: ProtocolOpts {
        devirtualize_iterate: false,
        specialize_compare: true,
        inline_hash: false,
        inline_to_string: false,
    },
};

pub const HASHABLE_DEF: ProtocolDef = ProtocolDef {
    id: HASHABLE,
    name: "Hashable",
    required: &["hashCode"],
    provided: &[],
    opts: ProtocolOpts {
        devirtualize_iterate: false,
        specialize_compare: false,
        inline_hash: true,
        inline_to_string: false,
    },
};

pub const STRINGABLE_DEF: ProtocolDef = ProtocolDef {
    id: STRINGABLE,
    name: "Stringable",
    required: &["toString"],
    provided: &[],
    opts: ProtocolOpts {
        devirtualize_iterate: false,
        specialize_compare: false,
        inline_hash: false,
        inline_to_string: true,
    },
};

/// All built-in protocols, in order of their IDs.
pub const BUILTIN_PROTOCOLS: &[ProtocolDef] = &[
    SEQUENCE_DEF,
    COMPARABLE_DEF,
    HASHABLE_DEF,
    STRINGABLE_DEF,
];

// ---------------------------------------------------------------------------
// Protocol set (bitset)
// ---------------------------------------------------------------------------

/// A compact bitset tracking which protocols a class conforms to.
/// Supports up to 32 protocols (more than enough for the foreseeable future).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ProtocolSet(pub u32);

impl ProtocolSet {
    pub const EMPTY: ProtocolSet = ProtocolSet(0);

    /// Check if this set contains the given protocol.
    pub const fn has(self, p: ProtocolId) -> bool {
        self.0 & (1 << p.0) != 0
    }

    /// Add a protocol to the set.
    pub const fn with(self, p: ProtocolId) -> ProtocolSet {
        ProtocolSet(self.0 | (1 << p.0))
    }

    /// Union two protocol sets (a class inherits its superclass's protocols).
    pub const fn union(self, other: ProtocolSet) -> ProtocolSet {
        ProtocolSet(self.0 | other.0)
    }

    /// Iterate over the protocol IDs in this set.
    pub fn iter(self) -> impl Iterator<Item = ProtocolId> {
        (0u8..32).filter(move |i| self.0 & (1 << i) != 0).map(ProtocolId)
    }

    /// Is this set empty?
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Number of protocols in the set.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl std::fmt::Display for ProtocolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.iter()
            .filter_map(|pid| BUILTIN_PROTOCOLS.get(pid.0 as usize).map(|p| p.name))
            .collect();
        write!(f, "{{{}}}", names.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Built-in class → protocol mappings
// ---------------------------------------------------------------------------

/// Returns the protocol set for a built-in class by name.
/// This is called during class initialization to pre-populate conformance.
pub fn builtin_protocols_for(class_name: &str) -> ProtocolSet {
    match class_name {
        // Sequence types: conform to Sequence + Stringable
        "String" => ProtocolSet::EMPTY
            .with(SEQUENCE)
            .with(COMPARABLE)
            .with(HASHABLE)
            .with(STRINGABLE),
        "List" => ProtocolSet::EMPTY
            .with(SEQUENCE)
            .with(STRINGABLE),
        "Map" => ProtocolSet::EMPTY
            .with(SEQUENCE)
            .with(STRINGABLE),
        "Range" => ProtocolSet::EMPTY
            .with(SEQUENCE)
            .with(STRINGABLE),
        // Num: comparable, hashable, stringable
        "Num" => ProtocolSet::EMPTY
            .with(COMPARABLE)
            .with(HASHABLE)
            .with(STRINGABLE),
        // Bool: hashable, stringable
        "Bool" => ProtocolSet::EMPTY
            .with(HASHABLE)
            .with(STRINGABLE),
        // Null: stringable
        "Null" => ProtocolSet::EMPTY
            .with(STRINGABLE),
        // Fn: no protocols
        "Fn" | "Fiber" | "System" => ProtocolSet::EMPTY,
        // Object: base, stringable (toString is defined on Object)
        "Object" => ProtocolSet::EMPTY
            .with(STRINGABLE)
            .with(HASHABLE),
        // Sequence itself: conformance comes from Sequence protocol
        "Sequence" => ProtocolSet::EMPTY
            .with(SEQUENCE),
        // Unknown: empty
        _ => ProtocolSet::EMPTY,
    }
}

// ---------------------------------------------------------------------------
// Conformance checking
// ---------------------------------------------------------------------------

/// Check whether a set of method signatures satisfies a protocol's requirements.
/// Returns a list of missing required methods, or empty if satisfied.
pub fn check_conformance(
    protocol: &ProtocolDef,
    methods: &[&str],
) -> Vec<&'static str> {
    protocol.required.iter()
        .filter(|req| !methods.contains(req))
        .copied()
        .collect()
}

/// Result of checking protocol conformance for a class.
#[derive(Debug, Clone)]
pub struct ConformanceResult {
    /// Protocols the class fully satisfies.
    pub conforms: ProtocolSet,
    /// Protocols that are partially satisfied (some methods missing).
    pub partial: Vec<(ProtocolId, Vec<&'static str>)>,
}

/// Check which protocols a class conforms to based on its method signatures,
/// including methods inherited from its superclass's protocol set.
pub fn check_all_conformance(
    methods: &[&str],
    superclass_protocols: ProtocolSet,
) -> ConformanceResult {
    let mut conforms = superclass_protocols;
    let mut partial = Vec::new();

    for proto in BUILTIN_PROTOCOLS {
        if conforms.has(proto.id) {
            continue; // Already inherited
        }
        let missing = check_conformance(proto, methods);
        if missing.is_empty() {
            conforms = conforms.with(proto.id);
        } else if missing.len() < proto.required.len() {
            partial.push((proto.id, missing));
        }
    }

    ConformanceResult { conforms, partial }
}

// ---------------------------------------------------------------------------
// Auto-derive support
// ---------------------------------------------------------------------------

/// Which protocols can be auto-derived for a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoDerive {
    /// Hashable: auto-generate hashCode from identity hash (default).
    Hashable,
    /// Stringable: auto-generate toString as "instance of ClassName".
    Stringable,
}

/// Determine which protocols can be auto-derived for a class.
/// Auto-derivation generates default method implementations at compile time.
pub fn auto_derivable(
    class_name: &str,
    current_methods: &[&str],
    superclass_protocols: ProtocolSet,
) -> Vec<AutoDerive> {
    let mut derivable = Vec::new();

    // Stringable: auto-derive if no explicit toString
    if !superclass_protocols.has(STRINGABLE)
        && !current_methods.contains(&"toString")
    {
        derivable.push(AutoDerive::Stringable);
    }

    // Hashable: auto-derive identity hash if no explicit hashCode
    if !superclass_protocols.has(HASHABLE)
        && !current_methods.contains(&"hashCode")
    {
        derivable.push(AutoDerive::Hashable);
    }

    let _ = class_name; // May be used for more complex derivation rules later
    derivable
}

// ---------------------------------------------------------------------------
// InferredType → ProtocolSet mapping
// ---------------------------------------------------------------------------

use crate::sema::types::InferredType;

/// Given a statically known InferredType, return the protocols it conforms to.
/// This is the bridge between type inference and protocol-based optimization.
pub fn protocols_of_type(ty: &InferredType) -> ProtocolSet {
    match ty {
        InferredType::Num => builtin_protocols_for("Num"),
        InferredType::Bool => builtin_protocols_for("Bool"),
        InferredType::Null => builtin_protocols_for("Null"),
        InferredType::String => builtin_protocols_for("String"),
        InferredType::List => builtin_protocols_for("List"),
        InferredType::Map => builtin_protocols_for("Map"),
        InferredType::Range => builtin_protocols_for("Range"),
        InferredType::Fn => ProtocolSet::EMPTY,
        InferredType::Class(_) => {
            // For user-defined classes, we'd need to look up their ClassMir.
            // For now, all objects get Object's protocols.
            builtin_protocols_for("Object")
        }
        InferredType::Any => ProtocolSet::EMPTY,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_set_basic() {
        let empty = ProtocolSet::EMPTY;
        assert!(empty.is_empty());
        assert_eq!(empty.count(), 0);
        assert!(!empty.has(SEQUENCE));

        let seq = empty.with(SEQUENCE);
        assert!(!seq.is_empty());
        assert_eq!(seq.count(), 1);
        assert!(seq.has(SEQUENCE));
        assert!(!seq.has(COMPARABLE));
    }

    #[test]
    fn test_protocol_set_union() {
        let a = ProtocolSet::EMPTY.with(SEQUENCE);
        let b = ProtocolSet::EMPTY.with(COMPARABLE).with(HASHABLE);
        let combined = a.union(b);

        assert!(combined.has(SEQUENCE));
        assert!(combined.has(COMPARABLE));
        assert!(combined.has(HASHABLE));
        assert!(!combined.has(STRINGABLE));
        assert_eq!(combined.count(), 3);
    }

    #[test]
    fn test_protocol_set_iter() {
        let set = ProtocolSet::EMPTY
            .with(SEQUENCE)
            .with(HASHABLE);
        let ids: Vec<ProtocolId> = set.iter().collect();
        assert_eq!(ids, vec![SEQUENCE, HASHABLE]);
    }

    #[test]
    fn test_protocol_set_display() {
        let set = ProtocolSet::EMPTY.with(SEQUENCE).with(COMPARABLE);
        let s = format!("{}", set);
        assert!(s.contains("Sequence"));
        assert!(s.contains("Comparable"));
    }

    #[test]
    fn test_builtin_protocols_for_string() {
        let ps = builtin_protocols_for("String");
        assert!(ps.has(SEQUENCE));
        assert!(ps.has(COMPARABLE));
        assert!(ps.has(HASHABLE));
        assert!(ps.has(STRINGABLE));
    }

    #[test]
    fn test_builtin_protocols_for_num() {
        let ps = builtin_protocols_for("Num");
        assert!(!ps.has(SEQUENCE));
        assert!(ps.has(COMPARABLE));
        assert!(ps.has(HASHABLE));
        assert!(ps.has(STRINGABLE));
    }

    #[test]
    fn test_builtin_protocols_for_list() {
        let ps = builtin_protocols_for("List");
        assert!(ps.has(SEQUENCE));
        assert!(!ps.has(COMPARABLE));
        assert!(ps.has(STRINGABLE));
    }

    #[test]
    fn test_check_conformance_full() {
        let methods = &["iterate(_)", "iteratorValue(_)"];
        let missing = check_conformance(&SEQUENCE_DEF, methods);
        assert!(missing.is_empty());
    }

    #[test]
    fn test_check_conformance_partial() {
        let methods: &[&str] = &["iterate(_)"];
        let missing = check_conformance(&SEQUENCE_DEF, methods);
        assert_eq!(missing, vec!["iteratorValue(_)"]);
    }

    #[test]
    fn test_check_conformance_empty() {
        let methods: &[&str] = &[];
        let missing = check_conformance(&SEQUENCE_DEF, methods);
        assert_eq!(missing.len(), 2);
    }

    #[test]
    fn test_check_all_conformance_inherits() {
        let methods = &["iterate(_)", "iteratorValue(_)", "toString"];
        let result = check_all_conformance(methods, ProtocolSet::EMPTY);
        assert!(result.conforms.has(SEQUENCE));
        assert!(result.conforms.has(STRINGABLE));
    }

    #[test]
    fn test_check_all_conformance_superclass() {
        // Superclass already has Sequence; don't re-check
        let methods: &[&str] = &[];
        let super_protos = ProtocolSet::EMPTY.with(SEQUENCE);
        let result = check_all_conformance(methods, super_protos);
        assert!(result.conforms.has(SEQUENCE));
    }

    #[test]
    fn test_auto_derivable() {
        let methods: &[&str] = &["iterate(_)", "iteratorValue(_)"];
        let derivable = auto_derivable("MySeq", methods, ProtocolSet::EMPTY);
        assert!(derivable.contains(&AutoDerive::Stringable));
        assert!(derivable.contains(&AutoDerive::Hashable));
    }

    #[test]
    fn test_auto_derivable_already_has() {
        let methods: &[&str] = &["toString", "hashCode"];
        let derivable = auto_derivable("Foo", methods, ProtocolSet::EMPTY);
        assert!(derivable.is_empty());
    }

    #[test]
    fn test_protocols_of_inferred_type() {
        let ps = protocols_of_type(&InferredType::Num);
        assert!(ps.has(COMPARABLE));
        assert!(ps.has(HASHABLE));
        assert!(!ps.has(SEQUENCE));

        let ps = protocols_of_type(&InferredType::List);
        assert!(ps.has(SEQUENCE));

        let ps = protocols_of_type(&InferredType::Any);
        assert!(ps.is_empty());
    }
}
