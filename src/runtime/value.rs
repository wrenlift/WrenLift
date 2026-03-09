/// NaN-boxed 64-bit value representation for the Wren runtime.
///
/// IEEE 754 double-precision NaN has the form:
///   sign(1) | exponent(11, all 1s) | quiet(1) | payload(51)
///
/// We use quiet NaN payloads to encode non-number values:
///
/// ```text
/// Normal f64:  any bit pattern that is NOT a quiet NaN with our tag
/// Null:        0x7FFC_0000_0000_0000
/// False:       0x7FFC_0000_0000_0001
/// True:        0x7FFC_0000_0000_0002
/// Undefined:   0x7FFC_0000_0000_0003  (internal sentinel)
/// Object ptr:  0xFFFC_0000_0000_0000 | (ptr & 0x0000_FFFF_FFFF_FFFF)
/// ```
use std::fmt;

// Tag constants
const QNAN: u64 = 0x7FFC_0000_0000_0000;
const TAG_NULL: u64 = QNAN;
const TAG_FALSE: u64 = QNAN | 1;
const TAG_TRUE: u64 = QNAN | 2;
const TAG_UNDEFINED: u64 = QNAN | 3;

/// Sign bit set + QNAN = object pointer tag.
const SIGN_BIT: u64 = 1 << 63;
const TAG_OBJ: u64 = SIGN_BIT | QNAN;

/// Mask for extracting the 48-bit pointer from a tagged object.
const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// A NaN-boxed Wren value. Exactly 8 bytes, `Copy`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Value(u64);

impl Value {
    // -- Constructors -------------------------------------------------------

    /// Box an f64 number.
    #[inline(always)]
    pub fn num(n: f64) -> Self {
        Value(n.to_bits())
    }

    /// The `null` value.
    #[inline(always)]
    pub fn null() -> Self {
        Value(TAG_NULL)
    }

    /// Box a boolean.
    #[inline(always)]
    pub fn bool(b: bool) -> Self {
        if b { Value(TAG_TRUE) } else { Value(TAG_FALSE) }
    }

    /// The internal undefined sentinel (never exposed to user code).
    #[inline(always)]
    pub fn undefined() -> Self {
        Value(TAG_UNDEFINED)
    }

    /// Box a heap object pointer.
    ///
    /// # Safety
    /// The pointer must be a valid, non-null, GC-managed object pointer.
    /// Only the lower 48 bits are stored; the upper 16 must be sign-extension
    /// (which is true for all userspace pointers on x86_64 and aarch64).
    #[inline(always)]
    pub fn object(ptr: *mut u8) -> Self {
        let addr = ptr as u64;
        debug_assert!(
            addr & !PTR_MASK == 0 || addr & !PTR_MASK == !PTR_MASK,
            "pointer has non-canonical upper bits"
        );
        Value(TAG_OBJ | (addr & PTR_MASK))
    }

    // -- Type checks --------------------------------------------------------

    /// Is this value a number (any f64, including NaN)?
    #[inline(always)]
    pub fn is_num(self) -> bool {
        // A value is a number if it's NOT one of our tagged values.
        // Our tagged values all have the QNAN bits set AND are either
        // a singleton (null/false/true/undefined) or an object pointer.
        (self.0 & QNAN) != QNAN || self.is_num_nan()
    }

    /// Check if this is an actual f64 NaN (not one of our tag patterns).
    #[inline(always)]
    fn is_num_nan(self) -> bool {
        // f64 NaN: exponent all 1s, mantissa non-zero, but NOT matching our
        // specific tag patterns. IEEE 754 NaN has many bit patterns; ours
        // specifically use QNAN (0x7FFC...) base. A "real" f64 NaN from
        // arithmetic is typically 0x7FF8_0000_0000_0000 (canonical qNaN)
        // which does NOT have bit 50 set, so (val & QNAN) != QNAN.
        // Edge case: user creates f64::NAN — its bits are 0x7FF8..., safe.
        false // handled by is_num's primary check
    }

    /// Is this `null`?
    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0 == TAG_NULL
    }

    /// Is this a boolean (`true` or `false`)?
    #[inline(always)]
    pub fn is_bool(self) -> bool {
        self.0 == TAG_TRUE || self.0 == TAG_FALSE
    }

    /// Is this a heap object pointer?
    #[inline(always)]
    pub fn is_object(self) -> bool {
        (self.0 & TAG_OBJ) == TAG_OBJ
    }

    /// Is this the internal undefined sentinel?
    #[inline(always)]
    pub fn is_undefined(self) -> bool {
        self.0 == TAG_UNDEFINED
    }

    /// Is this value falsy? In Wren, only `false` and `null` are falsy.
    #[inline(always)]
    pub fn is_falsy(self) -> bool {
        self.0 == TAG_FALSE || self.0 == TAG_NULL
    }

    // -- Extractors ---------------------------------------------------------

    /// Extract the f64 if this is a number.
    #[inline(always)]
    pub fn as_num(self) -> Option<f64> {
        if self.is_num() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    /// Extract the boolean if this is a bool.
    #[inline(always)]
    pub fn as_bool(self) -> Option<bool> {
        if self.0 == TAG_TRUE {
            Some(true)
        } else if self.0 == TAG_FALSE {
            Some(false)
        } else {
            None
        }
    }

    /// Extract the object pointer if this is an object.
    #[inline(always)]
    pub fn as_object(self) -> Option<*mut u8> {
        if self.is_object() {
            // Sign-extend from 48 bits for canonical pointer form.
            let raw = self.0 & PTR_MASK;
            // If bit 47 is set, sign-extend upper bits (kernel pointers).
            let ptr = if raw & (1 << 47) != 0 {
                raw | !PTR_MASK
            } else {
                raw
            };
            Some(ptr as *mut u8)
        } else {
            None
        }
    }

    /// Get the raw u64 bit representation.
    #[inline(always)]
    pub fn to_bits(self) -> u64 {
        self.0
    }

    /// Reconstruct a Value from its raw bit representation.
    #[inline(always)]
    pub fn from_bits(bits: u64) -> Self {
        Value(bits)
    }

    /// Returns true if this is an object of type String.
    #[inline]
    pub fn is_string_object(self) -> bool {
        if let Some(ptr) = self.as_object() {
            unsafe { (*(ptr as *const super::object::ObjHeader)).obj_type == super::object::ObjType::String }
        } else {
            false
        }
    }

    // -- Equality -----------------------------------------------------------

    /// Wren value equality. Numbers use IEEE equality (NaN != NaN).
    /// Strings compare by content. Other objects use pointer identity.
    #[inline]
    pub fn equals(self, other: Value) -> bool {
        if self.0 == other.0 {
            // Fast path: identical bits (same pointer, same bool, same null, or same num bits).
            // NaN != NaN in Wren, but two NaN bit patterns can't be equal here
            // unless they're the exact same NaN bits, which we allow.
            return !self.is_num() || !f64::from_bits(self.0).is_nan();
        }
        if self.is_num() && other.is_num() {
            let a = f64::from_bits(self.0);
            let b = f64::from_bits(other.0);
            return a == b;
        }
        // Different object pointers: compare string contents
        if self.is_object() && other.is_object() {
            let pa = self.as_object().unwrap();
            let pb = other.as_object().unwrap();
            unsafe {
                let ha = &*(pa as *const super::object::ObjHeader);
                let hb = &*(pb as *const super::object::ObjHeader);
                if ha.obj_type == super::object::ObjType::String
                    && hb.obj_type == super::object::ObjType::String
                {
                    let sa = &*(pa as *const super::object::ObjString);
                    let sb = &*(pb as *const super::object::ObjString);
                    return sa.value == sb.value;
                }
            }
        }
        false
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "null")
        } else if let Some(b) = self.as_bool() {
            write!(f, "{}", b)
        } else if self.is_undefined() {
            write!(f, "<undefined>")
        } else if self.is_object() {
            write!(f, "<object {:p}>", self.as_object().unwrap())
        } else if let Some(n) = self.as_num() {
            write!(f, "{}", n)
        } else {
            write!(f, "<unknown 0x{:016X}>", self.0)
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.equals(*other)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_size() {
        assert_eq!(std::mem::size_of::<Value>(), 8);
    }

    #[test]
    fn test_value_num_zero() {
        let v = Value::num(0.0);
        assert!(v.is_num());
        assert_eq!(v.as_num(), Some(0.0));
        assert!(!v.is_null());
        assert!(!v.is_bool());
        assert!(!v.is_object());
    }

    #[test]
    fn test_value_num_positive() {
        let v = Value::num(42.5);
        assert!(v.is_num());
        assert_eq!(v.as_num(), Some(42.5));
    }

    #[test]
    fn test_value_num_negative() {
        let v = Value::num(-1.0);
        assert!(v.is_num());
        assert_eq!(v.as_num(), Some(-1.0));
    }

    #[test]
    fn test_value_num_infinity() {
        let v = Value::num(f64::INFINITY);
        assert!(v.is_num());
        assert_eq!(v.as_num(), Some(f64::INFINITY));

        let v = Value::num(f64::NEG_INFINITY);
        assert!(v.is_num());
        assert_eq!(v.as_num(), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn test_value_num_nan() {
        let v = Value::num(f64::NAN);
        assert!(v.is_num());
        assert!(v.as_num().unwrap().is_nan());
    }

    #[test]
    fn test_value_null() {
        let v = Value::null();
        assert!(v.is_null());
        assert!(!v.is_num());
        assert!(!v.is_bool());
        assert!(!v.is_object());
        assert_eq!(v.as_num(), None);
        assert_eq!(v.as_bool(), None);
    }

    #[test]
    fn test_value_true() {
        let v = Value::bool(true);
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(true));
        assert!(!v.is_num());
        assert!(!v.is_null());
    }

    #[test]
    fn test_value_false() {
        let v = Value::bool(false);
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(false));
        assert!(!v.is_num());
        assert!(!v.is_null());
    }

    #[test]
    fn test_value_falsy_null() {
        assert!(Value::null().is_falsy());
    }

    #[test]
    fn test_value_falsy_false() {
        assert!(Value::bool(false).is_falsy());
    }

    #[test]
    fn test_value_truthy_true() {
        assert!(!Value::bool(true).is_falsy());
    }

    #[test]
    fn test_value_truthy_zero() {
        // In Wren, 0 is truthy!
        assert!(!Value::num(0.0).is_falsy());
    }

    #[test]
    fn test_value_truthy_object() {
        let mut data: u64 = 0xDEAD;
        let ptr = &mut data as *mut u64 as *mut u8;
        let v = Value::object(ptr);
        assert!(!v.is_falsy());
    }

    #[test]
    fn test_value_object_roundtrip() {
        let mut data: u64 = 0xCAFE_BABE;
        let ptr = &mut data as *mut u64 as *mut u8;
        let v = Value::object(ptr);
        assert!(v.is_object());
        assert!(!v.is_num());
        assert!(!v.is_null());
        assert!(!v.is_bool());
        let recovered = v.as_object().unwrap();
        assert_eq!(recovered, ptr);
    }

    #[test]
    fn test_value_object_high_bits() {
        // Test pointer with high bits within 48-bit range
        let mut data: u64 = 0;
        let ptr = &mut data as *mut u64 as *mut u8;
        let v = Value::object(ptr);
        assert!(v.is_object());
        assert_eq!(v.as_object().unwrap(), ptr);
    }

    #[test]
    fn test_value_type_discrimination() {
        let num = Value::num(1.0);
        let null = Value::null();
        let t = Value::bool(true);
        let f = Value::bool(false);
        let undef = Value::undefined();
        let mut data: u64 = 0;
        let obj = Value::object(&mut data as *mut u64 as *mut u8);

        // num
        assert!(num.is_num());
        assert!(!num.is_null());
        assert!(!num.is_bool());
        assert!(!num.is_object());
        assert!(!num.is_undefined());

        // null
        assert!(!null.is_num());
        assert!(null.is_null());
        assert!(!null.is_bool());
        assert!(!null.is_object());

        // true
        assert!(!t.is_num());
        assert!(!t.is_null());
        assert!(t.is_bool());
        assert!(!t.is_object());

        // false
        assert!(!f.is_num());
        assert!(!f.is_null());
        assert!(f.is_bool());
        assert!(!f.is_object());

        // undefined
        assert!(!undef.is_num());
        assert!(!undef.is_null());
        assert!(!undef.is_bool());
        assert!(!undef.is_object());
        assert!(undef.is_undefined());

        // object
        assert!(!obj.is_num());
        assert!(!obj.is_null());
        assert!(!obj.is_bool());
        assert!(obj.is_object());
    }

    #[test]
    fn test_value_equality_nums() {
        assert!(Value::num(42.0).equals(Value::num(42.0)));
        assert!(!Value::num(42.0).equals(Value::num(43.0)));
        // IEEE 754: 0.0 == -0.0 is true
        assert!(Value::num(0.0).equals(Value::num(-0.0)));
    }

    #[test]
    fn test_value_equality_nan() {
        // IEEE 754: NaN != NaN
        assert!(!Value::num(f64::NAN).equals(Value::num(f64::NAN)));
    }

    #[test]
    fn test_value_equality_bools() {
        assert!(Value::bool(true).equals(Value::bool(true)));
        assert!(Value::bool(false).equals(Value::bool(false)));
        assert!(!Value::bool(true).equals(Value::bool(false)));
    }

    #[test]
    fn test_value_equality_null() {
        assert!(Value::null().equals(Value::null()));
    }

    #[test]
    fn test_value_equality_objects() {
        let mut a: u64 = 1;
        let mut b: u64 = 2;
        let pa = &mut a as *mut u64 as *mut u8;
        let pb = &mut b as *mut u64 as *mut u8;

        assert!(Value::object(pa).equals(Value::object(pa)));
        assert!(!Value::object(pa).equals(Value::object(pb)));
    }

    #[test]
    fn test_value_equality_cross_type() {
        assert!(!Value::num(0.0).equals(Value::null()));
        assert!(!Value::num(0.0).equals(Value::bool(false)));
        assert!(!Value::null().equals(Value::bool(false)));
        assert!(!Value::bool(true).equals(Value::num(1.0)));
    }

    #[test]
    fn test_value_num_safe_integers() {
        // Verify i32 range roundtrips through f64
        for &i in &[0i32, 1, -1, 42, -42, i32::MIN, i32::MAX, 1000000, -1000000] {
            let v = Value::num(i as f64);
            let n = v.as_num().unwrap();
            assert_eq!(n as i32, i);
        }
    }

    #[test]
    fn test_value_debug_format() {
        assert_eq!(format!("{:?}", Value::null()), "null");
        assert_eq!(format!("{:?}", Value::bool(true)), "true");
        assert_eq!(format!("{:?}", Value::bool(false)), "false");
        assert_eq!(format!("{:?}", Value::num(42.0)), "42");
        assert_eq!(format!("{:?}", Value::undefined()), "<undefined>");
    }

    #[test]
    fn test_value_bitpattern_no_overlap() {
        let patterns = [TAG_NULL, TAG_FALSE, TAG_TRUE, TAG_UNDEFINED];
        for (i, a) in patterns.iter().enumerate() {
            for (j, b) in patterns.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "bit patterns must be distinct");
                }
            }
        }
    }

    #[test]
    fn test_value_num_special_values() {
        // Subnormals, max, min
        for &n in &[
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            f64::EPSILON,
            1.0e-308, // subnormal
            std::f64::consts::PI,
            std::f64::consts::E,
        ] {
            let v = Value::num(n);
            assert!(v.is_num(), "failed for {}", n);
            assert_eq!(v.as_num().unwrap(), n, "roundtrip failed for {}", n);
        }
    }
}
