//! Built-in `Num` class — every Wren number is a 64-bit IEEE-754
//! float. Methods are auto-imported; `42.toString` works without
//! an import. Bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)
//! treat the operands as `u32`.

/// Numeric scalar.
class Num {
  /// Constants.
  static infinity {}
  static nan {}
  static pi {}
  static tau {}
  /// `Num.largest` is `f64::MAX` (~1.797e308).
  static largest {}
  /// Smallest *positive* normal value (~2.225e-308).
  static smallest {}
  /// `2^53 - 1` and `-(2^53 - 1)` — bounds where every integer
  /// round-trips exactly through a 64-bit float.
  static maxSafeInteger {}
  static minSafeInteger {}

  /// Parse a decimal / hex / scientific string into a Num.
  /// Returns `null` when the input doesn't parse.
  ///
  /// ```wren
  /// Num.fromString("3.14")   // 3.14
  /// Num.fromString("0xff")   // 255
  /// Num.fromString("oops")   // null
  /// ```
  static fromString(text) {}

  /// Arithmetic.
  +(other) {}
  -(other) {}
  *(other) {}
  /(other) {}
  %(other) {}

  /// Comparison.
  <(other)  {}
  >(other)  {}
  <=(other) {}
  >=(other) {}
  ==(other) {}
  !=(other) {}

  /// Bitwise (operands cast to u32).
  &(other) {}
  |(other) {}
  ^(other) {}
  <<(other) {}
  >>(other) {}
  ~ {}

  /// Unary negate.
  - {}

  /// Decimal-string representation. Integer-valued floats render
  /// without a trailing `.0`.
  toString {}

  /// Absolute value.
  abs {}

  /// Sign: `1` if positive, `-1` if negative, `0` if zero.
  sign {}

  /// Truncate toward zero.
  truncate {}

  /// Round to the nearest integer; ties round half-away-from-zero.
  round {}

  /// Largest integer ≤ this number.
  floor {}

  /// Smallest integer ≥ this number.
  ceil {}

  /// Fractional part — `this - this.truncate`.
  fraction {}

  /// `true` when the value is `Infinity` or `-Infinity`.
  isInfinity {}

  /// `true` when the value has no fractional part and isn't
  /// `Infinity` / `NaN`.
  isInteger {}

  /// `true` when the value is `NaN`.
  isNan {}

  /// Square / cube root.
  sqrt {}
  cbrt {}

  /// Trig functions. Angles are in radians.
  sin {}
  cos {}
  tan {}
  asin {}
  acos {}
  atan {}

  /// `atan2(this, x)` — angle of the vector `(x, this)`.
  atan(x) {}

  /// Natural log, base-2 log, and `e^this`.
  log {}
  log2 {}
  exp {}

  /// Powers and clamping.
  pow(exponent) {}
  min(other) {}
  max(other) {}
  clamp(low, high) {}

  /// Inclusive range `this..end` (`Range`).
  ..(end) {}
  /// Exclusive range `this...end`.
  ...(end) {}
}
