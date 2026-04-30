//! Built-in `Num` class — every Wren number is a 64-bit IEEE-754
//! float. Methods are auto-imported; `42.toString` works without
//! an import. Bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)
//! treat the operands as `u32`.

/// Numeric scalar.
class Num {
  /// Decimal-string representation. Integer-valued floats render
  /// without a `.0` (`(3).toString` → `"3"`); fractional values
  /// use the shortest round-trippable form.
  toString {}

  /// Absolute value.
  abs {}

  /// Sign: `1` if positive, `-1` if negative, `0` if zero.
  sign {}

  /// Truncate toward zero.
  truncate {}

  /// Round to the nearest integer; ties round half-away-from-zero.
  round {}

  /// Largest integer less than or equal to this number.
  floor {}

  /// Smallest integer greater than or equal to this number.
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

  /// Square root.
  sqrt {}

  /// Cube root.
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
