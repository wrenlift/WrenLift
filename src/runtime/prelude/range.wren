//! Built-in `Range` class — endpoint pair created by the `..`
//! and `...` operators on `Num`. `1..5` is inclusive (5 in);
//! `1...5` is exclusive (5 out). Iterates one integer per step.

class Range {
  /// Lower bound.
  from {}

  /// Upper bound.
  to {}

  /// Smaller of `from` / `to` (ranges can be reversed).
  min {}

  /// Larger of `from` / `to`.
  max {}

  /// `true` for `..`, `false` for `...`.
  isInclusive {}

  /// Iteration (`for (i in 1..5) ...`).
  iterate(iterator) {}
  iteratorValue(iterator) {}

  /// `"1..5"` / `"1...5"`.
  toString {}
}
