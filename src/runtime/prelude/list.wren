//! Built-in `List` class — dense, growable array of values.
//! Indexed reads / writes are O(1); insert / remove at the
//! middle is O(n). Capacity grows geometrically on `add`.

class List {
  /// Empty list.
  construct new() {}

  /// `count`-element list with every slot set to `value`.
  static filled(count, value) {}

  /// Element count.
  count {}

  /// Indexed access — returns `Subscript out of bounds` on
  /// out-of-range indices. Negative indices count from the end.
  [index] {}

  /// Slice view via a `Range`.
  [range] {}

  /// Mutating ops.
  add(value) {}
  insert(index, value) {}
  removeAt(index) {}
  remove(value) {}
  clear() {}

  /// Search.
  contains(value) {}
  indexOf(value) {}

  /// Concatenation — returns a new list combining `this` and
  /// `other`. Both lists' elements are shallow-copied.
  +(other) {}

  /// Iteration (`for (x in list)`).
  iterate(iterator) {}
  iteratorValue(iterator) {}

  /// Functional helpers — return new lists / sequences.
  map(callback) {}
  where(callback) {}
  reduce(seed, callback) {}
  any(callback) {}
  all(callback) {}

  toString {}
}
