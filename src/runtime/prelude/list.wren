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

  /// Slice via a `Range`.
  [range] {}

  /// Indexed write.
  [index]=(value) {}

  /// Append a value. Grows the list.
  add(value) {}

  /// Append every element of another list / sequence.
  addAll(other) {}

  /// Insert `value` at `index`; shifts later elements right.
  insert(index, value) {}

  /// Remove and return the element at `index`. Negative
  /// indices count from the end.
  removeAt(index) {}

  /// Remove the first element equal to `value`. Returns the
  /// value when removed, `null` otherwise.
  remove(value) {}

  /// First codepoint-equal index of `value`, or `-1`.
  indexOf(value) {}

  /// Swap two elements in place.
  swap(a, b) {}

  /// Drop every element.
  clear() {}

  /// Sort in place using the natural comparison operator (`<`).
  sort() {}

  /// Sort in place by a callback returning a Num — negative
  /// for less, positive for greater, zero for equal.
  sort(comparator) {}

  /// Concatenation — `[1,2] + [3,4]` → `[1,2,3,4]`.
  +(other) {}

  /// Repeat — `[1,2] * 3` → `[1,2,1,2,1,2]`.
  *(count) {}

  /// `true` when `value` appears anywhere in the list.
  contains(value) {}

  /// Iteration (`for (x in list) ...`).
  iterate(iterator) {}
  iteratorValue(iterator) {}

  /// Functional helpers (defined on `Sequence`, inherited
  /// by `List`).
  map(callback) {}
  where(callback) {}
  reduce(seed, callback) {}
  any(callback) {}
  all(callback) {}
  each(callback) {}
  count(callback) {}
  skip(count) {}
  take(count) {}
  join() {}
  join(separator) {}

  toString {}
}
