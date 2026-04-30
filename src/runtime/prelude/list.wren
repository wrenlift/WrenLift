//! Built-in `List` class — dense, growable array of values.
//! Indexed reads / writes are O(1); insert / remove at the
//! middle is O(n). Capacity grows geometrically on `add`.

class List {
  /// Empty list.
  ///
  /// @returns {List} empty list
  construct new() {}

  /// `count`-element list with every slot set to `value`.
  ///
  /// @param {Num}    count — element count, ≥ 0
  /// @param {Object} value — initial value for every slot
  /// @returns {List} new list of length `count`
  static filled(count, value) {}

  /// Element count.
  ///
  /// @returns {Num}
  count {}

  /// Indexed access — returns `Subscript out of bounds` on
  /// out-of-range indices. Negative indices count from the end.
  ///
  /// @param   {Num}    index
  /// @returns {Object} element at `index`
  [index] {}

  /// Slice via a `Range`.
  ///
  /// @param   {Range} range
  /// @returns {List}  new list view of the range
  [range] {}

  /// Indexed write.
  ///
  /// @param {Num}    index
  /// @param {Object} value
  [index]=(value) {}

  /// Append a value. Grows the list.
  ///
  /// @param {Object} value
  add(value) {}

  /// Append every element of another list / sequence.
  ///
  /// @param {Sequence} other
  addAll(other) {}

  /// Insert `value` at `index`; shifts later elements right.
  ///
  /// @param {Num}    index
  /// @param {Object} value
  insert(index, value) {}

  /// Remove and return the element at `index`. Negative
  /// indices count from the end.
  ///
  /// @param   {Num}    index
  /// @returns {Object}
  removeAt(index) {}

  /// Remove the first element equal to `value`. Returns the
  /// value when removed, `null` otherwise.
  ///
  /// @param   {Object} value
  /// @returns {Object} the removed value, or null
  remove(value) {}

  /// First codepoint-equal index of `value`, or `-1`.
  ///
  /// @param   {Object} value
  /// @returns {Num}    index, or -1 when absent
  indexOf(value) {}

  /// Swap two elements in place.
  ///
  /// @param {Num} a
  /// @param {Num} b
  swap(a, b) {}

  /// Drop every element.
  clear() {}

  /// Sort in place using the natural comparison operator (`<`).
  sort() {}

  /// Sort in place by a callback returning a Num — negative
  /// for less, positive for greater, zero for equal.
  ///
  /// @param {Fn} comparator — `(a, b) → Num`
  sort(comparator) {}

  /// Concatenation — `[1,2] + [3,4]` → `[1,2,3,4]`.
  ///
  /// @param   {List} other
  /// @returns {List}
  +(other) {}

  /// Repeat — `[1,2] * 3` → `[1,2,1,2,1,2]`.
  ///
  /// @param   {Num}  count
  /// @returns {List}
  *(count) {}

  /// `true` when `value` appears anywhere in the list.
  ///
  /// @param   {Object} value
  /// @returns {Bool}
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
