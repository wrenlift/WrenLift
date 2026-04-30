//! Built-in `Map` class — hash-keyed associative array. Keys
//! must be comparable + hashable (Num, String, Bool, Null,
//! Class, Range). Values are arbitrary.

class Map {
  /// Empty map.
  construct new() {}

  /// Entry count.
  count {}

  /// Subscript get / set. Missing keys return `null`.
  ///
  /// ```wren
  /// var m = {}
  /// m["x"] = 1
  /// m["x"]    // 1
  /// m["y"]    // null
  /// ```
  [key] {}
  [key]=(value) {}

  /// Membership test. Distinguishes a stored `null` value from
  /// an absent key — `m[k]` returns `null` for both.
  containsKey(key) {}

  /// Insert without overwriting an existing entry.
  add(key, value) {}

  /// Remove an entry. Returns the removed value, or `null` when
  /// the key wasn't present.
  remove(key) {}

  /// Drop every entry.
  clear() {}

  /// Sequences of keys / values (lazy — iterate once).
  keys {}
  values {}

  /// Iteration (`for (k in map.keys)`).
  iterate(iterator) {}
  iteratorValue(iterator) {}

  toString {}
}
