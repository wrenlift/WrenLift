//! Built-in `Sequence` class — abstract base for anything you
//! can `for (x in seq)` over. `List`, `Map.keys`, `Map.values`,
//! `String.bytes`, `String.codePoints`, `Range`, and the lazy
//! `map` / `where` / `take` / `skip` views all inherit these
//! methods.

class Sequence {
  /// Element count. Walks the sequence once.
  count {}

  /// Count elements matching a predicate.
  count(callback) {}

  /// `true` when the sequence has no elements.
  isEmpty {}

  /// `true` when at least one element matches.
  any(callback) {}

  /// `true` when every element matches.
  all(callback) {}

  /// `true` when the sequence contains a value equal to
  /// `value`.
  contains(value) {}

  /// Run `callback` for every element; returns `null`. Use
  /// when you only care about side effects.
  each(callback) {}

  /// Lazy: yields `callback(x)` for every element.
  ///
  /// ```wren
  /// var doubled = [1, 2, 3].map { |x| x * 2 }
  /// ```
  map(callback) {}

  /// Lazy: yields elements that match.
  where(callback) {}

  /// Lazy: drop the first `count` elements.
  skip(count) {}

  /// Lazy: yield only the first `count` elements.
  take(count) {}

  /// Eager fold. With one arg, the seed is the first element.
  reduce(callback) {}
  reduce(seed, callback) {}

  /// Concatenated string. `[1,2,3].join` → `"123"`.
  join() {}

  /// With separator: `[1,2,3].join(", ")` → `"1, 2, 3"`.
  join(separator) {}

  /// Materialise into a `List`.
  toList {}
}
