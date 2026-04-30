//! Built-in `Object` class — every Wren value's root class.
//! Methods declared here are inherited by every other class
//! unless overridden.

class Object {
  /// Identity comparison. `Object.same(a, b)` returns `true`
  /// when `a` and `b` are the same value (no operator overload
  /// dispatch).
  static same(a, b) {}

  /// Logical NOT (every value is truthy by default).
  ! {}

  /// Equality. Default impl is identity; classes override with
  /// `==(other) { ... }` to add structural compare.
  ==(other) {}
  !=(other) {}

  /// `true` when `this` is an instance of `klass` (or a
  /// subclass).
  is(klass) {}

  /// Default string conversion — class name + identity hash.
  /// Override on your class to render meaningful values.
  toString {}

  /// The receiver's class. `(1).type` → `Num`.
  type {}

  /// Stable hash code suitable for use as a Map key.
  hashCode {}
}
