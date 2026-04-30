//! Built-in `Fn` class — first-class function objects. Wren
//! literal closures (`{ |x| x + 1 }`) are `Fn` instances
//! created via `Fn.new`. Pass them to higher-order methods on
//! `List`, `Sequence`, etc.

class Fn {
  /// Wrap a closure literal as an `Fn`.
  ///
  /// ```wren
  /// var add = Fn.new {|a, b| a + b }
  /// add.call(1, 2)   // 3
  /// ```
  static new(body) {}

  /// Number of declared parameters.
  arity {}

  /// Call with the given args.
  call() {}
  call(a) {}
  call(a, b) {}
  call(a, b, c) {}
  call(a, b, c, d) {}
  call(a, b, c, d, e) {}
  call(a, b, c, d, e, f) {}
  call(a, b, c, d, e, f, g) {}
  call(a, b, c, d, e, f, g, h) {}
}
