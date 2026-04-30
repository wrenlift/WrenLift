//! Built-in `Class` class — meta-class. Every Wren class's
//! type is `Class`; `Foo.type` returns `Class`. Methods here
//! introspect a class's definition.

class Class {
  /// Class name as a String.
  name {}

  /// Parent class, or `null` for `Object`.
  supertype {}

  /// Render as `name`.
  toString {}

  /// `#!` and `#` attribute blocks attached to the class
  /// declaration, returned as a `Map<String, value>`.
  attributes {}

  /// Per-method attributes — `Map<methodName, Map<key, value>>`.
  methodAttributes {}
}
