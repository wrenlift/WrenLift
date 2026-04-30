//! Built-in `Null` class — the singleton type whose only
//! instance is the `null` literal.

class Null {
  /// Logical NOT — `null` is falsy, so `!null` is `true`.
  ! {}

  /// Always returns `"null"`.
  toString {}
}
