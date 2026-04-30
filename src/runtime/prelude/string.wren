//! Built-in `String` class — UTF-8 internally; methods that
//! count characters or slice operate on Unicode codepoints.
//! Strings are immutable.

/// Immutable byte/codepoint sequence.
class String {
  /// Codepoint count (NOT byte count). Use `bytes.count` for
  /// raw byte length.
  count {}

  /// Indexed read by codepoint position. Returns a single-char
  /// String. Out-of-bounds raises `String index N out of bounds.`
  ///
  /// ```wren
  /// "hello"[0]   // "h"
  /// "héllo"[1]   // "é"
  /// ```
  [index] {}

  /// Substring view via a `Range`.
  ///
  /// ```wren
  /// "hello"[1..3]   // "ell"
  /// ```
  [range] {}

  /// `true` when the receiver contains `needle` anywhere.
  contains(needle) {}

  /// Prefix / suffix check.
  startsWith(prefix) {}
  endsWith(suffix) {}

  /// First / last byte-offset of `needle`, or `-1` when absent.
  indexOf(needle) {}

  /// Replace every occurrence of `needle` with `replacement`.
  /// Returns a new string.
  replace(needle, replacement) {}

  /// Split on `separator` into a `List`.
  split(separator) {}

  /// Strip leading / trailing whitespace.
  trim {}
  trimStart {}
  trimEnd {}

  /// Case-mapped copies.
  toLowercase {}
  toUppercase {}

  /// Iterate codepoints (`for (c in s)`).
  iterate(iterator) {}
  iteratorValue(iterator) {}

  /// Identity getter — kept for parity with `Num.toString`.
  toString {}

  /// View the underlying UTF-8 byte stream as a `Sequence` of
  /// integer codepoints.
  bytes {}
  codePoints {}
}
