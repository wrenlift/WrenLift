//! Built-in `String` class — UTF-8 internally; methods that
//! count characters or slice operate on Unicode codepoints.
//! Strings are immutable.

/// Immutable byte/codepoint sequence.
class String {
  /// Single-character string from a Unicode codepoint integer.
  ///
  /// ```wren
  /// String.fromCodePoint(0x1F34F)   // "🍏"
  /// ```
  static fromCodePoint(codepoint) {}

  /// Single-byte string from an integer 0..255. The result is
  /// not guaranteed to be valid UTF-8 — useful for byte-level
  /// protocols where the caller will glue many bytes together.
  static fromByte(byte) {}

  /// Codepoint count (NOT byte count). Use `bytes.count` for
  /// raw byte length.
  count {}

  /// `true` when `count == 0`.
  isEmpty {}

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

  /// Concatenation. `"foo" + "bar"` → `"foobar"`.
  +(other) {}

  /// Repeat. `"-" * 5` → `"-----"`.
  *(count) {}

  /// `true` when the receiver contains `needle` anywhere.
  contains(needle) {}

  /// Prefix / suffix check.
  startsWith(prefix) {}
  endsWith(suffix) {}

  /// First codepoint-offset of `needle`, or `-1` when absent.
  indexOf(needle) {}

  /// Search for `needle` starting at codepoint `start`.
  indexOf(needle, start) {}

  /// Replace every occurrence of `needle` with `replacement`.
  /// Returns a new string.
  replace(needle, replacement) {}

  /// Split on `separator` into a `List` of strings.
  split(separator) {}

  /// Strip whitespace from both ends.
  trim() {}

  /// Strip every character that appears in `chars` from both
  /// ends. `"--ok--".trim("-")` → `"ok"`.
  trim(chars) {}

  /// Strip from the start only.
  trimStart() {}
  trimStart(chars) {}

  /// Strip from the end only.
  trimEnd() {}
  trimEnd(chars) {}

  /// Identity — kept for parity with `Num.toString`.
  toString {}

  /// View the underlying UTF-8 byte stream as a `Sequence` of
  /// integer byte values.
  bytes {}

  /// `Sequence` of integer codepoints.
  codePoints {}

  /// Iterator hooks (`for (c in s) ...`) — yields one-codepoint
  /// strings.
  iterate(iterator) {}
  iteratorValue(iterator) {}
}
