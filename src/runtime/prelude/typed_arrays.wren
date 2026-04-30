//! Built-in typed-array classes — packed numeric buffers with
//! `O(1)` indexed access and a flat memory layout suitable for
//! GPU upload / wasm-host roundtrips.

/// `count`-element u8 buffer. Wren-side numbers are floats;
/// reads return Num, writes truncate to byte.
class ByteArray {
  /// Zero-filled buffer of `count` bytes.
  static new(count) {}

  /// Buffer with one byte per element of `list` (each cast to
  /// u8).
  static fromList(list) {}

  /// UTF-8 byte snapshot of `text`.
  static fromString(text) {}

  /// Element count (== byte length for `ByteArray`).
  count {}

  /// Total byte length. Identical to `count` for u8 arrays.
  byteLength {}

  /// Indexed read / write (0..count-1).
  [index] {}
  [index]=(value) {}

  /// Iteration (`for (b in bytes)`).
  iterate(iterator) {}
  iteratorValue(iterator) {}

  /// Copy contents into a `List<Num>`.
  toList {}

  toString {}
}

/// `count`-element f32 buffer. Same shape as `ByteArray`;
/// element width is 4 bytes.
class Float32Array {
  static new(count) {}
  static fromList(list) {}
  count {}
  byteLength {}
  [index] {}
  [index]=(value) {}
  iterate(iterator) {}
  iteratorValue(iterator) {}
  toList {}
  toString {}
}

/// `count`-element f64 buffer. Element width is 8 bytes.
class Float64Array {
  static new(count) {}
  static fromList(list) {}
  count {}
  byteLength {}
  [index] {}
  [index]=(value) {}
  iterate(iterator) {}
  iteratorValue(iterator) {}
  toList {}
  toString {}
}
