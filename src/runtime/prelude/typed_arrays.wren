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

  /// Read one little-endian 32-bit float at `byteOffset`. Native
  /// — significantly faster than reconstructing the IEEE-754 value
  /// from `[byte]` indices in Wren. Use this in binary-asset hot
  /// paths (glTF / .glb / OBJ).
  readF32LE(byteOffset) {}

  /// Read one little-endian unsigned 16-bit value, widened to Num.
  readU16LE(byteOffset) {}

  /// Read one little-endian unsigned 32-bit value, widened to Num.
  readU32LE(byteOffset) {}

  /// Bulk decode: read `count` little-endian f32 values starting
  /// at byte offset `srcByteOffset`, writing them into `dst` (a
  /// `Float32Array`) starting at element offset `dstOffset`.
  /// `stride` is the byte stride between consecutive source f32s
  /// (≥ 4; 0 / 4 means tight packing).
  ///
  /// One FFI call covers the entire accessor — the inner loop is
  /// a tight Rust copy.
  copyToFloat32Array(srcByteOffset, count, dst, dstOffset, stride) {}

  /// Bulk decode: read `count` little-endian u16 values, widen to
  /// i32, write into `dst` (an `Int32Array`). `stride` ≥ 2.
  /// Typical use: glTF JOINTS_0 with `componentType: 5123`.
  copyU16LEToInt32Array(srcByteOffset, count, dst, dstOffset, stride) {}

  /// Bulk decode: read `count` u8 values, widen to i32, write into
  /// `dst` (an `Int32Array`). `stride` ≥ 1.
  copyU8ToInt32Array(srcByteOffset, count, dst, dstOffset, stride) {}
}

/// `count`-element i32 buffer. Same shape as `ByteArray`;
/// element width is 4 bytes.
class Int32Array {
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

/// Fixed-width 4-lane SIMD value. Shared helpers live here; concrete
/// element kind lives on `Simd4f` and `Simd4i`.
class Simd {
  count {}
  byteLength {}
  [index] {}
  iterate(iterator) {}
  iteratorValue(iterator) {}
  replaceLane(index, value) {}
  shuffle(i0, i1, i2, i3) {}
  toString {}
}

/// 4-lane `f32` SIMD value with typed-array load/store interop.
class Simd4f is Simd {
  static new(x, y, z, w) {}
  static splat(value) {}
  static load(array, offset) {}
  static select(mask, whenTrue, whenFalse) {}
  store(array, offset) {}
  toFloat32Array {}
  reinterpretAsInt {}
  +(other) {}
  -(other) {}
  *(other) {}
  /(other) {}
  - {}
  min(other) {}
  max(other) {}
  abs {}
  sqrt {}
  ==(other) {}
  !=(other) {}
  <(other) {}
  <=(other) {}
  >(other) {}
  >=(other) {}
}

/// 4-lane signed `i32` SIMD value with typed-array load/store interop.
class Simd4i is Simd {
  static new(x, y, z, w) {}
  static splat(value) {}
  static load(array, offset) {}
  static select(mask, whenTrue, whenFalse) {}
  store(array, offset) {}
  toInt32Array {}
  reinterpretAsFloat {}
  +(other) {}
  -(other) {}
  *(other) {}
  - {}
  &(other) {}
  |(other) {}
  ^(other) {}
  ~ {}
  <<(count) {}
  >>(count) {}
  min(other) {}
  max(other) {}
  abs {}
  allTrue {}
  anyTrue {}
  bitmask {}
  ==(other) {}
  !=(other) {}
  <(other) {}
  <=(other) {}
  >(other) {}
  >=(other) {}
}
