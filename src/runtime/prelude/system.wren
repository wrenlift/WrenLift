//! Built-in `System` class — process-level I/O and runtime
//! introspection. Every Wren program has this in scope; no
//! import needed.

/// Host-level standard-output and time-of-day primitives.
class System {
  /// Print a value followed by a newline. Each value is
  /// converted via its `toString`.
  ///
  /// @param  {Object} value — any value; converted via `.toString`
  /// @returns {Object} the printed value (handy for chaining)
  ///
  /// ```wren
  /// System.print("hello, world")
  /// System.print(1 + 2)              // 3
  /// ```
  static print(value) {}

  /// Print a blank line.
  static print {}

  /// Print every element of `seq`, joined into one line, then
  /// a newline.
  ///
  /// @param {Sequence} seq — anything iterable
  static printAll(seq) {}

  /// Print without a trailing newline.
  ///
  /// @param {Object} value
  static write(value) {}

  /// Concatenated `write` over a sequence.
  ///
  /// @param {Sequence} seq
  static writeAll(seq) {}

  /// Wall-clock seconds since the runtime started. Monotonic;
  /// useful for benchmarking but not for time-of-day.
  ///
  /// @returns {Num} elapsed seconds (fractional)
  ///
  /// ```wren
  /// var t = System.clock
  /// expensive()
  /// System.print((System.clock - t) * 1000)
  /// ```
  static clock {}

  /// Trigger a garbage collection cycle.
  static gc() {}
}
