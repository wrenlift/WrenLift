//! Built-in `System` class — process-level I/O and runtime
//! introspection. Every Wren program has this in scope; no
//! import needed.

/// Host-level standard-output and time-of-day primitives.
class System {
  /// Print a value followed by a newline. Each value is
  /// converted via its `toString`.
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
  static printAll(seq) {}

  /// Print without a trailing newline.
  static write(value) {}

  /// Concatenated `write` over a sequence.
  static writeAll(seq) {}

  /// Wall-clock seconds since the runtime started. Monotonic;
  /// useful for benchmarking but not for time-of-day.
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
