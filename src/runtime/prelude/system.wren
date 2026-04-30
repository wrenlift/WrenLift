//! Built-in `System` class — process-level I/O and runtime
//! introspection. Every Wren program has this in scope; no
//! import needed.

/// The host-level standard-output and time-of-day primitives.
class System {
  /// Print one or more values followed by a newline. Each value
  /// is converted via its `toString` getter; objects without one
  /// fall back to a class-name placeholder. Returns the value
  /// printed (the last one when multiple are passed).
  ///
  /// ## Example
  ///
  /// ```wren
  /// System.print("hello, world")
  /// System.print(1 + 2)              // 3
  /// ```
  static print(value) {}

  /// Print without a trailing newline. Otherwise identical to
  /// `print`.
  static write(value) {}

  /// Wall-clock seconds since the runtime started. Monotonic;
  /// useful for benchmarking but not for time-of-day.
  ///
  /// ```wren
  /// var t = System.clock
  /// expensive()
  /// System.print((System.clock - t) * 1000)
  /// ```
  static clock {}

  /// Trigger a garbage collection cycle. Mostly a no-op on the
  /// generational + arena GCs; the mark-sweep one runs a full
  /// pass. Returns null.
  static gc() {}
}
