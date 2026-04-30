//! Built-in `Fiber` class — cooperative coroutines. The current
//! fiber is the one currently running; `Fiber.new {}` creates a
//! suspended one that runs when something calls `.call(...)` or
//! `.try(...)` on it.

class Fiber {
  /// Wrap a closure as a new fiber. The body runs the first time
  /// `call` / `try` resumes the fiber.
  ///
  /// ```wren
  /// var f = Fiber.new { System.print("inside") }
  /// f.call()
  /// ```
  static new(body) {}

  /// Return the fiber that's currently running.
  static current {}

  /// Suspend the running fiber's enclosing transfer. Resumes
  /// whatever scheduled this fiber next time control returns.
  static suspend() {}

  /// Yield a value to the caller. The yielding fiber stays
  /// alive; its caller's `call` returns `value`.
  static yield(value) {}
  static yield() {}

  /// Abort the fiber with an error message. Up the stack,
  /// `.try { ... }` catches it as the fiber's error slot.
  static abort(error) {}

  /// Run this fiber to its next yield / return / abort.
  call() {}
  call(arg) {}

  /// Like `call`, but catch aborts. Returns the value yielded
  /// or returned; on abort, returns the error.
  try() {}
  try(arg) {}

  /// Last error stashed by an abort under `try`. `null` when
  /// the fiber finished cleanly.
  error {}

  /// `true` when the fiber has run to completion.
  isDone {}
}
