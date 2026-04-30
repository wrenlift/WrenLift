//! Built-in `Fiber` class — cooperative coroutines. The current
//! fiber is the one currently running; `Fiber.new {}` creates a
//! suspended one that runs when something calls `.call(...)` or
//! `.try(...)` on it.

class Fiber {
  /// Wrap a closure as a new fiber.
  ///
  /// @param   {Fn}    body — closure to run on first resume
  /// @returns {Fiber} suspended fiber
  ///
  /// ```wren
  /// var f = Fiber.new { System.print("inside") }
  /// f.call()
  /// ```
  static new(body) {}

  /// Return the fiber that's currently running.
  ///
  /// @returns {Fiber}
  static current {}

  /// Suspend the running fiber's enclosing transfer.
  static suspend() {}

  /// Yield to the caller. The yielding fiber stays alive; its
  /// caller's `.call` returns `null` (or `value`).
  static yield() {}
  /// @param {Object} value — value handed back to `.call`
  static yield(value) {}

  /// Abort the fiber with an error. Up the stack, `.try { ... }`
  /// catches it as the fiber's `.error` slot.
  ///
  /// @param {Object} error — message or value to surface
  static abort(error) {}

  /// Cooperative-cancel hook. Long-running compute can poll
  /// `Fiber.isCancelled` and bail early.
  ///
  /// @returns {Bool}
  static isCancelled {}
  static cancel() {}

  /// Deadline in milliseconds (host-monotonic).
  ///
  /// @returns {Num}
  static deadlineMs {}
  /// @param {Num} ms
  static setDeadlineMs(ms) {}

  /// Per-fiber context bag — arbitrary user-supplied state
  /// stashed for the duration of the fiber.
  ///
  /// @returns {Object}
  static context {}

  /// Run this fiber to its next yield / return / abort.
  call() {}
  /// @param {Object} arg — value made available as the body's
  ///                       first parameter on first resume
  call(arg) {}

  /// Like `call`, but catch aborts. Returns the value yielded
  /// or returned; on abort, returns the error.
  ///
  /// @returns {Object} yielded / returned value, or the abort error
  try() {}
  try(arg) {}

  /// Transfer control to another fiber. Unlike `call`, the
  /// transferring fiber doesn't resume on the next `yield` —
  /// control only returns when some other fiber transfers back.
  transfer() {}
  transfer(arg) {}

  /// Transfer with an error — receiver re-aborts on resume.
  ///
  /// @param {Object} error
  transferError(error) {}

  /// Last error stashed by an abort under `try`. `null` when
  /// the fiber finished cleanly.
  ///
  /// @returns {Object}
  error {}

  /// `true` when the fiber has run to completion.
  ///
  /// @returns {Bool}
  isDone {}

  /// Recorded stack trace, if available.
  stackTrace {}

  /// Per-instance versions of the cancel / deadline / context
  /// hooks above.
  context {}
  cancel() {}
  isCancelled {}
  deadlineMs {}
  setDeadlineMs(ms) {}
}
