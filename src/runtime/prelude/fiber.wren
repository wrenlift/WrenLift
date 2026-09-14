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

  /// Hand a fiber to the scheduler. It runs when the driver ticks,
  /// a turn at a time: until it yields, sleeps, parks or finishes.
  /// An abort ends the task and stays in its `error`, as under `try`.
  ///
  /// @param   {Fn}    body — closure to run as the task
  /// @returns {Fiber} the task's fiber
  ///
  /// ```wren
  /// var t = Fiber.spawn { Fiber.sleep(10); System.print("later") }
  /// while (Fiber.tick(0)) Fiber.idle(100)
  /// ```
  static spawn(body) {}

  /// Wait `ms` milliseconds. On a task this parks on a timer and
  /// the other tasks run meanwhile; anywhere else the caller drives
  /// the scheduler until the time is up.
  ///
  /// @param {Num} ms — milliseconds; null waits forever
  static sleep(ms) {}

  /// A fresh wait token for `park`. Single use.
  ///
  /// @returns {Num}
  static waiter {}

  /// Park until `token` is woken or `ms` milliseconds pass. A wake
  /// that came before the park returns at once.
  ///
  /// @param   {Num}  token — from `Fiber.waiter`
  /// @param   {Num}  ms — timeout in milliseconds; null waits forever
  /// @returns {Bool} true when woken, false on timeout
  static park(token, ms) {}

  /// Wake the fiber parked on `token`, from any thread. Only the first
  /// wake of a token counts.
  ///
  /// @param   {Num}  token
  /// @returns {Bool} true when the wake was delivered
  static wake(token) {}

  /// Run the tasks that are ready, a turn each, and again while some
  /// are still ready and `ms` milliseconds have not passed. Cannot
  /// be called from a task.
  ///
  /// @param   {Num}  ms — time budget; 0 runs one round, null runs
  ///                      until nothing is ready
  /// @returns {Bool} true while tasks remain
  static tick(ms) {}

  /// Block until a task is ready, a timer is due, a wake arrives or
  /// `ms` milliseconds pass. Cannot be called from a task.
  ///
  /// @param {Num} ms — null waits forever
  static idle(ms) {}

  /// Tasks that have not finished.
  ///
  /// @returns {Num}
  static live {}

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
