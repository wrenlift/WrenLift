//! Built-in module `thread` — `import "thread" for Thread, Mutex, Lock, Deque`.
//! Tasks on the program's worker threads, one heap, in the shape of
//! Haxe's `sys.thread`: `Thread.create` runs a function on a worker;
//! `Mutex`, `Lock` and `Deque` let tasks wait for each other. A wait
//! parks the task on its worker's scheduler, never the OS thread, so
//! the worker runs other tasks meanwhile.

class Thread {
  /// Run `body` as a task on a worker thread. The workers start on
  /// the first call, one per hardware thread; the task lands on the
  /// least loaded. It shares the heap with the caller: the values it
  /// can reach are the caller's, and writes to them race the way
  /// they do between threads anywhere — guard shared state with a
  /// `Mutex`. An abort ends the task and prints nothing.
  ///
  /// @param {Fn} body — function to run as the task
  ///
  /// ```wren
  /// import "thread" for Thread, Lock
  /// var done = Lock.new()
  /// for (i in 0...4) Thread.create { work(i); done.release() }
  /// for (i in 0...4) done.wait()
  /// ```
  static create(body) {}

  /// The fiber running this task.
  ///
  /// @returns {Fiber}
  static current {}

  /// Let the other tasks of this worker run.
  static yield() {}

  /// Worker threads started.
  ///
  /// @returns {Num}
  static count {}
}

class Mutex {
  /// A mutual-exclusion lock. Not reentrant.
  ///
  /// @returns {Mutex}
  construct new() {}

  /// Take the lock, waiting for its holder to release it.
  acquire() {}

  /// Take the lock if it is free.
  ///
  /// @returns {Bool} whether it was taken
  tryAcquire() {}

  /// Give the lock up; the longest waiter takes it.
  release() {}
}

class Lock {
  /// A counting lock: every `release` adds a unit, every `wait`
  /// takes one, waiting for it if none is there. Starts at zero.
  ///
  /// @returns {Lock}
  construct new() {}

  /// Take a unit, waiting up to `ms` milliseconds for one.
  ///
  /// @param   {Num}  ms — how long to wait; without it, until a unit comes
  /// @returns {Bool} true with a unit, false when the wait ran out
  wait(ms) {}
  wait() {}

  /// Add a unit; the longest waiter takes it.
  release() {}
}

class Deque {
  /// A queue any task can add to or take from.
  ///
  /// @returns {Deque}
  construct new() {}

  /// Add `value` at the back.
  ///
  /// @param {Object} value
  add(value) {}

  /// Add `value` at the front.
  ///
  /// @param {Object} value
  push(value) {}

  /// Take the front item. With `block`, wait for one; without, null
  /// when the deque is empty.
  ///
  /// @param   {Bool} block — whether to wait for an item
  /// @returns {Object}
  pop(block) {}

  /// Items queued.
  ///
  /// @returns {Num}
  count {}
}
