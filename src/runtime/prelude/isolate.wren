//! Built-in module `isolate` — `import "isolate" for Isolate, Channel`.
//! An isolate is a VM on a thread of its own, with its own heap and
//! scheduler; values cross between isolates by copy, never by
//! reference, and a channel carries them.

class Isolate {
  /// Run `module` on a new thread. The thread builds a fresh VM,
  /// imports `module` and runs its top level; the module reads
  /// `arg` (copied: null, Bool, Num, String, List, Map, Channel,
  /// Isolate) through `Isolate.arg`. An abort or a module that
  /// will not load ends the isolate with `error` set.
  ///
  /// @param   {String}  module — module name, as an import would name it
  /// @param   {Object}  arg — value copied into the isolate
  /// @returns {Isolate}
  ///
  /// ```wren
  /// import "isolate" for Isolate, Channel
  /// var results = Channel.new()
  /// for (i in 0...Isolate.cpus) Isolate.spawn("worker", {"part": i, "out": results})
  /// for (i in 0...Isolate.cpus) System.print(results.receive())
  /// ```
  static spawn(module, arg) {}
  static spawn(module) {}

  /// The value this isolate was spawned with; null on the main one.
  static arg {}

  /// Hardware threads available to the process.
  ///
  /// @returns {Num}
  static cpus {}

  /// Whether the isolate's module has finished running.
  ///
  /// @returns {Bool}
  isDone {}

  /// The error the isolate ended with, or null.
  ///
  /// @returns {String}
  error {}

  /// Wait for the isolate to finish. On a scheduler task this parks
  /// the task; elsewhere the caller drives the scheduler meanwhile.
  ///
  /// @param   {Num}  ms — how long to wait; without it, until done
  /// @returns {Bool} whether the isolate has finished
  join(ms) {}
  join() {}
}

class Channel {
  /// An unbounded queue any isolate holding it can send to or
  /// receive from. A channel crosses to another isolate as a value.
  ///
  /// @returns {Channel}
  construct new() {}

  /// Queue a copy of `value`, waking a receiver.
  ///
  /// @param   {Object} value — copied, as `Isolate.spawn`'s arg is
  /// @returns {Bool}   false once the channel is closed
  send(value) {}

  /// The next value, waiting for one. Returns null when the channel
  /// is closed, or when `ms` milliseconds pass first; a sent null is
  /// not told apart from either — check `isClosed`.
  ///
  /// @param   {Num} ms — how long to wait; without it, until a value
  /// @returns {Object}
  receive(ms) {}
  receive() {}

  /// The next value if one is queued, else null.
  ///
  /// @returns {Object}
  tryReceive() {}

  /// Close the channel: sends fail and parked receivers return null.
  close() {}

  /// @returns {Bool}
  isClosed {}

  /// Values queued and not yet received.
  ///
  /// @returns {Num}
  count {}

  /// Release the channel. Every instance of it, in every isolate,
  /// is stale from here.
  drop() {}
}
