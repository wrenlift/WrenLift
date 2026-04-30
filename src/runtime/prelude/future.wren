//! Built-in `Future` class — handle to an async result the host
//! will eventually settle. Returned by `Browser.fetch`,
//! `Browser.setTimeout`, every `Dom.*` method, etc.
//!
//! Bind a future with `.await` inside a fiber to read its value:
//!
//! ```wren
//! var t = Browser.setTimeout(150).await
//! var html = Dom.text("#title").await
//! ```

class Future {
  /// Numeric handle the host uses to address this future.
  /// Mostly an implementation detail; useful when bridging
  /// custom JS shims.
  handle {}

  /// Settle status: `0` pending, `1` resolved, `2` rejected.
  /// Useful for non-blocking polling — e.g. peeking before a
  /// `Fiber.yield`.
  state {}

  /// Park the current fiber until the future settles, then
  /// either return the resolved value or `Fiber.abort` with the
  /// rejection message. Resumed by the runtime's scheduler when
  /// the host calls `resolve_future` / `reject_future`.
  await {}
}

/// Binary-flavoured `Future`. Same parking + scheduling shape,
/// but `.await` resolves to a `ByteArray` instead of a `String`.
/// Returned by `Browser.fetchBytes` for raw asset loads.
class ByteFuture {
  handle {}
  state {}

  /// Park until settled, return the bytes (or abort with the
  /// rejection message).
  await {}
}
