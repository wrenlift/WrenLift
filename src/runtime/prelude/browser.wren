//! Built-in `Browser` class — JS-side bridges injected into the
//! wasm runtime. Async work (`setTimeout`, `fetch`, `nextFrame`,
//! `connect`) returns a `Future`; bind the result with `.await`
//! inside a fiber.

/// Browser-only async primitives. All methods are static.
class Browser {
  /// Sleep for `ms` milliseconds. Returns a `Future` that
  /// resolves after the timeout fires.
  ///
  /// ```wren
  /// Browser.setTimeout(500).await
  /// ```
  static setTimeout(ms) {}

  /// Yield until the next browser paint. Returns a `Future`
  /// driven by `requestAnimationFrame`, so it's vsync-paced and
  /// pauses when the tab is hidden — the right primitive for
  /// game loops.
  ///
  /// ```wren
  /// while (true) {
  ///   draw()
  ///   Browser.nextFrame.await
  /// }
  /// ```
  static nextFrame {}

  /// HTTP GET. Resolves to the response body as a `String`.
  ///
  /// ```wren
  /// var json = Browser.fetch("./atlas.json").await
  /// ```
  static fetch(url) {}

  /// HTTP GET, binary. Resolves to a `ByteArray` — useful for
  /// loading raw asset bytes straight into a GPU texture
  /// without a UTF-8 round-trip.
  ///
  /// ```wren
  /// var bytes = Browser.fetchBytes("./atlas.rgba8").await
  /// device.writeTexture(tex, bytes, descriptor)
  /// ```
  static fetchBytes(url) {}

  /// Open a WebSocket. Returns a `WebSocket` object whose `send`
  /// / `recv` / `close` operate on the live socket.
  static connect(url) {}
}
