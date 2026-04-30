//! Built-in `Browser` class — JS-side bridges injected into the
//! wasm runtime. Async work (`setTimeout`, `fetch`, `nextFrame`,
//! `connect`) returns a `Future`; bind the result with `.await`
//! inside a fiber.

/// Browser-only async primitives. All methods are static.
class Browser {
  /// Sleep for `ms` milliseconds.
  ///
  /// @param   {Num}    ms — millisecond delay (≥ 0)
  /// @returns {Future} settles after the timeout fires
  ///
  /// ```wren
  /// Browser.setTimeout(500).await
  /// ```
  static setTimeout(ms) {}

  /// Yield until the next browser paint. Backed by
  /// `requestAnimationFrame` — vsync-paced and pauses when the
  /// tab is hidden, which is the right primitive for game loops.
  ///
  /// @returns {Future} resolves on the next animation frame
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
  /// @param   {String} url
  /// @returns {Future} resolves to the body as `String`
  ///
  /// ```wren
  /// var json = Browser.fetch("./atlas.json").await
  /// ```
  static fetch(url) {}

  /// HTTP GET, binary. Resolves to a `ByteArray` — useful for
  /// loading raw asset bytes straight into a GPU texture.
  ///
  /// @param   {String}     url
  /// @returns {ByteFuture} resolves to a `ByteArray`
  ///
  /// ```wren
  /// var bytes = Browser.fetchBytes("./atlas.rgba8").await
  /// device.writeTexture(tex, bytes, descriptor)
  /// ```
  static fetchBytes(url) {}

  /// Open a WebSocket.
  ///
  /// @param   {String}    url — `ws://...` or `wss://...`
  /// @returns {WebSocket} live socket handle
  static connect(url) {}
}
