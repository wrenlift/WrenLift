//! Built-in `WebSocket` class — handle to a live JS-side
//! WebSocket. `Browser.connect(url)` returns one; the methods
//! on it are blocking-shaped via `Future`-await.

class WebSocket {
  /// Numeric handle.
  handle {}

  /// Send a text frame. Returns synchronously; the underlying
  /// JS shim queues the send.
  ///
  /// ```wren
  /// var ws = Browser.connect("wss://example.com/echo")
  /// ws.send("hello")
  /// var reply = ws.recv.await
  /// ```
  send(text) {}

  /// Receive the next inbound frame as a `Future`. `.await` it
  /// to block until the host pushes a message.
  recv {}

  /// Close the socket.
  close {}
}
