//! Built-in `Dom` class — read / mutate the host page from Wren
//! over postMessage. Every method returns a `Future`; bind it
//! with `.await` inside a fiber.

/// DOM bridges. Static methods only.
class Dom {
  /// Text content of the first node matching `selector`.
  ///
  /// ```wren
  /// var t = Dom.text("#title").await
  /// ```
  static text(selector) {}

  /// Set the text content of the first match. Resolves to the
  /// new value (the host echoes back so failures are visible).
  static setText(selector, value) {}

  /// Read an attribute. Resolves to the string value or null
  /// when the attribute is unset.
  static getAttribute(selector, name) {}

  /// Set an attribute.
  static setAttribute(selector, name, value) {}

  /// Add / remove a CSS class. The matched node receives the
  /// class on the next frame.
  static addClass(selector, name) {}
  static removeClass(selector, name) {}

  /// Run a query and resolve to a `List` of selector strings,
  /// one per match. The host returns selectors (not handles)
  /// because nodes don't survive the postMessage boundary.
  static queryAll(selector) {}
}
