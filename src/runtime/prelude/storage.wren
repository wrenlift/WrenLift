//! Built-in `LocalStorage` / `SessionStorage` classes — JS-side
//! persistent key/value buckets, exposed through the same
//! `Future`-await shape as the rest of the browser bridges.

/// `window.localStorage` wrapper. Values survive page reloads.
class LocalStorage {
  /// Read a key. Resolves to the stored string or `null`.
  static get(key) {}

  /// Write a key. Resolves to the new value.
  static set(key, value) {}

  /// Drop a key.
  static remove(key) {}

  /// Drop every key.
  static clear {}
}

/// `window.sessionStorage` wrapper. Same shape; values are
/// scoped to the current browser tab and cleared on close.
class SessionStorage {
  static get(key) {}
  static set(key, value) {}
  static remove(key) {}
  static clear {}
}
