/// `@hatch:fmt`: terminal-output helpers.
///
/// ```wren
/// import "@hatch:fmt" for Fmt
///
/// Fmt.green("ok")                    // wraps in ANSI green
/// Fmt.bold(Fmt.red("FAIL"))          // nestable
/// Fmt.padLeft("3", 4)                // "   3"
/// Fmt.padRight("3", 4)               // "3   "
/// Fmt.center("hi", 6)                // "  hi  "
/// Fmt.hex(255)                       // "0xff"
/// Fmt.fixed(3.14159, 2)              // "3.14"
/// Fmt.duration(3670)                 // "1h 1m 10s"
/// ```
///
/// Wren has string interpolation already (`"%(x)"`) so there's
/// no printf here. The helpers focus on things Wren's core
/// doesn't cover: ANSI styling, width-padding, and a couple of
/// numeric shorthands.
///
/// Colors can be disabled globally for piped or non-TTY output:
///
/// ```wren
/// Fmt.enabled = false
/// Fmt.green("x")    // → "x"  (no escape codes)
/// ```
///
/// Pair the flag with `@hatch:os`'s `Os.isatty` for automatic
/// detection: `Fmt.enabled = Os.isatty(Os.STDOUT)`.

class Fmt {
  /// -- Global toggle ------------------------------------------------------

  static enabled {
    if (__enabled == null) __enabled = true
    return __enabled
  }
  static enabled=(v) { __enabled = v }

  /// -- ANSI escapes -------------------------------------------------------
  ///
  /// `\x1b` is the single ESC byte; each code is `\x1b[<n>m`. `reset`
  /// clears all styling so the next character starts fresh.

  static reset { "\x1b[0m" }

  /// Foreground colors.
  static red(s)     { wrap_(s, "\x1b[31m") }
  static green(s)   { wrap_(s, "\x1b[32m") }
  static yellow(s)  { wrap_(s, "\x1b[33m") }
  static blue(s)    { wrap_(s, "\x1b[34m") }
  static magenta(s) { wrap_(s, "\x1b[35m") }
  static cyan(s)    { wrap_(s, "\x1b[36m") }
  static white(s)   { wrap_(s, "\x1b[37m") }
  static gray(s)    { wrap_(s, "\x1b[90m") }

  /// Styles. `bold`, `dim`, `italic`, `underline` compose on top of
  /// colors. Wrap inside-out (`Fmt.bold(Fmt.red("x"))`).
  static bold(s)      { wrap_(s, "\x1b[1m") }
  static dim(s)       { wrap_(s, "\x1b[2m") }
  static italic(s)    { wrap_(s, "\x1b[3m") }
  static underline(s) { wrap_(s, "\x1b[4m") }

  // Convert any value to a string, optionally wrapped in an ANSI
  // code. Drops the wrapping when `enabled` is false so callers
  // don't have to branch.
  static wrap_(s, code) {
    var text = toString_(s)
    if (!enabled) return text
    return code + text + reset
  }

  // -- Padding ------------------------------------------------------------

  /// Right-align `s` in a field of `width` spaces.
  static padLeft(s, width) {
    var t = toString_(s)
    if (t.count >= width) return t
    return " " * (width - t.count) + t
  }

  /// Left-align `s` in a field of `width` spaces.
  static padRight(s, width) {
    var t = toString_(s)
    if (t.count >= width) return t
    return t + " " * (width - t.count)
  }

  /// Center `s` in a field of `width` spaces; excess space lands on
  /// the right when the gap is odd.
  static center(s, width) {
    var t = toString_(s)
    if (t.count >= width) return t
    var total = width - t.count
    var left = (total / 2).floor
    var right = total - left
    return " " * left + t + " " * right
  }

  // -- Numeric helpers ----------------------------------------------------

  /// Unsigned hex. Negative numbers are formatted against their
  /// absolute value with a leading `-`. Enough for debug prints,
  /// not an arbitrary-precision bignum helper.
  static hex(n) {
    if (n == 0) return "0x0"
    var sign = ""
    var v = n.floor
    if (v < 0) {
      sign = "-"
      v = -v
    }
    var digits = "0123456789abcdef"
    var s = ""
    while (v > 0) {
      s = digits[(v % 16).floor] + s
      v = (v / 16).floor
    }
    return sign + "0x" + s
  }

  /// Fixed-point decimal. Rounds half-up.
  static fixed(n, decimals) {
    if (decimals < 0) Fiber.abort("decimals must be >= 0")
    var mult = (10).pow(decimals)
    // Round half away from zero: add 0.5 before truncating.
    var sign = n < 0 ? -1 : 1
    var scaled = ((n * sign) * mult + 0.5).floor
    var whole = (scaled / mult).floor
    var frac = scaled - whole * mult
    var signStr = sign < 0 && (whole != 0 || frac != 0) ? "-" : ""
    if (decimals == 0) return "%(signStr)%(whole)"
    // Pad fractional part with leading zeros to `decimals` digits.
    var fracStr = "%(frac)"
    fracStr = "0" * (decimals - fracStr.count) + fracStr
    return "%(signStr)%(whole).%(fracStr)"
  }

  /// Duration from a whole-seconds count. Single-unit output for
  /// small values, space-joined for larger ones (e.g. `"1h 1m 10s"`).
  /// Floors to whole seconds; fractional seconds could round up a
  /// minute boundary and read misleadingly.
  static duration(seconds) {
    var s = seconds.floor
    if (s < 0) return "-" + duration(-s)
    if (s < 60) return "%(s)s"
    var m = (s / 60).floor
    s = s - m * 60
    if (m < 60) return "%(m)m %(s)s"
    var h = (m / 60).floor
    m = m - h * 60
    if (h < 24) return "%(h)h %(m)m %(s)s"
    var d = (h / 24).floor
    h = h - d * 24
    return "%(d)d %(h)h %(m)m %(s)s"
  }

  // -- Internals ----------------------------------------------------------

  static toString_(v) {
    if (v is String) return v
    return "%(v)"
  }
}
