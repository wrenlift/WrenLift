import "./fmt" for Fmt
import "@hatch:test" for Test
import "@hatch:assert" for Expect

// Colors write escape codes when enabled, pass through when not.
// Most tests below run with `Fmt.enabled = false` so assertions
// don't have to carry raw ESC bytes; a dedicated group re-enables
// to verify the wrapping bytes on both ends.

Fmt.enabled = false

Test.describe("ANSI wrappers (disabled)") {
  Test.it("color helpers return the text unchanged when disabled") {
    Expect.that(Fmt.red("err")).toBe("err")
    Expect.that(Fmt.green("ok")).toBe("ok")
    Expect.that(Fmt.bold("hi")).toBe("hi")
  }
  Test.it("color helpers stringify non-strings") {
    Expect.that(Fmt.red(42)).toBe("42")
  }
}

Test.describe("ANSI wrappers (enabled)") {
  Test.it("wraps input in the expected color + reset") {
    Fmt.enabled = true
    var painted = Fmt.red("x")
    Expect.that(painted).toContain("\x1b[31m")
    Expect.that(painted).toContain("x")
    Expect.that(painted).toContain("\x1b[0m")
    Fmt.enabled = false
  }
  Test.it("styles compose by nesting") {
    Fmt.enabled = true
    var painted = Fmt.bold(Fmt.red("x"))
    Expect.that(painted).toContain("\x1b[1m")
    Expect.that(painted).toContain("\x1b[31m")
    Fmt.enabled = false
  }
}

Test.describe("padding") {
  Test.it("padLeft right-aligns") {
    Expect.that(Fmt.padLeft("3", 4)).toBe("   3")
    Expect.that(Fmt.padLeft("abcd", 4)).toBe("abcd")   // no-op when at width
    Expect.that(Fmt.padLeft("abcde", 4)).toBe("abcde") // no truncation
  }
  Test.it("padRight left-aligns") {
    Expect.that(Fmt.padRight("3", 4)).toBe("3   ")
  }
  Test.it("center splits evenly, extra space on the right for odd gaps") {
    Expect.that(Fmt.center("hi", 6)).toBe("  hi  ")
    Expect.that(Fmt.center("hi", 5)).toBe(" hi  ")
  }
  Test.it("padding stringifies non-strings") {
    Expect.that(Fmt.padLeft(7, 3)).toBe("  7")
  }
}

Test.describe("hex") {
  Test.it("zero renders as 0x0") {
    Expect.that(Fmt.hex(0)).toBe("0x0")
  }
  Test.it("positive numbers round-trip through base 16") {
    Expect.that(Fmt.hex(15)).toBe("0xf")
    Expect.that(Fmt.hex(255)).toBe("0xff")
    Expect.that(Fmt.hex(4096)).toBe("0x1000")
  }
  Test.it("negative numbers carry a sign") {
    Expect.that(Fmt.hex(-16)).toBe("-0x10")
  }
  Test.it("floors fractional inputs") {
    Expect.that(Fmt.hex(255.9)).toBe("0xff")
  }
}

Test.describe("fixed") {
  Test.it("rounds half away from zero") {
    Expect.that(Fmt.fixed(3.14159, 2)).toBe("3.14")
    Expect.that(Fmt.fixed(3.145, 2)).toBe("3.15")
    Expect.that(Fmt.fixed(0.005, 2)).toBe("0.01")
  }
  Test.it("pads fractional digits with leading zeros") {
    Expect.that(Fmt.fixed(0.1, 3)).toBe("0.100")
    Expect.that(Fmt.fixed(1.05, 3)).toBe("1.050")
  }
  Test.it("zero decimals drops the dot") {
    Expect.that(Fmt.fixed(3.7, 0)).toBe("4")
  }
  Test.it("handles negative numbers") {
    Expect.that(Fmt.fixed(-3.14, 2)).toBe("-3.14")
  }
}

Test.describe("duration") {
  Test.it("seconds") {
    Expect.that(Fmt.duration(0)).toBe("0s")
    Expect.that(Fmt.duration(59)).toBe("59s")
  }
  Test.it("minutes") {
    Expect.that(Fmt.duration(60)).toBe("1m 0s")
    Expect.that(Fmt.duration(125)).toBe("2m 5s")
  }
  Test.it("hours") {
    Expect.that(Fmt.duration(3600)).toBe("1h 0m 0s")
    Expect.that(Fmt.duration(3670)).toBe("1h 1m 10s")
  }
  Test.it("days") {
    Expect.that(Fmt.duration(90061)).toBe("1d 1h 1m 1s")
  }
  Test.it("negative durations carry the sign") {
    Expect.that(Fmt.duration(-125)).toBe("-2m 5s")
  }
}

Test.run()
