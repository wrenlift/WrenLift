// Benchmark: Mandelbrot
// Measures floating-point arithmetic on short-lived objects: every
// iteration of the inner loop allocates a Complex, and every pixel
// picks a palette entry. The checksum sums the palette channels;
// it moves a little when compiled code fuses a multiply and an add,
// so the workflow compares it within a band.
// Based on the same benchmark in Ash's suite, at a size that fits
// the reference Wren implementation in the workflow's window.

class RGB {
  construct new(r, g, b) {
    _r = r
    _g = g
    _b = b
  }
  r { _r }
  g { _g }
  b { _b }
}

class Complex {
  construct new(i, j) {
    _i = i
    _j = j
  }
  i { _i }
  j { _j }
}

class Mandelbrot {
  static palette(fraction) {
    var r = (fraction * 255).truncate
    var g = ((1 - fraction) * 255).truncate
    var b = ((0.5 - (fraction - 0.5).abs) * 2 * 255).truncate
    return RGB.new(r, g, b)
  }

  static run() {
    var size = 25
    var maxIterations = 200
    var maxRad = 65536
    var width = 350
    var height = 200
    var palette = []
    for (i in 0..maxIterations) palette.add(Mandelbrot.palette(i / maxIterations))
    var scale = 0.25 / size
    var checksum = 0
    for (y in 0...height) {
      for (x in 0...width) {
        var iteration = 0
        var offset = Complex.new(x * scale - 2.5, y * scale - 1)
        var val = Complex.new(0.0, 0.0)
        while (val.i * val.i + val.j * val.j < maxRad && iteration < maxIterations) {
          val = Complex.new(val.i * val.i - val.j * val.j + offset.i, 2.0 * val.i * val.j + offset.j)
          iteration = iteration + 1
        }
        var color = palette[iteration]
        checksum = checksum + color.r + color.g + color.b
      }
    }
    return checksum
  }
}

var start = System.clock
var checksum = Mandelbrot.run()
var elapsed = System.clock - start

// Print the checksum so the bench-correctness step can diff our stdout
// against standard Wren.
System.print("checksum: %(checksum)")
System.print("elapsed: %(elapsed)")
