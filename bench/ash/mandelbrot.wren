// Mirrors Ash's Mandelbrot: 875x500 at 1000 iterations with a palette
// object per pixel; the checksum sums palette channels. Ash records a
// band of accepted checksums because FMA contraction changes it; the
// unfused value is the reference. Expected: Checksum: 111214112
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
    var maxIterations = 1000
    var maxRad = 65536
    var width = 875
    var height = 500
    var palette = []
    for (i in 0..maxIterations) palette.add(Mandelbrot.palette(i / maxIterations))
    var scale = 0.1 / size
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
System.print("Checksum: %(Mandelbrot.run())")
System.print("elapsed: %(System.clock - start)")
