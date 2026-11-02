// From github.com/Brugarolas/WrenJIT scripts/jit.wren at 10c9421 (MIT).
// Unchanged apart from this header.
var start = System.clock

class JitBench {
  static integratePi(steps) {
    var sum = 0.0
    var i = 0

    while (i < steps) {
      var x = (i + 0.5) / steps
      sum = sum + 4.0 / (1.0 + x * x)
      i = i + 1
    }

    return sum / steps
  }

  static xorshift(iterations) {
    var x = 123456789
    var checksum = 0
    var i = 0

    while (i < iterations) {
      x = x ^ (x << 13)
      x = x ^ (x >> 17)
      x = x ^ (x << 5)

      x = x & 2147483647

      if ((x & 255) < 32) {
        checksum = checksum + (x & 1023)
      } else {
        checksum = checksum - (x & 31)
      }

      i = i + 1
    }

    return checksum + x
  }

  static mandelbrot(width, height, maxIter) {
    var checksum = 0
    var escaped = 0
    var y = 0

    while (y < height) {
      var ci = (y / height) * 2.0 - 1.0
      var x = 0

      while (x < width) {
        var cr = (x / width) * 3.5 - 2.5

        var zr = 0.0
        var zi = 0.0
        var zr2 = 0.0
        var zi2 = 0.0
        var i = 0

        while (i < maxIter && (zr2 + zi2) <= 4.0) {
          zi = 2.0 * zr * zi + ci
          zr = zr2 - zi2 + cr

          zr2 = zr * zr
          zi2 = zi * zi

          i = i + 1
        }

        if (i < maxIter) {
          escaped = escaped + 1
        }

        checksum = checksum + i * (x + 1) * (y + 1)
        x = x + 1
      }

      y = y + 1
    }

    return checksum + escaped
  }
}

JitBench.integratePi(1000000)
JitBench.xorshift(1000000)
JitBench.mandelbrot(800, 600, 1000)

var elapsed = System.clock - start
System.print("elapsed: %(elapsed)")