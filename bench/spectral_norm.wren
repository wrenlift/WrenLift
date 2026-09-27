// Benchmark: Spectral-norm
// Measures nested numeric loops over lists: ten rounds of multiplying a
// vector by A and then by A's transpose, A being the Benchmarks Game's
// infinite matrix a(i, j) = 1 / ((i + j)(i + j + 1) / 2 + i + 1),
// truncated to 500 rows. Prints the norm the vectors converge to.

class SpectralNorm {
  // The denominator of a(i, j).
  static denominator(i, j) {
    var ij = i + j
    return (ij * (ij + 1) / 2).floor + i + 1
  }

  // out = A * v
  static multiplyAv(v, out) {
    var n = v.count
    for (i in 0...n) {
      var sum = 0
      for (j in 0...n) sum = sum + v[j] / denominator(i, j)
      out[i] = sum
    }
  }

  // out = A' * v
  static multiplyAtv(v, out) {
    var n = v.count
    for (i in 0...n) {
      var sum = 0
      for (j in 0...n) sum = sum + v[j] / denominator(j, i)
      out[i] = sum
    }
  }

  static run(n) {
    var u = List.filled(n, 1)
    var v = List.filled(n, 0)
    var tmp = List.filled(n, 0)
    for (round in 0...10) {
      multiplyAv(u, tmp)
      multiplyAtv(tmp, v)
      multiplyAv(v, tmp)
      multiplyAtv(tmp, u)
    }
    var vBv = 0
    var vv = 0
    for (i in 0...n) {
      vBv = vBv + u[i] * v[i]
      vv = vv + v[i] * v[i]
    }
    return (vBv / vv).sqrt
  }
}

var start = System.clock
var result = SpectralNorm.run(500)
System.print("result: %(result)")
System.print("elapsed: %(System.clock - start)")
