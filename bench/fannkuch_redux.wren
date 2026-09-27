// Benchmark: Fannkuch-redux
// Measures small-list indexing and swaps: every permutation of 0..n-1,
// in the Benchmarks Game's order, is flipped (its first k + 1 elements
// reversed, k being the first) until it starts with 0. Prints the
// alternating checksum of the flip counts and the most flips any
// permutation took; n = 10 gives 73196 and 38.

class Fannkuch {
  static run(n) {
    var perm = (0...n).toList
    var count = List.filled(n, 0)
    var copy = List.filled(n, 0)
    var maxFlips = 0
    var checksum = 0
    var sign = 1
    var r = n
    while (true) {
      while (r != 1) {
        count[r - 1] = r
        r = r - 1
      }
      if (perm[0] != 0) {
        for (i in 0...n) copy[i] = perm[i]
        var flips = 0
        var k = copy[0]
        while (k != 0) {
          var lo = 0
          var hi = k
          while (lo < hi) {
            var t = copy[lo]
            copy[lo] = copy[hi]
            copy[hi] = t
            lo = lo + 1
            hi = hi - 1
          }
          flips = flips + 1
          k = copy[0]
        }
        if (flips > maxFlips) maxFlips = flips
        checksum = checksum + sign * flips
      }
      // The next permutation: rotate the first r + 1 elements until a
      // counter has room.
      while (true) {
        if (r == n) return [checksum, maxFlips]
        var first = perm[0]
        for (i in 0...r) perm[i] = perm[i + 1]
        perm[r] = first
        count[r] = count[r] - 1
        if (count[r] > 0) break
        r = r + 1
      }
      sign = -sign
    }
  }
}

var start = System.clock
var result = Fannkuch.run(10)
System.print("checksum: %(result[0])")
System.print("max flips: %(result[1])")
System.print("elapsed: %(System.clock - start)")
