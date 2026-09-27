// Benchmark: FASTA
// Measures a tight numeric loop with a table lookup: the Benchmarks
// Game's linear-congruential generator picks 500,000 symbols from the
// IUB table by cumulative probability. The output is folded into a
// checksum, so the terminal's speed stays out of the timing.

class Fasta {
  static run(n) {
    var codes = "acgtBDHKMNRSVWY".bytes.toList
    var cumulative = [0.27, 0.39, 0.51, 0.78, 0.80, 0.82, 0.84, 0.86,
      0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1]
    var seed = 42
    var checksum = 0
    for (i in 0...n) {
      seed = (seed * 3877 + 29573) % 139968
      var r = seed / 139968
      var j = 0
      while (r >= cumulative[j]) j = j + 1
      checksum = (checksum + codes[j]) % 1000000007
    }
    return checksum
  }
}

var n = 500000
var start = System.clock
var checksum = Fasta.run(n)
System.print("length: %(n)")
System.print("checksum: %(checksum)")
System.print("elapsed: %(System.clock - start)")
