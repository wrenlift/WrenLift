// Benchmark: Reverse-complement
// Measures list element reads and writes from both ends: a million
// bases, encoded 0 to 3 so each base's complement is 3 minus it, are
// reverse-complemented in place and then checksummed. The sequence
// comes from the FASTA generator and is built before the clock starts.

class ReverseComplement {
  static sequence(n) {
    var dna = List.filled(n, 0)
    var seed = 42
    for (i in 0...n) {
      seed = (seed * 3877 + 29573) % 139968
      dna[i] = (seed * 4 / 139968).floor
    }
    return dna
  }

  static apply(dna) {
    var lo = 0
    var hi = dna.count - 1
    while (lo <= hi) {
      var left = 3 - dna[hi]
      dna[hi] = 3 - dna[lo]
      dna[lo] = left
      lo = lo + 1
      hi = hi - 1
    }
  }

  static checksum(dna) {
    var sum = 0
    for (base in dna) sum = (sum * 5 + base) % 1000000007
    return sum
  }
}

var dna = ReverseComplement.sequence(1000000)
var start = System.clock
ReverseComplement.apply(dna)
var checksum = ReverseComplement.checksum(dna)
System.print("length: %(dna.count)")
System.print("checksum: %(checksum)")
System.print("elapsed: %(System.clock - start)")
