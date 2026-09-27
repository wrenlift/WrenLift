// Benchmark: K-nucleotide
// Measures map lookups and inserts keyed by numbers: every k-mer of a
// 250,000-base sequence, for k = 1, 2, 3, 4, 6, 12 and 18, is counted
// by its base-4 key. The sequence comes from the FASTA generator and
// is built before the clock starts.

class KNucleotide {
  // n bases, 0 to 3.
  static sequence(n) {
    var dna = List.filled(n, 0)
    var seed = 42
    for (i in 0...n) {
      seed = (seed * 3877 + 29573) % 139968
      dna[i] = (seed * 4 / 139968).floor
    }
    return dna
  }

  // Each k-mer's count, by its key.
  static frequencies(dna, k) {
    var counts = {}
    var high = 4.pow(k - 1)
    var key = 0
    for (j in 0...k) key = key * 4 + dna[j]
    counts[key] = 1
    for (i in k...dna.count) {
      key = key % high * 4 + dna[i]
      var seen = counts[key]
      counts[key] = seen == null ? 1 : seen + 1
    }
    return counts
  }
}

var dna = KNucleotide.sequence(250000)
var start = System.clock
var checksum = 0
for (k in [1, 2, 3, 4, 6, 12, 18]) {
  for (count in KNucleotide.frequencies(dna, k).values) {
    checksum = (checksum + count * k) % 1000000007
  }
}
System.print("length: %(dna.count)")
System.print("checksum: %(checksum)")
System.print("elapsed: %(System.clock - start)")
