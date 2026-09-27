// Benchmark: Regex-redux
// Measures bitwise tests in nested loops: the nine variant patterns of
// the Benchmarks Game's regex-redux, each an 8-base pattern and its
// reverse complement with some positions allowing several bases, are
// counted over 250,000 bases. Wren has no regular expressions, so a
// pattern is a list of base masks, and every language here runs the
// same matcher.

class RegexRedux {
  static sequence(n) {
    var dna = List.filled(n, 0)
    var seed = 42
    for (i in 0...n) {
      seed = (seed * 3877 + 29573) % 139968
      dna[i] = (seed * 4 / 139968).floor
    }
    return dna
  }

  // Whether pattern matches dna from position i.
  static matchesAt(dna, i, pattern) {
    for (j in 0...pattern.count) {
      if (pattern[j] & (1 << dna[i + j]) == 0) return false
    }
    return true
  }

  // Positions where either of alternatives matches, each counted per
  // alternative that does.
  static count(dna, alternatives) {
    var n = 0
    for (i in 0..dna.count - 8) {
      for (pattern in alternatives) {
        if (matchesAt(dna, i, pattern)) n = n + 1
      }
    }
    return n
  }
}

var A = 1
var C = 2
var G = 4
var T = 8
var variants = [
  [[A, G, G, G, T, A, A, A], [T, T, T, A, C, C, C, T]],
  [[C | G | T, G, G, G, T, A, A, A], [T, T, T, A, C, C, C, A | C | G]],
  [[A, A | C | T, G, G, T, A, A, A], [T, T, T, A, C, C, A | G | T, T]],
  [[A, G, A | C | T, G, T, A, A, A], [T, T, T, A, C, A | G | T, C, T]],
  [[A, G, G, A | C | T, T, A, A, A], [T, T, T, A, A | G | T, C, C, T]],
  [[A, G, G, G, A | C | G, A, A, A], [T, T, T, C | G | T, C, C, C, T]],
  [[A, G, G, G, T, C | G | T, A, A], [T, T, A | C | G, A, C, C, C, T]],
  [[A, G, G, G, T, A, C | G | T, A], [T, A | C | G, T, A, C, C, C, T]],
  [[A, G, G, G, T, A, A, C | G | T], [A | C | G, T, T, A, C, C, C, T]]
]

var dna = RegexRedux.sequence(250000)
var start = System.clock
var checksum = 0
for (alternatives in variants) {
  checksum = checksum * 31 + RegexRedux.count(dna, alternatives)
}
System.print("length: %(dna.count)")
System.print("checksum: %(checksum)")
System.print("elapsed: %(System.clock - start)")
