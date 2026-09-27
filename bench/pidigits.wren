// Benchmark: Pidigits
// Measures integer arithmetic on doubles over a list: the first 2,000
// digits of pi from a spigot, each digit a pass over the whole list of
// remainders. The Benchmarks Game's version uses bignums; this one
// needs none. The digits are folded into a checksum.

class Pidigits {
  static run(digits) {
    var boxes = (digits * 10 / 3).floor + 1
    var a = List.filled(boxes, 2)
    var predigit = 0
    var nines = 0
    var checksum = 0
    var produced = 0
    for (d in 0...digits) {
      var q = 0
      var i = boxes
      while (i > 0) {
        var x = 10 * a[i - 1] + q * i
        var denominator = 2 * i - 1
        a[i - 1] = x % denominator
        q = (x / denominator).floor
        i = i - 1
      }
      a[0] = q % 10
      q = (q / 10).floor
      // A 9 waits to see whether the next digit carries into it.
      if (q == 9) {
        nines = nines + 1
      } else {
        var carry = q == 10 ? 1 : 0
        checksum = (checksum * 10 + predigit + carry) % 1000000007
        produced = produced + 1
        for (j in 0...nines) {
          checksum = (checksum * 10 + (carry == 1 ? 0 : 9)) % 1000000007
          produced = produced + 1
        }
        predigit = carry == 1 ? 0 : q
        nines = 0
      }
    }
    checksum = (checksum * 10 + predigit) % 1000000007
    return [produced + 1, checksum]
  }
}

var start = System.clock
var result = Pidigits.run(2000)
System.print("digits: %(result[0])")
System.print("checksum: %(result[1])")
System.print("elapsed: %(System.clock - start)")
