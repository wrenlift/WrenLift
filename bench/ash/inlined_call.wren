// Mirrors Ash's BenchInlinedCall: 100M iterations of the accumulator
// with no call. Loop state lives in locals of one method, as Haxe's
// main() does. Expected: BenchInlinedCall -1737346944
class Bench {
  static run() {
    var sum = 0
    var i = 0
    while (i < 100000000) {
      sum = (sum * 31 + (i % 8)) % 4294967296
      i = i + 1
    }
    if (sum >= 2147483648) sum = sum - 4294967296
    return sum
  }
}
var start = System.clock
System.print("BenchInlinedCall %(Bench.run())")
System.print("elapsed: %(System.clock - start)")
