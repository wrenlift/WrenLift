// Mirrors Ash's BenchFreeCall: the accumulator behind a static call.
// Expected: BenchFreeCall -1737346944
class Bench {
  static step(acc, i) { (acc * 31 + (i % 8)) % 4294967296 }
}
var start = System.clock
var sum = 0
var i = 0
while (i < 100000000) {
  sum = Bench.step(sum, i)
  i = i + 1
}
if (sum >= 2147483648) sum = sum - 4294967296
System.print("BenchFreeCall %(sum)")
System.print("elapsed: %(System.clock - start)")
