// Mirrors Ash's BenchClosureCall: the accumulator behind a closure
// chosen at run time. Expected: BenchClosureCall -1737346944
class Bench {
  static run() {
    var step = Fn.new { |acc, i| (acc * 31 + (i % 8)) % 4294967296 }
    var other = Fn.new { |a, b| a }
    var f = System.clock < 0 ? other : step
    var sum = 0
    var i = 0
    while (i < 100000000) {
      sum = f.call(sum, i)
      i = i + 1
    }
    if (sum >= 2147483648) sum = sum - 4294967296
    return sum
  }
}
var start = System.clock
System.print("BenchClosureCall %(Bench.run())")
System.print("elapsed: %(System.clock - start)")
