// Mirrors Ash's BenchMethodCall: the accumulator behind a virtual call
// whose receiver class is chosen at run time from two live subclasses,
// so the site cannot be devirtualised statically.
// Expected: BenchMethodCall -1737346944
class Stepper {
  construct new() {}
  step(acc, i) { (acc * 31 + (i % 8)) % 4294967296 }
}
class Stepper2 is Stepper {
  construct new() {}
  step(acc, i) { (acc * 31 + (i % 8) + 0) % 4294967296 }
}
class Bench {
  static run() {
    var s = System.clock < 0 ? Stepper2.new() : Stepper.new()
    var sum = 0
    var i = 0
    while (i < 100000000) {
      sum = s.step(sum, i)
      i = i + 1
    }
    if (sum >= 2147483648) sum = sum - 4294967296
    return sum
  }
}
var start = System.clock
System.print("BenchMethodCall %(Bench.run())")
System.print("elapsed: %(System.clock - start)")
