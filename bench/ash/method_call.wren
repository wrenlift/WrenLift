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
var start = System.clock
var s = System.clock < 0 ? Stepper2.new() : Stepper.new()
var sum = 0
var i = 0
while (i < 100000000) {
  sum = s.step(sum, i)
  i = i + 1
}
if (sum >= 2147483648) sum = sum - 4294967296
System.print("BenchMethodCall %(sum)")
System.print("elapsed: %(System.clock - start)")
