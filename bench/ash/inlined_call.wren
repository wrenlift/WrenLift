// Mirrors Ash's BenchInlinedCall: 100M iterations of the accumulator
// with no call. Expected: BenchInlinedCall -1737346944
var start = System.clock
var sum = 0
var i = 0
while (i < 100000000) {
  sum = (sum * 31 + (i % 8)) % 4294967296
  i = i + 1
}
if (sum >= 2147483648) sum = sum - 4294967296
System.print("BenchInlinedCall %(sum)")
System.print("elapsed: %(System.clock - start)")
