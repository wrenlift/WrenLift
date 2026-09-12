// Mirrors Ash's BenchFib: fib(40), one call. Expected: BenchFib 102334155
class Bench {
  static fib(n) {
    if (n < 2) return n
    return fib(n - 1) + fib(n - 2)
  }
}
var start = System.clock
System.print("BenchFib %(Bench.fib(40))")
System.print("elapsed: %(System.clock - start)")
