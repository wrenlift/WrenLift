// Benchmark: Recursive Fibonacci
// Measures function call overhead with deep recursion.
// Based on the standard Wren benchmark from wren-lang/wren.

class Fib {
  static calc(n) {
    if (n < 2) return n
    return Fib.calc(n - 1) + Fib.calc(n - 2)
  }
}

var start = System.clock

for (i in 0...5) {
  Fib.calc(28)
}

var elapsed = System.clock - start
System.print("elapsed: %(elapsed)")
