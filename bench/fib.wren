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

var result = 0
for (i in 0...5) {
  result = Fib.calc(28)
}

// Print the result so the bench-correctness step can diff our stdout
// against standard Wren. fib(28) = 317811 — any silent miscompile
// (tail-call mangling, off-by-one in recursion, etc.) will diverge.
System.print(result)

var elapsed = System.clock - start
System.print("elapsed: %(elapsed)")
