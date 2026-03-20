// Minimal test for nested JIT dispatch correctness.
// Tests: method calls, field access, super() chains, closures.

class Base {
  construct new(val) {
    _val = val
  }
  value { _val }
  value=(v) { _val = v }
  compute() { _val * 2 }
}

class Child is Base {
  construct new(val, offset) {
    super(val)
    _offset = offset
  }
  compute() { super.compute() + _offset }
  offset { _offset }
  offset=(v) { _offset = v }
}

class Container {
  construct new(child) {
    _child = child
  }
  child { _child }
  result() { _child.compute() }
}

// Test 1: basic nested method calls
var c = Child.new(10, 5)
var container = Container.new(c)
var r = container.result()
if (r != 25) System.print("FAIL Test 1: expected 25 got %(r)")

// Test 2: modify and recompute
c.value = 20
r = container.result()
if (r != 45) System.print("FAIL Test 2: expected 45 got %(r)")

// Test 3: modify offset
c.offset = 100
r = container.result()
if (r != 140) System.print("FAIL Test 3: expected 140 got %(r)")

// Test 4: loop to trigger JIT compilation
var total = 0
for (i in 0...1000) {
  var child = Child.new(i, i * 2)
  var cont = Container.new(child)
  total = total + cont.result()
}
if (total != 1998000) System.print("FAIL Test 4: expected 1998000 got %(total)")

// Test 5: repeated modify + compute (exercises nested JIT with state changes)
var obj = Child.new(0, 1000)
var cont2 = Container.new(obj)
var sum = 0
for (i in 0...100) {
  obj.value = i
  obj.offset = i * 5 + 2000
  sum = sum + cont2.result()
}
// sum = Σ(i*2 + i*5 + 2000) for i=0..99 = Σ(7i + 2000) = 7*4950 + 200000 = 34650 + 200000 = 234650
if (sum != 234650) System.print("FAIL Test 5: expected 234650 got %(sum)")

System.print("All tests completed")
