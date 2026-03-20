// Reproduction of delta_blue's nested JIT failure pattern.
// Tests: field objects, calling methods on field-referenced objects,
// mutation through nested calls.

class Variable {
  construct new(name, val) {
    _name = name
    _val = val
  }
  value { _val }
  value=(v) { _val = v }
  name { _name }
}

class Constraint {
  construct new(v1, v2) {
    _v1 = v1
    _v2 = v2
  }
  v1 { _v1 }
  v2 { _v2 }
  execute() {
    // Subclasses override this
  }
}

class ScaleConstraint is Constraint {
  construct new(v1, scale, offset, v2) {
    super(v1, v2)
    _scale = scale
    _offset = offset
  }
  execute() {
    _v2.value = _v1.value * _scale.value + _offset.value
  }
}

class Plan {
  construct new() {
    _constraints = []
  }
  add(c) { _constraints.add(c) }
  execute() {
    for (c in _constraints) {
      c.execute()
    }
  }
}

// Setup
var scale = Variable.new("scale", 10)
var offset = Variable.new("offset", 1000)
var src = Variable.new("src", 5)
var dst = Variable.new("dst", 0)

var sc = ScaleConstraint.new(src, scale, offset, dst)
var plan = Plan.new()
plan.add(sc)

// Test: execute plan, check dst value
plan.execute()
if (dst.value != 1050) System.print("FAIL initial: expected 1050, got %(dst.value)")

// Test: change offset and re-execute
offset.value = 2000
plan.execute()
if (dst.value != 2050) System.print("FAIL after offset change: expected 2050, got %(dst.value)")

// Hot loop to trigger JIT
for (i in 0...2000) {
  src.value = i
  plan.execute()
}

// After hot loop, change offset again and verify
offset.value = 3000
plan.execute()
var expected = (2000 - 1) * 10 + 3000
if (dst.value != expected) System.print("FAIL post-loop: expected %(expected), got %(dst.value)")

// The critical test: change offset AFTER JIT compilation
// This is what delta_blue's Projection 4 tests
offset.value = 5000
for (i in 0...100) {
  src.value = i
  plan.execute()
  var exp = i * 10 + 5000
  if (dst.value != exp) {
    System.print("FAIL loop2 i=%(i): expected %(exp), got %(dst.value)")
    break
  }
}

System.print("Done")
