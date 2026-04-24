// Benchmark: Method Call
// Measures method dispatch overhead with a tight loop of simple method calls.
// Based on the standard Wren benchmark from wren-lang/wren.

class Toggle {
  construct new(startState) {
    _state = startState
  }

  value { _state }

  activate {
    _state = !_state
    return this
  }
}

class NthToggle {
  construct new(startState, maxCounter) {
    _state = startState
    _countMax = maxCounter
    _count = 0
  }

  value { _state }

  activate {
    _count = _count + 1
    if (_count >= _countMax) {
      _state = !_state
      _count = 0
    }
    return this
  }
}

var start = System.clock

var n = 100000
var val = true
var toggle = Toggle.new(val)

for (i in 0...n) {
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
}

var ntoggle = NthToggle.new(val, 3)

for (i in 0...n) {
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
}

// Print the final val so the bench-correctness step can diff our
// stdout against standard Wren. Without this, a miscompile that
// silently bails out of the toggle loop would look like a huge speedup.
System.print(val)

var elapsed = System.clock - start
System.print("elapsed: %(elapsed)")
