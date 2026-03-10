// Benchmark: Binary Trees
// Measures allocation performance and GC pressure.
// Based on the standard Wren benchmark from wren-lang/wren.

class Tree {
  construct new(item, depth) {
    _item = item
    if (depth > 0) {
      var item2 = item + item
      depth = depth - 1
      _left = Tree.new(item2 - 1, depth)
      _right = Tree.new(item2, depth)
    }
  }

  check {
    if (_left == null) return _item
    return _item + _left.check - _right.check
  }
}

var start = System.clock

var minDepth = 4
var maxDepth = 14
var stretchDepth = maxDepth + 1

System.print("stretch tree of depth %(stretchDepth) check: %(Tree.new(0, stretchDepth).check)")

var longLivedTree = Tree.new(0, maxDepth)

var iterations = 1
var d = 0
while (d < maxDepth) {
  iterations = iterations * 2
  d = d + 1
}

var depth = minDepth
while (depth <= maxDepth) {
  var check = 0
  var i = 1
  while (i <= iterations) {
    check = check + Tree.new(i, depth).check + Tree.new(-i, depth).check
    i = i + 1
  }

  System.print("%(iterations * 2) trees of depth %(depth) check: %(check)")
  iterations = (iterations / 4).floor
  depth = depth + 2
}

System.print("long lived tree of depth %(maxDepth) check: %(longLivedTree.check)")

var elapsed = System.clock - start
System.print("elapsed: %(elapsed)")
