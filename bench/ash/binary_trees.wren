// Mirrors Ash's BenchBinaryTrees (HashLink's BinaryTrees at n = 16):
// one long-lived tree stays rooted while millions of short-lived
// perfect trees are built and checked. Checksum folds every depth's
// check with XOR as a 32-bit signed int. Expected: Checksum: -104864
class TreeNode {
  construct new(left, right, item) {
    _left = left
    _right = right
    _item = item
  }
  itemCheck {
    if (_left == null) return _item
    return _item + _left.itemCheck - _right.itemCheck
  }
}

class Bench {
  static bottomUpTree(item, depth) {
    if (depth > 0) {
      return TreeNode.new(bottomUpTree(2 * item - 1, depth - 1), bottomUpTree(2 * item, depth - 1), item)
    }
    return TreeNode.new(null, null, item)
  }

  // XOR on 32-bit two's complement values, kept as a non-negative
  // Num in [0, 2^32) between folds.
  static xor32(a, b) {
    var ua = a % 4294967296
    if (ua < 0) ua = ua + 4294967296
    var ub = b % 4294967296
    if (ub < 0) ub = ub + 4294967296
    return ua ^ ub
  }

  static signed32(v) {
    if (v >= 2147483648) return v - 4294967296
    return v
  }

  static run() {
    var minDepth = 4
    var n = 16
    var maxDepth = n
    if (minDepth + 2 > maxDepth) maxDepth = minDepth + 2
    var stretchDepth = maxDepth + 1
    var check = bottomUpTree(0, stretchDepth).itemCheck
    var result = check

    var longLivedTree = bottomUpTree(0, maxDepth)
    var depth = minDepth
    while (depth <= maxDepth) {
      var iterations = 1
      var shift = maxDepth - depth + minDepth
      while (shift > 0) {
        iterations = iterations * 2
        shift = shift - 1
      }
      check = 0
      for (i in 0...iterations) {
        check = check + bottomUpTree(i, depth).itemCheck
        check = check + bottomUpTree(-i, depth).itemCheck
      }
      result = xor32(result, check)
      depth = depth + 2
    }

    result = xor32(result, longLivedTree.itemCheck)
    return signed32(result)
  }
}

var start = System.clock
System.print("Checksum: %(Bench.run())")
System.print("elapsed: %(System.clock - start)")
