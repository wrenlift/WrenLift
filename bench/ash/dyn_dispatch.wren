// Mirrors Ash's BenchDynDispatch: calls through a structural type (duck
// typed here) where every call allocates, against a large retained
// graph so each collection has a real live set to walk. The checksum is
// the accumulator wrapped to a 32-bit signed int. Expected: Checksum: 1148967109
class Prop {
  construct new(id) {
    _id = id
    _cache = []
    for (i in 0...4) _cache.add(i * id)
  }
  id { _id }
  step(dt) {
    var acc = 0
    for (c in _cache) acc = acc + c * dt
    return acc
  }
  tag() { "prop:" + _id.toString }
  children() { [_id, _id + 1] }
}

class Mover {
  construct new(id) { _id = id }
  id { _id }
  step(dt) { _id * dt + (_id % 3) }
  tag() { "mover:" + _id.toString }
  children() {
    var out = []
    for (i in 0...3) out.add(_id * i)
    return out
  }
}

class Bench {
  static tick(world, dt) {
    var sum = 0
    for (e in world) {
      sum = sum + e.step(dt)
      var kids = e.children()
      for (k in kids) sum = sum + k
    }
    return sum
  }

  static names(world) {
    var n = 0
    for (e in world) n = n + e.tag().count
    return n
  }

  static signed32(v) {
    var u = v % 4294967296
    if (u < 0) u = u + 4294967296
    if (u >= 2147483648) return u - 4294967296
    return u
  }

  static run() {
    var retained = []
    for (i in 0...40000) {
      var row = []
      for (j in 0...8) row.add(i + j)
      retained.add(row)
    }
    var world = []
    for (i in 0...64) world.add(i % 2 == 0 ? Prop.new(i) : Mover.new(i))
    var sum = 0
    for (round in 0...20000) {
      sum = sum + tick(world, round % 16)
      if (round % 64 == 0) sum = sum + names(world)
    }
    sum = sum + retained[retained.count - 1][0]
    return signed32(sum)
  }
}

var start = System.clock
System.print("Checksum: %(Bench.run())")
System.print("elapsed: %(System.clock - start)")
