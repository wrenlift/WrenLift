// From github.com/Brugarolas/WrenJIT scripts/gc.wren at 10c9421 (MIT).
// Unchanged apart from this header.
var start = System.clock

class GcBench {
  static run(iterations) {
    var old = []
    var i = 0
    while (i < 256) {
      old.add(null)
      i = i + 1
    }
    System.gc()

    var checksum = 0
    i = 0
    while (i < iterations) {
      var index = i % 256
      old[index] = [i, i + 1]
      checksum = checksum + old[index][1]
      i = i + 1
    }
    System.gc()
    return checksum + old.count
  }
}

System.print(GcBench.run(500000))
System.print("elapsed: %(System.clock - start)")
