// Fibers nothing holds park on timers while another fiber collects;
// each must come back with its own state intact.
var results = List.filled(8, null)
var done = 0
var sleeper = Fn.new {|tag|
  Fiber.new {
    var mine = [tag, tag + 1]
    var n = 0
    while (n < 3) {
      Browser.setTimeout(15).await
      n = n + 1
    }
    results[tag] = mine[1] * 10 + n
    done = done + 1
  }.call()
}
for (t in 0...8) sleeper.call(t)
Fiber.new {
  var round = 0
  while (round < 40 || done < 8) {
    var junk = []
    for (i in 0...3000) junk.add([i, i])
    System.gc()
    Browser.setTimeout(1).await
    round = round + 1
  }
  System.print(results)
}.call()
