// A compiled body keeps a fresh list in a wasm local across a call
// that allocates enough to collect; the list must survive.
var churn = Fn.new {|n|
  var s = 0
  var i = 0
  while (i < n) {
    var t = [i, i + 1]
    s = s + t[0]
    i = i + 1
  }
  return s
}
var hold = Fn.new {|i|
  var keep = [i, i * 2]
  var c = churn.call(2000)
  return keep[0] + keep[1] + c
}
var total = 0
var k = 0
while (k < 3000) {
  total = total + hold.call(k)
  k = k + 1
}
System.print(total)
