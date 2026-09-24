// A compiled body's string literals are strings, alone and around an
// interpolated value.
var lit = Fn.new {|i| "lit" }
var around = Fn.new {|i| "a%(i)b" }
var s = null
var t = null
var k = 0
while (k < 300) {
  s = lit.call(k)
  t = around.call(k)
  k = k + 1
}
System.print(s)
System.print(t)
