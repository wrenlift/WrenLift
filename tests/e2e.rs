/// End-to-end integration tests for the WrenLift runtime.
///
/// These tests exercise complex Wren programs that combine multiple language
/// features. Each test verifies correctness (expected output), stability
/// (no panics), and profiles execution time.
use std::time::Instant;
use wren_lift::runtime::engine::{ExecutionMode, InterpretResult};
use wren_lift::runtime::gc_trait::GcStrategy;
use wren_lift::runtime::vm::{VMConfig, VM};

// ---------------------------------------------------------------------------
// Harness with timing
// ---------------------------------------------------------------------------

/// Run a Wren program and return (result, output, elapsed).
fn run(source: &str) -> (InterpretResult, String, std::time::Duration) {
    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let start = Instant::now();
    let result = vm.interpret("main", source);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    (result, output, elapsed)
}

fn run_with_config(
    source: &str,
    config: VMConfig,
) -> (InterpretResult, String, std::time::Duration) {
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let start = Instant::now();
    let result = vm.interpret("main", source);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    (result, output, elapsed)
}

fn fmt_elapsed(d: std::time::Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.1}ms", ms)
    }
}

fn assert_output(source: &str, expected: &str) {
    let (result, output, elapsed) = run(source);
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "Expected success, got {:?} ({})\nSource:\n{}\nOutput:\n{}",
        result,
        t,
        source,
        output
    );
    assert_eq!(
        output.trim_end(),
        expected.trim_end(),
        "\nSource:\n{}\nElapsed: {}",
        source,
        t
    );
    eprintln!("  [{}]", t);
}

fn assert_success(source: &str) {
    let (result, output, elapsed) = run(source);
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "Expected success, got {:?} ({})\nSource:\n{}\nOutput:\n{}",
        result,
        t,
        source,
        output
    );
    eprintln!("  [{}]", t);
}

fn assert_runtime_error(source: &str) {
    let (result, _, elapsed) = run(source);
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::RuntimeError),
        "Expected runtime error ({}) for:\n{}",
        t,
        source
    );
    eprintln!("  [{}]", t);
}

// ===========================================================================
// 1. Arithmetic & variables
// ===========================================================================

#[test]
fn e2e_arithmetic_expressions() {
    assert_output(
        r#"
var a = 10
var b = 3
System.print(a + b)
System.print(a - b)
System.print(a * b)
System.print(a / b)
System.print(a % b)
System.print(-a)
"#,
        "13\n7\n30\n3.3333333333333335\n1\n-10",
    );
}

#[test]
fn e2e_bitwise_operations() {
    assert_output(
        r#"
System.print(0xff & 0x0f)
System.print(0xf0 | 0x0f)
System.print(0xff ^ 0x0f)
System.print(~0)
System.print(1 << 4)
System.print(256 >> 4)
"#,
        "15\n255\n240\n-1\n16\n16",
    );
}

#[test]
fn e2e_variable_scoping() {
    assert_output(
        r#"
var x = "outer"
{
    var x = "inner"
    System.print(x)
}
System.print(x)
"#,
        "inner\nouter",
    );
}

// ===========================================================================
// 2. Control flow
// ===========================================================================

#[test]
fn e2e_if_else_chain() {
    assert_output(
        r#"
var classify = Fn.new {|n|
    if (n < 0) {
        return "negative"
    } else if (n == 0) {
        return "zero"
    } else {
        return "positive"
    }
}
System.print(classify.call(-5))
System.print(classify.call(0))
System.print(classify.call(42))
"#,
        "negative\nzero\npositive",
    );
}

#[test]
fn e2e_while_loop_accumulator() {
    assert_output(
        r#"
var sum = 0
var i = 1
while (i <= 100) {
    sum = sum + i
    i = i + 1
}
System.print(sum)
"#,
        "5050",
    );
}

#[test]
fn e2e_for_in_range() {
    // 1..10 is inclusive in Wren (1 through 10), use 1...10 for exclusive
    assert_output(
        r#"
var sum = 0
for (i in 1..10) {
    sum = sum + i
}
System.print(sum)
"#,
        "55",
    );
}

#[test]
fn e2e_for_in_list() {
    assert_output(
        r#"
var words = ["hello", "world", "wren"]
var result = ""
for (w in words) {
    if (result != "") result = result + " "
    result = result + w
}
System.print(result)
"#,
        "hello world wren",
    );
}

// ===========================================================================
// 3. Classes — construct, fields, methods, getters, setters
// ===========================================================================

#[test]
fn e2e_class_point() {
    assert_output(
        r#"
class Point {
    construct new(x, y) {
        _x = x
        _y = y
    }
    x { _x }
    y { _y }
    toString { "(" + _x.toString + ", " + _y.toString + ")" }
    + (other) { Point.new(_x + other.x, _y + other.y) }
    == (other) { _x == other.x && _y == other.y }
}

var a = Point.new(1, 2)
var b = Point.new(3, 4)
var c = a + b
System.print(c.toString)
System.print(a == Point.new(1, 2))
System.print(a == b)
"#,
        "(4, 6)\ntrue\nfalse",
    );
}

#[test]
fn e2e_class_setter() {
    assert_output(
        r#"
class Counter {
    construct new() { _count = 0 }
    count { _count }
    count=(value) { _count = value }
    increment() { _count = _count + 1 }
}

var c = Counter.new()
c.increment()
c.increment()
c.increment()
System.print(c.count)
c.count = 10
System.print(c.count)
"#,
        "3\n10",
    );
}

// ===========================================================================
// 4. Inheritance & super
// ===========================================================================

#[test]
fn e2e_inheritance_chain() {
    assert_output(
        r#"
class Animal {
    construct new(name) { _name = name }
    name { _name }
    speak() { return "..." }
    toString { return _name + " says " + this.speak() }
}

class Dog is Animal {
    construct new(name) { super(name) }
    speak() { return "woof" }
}

class Cat is Animal {
    construct new(name) { super(name) }
    speak() { return "meow" }
}

var animals = [Dog.new("Rex"), Cat.new("Whiskers"), Dog.new("Buddy")]
for (a in animals) {
    System.print(a.toString)
}
"#,
        "Rex says woof\nWhiskers says meow\nBuddy says woof",
    );
}

#[test]
fn e2e_super_in_method() {
    assert_output(
        r#"
class Base {
    construct new() {}
    greet(name) { "Hello, " + name }
}

class Derived is Base {
    construct new() { super() }
    greet(name) { super.greet(name) + "!" }
}

System.print(Derived.new().greet("world"))
"#,
        "Hello, world!",
    );
}

#[test]
fn e2e_is_operator_hierarchy() {
    // Test is with 2-level hierarchy (3-level super chain not yet supported)
    assert_output(
        r#"
class A {
    construct new() {}
}
class B is A {
    construct new() { super() }
}

var b = B.new()
System.print(b is B)
System.print(b is A)
System.print(b is Object)
System.print(b is Num)
"#,
        "true\ntrue\ntrue\nfalse",
    );
}

// ===========================================================================
// 5. Closures & upvalues
// ===========================================================================

#[test]
fn e2e_closure_counter() {
    assert_output(
        r#"
var makeCounter = Fn.new {
    var count = 0
    return Fn.new {
        count = count + 1
        return count
    }
}

var counter = makeCounter.call()
System.print(counter.call())
System.print(counter.call())
System.print(counter.call())
"#,
        "1\n2\n3",
    );
}

#[test]
fn e2e_closure_captures_loop_var() {
    assert_output(
        r#"
var fns = []
for (i in 0...3) {
    var captured = i
    fns.add(Fn.new { captured })
}
for (f in fns) {
    System.print(f.call())
}
"#,
        "0\n1\n2",
    );
}

#[test]
fn e2e_closure_as_callback() {
    // Test map with closure callback
    assert_output(
        r#"
class MyList {
    construct new(items) { _items = items }
    map(fn) {
        var result = []
        for (item in _items) {
            result.add(fn.call(item))
        }
        return result
    }
}

var nums = MyList.new([1, 2, 3, 4, 5])
var doubled = nums.map(Fn.new {|x| x * 2 })
System.print(doubled)
"#,
        "[2, 4, 6, 8, 10]",
    );
}

// ===========================================================================
// 6. Fibers
// ===========================================================================

#[test]
fn e2e_fiber_generator() {
    assert_output(
        r#"
var gen = Fiber.new {
    Fiber.yield(1)
    Fiber.yield(2)
    Fiber.yield(3)
}
System.print(gen.call())
System.print(gen.call())
System.print(gen.call())
System.print(gen.isDone)
"#,
        "1\n2\n3\nfalse",
    );
}

#[test]
fn e2e_fiber_coroutine_ping_pong() {
    assert_output(
        r#"
var log = []

var worker = Fiber.new {
    log.add("worker: started")
    var input = Fiber.yield("ready")
    log.add("worker: got " + input)
    Fiber.yield("done")
}

log.add("main: starting worker")
var status = worker.call()
log.add("main: worker said " + status)
var result = worker.call("task-1")
log.add("main: worker said " + result)

for (entry in log) {
    System.print(entry)
}
"#,
        "main: starting worker\nworker: started\nmain: worker said ready\nworker: got task-1\nmain: worker said done",
    );
}

#[test]
fn e2e_fiber_is_done_lifecycle() {
    assert_output(
        r#"
var fib = Fiber.new { Fiber.yield(42) }
System.print(fib.isDone)
fib.call()
System.print(fib.isDone)
fib.call()
System.print(fib.isDone)
"#,
        "false\nfalse\ntrue",
    );
}

// ===========================================================================
// 7. Strings & interpolation
// ===========================================================================

#[test]
fn e2e_string_interpolation() {
    assert_output(
        r#"
var name = "Wren"
System.print("Hello, %(name)!")
System.print("2 + 3 = %(2 + 3)")
"#,
        "Hello, Wren!\n2 + 3 = 5",
    );
}

#[test]
fn e2e_string_methods() {
    assert_output(
        r#"
var s = "Hello, World!"
System.print(s.count)
System.print(s.contains("World"))
System.print(s.contains("wren"))
"#,
        "13\ntrue\nfalse",
    );
}

// ===========================================================================
// 8. Lists & maps
// ===========================================================================

#[test]
fn e2e_list_operations() {
    assert_output(
        r#"
var list = [3, 1, 4, 1, 5, 9, 2, 6]
System.print(list.count)
list.sort()
var s = ""
for (n in list) s = s + n.toString + " "
System.print(s)
System.print(list[0])
System.print(list[-1])
"#,
        "8\n1 1 2 3 4 5 6 9 \n1\n9",
    );
}

#[test]
fn e2e_map_operations() {
    assert_output(
        r#"
var map = {"name": "Wren", "version": 1}
System.print(map["name"])
System.print(map.count)
map["author"] = "Bob"
System.print(map.count)
System.print(map.containsKey("author"))
System.print(map.containsKey("missing"))
"#,
        "Wren\n2\n3\ntrue\nfalse",
    );
}

// ===========================================================================
// 9. Recursive algorithms
// ===========================================================================

#[test]
fn e2e_recursive_fibonacci() {
    assert_output(
        r#"
class Math {
    static fib(n) {
        if (n <= 1) return n
        return Math.fib(n - 1) + Math.fib(n - 2)
    }
}
System.print(Math.fib(0))
System.print(Math.fib(1))
System.print(Math.fib(10))
System.print(Math.fib(15))
"#,
        "0\n1\n55\n610",
    );
}

#[test]
fn e2e_recursive_factorial() {
    assert_output(
        r#"
class Math {
    static factorial(n) {
        if (n <= 1) return 1
        return n * Math.factorial(n - 1)
    }
}
System.print(Math.factorial(1))
System.print(Math.factorial(5))
System.print(Math.factorial(10))
"#,
        "1\n120\n3628800",
    );
}

// ===========================================================================
// 10. Complex multi-feature programs
// ===========================================================================

#[test]
fn e2e_linked_list() {
    assert_output(
        r#"
class Node {
    construct new(value, next) {
        _value = value
        _next = next
    }
    value { _value }
    next { _next }
}

class LinkedList {
    construct new() {
        _head = null
        _count = 0
    }
    count { _count }
    push(value) {
        _head = Node.new(value, _head)
        _count = _count + 1
    }
    toList() {
        var result = []
        var node = _head
        while (node != null) {
            result.add(node.value)
            node = node.next
        }
        return result
    }
}

var list = LinkedList.new()
list.push(3)
list.push(2)
list.push(1)
System.print(list.count)
System.print(list.toList())
"#,
        "3\n[1, 2, 3]",
    );
}

#[test]
fn e2e_state_machine() {
    assert_output(
        r#"
class StateMachine {
    construct new() {
        _state = "idle"
        _log = []
    }
    state { _state }
    transition(event) {
        if (_state == "idle" && event == "start") {
            _state = "running"
            _log.add(_state)
        } else if (_state == "running" && event == "pause") {
            _state = "paused"
            _log.add(_state)
        } else if (_state == "paused" && event == "resume") {
            _state = "running"
            _log.add(_state)
        } else if (_state == "running" && event == "stop") {
            _state = "idle"
            _log.add(_state)
        } else {
            _log.add("invalid: " + _state + " + " + event)
        }
    }
    log { _log }
}

var sm = StateMachine.new()
sm.transition("start")
sm.transition("pause")
sm.transition("resume")
sm.transition("stop")
sm.transition("pause")
for (entry in sm.log) {
    System.print(entry)
}
"#,
        "running\npaused\nrunning\nidle\ninvalid: idle + pause",
    );
}

#[test]
fn e2e_iterator_protocol() {
    assert_output(
        r#"
class Range2 {
    construct new(from, to) {
        _from = from
        _to = to
    }
    iterate(iter) {
        if (iter == null) return _from
        var next = iter + 1
        if (next >= _to) return false
        return next
    }
    iteratorValue(iter) { iter }
}

var sum = 0
for (i in Range2.new(0, 5)) {
    sum = sum + i
}
System.print(sum)
"#,
        "10",
    );
}

#[test]
fn e2e_observer_pattern() {
    assert_output(
        r#"
class EventEmitter {
    construct new() { _listeners = {} }
    on(event, fn) {
        if (!_listeners.containsKey(event)) {
            _listeners[event] = []
        }
        _listeners[event].add(fn)
    }
    emit(event, data) {
        if (_listeners.containsKey(event)) {
            for (fn in _listeners[event]) {
                fn.call(data)
            }
        }
    }
}

var log = []
var emitter = EventEmitter.new()
emitter.on("greet", Fn.new {|name| log.add("Hello, " + name + "!") })
emitter.on("greet", Fn.new {|name| log.add("Hi " + name) })
emitter.on("bye", Fn.new {|name| log.add("Goodbye, " + name) })

emitter.emit("greet", "Alice")
emitter.emit("bye", "Bob")
emitter.emit("unknown", "X")

for (entry in log) {
    System.print(entry)
}
"#,
        "Hello, Alice!\nHi Alice\nGoodbye, Bob",
    );
}

#[test]
fn e2e_builder_pattern() {
    assert_output(
        r#"
class QueryBuilder {
    construct new() {
        _table = ""
        _conditions = []
        _limit = null
    }
    from(table) {
        _table = table
        return this
    }
    where_(cond) {
        _conditions.add(cond)
        return this
    }
    limit(n) {
        _limit = n
        return this
    }
    build() {
        var q = "SELECT * FROM " + _table
        if (_conditions.count > 0) {
            q = q + " WHERE " + _conditions[0]
            for (i in 1..._conditions.count) {
                q = q + " AND " + _conditions[i]
            }
        }
        if (_limit != null) {
            q = q + " LIMIT " + _limit.toString
        }
        return q
    }
}

var query = QueryBuilder.new().from("users").where_("age > 18").where_("active = true").limit(10).build()
System.print(query)
"#,
        "SELECT * FROM users WHERE age > 18 AND active = true LIMIT 10",
    );
}

// ===========================================================================
// 11. Edge cases & stability
// ===========================================================================

#[test]
fn e2e_empty_class() {
    assert_success(
        r#"
class Empty {
    construct new() {}
}
var e = Empty.new()
System.print(e is Empty)
"#,
    );
}

#[test]
fn e2e_deeply_nested_calls() {
    assert_output(
        r#"
class Wrapper {
    construct new(value) { _value = value }
    value { _value }
    wrap() { Wrapper.new(this) }
}

var w = Wrapper.new(42)
w = w.wrap().wrap().wrap().wrap().wrap()
System.print(w.value.value.value.value.value.value)
"#,
        "42",
    );
}

#[test]
fn e2e_many_local_variables() {
    assert_output(
        r#"
var a = 1
var b = 2
var c = 3
var d = 4
var e = 5
var f = 6
var g = 7
var h = 8
var i = 9
var j = 10
System.print(a + b + c + d + e + f + g + h + i + j)
"#,
        "55",
    );
}

#[test]
fn e2e_null_handling() {
    assert_output(
        r#"
System.print(null == null)
System.print(null != null)
System.print(null == false)
System.print(null == 0)
System.print(null.toString)
"#,
        "true\nfalse\nfalse\nfalse\nnull",
    );
}

#[test]
fn e2e_boolean_logic() {
    assert_output(
        r#"
System.print(true && true)
System.print(true && false)
System.print(false || true)
System.print(false || false)
System.print(!true)
System.print(!false)
System.print(!null)
System.print(!0)
"#,
        "true\nfalse\ntrue\nfalse\nfalse\ntrue\ntrue\nfalse",
    );
}

#[test]
fn e2e_type_checks_all_types() {
    assert_output(
        r#"
System.print(42 is Num)
System.print("hi" is String)
System.print(true is Bool)
System.print(null is Null)
System.print([1,2] is List)
System.print((1..3) is Range)
System.print(Fn.new {} is Fn)
"#,
        "true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue",
    );
}

// ===========================================================================
// 12. GC pressure — allocate many objects
// ===========================================================================

#[test]
fn e2e_gc_pressure_many_objects() {
    assert_output(
        r#"
class Box {
    construct new(value) { _value = value }
    value { _value }
}

var last = null
for (i in 0...1000) {
    last = Box.new(i)
}
System.print(last.value)
"#,
        "999",
    );
}

#[test]
fn e2e_gc_pressure_string_concat() {
    assert_output(
        r#"
var s = ""
for (i in 0...100) {
    s = s + "x"
}
System.print(s.count)
"#,
        "100",
    );
}

// ===========================================================================
// 13. Imports
// ===========================================================================

#[test]
fn e2e_module_import() {
    let config = VMConfig {
        load_module_fn: Some(Box::new(|name: &str| -> Option<String> {
            if name == "math_helpers" {
                Some(
                    r#"
class MathHelpers {
    static square(n) { n * n }
    static cube(n) { n * n * n }
}
"#
                    .to_string(),
                )
            } else {
                None
            }
        })),
        ..Default::default()
    };

    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let start = Instant::now();
    let result = vm.interpret(
        "main",
        r#"
import "math_helpers" for MathHelpers
System.print(MathHelpers.square(5))
System.print(MathHelpers.cube(3))
"#,
    );
    let elapsed = start.elapsed();
    let output = vm.take_output();
    let t = fmt_elapsed(elapsed);
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output.trim_end(), "25\n27");
    eprintln!("  [{}]", t);
}

// ===========================================================================
// 14. Error cases
// ===========================================================================

#[test]
fn e2e_error_undefined_variable() {
    let (result, _, elapsed) = run("System.print(undefined_var)");
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(
            result,
            InterpretResult::CompileError | InterpretResult::RuntimeError
        ),
        "Expected error for undefined variable ({})",
        t
    );
    eprintln!("  [{}]", t);
}

#[test]
fn e2e_error_method_not_found() {
    assert_runtime_error(
        r#"
var x = 42
x.nonExistentMethod()
"#,
    );
}

// ===========================================================================
// 15. Complex algorithmic programs
// ===========================================================================

#[test]
fn e2e_bubble_sort() {
    assert_output(
        r#"
class Sorter {
    static bubbleSort(list) {
        var n = list.count
        var i = 0
        while (i < n - 1) {
            var j = 0
            while (j < n - i - 1) {
                if (list[j] > list[j + 1]) {
                    var temp = list[j]
                    list[j] = list[j + 1]
                    list[j + 1] = temp
                }
                j = j + 1
            }
            i = i + 1
        }
        return list
    }
}

var arr = [64, 34, 25, 12, 22, 11, 90]
Sorter.bubbleSort(arr)
var s = ""
for (n in arr) s = s + n.toString + " "
System.print(s)
"#,
        "11 12 22 25 34 64 90 ",
    );
}

#[test]
fn e2e_binary_search() {
    assert_output(
        r#"
class Search {
    static binary(list, target) {
        var lo = 0
        var hi = list.count - 1
        while (lo <= hi) {
            var mid = ((lo + hi) / 2).floor
            if (list[mid] == target) return mid
            if (list[mid] < target) {
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        return -1
    }
}

var sorted = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
System.print(Search.binary(sorted, 23))
System.print(Search.binary(sorted, 2))
System.print(Search.binary(sorted, 91))
System.print(Search.binary(sorted, 99))
"#,
        "5\n0\n9\n-1",
    );
}

#[test]
fn e2e_fizzbuzz() {
    assert_output(
        r#"
for (i in 1...16) {
    if (i % 15 == 0) {
        System.print("FizzBuzz")
    } else if (i % 3 == 0) {
        System.print("Fizz")
    } else if (i % 5 == 0) {
        System.print("Buzz")
    } else {
        System.print(i)
    }
}
"#,
        "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz",
    );
}

#[test]
fn e2e_tower_of_hanoi() {
    assert_output(
        r#"
class Hanoi {
    construct new() { _moves = [] }
    moves { _moves }
    solve(n, from, to, aux) {
        if (n == 1) {
            _moves.add(from + "->" + to)
            return
        }
        this.solve(n - 1, from, aux, to)
        _moves.add(from + "->" + to)
        this.solve(n - 1, aux, to, from)
    }
}

var h = Hanoi.new()
h.solve(3, "A", "C", "B")
System.print(h.moves.count)
for (m in h.moves) System.print(m)
"#,
        "7\nA->C\nA->B\nC->B\nA->C\nB->A\nB->C\nA->C",
    );
}

// ===========================================================================
// 16. Fiber-based cooperative patterns
// ===========================================================================

#[test]
fn e2e_fiber_range_generator() {
    assert_output(
        r#"
var rangeGen = Fn.new {|from, to|
    return Fiber.new {
        var i = from
        while (i < to) {
            Fiber.yield(i)
            i = i + 1
        }
    }
}

var fib = rangeGen.call(5, 10)
var s = ""
while (!fib.isDone) {
    var val = fib.call()
    if (val != null) s = s + val.toString + " "
}
System.print(s)
"#,
        "5 6 7 8 9 ",
    );
}

// ===========================================================================
// 17. Class with multiple constructors
// ===========================================================================

#[test]
fn e2e_named_constructors() {
    assert_output(
        r#"
class Color {
    construct rgb(r, g, b) {
        _r = r
        _g = g
        _b = b
    }
    construct white() {
        _r = 255
        _g = 255
        _b = 255
    }
    construct black() {
        _r = 0
        _g = 0
        _b = 0
    }
    toString { "(" + _r.toString + ", " + _g.toString + ", " + _b.toString + ")" }
}

System.print(Color.rgb(128, 64, 32).toString)
System.print(Color.white().toString)
System.print(Color.black().toString)
"#,
        "(128, 64, 32)\n(255, 255, 255)\n(0, 0, 0)",
    );
}

// ===========================================================================
// 18. Operator overloading
// ===========================================================================

#[test]
fn e2e_operator_overloading() {
    assert_output(
        r#"
class Vec2 {
    construct new(x, y) {
        _x = x
        _y = y
    }
    x { _x }
    y { _y }
    + (other) { Vec2.new(_x + other.x, _y + other.y) }
    - (other) { Vec2.new(_x - other.x, _y - other.y) }
    * (scalar) { Vec2.new(_x * scalar, _y * scalar) }
    - { Vec2.new(-_x, -_y) }
    == (other) { _x == other.x && _y == other.y }
    toString { "<" + _x.toString + ", " + _y.toString + ">" }
}

var a = Vec2.new(1, 2)
var b = Vec2.new(3, 4)
System.print((a + b).toString)
System.print((b - a).toString)
System.print((a * 3).toString)
System.print((-a).toString)
System.print(a == Vec2.new(1, 2))
"#,
        "<4, 6>\n<2, 2>\n<3, 6>\n<-1, -2>\ntrue",
    );
}

// ===========================================================================
// 19. Subscript operator
// ===========================================================================

#[test]
fn e2e_subscript_overloading() {
    assert_output(
        r#"
class Matrix {
    construct new(rows, cols) {
        _rows = rows
        _cols = cols
        _data = []
        var i = 0
        while (i < rows * cols) {
            _data.add(0)
            i = i + 1
        }
    }
    [row, col] { _data[row * _cols + col] }
    [row, col]=(value) { _data[row * _cols + col] = value }
    rows { _rows }
    cols { _cols }
}

var m = Matrix.new(2, 3)
m[0, 0] = 1
m[0, 1] = 2
m[0, 2] = 3
m[1, 0] = 4
m[1, 1] = 5
m[1, 2] = 6
System.print(m[0, 0])
System.print(m[1, 2])
System.print(m[0, 1] + m[1, 0])
"#,
        "1\n6\n6",
    );
}

// ===========================================================================
// 20. Closures + classes combined
// ===========================================================================

#[test]
fn e2e_strategy_pattern() {
    assert_output(
        r#"
var doubler = Fn.new {|x| x * 2 }
var squarer = Fn.new {|x| x * x }

var input = [1, 2, 3, 4, 5]

var d = []
for (x in input) d.add(doubler.call(x))
System.print(d)

var s = []
for (x in input) s.add(squarer.call(x))
System.print(s)
"#,
        "[2, 4, 6, 8, 10]\n[1, 4, 9, 16, 25]",
    );
}

#[test]
fn e2e_memoized_fibonacci() {
    assert_output(
        r#"
var cache = {}
var fib = null
fib = Fn.new {|n|
    if (cache.containsKey(n)) return cache[n]
    var result
    if (n <= 1) {
        result = n
    } else {
        result = fib.call(n - 1) + fib.call(n - 2)
    }
    cache[n] = result
    return result
}

System.print(fib.call(20))
System.print(fib.call(30))
"#,
        "6765\n832040",
    );
}

// ===========================================================================
// 21. Performance benchmarks (timed)
// ===========================================================================

#[test]
fn e2e_bench_fib25() {
    assert_output(
        r#"
class Fib {
    static calc(n) {
        if (n <= 1) return n
        return Fib.calc(n - 1) + Fib.calc(n - 2)
    }
}
System.print(Fib.calc(25))
"#,
        "75025",
    );
}

#[test]
fn e2e_bench_loop_1m() {
    assert_output(
        r#"
var sum = 0
var i = 0
while (i < 1000000) {
    sum = sum + i
    i = i + 1
}
System.print(sum)
"#,
        "499999500000",
    );
}

#[test]
fn e2e_bench_gc_pressure_10k() {
    assert_output(
        r#"
class Node {
    construct new(v, n) {
        _v = v
        _n = n
    }
    value { _v }
    next { _n }
}

var head = null
for (i in 0...10000) {
    head = Node.new(i, head)
}

var count = 0
var cur = head
while (cur != null) {
    count = count + 1
    cur = cur.next
}
System.print(count)
"#,
        "10000",
    );
}

// ---------------------------------------------------------------------------
// Static fields
// ---------------------------------------------------------------------------

#[test]
fn e2e_static_fields() {
    assert_output(
        r#"
class Counter {
    static increment() {
        __count = __count + 1
    }
    static count { __count }
    static reset() {
        __count = 0
    }
}

Counter.reset()
Counter.increment()
Counter.increment()
Counter.increment()
System.print(Counter.count)
"#,
        "3",
    );
}

// ---------------------------------------------------------------------------
// Compound assignment on subscript
// ---------------------------------------------------------------------------

#[test]
fn e2e_compound_assign_subscript() {
    assert_output(
        r#"
var list = [10, 20, 30]
list[1] = list[1] + 5
System.print(list[1])
"#,
        "25",
    );
}

// ---------------------------------------------------------------------------
// Fiber.try — catches runtime errors
// ---------------------------------------------------------------------------

#[test]
fn e2e_fiber_try() {
    assert_output(
        r#"
var fiber = Fiber.new {
    Fiber.abort("something went wrong")
}
var result = fiber.try()
System.print(fiber.error)
"#,
        "something went wrong",
    );
}

// ---------------------------------------------------------------------------
// System.gc() — explicit garbage collection
// ---------------------------------------------------------------------------

#[test]
fn e2e_system_gc() {
    assert_output(
        r#"
System.print("before")
System.gc()
System.print("after")
"#,
        "before\nafter",
    );
}

// ---------------------------------------------------------------------------
// GC under allocation pressure — objects survive collection
// ---------------------------------------------------------------------------

#[test]
fn e2e_gc_objects_survive() {
    assert_output(
        r#"
var list = []
for (i in 0...1000) {
    list.add("item %(i)")
}
System.gc()
System.print(list.count)
System.print(list[999])
"#,
        "1000\nitem 999",
    );
}

// ---------------------------------------------------------------------------
// Circular import detection
// ---------------------------------------------------------------------------

#[test]
fn e2e_circular_import_detected() {
    let config = VMConfig {
        load_module_fn: Some(Box::new(|name: &str| -> Option<String> {
            match name {
                "module_a" => Some(r#"import "module_b" for B"#.to_string()),
                "module_b" => Some(r#"import "module_a" for A"#.to_string()),
                _ => None,
            }
        })),
        ..Default::default()
    };

    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", r#"import "module_a" for A"#);
    assert!(
        matches!(
            result,
            InterpretResult::CompileError | InterpretResult::RuntimeError
        ),
        "Expected error for circular import, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Configurable step limit
// ---------------------------------------------------------------------------

#[test]
fn e2e_custom_step_limit() {
    let config = VMConfig {
        step_limit: 100,
        ..Default::default()
    };

    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret(
        "main",
        r#"
while (true) {
    var x = 1
}
"#,
    );
    assert!(
        matches!(result, InterpretResult::RuntimeError),
        "Expected step limit error, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// JIT tiering e2e: nbody simulation
// ---------------------------------------------------------------------------

const NBODY_SRC: &str = r#"
class Vec3 {
    construct new(x, y, z) {
        _x = x
        _y = y
        _z = z
    }
    x { _x }
    y { _y }
    z { _z }
    x=(v) { _x = v }
    y=(v) { _y = v }
    z=(v) { _z = v }
}

class Body {
    construct new(x, y, z, vx, vy, vz, mass) {
        _pos = Vec3.new(x, y, z)
        _vel = Vec3.new(vx, vy, vz)
        _mass = mass
    }
    pos { _pos }
    vel { _vel }
    mass { _mass }
}

var PI = 3.141592653589793
var SOLAR_MASS = 4 * PI * PI
var DAYS_PER_YEAR = 365.24

var bodies = [
    Body.new(0, 0, 0, 0, 0, 0, SOLAR_MASS),
    Body.new(
        4.84143144246472090,
        -1.16032004402742839,
        -0.10362204447112311,
        0.00166007664274403694 * DAYS_PER_YEAR,
        0.00769901118419740425 * DAYS_PER_YEAR,
        -0.00006904600169720200 * DAYS_PER_YEAR,
        0.000954791938424326609 * SOLAR_MASS
    ),
    Body.new(
        8.34336671824457987,
        4.12479856412430479,
        -0.40352341895349131,
        -0.00276742510726862411 * DAYS_PER_YEAR,
        0.00499852801234917238 * DAYS_PER_YEAR,
        0.00023041729757376393 * DAYS_PER_YEAR,
        0.000285885980666130812 * SOLAR_MASS
    )
]

var n = bodies.count

// Advance simulation by dt
var advance = Fn.new { |dt|
    for (i in 0...n) {
        var bi = bodies[i]
        for (j in (i + 1)...n) {
            var bj = bodies[j]
            var dx = bi.pos.x - bj.pos.x
            var dy = bi.pos.y - bj.pos.y
            var dz = bi.pos.z - bj.pos.z
            var dist2 = dx * dx + dy * dy + dz * dz
            var dist = dist2.sqrt
            var mag = dt / (dist2 * dist)
            bi.vel.x = bi.vel.x - dx * bj.mass * mag
            bi.vel.y = bi.vel.y - dy * bj.mass * mag
            bi.vel.z = bi.vel.z - dz * bj.mass * mag
            bj.vel.x = bj.vel.x + dx * bi.mass * mag
            bj.vel.y = bj.vel.y + dy * bi.mass * mag
            bj.vel.z = bj.vel.z + dz * bi.mass * mag
        }
    }
    for (i in 0...n) {
        var b = bodies[i]
        b.pos.x = b.pos.x + dt * b.vel.x
        b.pos.y = b.pos.y + dt * b.vel.y
        b.pos.z = b.pos.z + dt * b.vel.z
    }
}

// Compute total energy
var energy = Fn.new {
    var e = 0
    for (i in 0...n) {
        var bi = bodies[i]
        var vx = bi.vel.x
        var vy = bi.vel.y
        var vz = bi.vel.z
        e = e + 0.5 * bi.mass * (vx * vx + vy * vy + vz * vz)
        for (j in (i + 1)...n) {
            var bj = bodies[j]
            var dx = bi.pos.x - bj.pos.x
            var dy = bi.pos.y - bj.pos.y
            var dz = bi.pos.z - bj.pos.z
            var dist = (dx * dx + dy * dy + dz * dz).sqrt
            e = e - bi.mass * bj.mass / dist
        }
    }
    return e
}

var e0 = energy.call
for (i in 0...200) {
    advance.call(0.01)
}
var e1 = energy.call

// Energy should be conserved (small drift okay)
var drift = (e1 - e0).abs
System.print(drift < 0.001)
System.print("done")
"#;

#[test]
fn e2e_jit_nbody() {
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 5,
        ..Default::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let start = Instant::now();
    let result = vm.interpret("main", NBODY_SRC);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "nbody failed: {:?} ({})\n{}",
        result,
        t,
        output
    );
    assert_eq!(output.trim(), "true\ndone", "nbody output mismatch ({})", t);
    eprintln!("  [nbody tiered {}]", t);
}

#[test]
fn e2e_tiered_closure_call_inside_promoted_function() {
    let source = r#"
var apply = Fn.new { |f, x|
  return f.call(x)
}

var inc = Fn.new { |x| x + 1 }

var total = 0
for (i in 0...20) {
  total = total + apply.call(inc, i)
}

System.print(total)
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered closure-call promotion failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "210",
        "tiered closure-call output mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_nested_nonleaf_closure_call_survives_gc_pressure() {
    let source = r#"
var outer = Fn.new { |f, list|
  for (i in 0...400) {
    var tmp = [i, i + 1, i + 2, i + 3]
  }
  return f.call(list)
}

var inner = Fn.new { |list|
  list.add(1)
  return list.count
}

var list = []
var total = 0
for (i in 0...40) {
  total = total + outer.call(inner, list)
}

System.print(total)
System.print(list.count)
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered nested non-leaf closure-call failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "820\n40",
        "tiered nested non-leaf closure-call output mismatch ({})",
        t
    );
}

// ---------------------------------------------------------------------------
// JIT tiering e2e: mandelbrot
// ---------------------------------------------------------------------------

const MANDELBROT_SRC: &str = r##"
var WIDTH = 16
var HEIGHT = 12
var MAX_ITER = 20

var output = ""
for (py in 0...HEIGHT) {
    var y0 = py / HEIGHT * 2.4 - 1.2
    for (px in 0...WIDTH) {
        var x0 = px / WIDTH * 3.5 - 2.5
        var x = 0
        var y = 0
        var iter = 0
        while (x * x + y * y <= 4 && iter < MAX_ITER) {
            var xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iter = iter + 1
        }
        if (iter == MAX_ITER) {
            output = output + "#"
        } else if (iter > 10) {
            output = output + "+"
        } else if (iter > 5) {
            output = output + "."
        } else {
            output = output + " "
        }
    }
    output = output + "\n"
}
System.print(output)
"##;

#[test]
fn e2e_jit_mandelbrot() {
    // Run in both interpreter and tiered mode, verify same output.
    let (interp_result, interp_output, _) = {
        let config = VMConfig {
            execution_mode: ExecutionMode::Interpreter,
            ..Default::default()
        };
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let start = Instant::now();
        let r = vm.interpret("main", MANDELBROT_SRC);
        let elapsed = start.elapsed();
        (r, vm.take_output(), elapsed)
    };

    let (tiered_result, tiered_output, tiered_elapsed) = {
        let config = VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 3,
            ..Default::default()
        };
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let start = Instant::now();
        let r = vm.interpret("main", MANDELBROT_SRC);
        let elapsed = start.elapsed();
        (r, vm.take_output(), elapsed)
    };

    let t = fmt_elapsed(tiered_elapsed);
    assert!(
        matches!(interp_result, InterpretResult::Success),
        "mandelbrot interpreter failed: {:?}",
        interp_result
    );
    assert!(
        matches!(tiered_result, InterpretResult::Success),
        "mandelbrot tiered failed: {:?} ({})",
        tiered_result,
        t
    );
    assert_eq!(
        interp_output, tiered_output,
        "mandelbrot output mismatch between interpreter and tiered ({})",
        t
    );
    // Verify we got non-empty output with expected characters
    assert!(
        tiered_output.contains('#') && tiered_output.contains(' '),
        "mandelbrot output seems wrong: {}",
        tiered_output
    );
    eprintln!("  [mandelbrot tiered {}]", t);
}

// ---------------------------------------------------------------------------
// Stack overflow detection
// ---------------------------------------------------------------------------

#[test]
fn e2e_stack_overflow() {
    let config = VMConfig {
        max_call_depth: 64,
        ..Default::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let source = r#"
class Boom {
    static go(n) {
        Boom.go(n + 1)
    }
}
Boom.go(0)
"#;
    let result = vm.interpret("main", source);
    // Should fail with a runtime error, not panic
    assert!(
        result != InterpretResult::Success,
        "infinite recursion should not succeed"
    );
    let output = vm.take_output();
    eprintln!(
        "  [stack overflow detected: result={:?}, output={:?}]",
        result, output
    );
}

// ---------------------------------------------------------------------------
// Optional module: random
// ---------------------------------------------------------------------------

#[test]
fn e2e_random_module() {
    let source = r#"
import "random" for Random

var rng = Random.new(12345)
var a = rng.float()
System.print(a > 0)       // true (float in [0,1))
System.print(a < 1)       // true

var b = rng.int(100)
System.print(b >= 0)      // true
System.print(b < 100)     // true

var c = rng.int(10, 20)
System.print(c >= 10)     // true
System.print(c < 20)      // true

var d = rng.float(5)
System.print(d >= 0)      // true
System.print(d < 5)       // true

var e = rng.float(2, 8)
System.print(e >= 2)      // true
System.print(e < 8)       // true
"#;
    let (result, output, elapsed) = run(source);
    assert!(
        matches!(result, InterpretResult::Success),
        "random module failed: {:?}",
        result
    );
    // All lines should be "true"
    for (i, line) in output.lines().enumerate() {
        assert_eq!(line, "true", "line {} was not true: {}", i, line);
    }
    eprintln!("  [random module {}]", fmt_elapsed(elapsed));
}

#[test]
fn e2e_random_deterministic() {
    // Same seed should produce same sequence
    let source = r#"
import "random" for Random

var rng1 = Random.new(42)
var rng2 = Random.new(42)

System.print(rng1.float() == rng2.float())
System.print(rng1.int(1000) == rng2.int(1000))
System.print(rng1.float(10, 20) == rng2.float(10, 20))
"#;
    let (result, output, _) = run(source);
    assert!(matches!(result, InterpretResult::Success), "{:?}", result);
    for line in output.lines() {
        assert_eq!(line, "true");
    }
}

#[test]
fn e2e_random_sample_shuffle() {
    let source = r#"
import "random" for Random

var rng = Random.new(99)
var list = [1, 2, 3, 4, 5]

var picked = rng.sample(list)
System.print(picked >= 1)
System.print(picked <= 5)

var sampled = rng.sample(list, 3)
System.print(sampled.count == 3)

rng.shuffle(list)
System.print(list.count == 5)
"#;
    let (result, output, _) = run(source);
    assert!(matches!(result, InterpretResult::Success), "{:?}", result);
    for line in output.lines() {
        assert_eq!(line, "true");
    }
}

// ---------------------------------------------------------------------------
// Optional module: meta
// ---------------------------------------------------------------------------

#[test]
fn e2e_meta_get_module_variables() {
    let source = r#"
import "meta" for Meta

var Foo = 42
var Bar = "hello"

var vars = Meta.getModuleVariables("main")
System.print(vars is List)
System.print(vars.count > 0)
// Should contain our variables
var hasFoo = false
var hasBar = false
for (v in vars) {
    if (v == "Foo") hasFoo = true
    if (v == "Bar") hasBar = true
}
System.print(hasFoo)
System.print(hasBar)
"#;
    let (result, output, elapsed) = run(source);
    assert!(
        matches!(result, InterpretResult::Success),
        "meta module failed: {:?}",
        result
    );
    for (i, line) in output.lines().enumerate() {
        assert_eq!(line, "true", "line {} was not true: {}", i, line);
    }
    eprintln!("  [meta module {}]", fmt_elapsed(elapsed));
}

#[test]
fn e2e_meta_eval() {
    let source = r#"
import "meta" for Meta

var a = 10
var b = 20
Meta.eval("System.print(a + b)")
"#;
    let (result, output, _) = run(source);
    assert!(
        matches!(result, InterpretResult::Success),
        "Meta.eval failed: {:?}",
        result
    );
    assert_eq!(output.trim(), "30");
}

#[test]
fn e2e_meta_compile() {
    let source = r#"
import "meta" for Meta

var closure = Meta.compile("System.print(\"compiled\")")
System.print(closure is Fn)
closure.call()
"#;
    let (result, output, _) = run(source);
    assert!(
        matches!(result, InterpretResult::Success),
        "Meta.compile failed: {:?}",
        result
    );
    assert!(output.contains("true"), "closure should be a Fn");
    assert!(output.contains("compiled"), "compiled code should execute");
}

#[test]
fn e2e_meta_compile_expression() {
    let source = r#"
import "meta" for Meta

var closure = Meta.compileExpression("2 + 3 * 4")
System.print(closure is Fn)
System.print(closure.call())
"#;
    let (result, output, _) = run(source);
    assert!(
        matches!(result, InterpretResult::Success),
        "Meta.compileExpression failed: {:?}",
        result
    );
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines[0], "true");
    assert_eq!(lines[1], "14");
}

// ---------------------------------------------------------------------------
// Lazy sequence wrappers (MapSequence, WhereSequence, SkipSequence, TakeSequence)
// ---------------------------------------------------------------------------

#[test]
fn e2e_sequence_map() {
    let source = r#"
var list = [1, 2, 3, 4]
var doubled = list.map {|x| x * 2 }
System.print(doubled.toList)
"#;
    assert_output(source, "[2, 4, 6, 8]\n");
}

#[test]
fn e2e_sequence_where() {
    let source = r#"
var list = [1, 2, 3, 4, 5, 6]
var evens = list.where {|x| x % 2 == 0 }
System.print(evens.toList)
"#;
    assert_output(source, "[2, 4, 6]\n");
}

#[test]
fn e2e_sequence_skip() {
    let source = r#"
var list = [10, 20, 30, 40, 50]
var skipped = list.skip(2)
System.print(skipped.toList)
"#;
    assert_output(source, "[30, 40, 50]\n");
}

#[test]
fn e2e_sequence_take() {
    let source = r#"
var list = [10, 20, 30, 40, 50]
var taken = list.take(3)
System.print(taken.toList)
"#;
    assert_output(source, "[10, 20, 30]\n");
}

#[test]
fn e2e_sequence_chain() {
    let source = r#"
var list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
var result = list.where {|x| x % 2 == 0 }.map {|x| x * 10 }.take(3).toList
System.print(result)
"#;
    assert_output(source, "[20, 40, 60]\n");
}

#[test]
fn e2e_sequence_skip_and_take() {
    let source = r#"
var list = [1, 2, 3, 4, 5, 6, 7, 8]
var result = list.skip(2).take(4).toList
System.print(result)
"#;
    assert_output(source, "[3, 4, 5, 6]\n");
}

// ---------------------------------------------------------------------------
// String byte and code point sequences
// ---------------------------------------------------------------------------

#[test]
fn e2e_string_bytes() {
    let source = r#"
var s = "ABC"
var bytes = s.bytes
System.print(bytes.toList)
"#;
    assert_output(source, "[65, 66, 67]\n");
}

#[test]
fn e2e_string_code_points() {
    let source = r#"
var s = "Hi!"
var cp = s.codePoints
System.print(cp.toList)
"#;
    assert_output(source, "[72, 105, 33]\n");
}

// ---------------------------------------------------------------------------
// String trim with custom characters
// ---------------------------------------------------------------------------

#[test]
fn e2e_string_trim_chars() {
    let source = r#"
System.print("***hello***".trim("*"))
System.print("xxhelloxx".trimStart("x"))
System.print("helloxx".trimEnd("x"))
"#;
    assert_output(source, "hello\nhelloxx\nhello\n");
}

// ---------------------------------------------------------------------------
// Map iteration with MapEntry
// ---------------------------------------------------------------------------

#[test]
fn e2e_map_entry() {
    let source = r#"
var map = {"a": 1}
for (entry in map) {
  System.print(entry.key)
  System.print(entry.value)
}
"#;
    assert_output(source, "a\n1\n");
}

// ---------------------------------------------------------------------------
// System.writeObject_
// ---------------------------------------------------------------------------

#[test]
fn e2e_system_write_object() {
    let source = r#"
System.writeObject_(42)
"#;
    assert_output(source, "42");
}

// ---------------------------------------------------------------------------
// DeltaBlue benchmark
// ---------------------------------------------------------------------------

#[test]
fn e2e_delta_blue() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");

    // Run with interpreter only (no JIT) to verify the program is correct
    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    });
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", &source);
    let output = vm.take_output();
    eprintln!("delta_blue interpreter output: {:?}", output.trim_end());
    assert!(
        matches!(result, InterpretResult::Success),
        "delta_blue interpreter failed: {:?}\nOutput:\n{}",
        result,
        output
    );
    assert!(
        !output.contains("failed"),
        "delta_blue interpreter has projection failures:\n{}",
        output
    );
    assert!(
        output.contains("14065400"),
        "delta_blue interpreter wrong total:\n{}",
        output
    );

    // Run with default JIT (skip known buggy non-leaf funcs)
    // No skip — run with JIT normally (IC fast path disabled in vm_interp.rs)
    let (result2, output2, elapsed2) = run(&source);
    let t2 = fmt_elapsed(elapsed2);
    eprintln!("delta_blue JIT output: {:?} ({})", output2.trim_end(), t2);
    assert!(
        matches!(result2, InterpretResult::Success),
        "delta_blue JIT failed: {:?}\nOutput:\n{}",
        result2,
        output2
    );
    assert!(
        !output2.contains("failed"),
        "delta_blue JIT has projection failures:\n{}",
        output2
    );
}

#[test]
#[ignore = "debugging tiered promotion regressions in delta_blue"]
fn e2e_delta_blue_projection_tiered_promotion_smoke() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");
    let prefix = source
        .split("var start = System.clock")
        .next()
        .expect("delta_blue benchmark footer must exist");
    let smoke = format!("{}projectionTest.call(5)\nSystem.print(total)\n", prefix);

    let (result, output, elapsed) = run_with_config(
        &smoke,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered projection smoke failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "tiered projection smoke has projection failures:\n{}",
        output
    );
}

#[test]
#[ignore = "debugging tiered promotion regressions in delta_blue"]
fn e2e_delta_blue_tiered_stress_smoke() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");
    let prefix = source
        .split("var start = System.clock")
        .next()
        .expect("delta_blue benchmark footer must exist");
    let smoke = format!(
        "{}for (i in 0...5) {{\n  chainTest.call(20)\n  projectionTest.call(20)\n}}\nSystem.print(total)\n",
        prefix
    );

    let (result, output, elapsed) = run_with_config(
        &smoke,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered stress smoke failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "tiered stress smoke has projection failures:\n{}",
        output
    );
}

#[test]
#[ignore = "debugging full non-leaf tiered execution under mark-sweep GC"]
fn e2e_delta_blue_mark_sweep_tiered_projection_smoke() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");
    let prefix = source
        .split("var start = System.clock")
        .next()
        .expect("delta_blue benchmark footer must exist");
    let smoke = format!("{}projectionTest.call(5)\nSystem.print(total)\n", prefix);

    let (result, output, elapsed) = run_with_config(
        &smoke,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::MarkSweep,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "mark-sweep tiered projection smoke failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "mark-sweep tiered projection smoke has projection failures:\n{}",
        output
    );
}

#[test]
#[ignore = "debugging full non-leaf tiered execution under mark-sweep GC"]
fn e2e_delta_blue_mark_sweep_tiered_stress_smoke() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");
    let prefix = source
        .split("var start = System.clock")
        .next()
        .expect("delta_blue benchmark footer must exist");
    let smoke = format!(
        "{}for (i in 0...5) {{\n  chainTest.call(20)\n  projectionTest.call(20)\n}}\nSystem.print(total)\n",
        prefix
    );

    let (result, output, elapsed) = run_with_config(
        &smoke,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::MarkSweep,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "mark-sweep tiered stress smoke failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "mark-sweep tiered stress smoke has projection failures:\n{}",
        output
    );
}
