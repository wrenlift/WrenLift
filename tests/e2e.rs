/// End-to-end integration tests for the WrenLift runtime.
///
/// These tests exercise complex Wren programs that combine multiple language
/// features. Each test verifies correctness (expected output), stability
/// (no panics), and profiles execution time.
use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};
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

fn osr_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn lock_osr_test() -> std::sync::MutexGuard<'static, ()> {
    osr_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
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
        "13\n7\n30\n3.3333333333333\n1\n-10",
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
        "15\n255\n240\n4294967295\n16\n16",
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

/// Constructor JIT dispatch under GC pressure with object-typed
/// args. The historic "Constructor JIT SIGSEGV under GC pressure"
/// shape: the JIT'd ctor body runs with arg pointers in registers,
/// and an allocator fired during the body moves the underlying
/// objects without updating the register-bound copies. This test
/// exercises the path by chaining `Pair.new(prev, i)` so each
/// ctor invocation receives a still-live pointer arg that's a
/// candidate for GC-promote during the body.
#[test]
fn e2e_gc_pressure_constructor_with_object_args() {
    let source = r#"
class Pair {
    construct new(left, right) {
        _left = left
        _right = right
    }
    left  { _left }
    right { _right }
}

var prev = null
var i = 0
while (i < 2000) {
    prev = Pair.new(prev, i)
    i = i + 1
}
// Walk back through the chain to dereference every still-live
// pointer the constructor stashed — proves none staled out.
var node = prev
var sum = 0
while (node != null) {
    sum = sum + node.right
    node = node.left
}
System.print(sum)
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let mut vm = VM::new(VMConfig {
            execution_mode: mode,
            jit_threshold: 5,
            ..Default::default()
        });
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", source);
        let output = vm.take_output();
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}: {:?}\n{}",
            mode,
            result,
            output
        );
        // Sum 0..1999 = 1999 * 2000 / 2 = 1999000
        assert_eq!(
            output.trim(),
            "1999000",
            "{:?} ctor-gc-pressure mismatch",
            mode
        );
    }
}

/// Verifies pure-user-method CSE survives across a hot loop in
/// tiered mode. `Helper.add(a, b)` is a pure user method (only
/// arithmetic + return). A naive CSE pass would flush its memory
/// cache on every dispatch; with the codegen-time purity-aware
/// CSE, the call doesn't invalidate the cache. We check
/// behavioural correctness (the answer matches the interpreter
/// reference) — performance is the side-effect, not the contract.
#[test]
fn e2e_pure_user_method_cse_through_call() {
    use crate::{ExecutionMode, InterpretResult, VMConfig, VM};
    let source = r#"
class Helper {
    static add(a, b) { a + b }
}

class Box {
    construct new(v) { _v = v }
    v { _v }
    v=(x) { _v = x }
}

var b = Box.new(42)
var sum = 0
var i = 0
while (i < 50000) {
    // Two reads of b.v straddle a pure user-defined call —
    // CSE must keep the second read merged with the first
    // when the callee is recognised as pure. We don't observe
    // the optimisation directly; the test just locks in that
    // the resulting program still computes correctly.
    sum = sum + b.v + Helper.add(b.v, i)
    i = i + 1
}
System.print(sum)
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let mut vm = VM::new(VMConfig {
            execution_mode: mode,
            jit_threshold: 5,
            ..Default::default()
        });
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", source);
        let output = vm.take_output();
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}: {:?}\n{}",
            mode,
            result,
            output
        );
        // sum = sum_{i=0..49999} (b.v + (b.v + i))
        //     = 50000*(2*42) + sum_{i=0..49999} i
        //     = 4_200_000 + 1_249_975_000
        //     = 1_254_175_000
        assert_eq!(
            output.trim(),
            "1254175000",
            "{:?} pure-user-cse mismatch",
            mode
        );
    }
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

// Regression: multi-`%()` string interpolation inside a closure went
// through the threaded interpreter, which only emitted op_string_concat
// for arity-2 chains and noop'd everything else — so any 3+ part
// interpolation in a closure returned `null` instead of the formatted
// string.
#[test]
fn e2e_threaded_multi_interp_in_closure() {
    assert_output(
        r#"
var f = Fn.new {|x, y, z|
  System.print("a %(x) b %(y) c %(z)")
}
f.call(1, 2, 3)
"#,
        "a 1 b 2 c 3",
    );
}

// ===========================================================================
// 13. Imports
// ===========================================================================

#[test]
fn e2e_module_import() {
    let config = VMConfig {
        load_module_fn: Some(Box::new(|name: &str, _from: &str| -> Option<String> {
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

// An error in a fiber run with `call()` aborts its caller too, up the
// chain to the nearest `try`, which answers with it; every fiber on
// the way is done with the error. Reference Wren's `runtimeError`.
#[test]
fn e2e_fiber_abort_reaches_the_callers_try() {
    assert_output(
        r#"
var c = Fiber.new { Fiber.abort("deep") }
var b = Fiber.new { c.call() }
var a = Fiber.new { b.call() }
System.print(a.try())
System.print("%(a.isDone) %(b.isDone) %(c.isDone)")
System.print("%(a.error) %(b.error) %(c.error)")
var inner = Fiber.new { Fiber.abort("mid") }
var mid = Fiber.new {
  var r = inner.try()
  System.print("mid caught %(r)")
  return "mid done"
}
System.print(Fiber.new { mid.call() }.try())
var y = Fiber.new {
  Fiber.yield(1)
  Fiber.abort("late")
}
System.print(y.call())
System.print(Fiber.new { y.call() }.try())
"#,
        "deep
true true true
deep deep deep
mid caught mid
mid done
1
late",
    );
}

// `Fiber.try` should catch method-not-found errors, not just
// `Fiber.abort` and native `runtime_error`. Used to abort the
// process — the dispatch site raised RuntimeError::MethodNotFound
// before checking the fiber's `is_try` flag.
#[test]
fn e2e_fiber_try_catches_missing_method() {
    assert_output(
        r#"
class B { construct new() {} }
var fiber = Fiber.new { B.new().missing() }
fiber.try()
System.print(fiber.error)
System.print(fiber.isDone)
"#,
        "B does not implement 'missing()'\ntrue",
    );
}

// Same path through `super` — the super-chain miss takes a
// different code site but should follow the same fiber-try
// routing.
// `Meta.compile` returns a closure whose body's bare-identifier
// getters (e.g. `obj.name`) used to misdispatch to `Class.name`
// when invoked from inside another class's static method —
// because the lookup resolved against the metaobject rather than
// the receiver's class. Locks current correct behaviour so a
// future regression in IC freshness or sema resolution doesn't
// re-introduce the bug.
#[test]
fn e2e_meta_compile_closure_dispatches_through_receiver() {
    assert_output(
        r#"
import "meta" for Meta
class Shape {
  construct new(n) { _name = n }
  name { _name }
}
var acc = Meta.compile("return Fn.new { |obj| obj.name }\n").call()
class Enc {
  static invoke(fn, obj) { return fn.call(obj) }
}
System.print(acc.call(Shape.new("alpha")))
System.print(Enc.invoke(acc, Shape.new("beta")))
"#,
        "alpha\nbeta",
    );
}

#[test]
fn e2e_fiber_try_catches_missing_super_method() {
    assert_output(
        r#"
class A { construct new() {} }
class B is A {
  construct new() {}
  callBad() { super.missing() }
}
var fiber = Fiber.new { B.new().callBad() }
fiber.try()
System.print(fiber.error)
"#,
        "A does not implement 'missing()'",
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
        load_module_fn: Some(Box::new(|name: &str, _from: &str| -> Option<String> {
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

// Energy should be conserved. The bound is wider than a
// pure-arithmetic invariant would need because FP rounding
// paths differ across CPU architectures + JIT register
// allocation; 0.01 is tight enough to flag a real
// correctness regression but absorbs the cross-platform
// drift noise.
var drift = (e1 - e0).abs
System.print(drift < 0.01)
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
fn e2e_tiered_backedge_does_not_restart_module_entry() {
    let source = r#"
System.print("setup")
var i = 0
while (i < 1000000) {
  i = i + 1
}
System.print("done")
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 5,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered back-edge run failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "setup\ndone",
        "tiered back-edge OSR must not re-run module setup ({})",
        t
    );
}

#[test]
fn e2e_tiered_backedge_enters_osr_entry() {
    let _guard = lock_osr_test();
    let source = r#"
var passes = 0
var total = 0
while (passes < 3) {
  var i = 0
  while (i < 4000000) {
    i = i + 1
  }
  total = total + i
  passes = passes + 1
}
System.print(total)
"#;

    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Tiered,
        // Start background compile as early as possible so the OSR entry
        // still lands under parallel `cargo test` load.
        jit_threshold: 1,
        ..VMConfig::default()
    });
    vm.engine.collect_tier_stats = true;
    vm.output_buffer = Some(String::new());

    let start = Instant::now();
    let result = vm.interpret("main", source);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    let t = fmt_elapsed(elapsed);

    assert!(
        matches!(result, InterpretResult::Success),
        "tiered OSR run failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "12000000",
        "tiered OSR output mismatch ({})",
        t
    );
    assert!(
        vm.engine
            .tier_stats
            .iter()
            .any(|stats| stats.osr_entries > 0),
        "expected at least one OSR entry ({})",
        t
    );
}

#[test]
fn e2e_tiered_backedge_enters_osr_entry_in_method() {
    let _guard = lock_osr_test();
    // A hot loop inside a user-defined method should also take an OSR entry
    // now that method frames are eligible. Threaded dispatch should fall back
    // to bytecode for functions with OSR safepoints.
    let source = r#"
class Counter {
  construct new() {}
  run() {
    var passes = 0
    var total = 0
    while (passes < 3) {
      var i = 0
      while (i < 2000000) {
        i = i + 1
      }
      total = total + i
      passes = passes + 1
    }
    return total
  }
}

var c = Counter.new()
System.print(c.run())
"#;

    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Tiered,
        // Same early tier-up as the top-level loop test: under a fully
        // parallel e2e run, a later compile trigger can miss the method
        // loop's OSR window even though the feature itself is healthy.
        jit_threshold: 1,
        ..VMConfig::default()
    });
    vm.engine.collect_tier_stats = true;
    vm.output_buffer = Some(String::new());

    let start = Instant::now();
    let result = vm.interpret("main", source);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    let t = fmt_elapsed(elapsed);

    assert!(
        matches!(result, InterpretResult::Success),
        "tiered method OSR run failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "6000000",
        "tiered method OSR output mismatch ({})",
        t
    );
    assert!(
        vm.engine
            .tier_stats
            .iter()
            .any(|stats| stats.osr_entries > 0),
        "expected at least one OSR entry from a method loop ({})",
        t
    );
}

#[test]
fn e2e_tiered_backedge_osr_survives_gc_pressure() {
    // Allocating inside a hot method loop forces multiple GC cycles while the
    // OSR entry is active. The receiver and loop-carried list must stay live
    // across each transfer, without globally disabling threaded dispatch.
    let source = r#"
class Accumulator {
  construct new() {
    _list = []
  }
  run() {
    var i = 0
    while (i < 500) {
      _list.add([i, i + 1, i + 2, i + 3])
      i = i + 1
    }
    return _list.count
  }
}

var total = 0
for (j in 0...8) {
  total = total + Accumulator.new().run()
}
System.print(total)
"#;

    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 5,
        ..VMConfig::default()
    });
    vm.engine.collect_tier_stats = true;
    vm.output_buffer = Some(String::new());

    let start = Instant::now();
    let result = vm.interpret("main", source);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    let t = fmt_elapsed(elapsed);

    assert!(
        matches!(result, InterpretResult::Success),
        "tiered method OSR GC stress run failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "4000",
        "method OSR under GC pressure output mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_backedge_osr_nested_inside_native_caller() {
    let _guard = lock_osr_test();
    // The module loop first OSRs into native code, then the native module frame
    // repeatedly calls a method whose loop also tiers up via OSR. The inner
    // transfer must work even though we're already nested in a native frame
    // (jit_depth > 0). Repeating the call gives the background compiler enough
    // runway under a fully parallel test run without weakening the guarantee.
    let source = r#"
class Worker {
  construct new() {}
  inner() {
    var j = 0
    while (j < 3000000) {
      j = j + 1
    }
    return j
  }
}

var w = Worker.new()
var passes = 0
while (passes < 3) {
  var i = 0
  while (i < 2000000) {
    i = i + 1
  }
  passes = passes + 1
}
var total = 0
for (k in 0...3) {
  total = total + w.inner()
}
System.print(total)
"#;

    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 1,
        ..VMConfig::default()
    });
    vm.engine.collect_tier_stats = true;
    vm.output_buffer = Some(String::new());

    let start = Instant::now();
    let result = vm.interpret("main", source);
    let elapsed = start.elapsed();
    let output = vm.take_output();
    let t = fmt_elapsed(elapsed);

    assert!(
        matches!(result, InterpretResult::Success),
        "nested OSR run failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "9000000",
        "nested OSR output mismatch ({})",
        t
    );
    let mut module_osr_entries = 0;
    let mut inner_osr_entries = 0;
    let mut inner_native_entries = 0;
    for (idx, stats) in vm.engine.tier_stats.iter().enumerate() {
        let Some(mir) = vm
            .engine
            .get_mir(wren_lift::runtime::engine::FuncId(idx as u32))
        else {
            continue;
        };
        let name = vm.interner.resolve(mir.name);
        if name == "<module>" {
            module_osr_entries += stats.osr_entries;
        } else if name == "inner()" {
            inner_osr_entries += stats.osr_entries;
            inner_native_entries += stats.baseline_entries + stats.optimized_entries;
        }
    }
    assert!(
        module_osr_entries > 0,
        "expected <module> to enter OSR before calling inner() ({})",
        t
    );
    assert!(
        inner_osr_entries > 0 || inner_native_entries > 0,
        "expected inner() to tier into native execution under the native caller ({})",
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

#[test]
fn e2e_tiered_nonleaf_loop_preserves_object_local() {
    let source = r#"
class Keeper {
  construct new(tag) {
    _tag = tag
  }

  tag { _tag }
}

var noop = Fn.new {}

var run = Fn.new { |keep, other|
  var saved = keep
  var noise = other
  for (i in 0...20) {
    noop.call()
  }
  System.print(saved.tag)
  System.print(noise.tag)
}

for (i in 0...20) {
  run.call(Keeper.new("keep"), Keeper.new("noise"))
}
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::Arena,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered non-leaf local preservation failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    let expected = (0..20)
        .map(|_| "keep\nnoise")
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        output.trim(),
        expected,
        "tiered non-leaf local preservation output mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_nonleaf_change_shape_preserves_receiver() {
    let source = r#"
class Variable {
  construct new(tag, value) {
    _tag = tag
    _value = value
  }

  tag { _tag }
  value { _value }
  value=(newValue) { _value = newValue }
}

class Edit {
  construct new(v) {
    _target = v
  }

  destroy() {
    System.print(_target.tag)
  }
}

class Plan {
  construct new(v, value) {
    _target = v
    _value = value
  }

  execute() {
    _target.value = _value
  }
}

var change = Fn.new { |v, newValue|
  var edit = Edit.new(v)
  var plan = Plan.new(v, newValue)
  for (i in 0...10) {
    plan.execute()
  }
  edit.destroy()
  System.print(v.value)
}

for (i in 0...20) {
  var v = Variable.new("var", 0)
  change.call(v, i)
}
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::Arena,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered change-shape preservation failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    let expected = (0..20)
        .flat_map(|i| ["var".to_string(), i.to_string()])
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        output.trim(),
        expected,
        "tiered change-shape preservation output mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_nonleaf_plan_execute_list_iteration() {
    let source = r#"
class Variable {
  construct new(value) {
    _value = value
  }

  value { _value }
  value=(newValue) { _value = newValue }
}

class Worker {
  construct new(v, value) {
    _target = v
    _value = value
  }

  execute() {
    _target.value = _value
  }
}

class Plan {
  construct new(worker) {
    _list = [worker]
  }

  execute() {
    for (constraint in _list) {
      constraint.execute()
    }
  }
}

var run = Fn.new { |value|
  var v = Variable.new(0)
  var worker = Worker.new(v, value)
  var plan = Plan.new(worker)
  for (i in 0...10) {
    plan.execute()
  }
  System.print(v.value)
}

for (i in 0...20) {
  run.call(i)
}
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::Arena,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered list-iteration execute failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    let expected = (0..20)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        output.trim(),
        expected,
        "tiered list-iteration execute output mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_nonleaf_inherited_destroy_after_loop() {
    let source = r#"
class Constraint {
  destroy() {
    removeFromGraph()
  }
}

class UnaryConstraint is Constraint {
  construct new(output) {
    _myOutput = output
  }

  removeFromGraph() {
    _myOutput.removeConstraint(this)
  }
}

class EditConstraint is UnaryConstraint {
  construct new(output) {
    super(output)
  }
}

class Variable {
  construct new(tag) {
    _tag = tag
  }

  removeConstraint(constraint) {
    System.print(_tag)
  }
}

class Worker {
  construct new() {}

  execute() {}
}

class Plan {
  construct new() {
    _list = [Worker.new()]
  }

  execute() {
    for (constraint in _list) {
      constraint.execute()
    }
  }
}

var change = Fn.new { |v|
  var edit = EditConstraint.new(v)
  var plan = Plan.new()
  for (i in 0...10) {
    plan.execute()
  }
  edit.destroy()
}

for (i in 0...20) {
  change.call(Variable.new("ok"))
}
"#;

    let (result, output, elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::Arena,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "tiered inherited destroy failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    let expected = (0..20).map(|_| "ok").collect::<Vec<_>>().join("\n");
    assert_eq!(
        output.trim(),
        expected,
        "tiered inherited destroy output mismatch ({})",
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
    // Run on a dedicated thread with an explicit 8 MiB stack. The
    // cargo-test runner's default thread stack varies by platform
    // (macOS 8 MiB, Linux 2 MiB) — a runaway recursion that the
    // VM's `max_call_depth` should catch can otherwise blow the
    // native stack first on Linux before the depth check fires,
    // turning a clean RuntimeError into a SIGSEGV that kills the
    // whole test binary. The test is asserting that the VM detects
    // unbounded recursion; the host thread's stack budget is
    // incidental, so pin it.
    let handle = std::thread::Builder::new()
        .name("e2e_stack_overflow".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
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
            assert!(
                result != InterpretResult::Success,
                "infinite recursion should not succeed"
            );
            let output = vm.take_output();
            eprintln!(
                "  [stack overflow detected: result={:?}, output={:?}]",
                result, output
            );
        })
        .expect("spawn e2e_stack_overflow worker");
    handle
        .join()
        .expect("e2e_stack_overflow worker panicked or aborted");
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
// Int32Array + Simd built-ins
// ---------------------------------------------------------------------------

#[test]
fn e2e_int32_array_basics() {
    let source = r#"
var ints = Int32Array.new(4)
ints[0] = 7
ints[1] = -2
ints[2] = 123
ints[3] = 0
System.print(ints.count)
System.print(ints.byteLength)
System.print(ints.toList)
System.print(Int32Array.fromList([9, 8, 7, 6]).toString)
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let mut vm = VM::new(VMConfig {
            execution_mode: mode,
            jit_threshold: 5,
            ..Default::default()
        });
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", source);
        let output = vm.take_output();
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}\n{}",
            result,
            output
        );
        assert_eq!(
            output.trim_end(),
            "4\n16\n[7, -2, 123, 0]\nInt32Array(4)",
            "{:?} Int32Array mismatch",
            mode
        );
    }
}

#[test]
fn e2e_simd_interop_and_ops() {
    let source = r#"
var ints = Int32Array.fromList([1, 2, 3, 4, 5, 6])
var vi = Simd4i.load(ints, 1)
System.print(vi[0])
System.print(vi[3])
var sum = vi + Simd4i.splat(10)
sum.store(ints, 0)
System.print(ints.toList)
var mask = sum > Simd4i.splat(12)
System.print(mask.bitmask)
System.print(mask.anyTrue)
System.print(mask.allTrue)

var floats = Float32Array.fromList([1.5, 2.5, 3.5, 4.5])
var vf = Simd4f.load(floats, 0)
var scaled = (vf * Simd4f.splat(2)).replaceLane(1, 9)
scaled.store(floats, 0)
System.print(floats.toList)
System.print(scaled.shuffle(3, 2, 1, 0).toString)
System.print((scaled >= Simd4f.splat(7)).bitmask)
System.print(Simd4i.new(-1, 0, 1065353216, 1073741824).reinterpretAsFloat[2])
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let mut vm = VM::new(VMConfig {
            execution_mode: mode,
            jit_threshold: 5,
            ..Default::default()
        });
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", source);
        let output = vm.take_output();
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}\n{}",
            result,
            output
        );
        assert_eq!(
            output.trim_end(),
            "2\n5\n[12, 13, 14, 15, 5, 6]\n14\ntrue\nfalse\n[3, 9, 7, 9]\nSimd4f(9, 7, 9, 3)\n14\n1",
            "{:?} Simd mismatch",
            mode
        );
    }
}

#[test]
fn e2e_simd_runtime_errors() {
    assert_runtime_error("Simd4i.new(1.5, 2, 3, 4)");
    assert_runtime_error("Simd4f.new(1, 2, Num.infinity, 4)");
    assert_runtime_error("Simd4f.load(Int32Array.new(4), 0)");
    assert_runtime_error("Simd4i.load(Int32Array.new(3), 0)");
}

#[test]
fn e2e_simd_hot_loop_tiered() {
    let source = r#"
var ints = Int32Array.fromList([1, 2, 3, 4, 5, 6, 7, 8])
var sum = 0
var i = 0
while (i < 20000) {
    var lanes = Simd4i.load(ints, 0) + Simd4i.splat(i)
    sum = sum + lanes[0] + lanes[1]
    i = i + 1
}
System.print(sum)
"#;
    let (result, output, _elapsed) = run_with_config(
        source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 5,
            ..Default::default()
        },
    );
    assert!(
        matches!(result, InterpretResult::Success),
        "{:?}\n{}",
        result,
        output
    );
    assert_eq!(output.trim_end(), "400040000");
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
fn e2e_delta_blue_generational_tiered_full_default_threshold() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");

    let (result, output, elapsed) = run_with_config(
        &source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 100,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "generational tiered full delta_blue failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "generational tiered full delta_blue has projection failures:\n{}",
        output
    );
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(
        lines.first().copied(),
        Some("14065400"),
        "generational tiered full delta_blue total mismatch ({})",
        t
    );
}

#[test]
fn e2e_delta_blue_generational_tiered_full_threshold_one() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");

    let (result, output, elapsed) = run_with_config(
        &source,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            ..VMConfig::default()
        },
    );
    let t = fmt_elapsed(elapsed);
    assert!(
        matches!(result, InterpretResult::Success),
        "generational tiered full delta_blue threshold-one failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "generational tiered full delta_blue threshold-one has projection failures:\n{}",
        output
    );
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(
        lines.first().copied(),
        Some("14065400"),
        "generational tiered full delta_blue threshold-one total mismatch ({})",
        t
    );
}

#[test]
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

#[test]
fn e2e_delta_blue_mark_sweep_tiered_full() {
    let source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");

    let (result, output, elapsed) = run_with_config(
        &source,
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
        "mark-sweep tiered full delta_blue failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert!(
        !output.contains("failed"),
        "mark-sweep tiered full delta_blue has projection failures:\n{}",
        output
    );
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(
        lines.first().copied(),
        Some("14065400"),
        "mark-sweep tiered full delta_blue total mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_mark_sweep_where_predicate_survives_explicit_gc() {
    let source = r#"
class Holder {
  construct new(values) {
    _constraints = values
  }

  constraints { _constraints }
  constraints=(value) { _constraints = value }
}

var holder = Holder.new([1, 2, 3, 4])
for (i in 0...10) {
  holder.constraints = holder.constraints.where { |x|
    System.gc()
    return x > 0
  }
  var total = 0
  for (value in holder.constraints) {
    total = total + value
  }
  System.print(total)
}
"#;

    let (result, output, elapsed) = run_with_config(
        source,
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
        "mark-sweep where predicate GC failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    let expected = (0..10).map(|_| "10").collect::<Vec<_>>().join("\n");
    assert_eq!(
        output.trim(),
        expected,
        "mark-sweep where predicate GC output mismatch ({})",
        t
    );
}

#[test]
fn e2e_tiered_mark_sweep_repeated_where_sequence_reassignment() {
    let source = r#"
class Holder {
  construct new(values) {
    _constraints = values
  }

  constraints { _constraints }
  constraints=(value) { _constraints = value }
}

var run = Fn.new {
  var holder = Holder.new([1, 2, 3, 4])
  for (i in 0...40) {
    holder.constraints = holder.constraints.where { |x| x > 0 }
    var total = 0
    for (value in holder.constraints) {
      total = total + value
    }
    if (total != 10) {
      System.print("bad")
      System.print(total)
    }
  }
}

for (i in 0...40) {
  run.call()
}
"#;

    let (result, output, elapsed) = run_with_config(
        source,
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
        "mark-sweep repeated where-sequence reassignment failed: {:?} ({})\nOutput:\n{}",
        result,
        t,
        output
    );
    assert_eq!(
        output.trim(),
        "",
        "mark-sweep repeated where-sequence reassignment output mismatch ({})",
        t
    );
}

#[test]
fn e2e_delta_blue_chain_then_projection_debug_values() {
    let mut source =
        std::fs::read_to_string("bench/delta_blue.wren").expect("bench/delta_blue.wren must exist");
    source = source
        .replace(
            r#"  if (dst.value != 1170) System.print("Projection 1 failed")"#,
            r#"  System.print("p1 dst=%(dst.value)")
  if (dst.value != 1170) System.print("Projection 1 failed")"#,
        )
        .replace(
            r#"  if (src.value != 5) System.print("Projection 2 failed")"#,
            r#"  System.print("p2 src=%(src.value)")
  if (src.value != 5) System.print("Projection 2 failed")"#,
        )
        .replace(
            r#"    if (dests[i].value != i * 5 + 1000) System.print("Projection 3 failed")"#,
            r#"    if (i < 3) System.print("p3 i=%(i) value=%(dests[i].value)")
    if (dests[i].value != i * 5 + 1000) System.print("Projection 3 failed")"#,
        )
        .replace(
            r#"    if (dests[i].value != i * 5 + 2000) System.print("Projection 4 failed")"#,
            r#"    if (i < 3) System.print("p4 i=%(i) value=%(dests[i].value)")
    if (dests[i].value != i * 5 + 2000) System.print("Projection 4 failed")"#,
        );
    let prefix = source
        .split("var start = System.clock")
        .next()
        .expect("delta_blue benchmark footer must exist");
    let smoke = format!(
        "{}chainTest.call(20)\nprojectionTest.call(20)\nSystem.print(total)\n",
        prefix
    );

    let (_result, output, _elapsed) = run_with_config(
        &smoke,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            gc_strategy: GcStrategy::MarkSweep,
            ..VMConfig::default()
        },
    );

    eprintln!("{}", output);
}

// ---------------------------------------------------------------------------
// .wlbc bytecode cache round-trip
// ---------------------------------------------------------------------------

/// Compiling to `.wlbc` and loading the resulting blob must produce the
/// same output as running the source directly. Guards against symbol-
/// table drift, closure-id remapping bugs, and class-var-slot mismatches.
#[test]
fn e2e_bytecode_cache_round_trip_matches_source() {
    // Program touches classes (with a constructor, a setter, and a
    // getter), closures via `for`, arithmetic, string interpolation,
    // and `System.print` — enough moving parts to catch most
    // serialization / install regressions.
    let source = r#"
class Counter {
  construct new() { _n = 0 }
  tick() { _n = _n + 1 }
  count { _n }
}

var c = Counter.new()
for (i in 0..9) c.tick()
System.print("count: %(c.count)")
"#;

    // Baseline run: fresh VM, normal source path.
    let mut vm_src = VM::new_default();
    vm_src.output_buffer = Some(String::new());
    let result_src = vm_src.interpret("main", source);
    let output_src = vm_src.take_output();
    assert!(
        matches!(result_src, InterpretResult::Success),
        "source path should succeed, got {:?}\n{}",
        result_src,
        output_src
    );

    // Compile-only run: produce a .wlbc blob.
    let mut vm_build = VM::new_default();
    let blob = vm_build
        .compile_source_to_blob(source)
        .expect("compile_source_to_blob");
    assert!(
        wren_lift::serialize::looks_like_wlbc(&blob),
        "emitted blob must start with the WLBC magic"
    );

    // Load + run path: fresh VM again, this time from the blob.
    let mut vm_load = VM::new_default();
    vm_load.output_buffer = Some(String::new());
    let result_load = vm_load.interpret_bytecode("main", &blob);
    let output_load = vm_load.take_output();
    assert!(
        matches!(result_load, InterpretResult::Success),
        "bytecode path should succeed, got {:?}\n{}",
        result_load,
        output_load
    );

    assert_eq!(
        output_load, output_src,
        "bytecode cache output must match source output"
    );
}

#[test]
fn e2e_bytecode_cache_rejects_garbage() {
    // Anything that isn't a WLBC header should cause a clean
    // CompileError, not a panic.
    let mut vm = VM::new_default();
    let result = vm.interpret_bytecode("main", b"not a cache file");
    assert!(
        matches!(result, InterpretResult::CompileError),
        "loader should reject non-wlbc bytes with CompileError"
    );
}

#[test]
fn e2e_hatch_package_round_trip_matches_source() {
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest, Section, SectionKind};

    // Build a hatch containing one compiled module.
    let source = r#"
class Counter {
  construct new() { _n = 0 }
  tick() { _n = _n + 1 }
  count { _n }
}

var c = Counter.new()
for (i in 0..2) c.tick()
System.print("main says %(c.count)")
"#;

    let mut vm_build = VM::new_default();
    let wlbc = vm_build
        .compile_source_to_blob(source)
        .expect("compile_source_to_blob");

    let hatch = Hatch {
        manifest: Manifest {
            name: "e2e".to_string(),
            version: "0.1.0".to_string(),
            entry: "main".to_string(),
            description: None,
            homepage: None,
            readme: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["main".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
            changelog: None,
        },
        sections: vec![Section {
            kind: SectionKind::Wlbc,
            name: "main".to_string(),
            data: wlbc,
        }],
    };
    let bytes = emit(&hatch).expect("emit hatch");
    assert!(
        wren_lift::hatch::looks_like_hatch(&bytes),
        "emitted bytes must start with HATCH magic"
    );

    // Source baseline for output comparison.
    let mut vm_src = VM::new_default();
    vm_src.output_buffer = Some(String::new());
    let result_src = vm_src.interpret("main", source);
    let output_src = vm_src.take_output();
    assert!(matches!(result_src, InterpretResult::Success));

    // Load + run the hatch in a fresh VM.
    let mut vm_load = VM::new_default();
    vm_load.output_buffer = Some(String::new());
    let result_load = vm_load.interpret_hatch(&bytes);
    let output_load = vm_load.take_output();
    assert!(
        matches!(result_load, InterpretResult::Success),
        "hatch run should succeed, got {:?}",
        result_load
    );

    assert_eq!(output_load, output_src, "hatch output must match source");
}

#[test]
fn e2e_hatch_rejects_missing_entry_module() {
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest};

    // Manifest claims `entry = "ghost"` but no such section exists.
    let hatch = Hatch {
        manifest: Manifest {
            name: "bad".to_string(),
            version: "0.1.0".to_string(),
            entry: "ghost".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["ghost".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![],
    };
    let bytes = emit(&hatch).expect("emit");

    let mut vm = VM::new_default();
    let result = vm.interpret_hatch(&bytes);
    assert!(
        matches!(result, InterpretResult::CompileError),
        "hatch with unresolved manifest module should fail cleanly"
    );
}

#[test]
fn e2e_hatch_cross_module_import_within_one_hatch() {
    // Two modules in the same hatch: `util` exports a class that
    // `main` imports and uses. The manifest lists them in dependency
    // order (util before main) so util's top-level runs first and its
    // class is visible via `find_imported_var` when main installs.
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest, Section, SectionKind};

    let util_src = r#"
class Greeter {
  construct new(who) { _who = who }
  hello { "hello, %(_who)!" }
}
"#;
    let main_src = r#"
import "util" for Greeter
var g = Greeter.new("hatch")
System.print(g.hello)
"#;

    let mut util_vm = VM::new_default();
    let util_wlbc = util_vm
        .compile_source_to_blob(util_src)
        .expect("util compile");
    let mut main_vm = VM::new_default();
    let main_wlbc = main_vm
        .compile_source_to_blob(main_src)
        .expect("main compile");

    let hatch = Hatch {
        manifest: Manifest {
            name: "cross-module-one-hatch".to_string(),
            version: "0.1.0".to_string(),
            entry: "main".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["util".to_string(), "main".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![
            Section {
                kind: SectionKind::Wlbc,
                name: "util".to_string(),
                data: util_wlbc,
            },
            Section {
                kind: SectionKind::Wlbc,
                name: "main".to_string(),
                data: main_wlbc,
            },
        ],
    };
    let bytes = emit(&hatch).expect("emit hatch");

    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let result = vm.interpret_hatch(&bytes);
    let output = vm.take_output();
    assert!(
        matches!(result, InterpretResult::Success),
        "cross-module hatch should succeed, got {:?}\n{}",
        result,
        output
    );
    assert_eq!(output.trim(), "hello, hatch!");
}

#[test]
fn e2e_hatch_cross_hatch_import_via_install_then_run() {
    // Simulate what hatch-cli will do for a dependency graph:
    // install the library hatch first via `install_hatch_modules`
    // (no entry required), then run the application hatch that
    // imports from it. Classes registered by the library hatch must
    // be visible to the application hatch at install time.
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest, Section, SectionKind};

    let lib_src = r#"
class Counter {
  construct new() { _n = 0 }
  bump { _n = _n + 1 }
  value { _n }
}
"#;
    let app_src = r#"
import "counter" for Counter
var c = Counter.new()
for (_ in 0..4) c.bump
System.print(c.value)
"#;

    let mut vm_build = VM::new_default();
    let lib_wlbc = vm_build
        .compile_source_to_blob(lib_src)
        .expect("lib compile");
    let mut vm_build = VM::new_default();
    let app_wlbc = vm_build
        .compile_source_to_blob(app_src)
        .expect("app compile");

    let lib_hatch = emit(&Hatch {
        manifest: Manifest {
            name: "libcounter".to_string(),
            version: "0.1.0".to_string(),
            entry: "counter".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["counter".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![Section {
            kind: SectionKind::Wlbc,
            name: "counter".to_string(),
            data: lib_wlbc,
        }],
    })
    .expect("emit lib hatch");

    let app_hatch = emit(&Hatch {
        manifest: Manifest {
            name: "app".to_string(),
            version: "0.1.0".to_string(),
            entry: "main".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["main".to_string()],
            dependencies: {
                let mut d = BTreeMap::new();
                d.insert(
                    "libcounter".to_string(),
                    wren_lift::hatch::Dependency::Version("0.1.0".to_string()),
                );
                d
            },
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![Section {
            kind: SectionKind::Wlbc,
            name: "main".to_string(),
            data: app_wlbc,
        }],
    })
    .expect("emit app hatch");

    // Install order: lib first (so `Counter` is registered), then app.
    // This is the exact sequence `hatch-cli` will orchestrate once
    // it's built.
    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let install = vm.install_hatch_modules(&lib_hatch);
    assert!(
        matches!(install, InterpretResult::Success),
        "lib install should succeed, got {:?}",
        install
    );
    let run = vm.interpret_hatch(&app_hatch);
    let output = vm.take_output();
    assert!(
        matches!(run, InterpretResult::Success),
        "cross-hatch run should succeed, got {:?}\n{}",
        run,
        output
    );
    assert_eq!(output.trim(), "5");
}

#[test]
fn e2e_hatch_dispatcher_entry_resolves_after_siblings() {
    // The unified `@hatch:gpu` / `@hatch:window` packages use an
    // entry-as-dispatcher pattern: the entry only does
    // `import "<backend>" for <Class>` to re-export the right
    // backend on each target. `manifest.modules` lists modules in
    // alphabetical order, so the dispatcher entry can land BEFORE
    // its backend; if installed in that order, the dispatcher's
    // import resolves to null and the consumer sees a null Class.
    // The install loop must hoist non-entry modules first so the
    // dispatcher's slot fills correctly on its own first install.
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest, Section, SectionKind};

    let backend_src = "class Foo { static greet() { \"from backend\" } }";
    let dispatcher_src = "import \"backend\" for Foo";
    let app_src = "import \"@pkg:dispatcher\" for Foo\nSystem.print(Foo.greet())";

    let mut vm_b = VM::new_default();
    let backend_wlbc = vm_b.compile_source_to_blob(backend_src).expect("backend");
    let mut vm_d = VM::new_default();
    let dispatcher_wlbc = vm_d
        .compile_source_to_blob(dispatcher_src)
        .expect("dispatcher");
    let mut vm_a = VM::new_default();
    let app_wlbc = vm_a.compile_source_to_blob(app_src).expect("app");

    // Modules listed dispatcher-first (the alphabetical case the
    // build emits when the entry's filename sorts ahead of its
    // siblings).
    let pkg_hatch = Hatch {
        manifest: Manifest {
            name: "@pkg:dispatcher".to_string(),
            version: "0.1.0".to_string(),
            entry: "@pkg:dispatcher".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["@pkg:dispatcher".to_string(), "backend".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![
            Section {
                kind: SectionKind::Wlbc,
                name: "@pkg:dispatcher".to_string(),
                data: dispatcher_wlbc,
            },
            Section {
                kind: SectionKind::Wlbc,
                name: "backend".to_string(),
                data: backend_wlbc,
            },
        ],
    };
    let pkg_bytes = emit(&pkg_hatch).expect("emit pkg");

    let app_hatch = Hatch {
        manifest: Manifest {
            name: "app".to_string(),
            version: "0.1.0".to_string(),
            entry: "main".to_string(),
            description: None,
            homepage: None,
            changelog: None,
            readme: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["main".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![Section {
            kind: SectionKind::Wlbc,
            name: "main".to_string(),
            data: app_wlbc,
        }],
    };
    let app_bytes = emit(&app_hatch).expect("emit app");

    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let install = vm.install_hatch_modules(&pkg_bytes);
    assert!(
        matches!(install, InterpretResult::Success),
        "pkg install should succeed, got {:?}",
        install
    );
    let run = vm.interpret_hatch(&app_bytes);
    let output = vm.take_output();
    assert!(
        matches!(run, InterpretResult::Success),
        "app run should succeed, got {:?}\n{}",
        run,
        output
    );
    assert_eq!(output.trim(), "from backend");
}

#[test]
fn e2e_hatch_extracts_native_lib_sections_to_disk() {
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest, Section, SectionKind};

    // A hatch carrying a `NativeLib` section must write that section
    // to a temp directory at load time, register a `<name> → path`
    // override in `native_lib_paths`, and prepend the temp directory
    // to `native_search_paths`. We stuff arbitrary bytes in (not a
    // real .dylib) — this test only checks the extraction, not
    // dlopen, so any payload works.
    let mut vm_compile = VM::new_default();
    let main_wlbc = vm_compile
        .compile_source_to_blob("System.print(\"ok\")")
        .expect("compile main");

    let hatch = Hatch {
        manifest: Manifest {
            name: "bundled".to_string(),
            version: "0.1.0".to_string(),
            entry: "main".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["main".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs: BTreeMap::new(),
            native_search_paths: Vec::new(),
            plugin_source: None,
            target: None,
        },
        sections: vec![
            Section {
                kind: SectionKind::Wlbc,
                name: "main".to_string(),
                data: main_wlbc,
            },
            Section {
                kind: SectionKind::NativeLib,
                name: "libdb".to_string(),
                data: b"MACH-O-or-ELF-bytes-here".to_vec(),
            },
        ],
    };
    let bytes = emit(&hatch).expect("emit");

    let mut vm = VM::new_default();
    let result = vm.interpret_hatch(&bytes);
    assert!(matches!(result, InterpretResult::Success));

    // The extraction dir must be the first entry in the search paths
    // so bundled libs win over ambient OS search.
    assert!(!vm.native_search_paths.is_empty());
    let extract_dir = &vm.native_search_paths[0];
    assert!(extract_dir.exists(), "extraction dir should live on disk");

    // The section name must map directly to the extracted file so a
    // matching `#!native = "libdb"` attribute finds it.
    let path = vm
        .native_lib_paths
        .get("libdb")
        .expect("libdb path registered");
    assert!(path.exists(), "extracted lib file must exist");
    let written = std::fs::read(path).expect("read back");
    assert_eq!(written, b"MACH-O-or-ELF-bytes-here");
}

// ===========================================================================
// Attribute reflection (Phase 2)
// ===========================================================================

#[test]
fn e2e_class_attributes_runtime_visible() {
    // Flag, value, and group attributes must all round-trip through MIR
    // and surface as a nested map via Class.attributes.
    let source = r#"
#runnable
#author = "Bob"
#doc(brief = "sum")
class Foo {}

var a = Foo.attributes
System.print(a[null]["runnable"][0])
System.print(a[null]["author"][0])
System.print(a["doc"]["brief"][0])
"#;
    assert_output(source, "null\nBob\nsum");
}

#[test]
fn e2e_compile_time_attributes_hidden() {
    // `#!` attributes live only for the compiler — the runtime must not see
    // them even as an empty group.
    let source = r#"
#!internal
class Foo {}
System.print(Foo.attributes)
"#;
    assert_output(source, "null");
}

#[test]
fn e2e_method_attributes_reflected() {
    let source = r#"
class C {
  #pinned
  foo() { 1 }
  bar() { 2 }
}
var m = C.methodAttributes
System.print(m["foo()"][null]["pinned"][0])
"#;
    assert_output(source, "null");
}

// ===========================================================================
// Foreign methods backed by dlopen/dlsym (Phase 3b / 3c-iii)
// ===========================================================================
//
// The dispatch bridge is unit-tested directly in src/runtime/foreign.rs.
// These tests drive the full Wren → #!native → dlsym → extern fn path.
// `.cargo/config.toml` enables `-Wl,-export_dynamic` on unix so the
// test binary's own `#[no_mangle]` symbols are reachable via
// `Library::this()` — a sentinel `#!native = "self"` targets that case.

#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn wrenlift_e2e_double(vm: *mut wren_lift::runtime::vm::VM) {
    unsafe {
        let slots = &mut (*vm).api_stack;
        let n = slots[1].as_num().unwrap_or(0.0);
        slots[0] = wren_lift::runtime::value::Value::num(n * 2.0);
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn wrenlift_e2e_add(vm: *mut wren_lift::runtime::vm::VM) {
    unsafe {
        let slots = &mut (*vm).api_stack;
        let a = slots[1].as_num().unwrap_or(0.0);
        let b = slots[2].as_num().unwrap_or(0.0);
        slots[0] = wren_lift::runtime::value::Value::num(a + b);
    }
}

#[cfg(unix)]
#[test]
fn e2e_foreign_class_binds_symbol_from_self() {
    let source = r#"
#!native = "self"
foreign class Doubler {
  #!symbol = "wrenlift_e2e_double"
  foreign static double(x)
}
System.print(Doubler.double(21))
"#;
    assert_output(source, "42");
}

#[cfg(unix)]
#[test]
fn e2e_foreign_symbol_defaults_to_method_name() {
    // With no `#!symbol` override, the loader falls back to the
    // method's base name — so a Wren method called `wrenlift_e2e_add`
    // maps directly to the `#[no_mangle] extern "C" fn` of the same
    // name in this test binary.
    let source = r#"
#!native = "self"
foreign class MathX {
  foreign static wrenlift_e2e_add(a, b)
}
System.print(MathX.wrenlift_e2e_add(3, 4))
"#;
    assert_output(source, "7");
}

#[cfg(unix)]
#[test]
fn e2e_foreign_missing_library_surfaces_error() {
    // A bogus library name must fail cleanly: the load error prints to
    // stderr and the foreign method simply isn't bound. Calling it then
    // surfaces as a normal "method not found" runtime error rather than
    // SEGV-ing the process.
    let source = r#"
#!native = "__wrenlift_missing_lib__"
foreign class Bogus {
  #!symbol = "nope"
  foreign static go()
}
Bogus.go()
"#;
    let (result, _output, _) = run(source);
    assert!(matches!(result, InterpretResult::RuntimeError));
}

// ===========================================================================
// Hatchfile [native_libs] + native_search_paths (Phase 3c-i)
// ===========================================================================

#[test]
fn e2e_hatch_manifest_applies_native_search_paths_and_overrides() {
    // Install a hatch whose manifest declares a `[native_libs]`
    // override and a custom search path, then confirm both have been
    // folded into the VM's foreign-loader state. This verifies the
    // manifest plumbing without needing a real shared library to load.
    use std::collections::BTreeMap;
    use wren_lift::hatch::{emit, Hatch, Manifest, NativeLibEntry, Section, SectionKind};

    // Build a tiny self-contained hatch so we exercise the real
    // install path end-to-end.
    let mut vm_compile = VM::new_default();
    let main_wlbc = vm_compile
        .compile_source_to_blob("System.print(\"ok\")")
        .expect("compile main");

    let mut native_libs = BTreeMap::new();
    native_libs.insert(
        "custom_db".to_string(),
        NativeLibEntry::Path("/opt/custom/libdb.dylib".to_string()),
    );

    let hatch_bytes = emit(&Hatch {
        manifest: Manifest {
            name: "native-decls".to_string(),
            version: "0.1.0".to_string(),
            entry: "main".to_string(),
            description: None,
            homepage: None,
            readme: None,
            changelog: None,
            bundled_versions: BTreeMap::new(),
            modules: vec!["main".to_string()],
            dependencies: BTreeMap::new(),
            spec_dependencies: BTreeMap::new(),
            native_libs,
            native_search_paths: vec!["/opt/homebrew/lib".to_string()],
            plugin_source: None,
            target: None,
        },
        sections: vec![Section {
            kind: SectionKind::Wlbc,
            name: "main".to_string(),
            data: main_wlbc,
        }],
    })
    .expect("emit");

    let mut vm = VM::new_default();
    let result = vm.interpret_hatch(&hatch_bytes);
    assert!(matches!(result, InterpretResult::Success));

    // The manifest's declarations must have seeded the loader state.
    assert!(vm
        .native_search_paths
        .iter()
        .any(|p| p == std::path::Path::new("/opt/homebrew/lib")));
    assert_eq!(
        vm.native_lib_paths.get("custom_db"),
        Some(&std::path::PathBuf::from("/opt/custom/libdb.dylib"))
    );
}

/// Scalar replacement of a loop-carried object: the compiled body must
/// agree with the interpreter whether the object is replaced (getters
/// only) or must stay an allocation (it escapes).
#[test]
fn e2e_tiered_scalar_replaced_loop_object_matches_interpreter() {
    let source = r#"
class Vec2 {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  y { _y }
  norm2 { _x * _x + _y * _y }
}
class Bench {
  static walk(n) {
    var p = Vec2.new(0.5, 0.25)
    var acc = 0
    var i = 0
    while (i < n) {
      p = Vec2.new(p.x * 0.5 + 1, p.y * 0.5 + 2)
      acc = acc + p.x + p.y
      i = i + 1
    }
    return acc
  }
  static escapes(n) {
    var kept = []
    var p = Vec2.new(1, 2)
    var i = 0
    while (i < n) {
      p = Vec2.new(p.x + 1, p.y + 1)
      if (i % 1000 == 0) kept.add(p)
      i = i + 1
    }
    var s = 0
    for (q in kept) s = s + q.norm2
    return s
  }
}
System.print(Bench.walk(20000))
System.print(Bench.escapes(20000))
"#;
    let (result, output, _) = run(source);
    assert!(matches!(result, InterpretResult::Success), "{:?}", result);
    let expected = {
        let config = VMConfig {
            execution_mode: ExecutionMode::Interpreter,
            ..Default::default()
        };
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let r = vm.interpret("main", source);
        assert!(matches!(r, InterpretResult::Success), "{:?}", r);
        vm.take_output()
    };
    assert_eq!(output.trim(), expected.trim());
}

// ===========================================================================
// Runtime errors raised by natives
// ===========================================================================

/// Run `source` with an error callback and return (result, output, errors).
fn run_collecting_errors(
    source: &str,
    mode: ExecutionMode,
) -> (InterpretResult, String, Vec<String>) {
    use std::sync::Arc;
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = errors.clone();
    let config = VMConfig {
        error_fn: Some(Box::new(move |_, _, _, msg| {
            sink.lock().unwrap().push(msg.to_string());
        })),
        execution_mode: mode,
        ..VMConfig::default()
    };
    let (result, output, _) = run_with_config(source, config);
    let errors = errors.lock().unwrap().clone();
    (result, output, errors)
}

#[test]
fn e2e_native_error_in_method_reports_once() {
    let src = r#"
class B {
  static run(seed) {
    var acc = seed
    for (i in 0...3) {
      acc = acc + i
    }
    return acc * 2
  }
}
System.print("before")
B.run("s")
System.print("after")
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let (result, output, errors) = run_collecting_errors(src, mode);
        assert!(
            matches!(result, InterpretResult::RuntimeError),
            "{:?}",
            mode
        );
        assert_eq!(output.trim(), "before", "{:?}", mode);
        assert_eq!(errors.len(), 1, "{:?}: {:?}", mode, errors);
        assert!(
            errors[0].contains("Right operand must be a string."),
            "{:?}: {}",
            mode,
            errors[0]
        );
        assert!(
            errors[0].contains("stack trace"),
            "{:?}: {}",
            mode,
            errors[0]
        );
    }
}

#[test]
fn e2e_native_error_caught_by_fiber_try() {
    let src = r#"
var f = Fiber.new { "a" + 1 }
f.try()
System.print("caught: %(f.error)")
var g = Fiber.new { null + 1 }
g.try()
System.print("caught: %(g.error)")
var h = Fiber.new { [1, 2][5] }
h.try()
System.print("caught: %(h.error)")
System.print("done")
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let (result, output, errors) = run_collecting_errors(src, mode);
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}: {:?}",
            mode,
            errors
        );
        assert_eq!(errors, Vec::<String>::new(), "{:?}", mode);
        assert_eq!(
            output.trim(),
            "caught: Right operand must be a string.\ncaught: Null does not implement '+(_)'\ncaught: Subscript 5 out of bounds (count 2).\ndone",
            "{:?}",
            mode
        );
    }
}

#[test]
fn e2e_tiered_inlined_known_calls_match_interpreter() {
    // Guarded inlining with loop versioning: receivers and closures
    // that change after warm-up, multi-block callees, implicit null
    // returns, breaks, nested loops, loop live-outs, errors raised
    // inside an inlined body, and a site outside any loop.
    let src = r#"class A {
  construct new() { _k = 1 }
  step(acc, i) { (acc * 3 + i + _k) % 1000003 }
  pick(x) {
    if (x > 5) return x * 2
    return x - 1
  }
  noret(x) {
    _k = x
    return null
  }
  k { _k }
}
class B is A {
  construct new() { super() }
  step(acc, i) { (acc * 5 + i) % 1000003 }
}
class C {
  static sf(a, b) { a * 2 + b }
  static boom(x) { x + "s" }
}
// 1. receiver switches mid-loop after warm-up
var objs = [A.new(), B.new()]
var acc = 0
var i = 0
var last = null
while (i < 3000) {
  var o = objs[i < 2000 ? 0 : 1]
  acc = o.step(acc, i)
  last = o
  i = i + 1
}
System.print("1: %(acc) %(i) %(last is B)")
// 2. closure switches mid-loop
var f1 = Fn.new { |a, b| a + b }
var f2 = Fn.new { |a, b| a - b }
var s = 0
for (j in 0...3000) {
  var f = j < 2500 ? f1 : f2
  s = f.call(s, j)
}
System.print("2: %(s)")
// 3. static call site with class receiver switching
class D { static sf(a, b) { a - b } }
var t = 0
for (j in 0...3000) {
  var cls = j % 2 == 0 ? C : D
  t = cls.sf(t, 1) % 97
}
System.print("3: %(t)")
// 4. multi-block callee, implicit null return, break, nested loops, live-outs
var a = A.new()
var total = 0
var seen = 0
for (x in 0...200) {
  var inner = 0
  for (y in 0...50) {
    inner = inner + a.pick(y)
    if (inner > 10000) break
  }
  total = total + inner
  seen = x
  var r = a.noret(x)
  if (r != null) System.print("bad")
}
System.print("4: %(total) %(seen) %(a.k)")
// 5. runtime error inside an inlined callee under try, after warm-up
var u = 0
for (j in 0...2000) u = C.sf(u, 1) % 1009
var fb = Fiber.new {
  var w = 0
  for (j in 0...500) w = C.sf(w, j)
  C.boom(w)
}
fb.try()
System.print("5: %(u) %(fb.error)")
// 6. site outside any loop
System.print("6: %(a.pick(9)) %(a.pick(2)) %(C.sf(3, 4))")
"#;
    let (result, expected, _) = run_with_config(
        src,
        VMConfig {
            execution_mode: ExecutionMode::Interpreter,
            ..VMConfig::default()
        },
    );
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(
        expected.trim(),
        "1: 499414 3000 true\n2: 1749000\n3: 0\n4: 485800 199 199\n5: 452 Right operand must be a number.\n6: 18 1 10"
    );
    let (result, output, _) = run_with_config(
        src,
        VMConfig {
            execution_mode: ExecutionMode::Tiered,
            ..VMConfig::default()
        },
    );
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output, expected);
}

#[test]
fn e2e_block_body_implicit_value() {
    // An expression body returns its expression. A block body returns
    // its tail expression statement (a WrenLift extension; the
    // reference returns null) and nothing else: a declaration or a
    // control-flow statement at the end yields null.
    let src = r#"
class A {
  construct new() { _k = 0 }
  one(x) { _k = x }
  two(x) {
    x + 1
  }
  three(x) {
    var y = x
  }
  four(x) {
    if (x > 0) x
  }
  five(x) {
    var y = x
    y * 2
  }
}
var a = A.new()
System.print(a.one(3))
System.print(a.two(3))
System.print(a.three(3))
System.print(a.four(3))
System.print(a.five(3))
var f = Fn.new { |x|
  var t = x
}
System.print(f.call(3))
var g = Fn.new { |x| x + 1 }
System.print(g.call(3))
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let (result, output, _) = run_collecting_errors(src, mode);
        assert!(matches!(result, InterpretResult::Success), "{:?}", mode);
        assert_eq!(output.trim(), "3\n4\nnull\nnull\n6\nnull\n4", "{:?}", mode);
    }
}

#[test]
fn e2e_tiered_integer_specialised_loops_match_reference() {
    // Loop counters and accumulators proven integral run as i64; every
    // case that could differ from f64 (negative zero from a product,
    // negation or remainder, growth past 2^53, fractional steps, bounds
    // that are not constants) stays float. Expected output is wren_cli's.
    let src = r#"class K {
  // negative zero from a product with zero
  static negProd() {
    var s = 0
    var i = 0
    while (i < 3000) {
      s = i * -1
      i = i + 1
      if (i == 3000) s = 0 * -1
    }
    return s
  }
  // remainder of a negative dividend
  static negRem() {
    var s = 0
    var i = 0
    while (i < 3000) {
      s = (0 - i) % 8
      i = i + 1
    }
    return s
  }
  // remainder by a non power of two
  static rem7() {
    var s = 0
    var i = 0
    while (i < 3000) {
      s = (s * 5 + i) % 7
      i = i + 1
    }
    return s
  }
  // grows past 2^53 and must stay float
  static big() {
    var x = 1
    var i = 0
    while (i < 60) {
      x = x * 3
      i = i + 1
    }
    return x
  }
  // counter bounded by a non-constant stays float but correct
  static bound(n) {
    var s = 0
    var i = 0
    while (i < n) {
      s = s + i
      i = i + 1
    }
    return s
  }
  // fractional step stays float
  static frac() {
    var s = 0
    var i = 0
    while (i < 3000) {
      s = s + 0.5
      i = i + 1
    }
    return s
  }
  // subtraction below zero, negation
  static neg() {
    var s = 0
    var i = 0
    while (i < 3000) {
      s = (10 - i) - (-i)
      i = i + 1
    }
    return s
  }
  // exit value used after the loop with float ops
  static mixed() {
    var s = 0
    var i = 0
    while (i < 3000) {
      s = (s * 31 + (i % 8)) % 4294967296
      i = i + 1
    }
    return s / 3 + i.sqrt
  }
}
System.print(K.negProd())
System.print(K.negRem())
System.print(K.rem7())
System.print(K.big())
System.print(K.bound(3000))
System.print(K.bound(2.5))
System.print(K.frac())
System.print(K.neg())
System.print(K.mixed())
System.print(1 / K.negProd())
"#;
    let expected =
        "-0\n-7\n6\n4.2391158275216e+28\n4498500\n3\n1500\n10\n1011574997.4389\n-infinity";
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let (result, output, errors) = run_collecting_errors(src, mode);
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}: {:?}",
            mode,
            errors
        );
        assert_eq!(output.trim(), expected, "{:?}", mode);
    }
}

#[test]
fn e2e_tiered_osr_entered_loop_runs_every_iteration() {
    // Nine toggles per iteration and an odd count make the final value
    // depend on the iteration count, which the reference benchmark's
    // even count hides. An OSR entry that skips iterations shows here.
    let src = r#"
class Toggle {
  construct new(startState) { _state = startState }
  value { _state }
  activate {
    _state = !_state
    return this
  }
}
var n = 100001
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
}
System.print(val)
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let (result, output, errors) = run_collecting_errors(src, mode);
        assert!(
            matches!(result, InterpretResult::Success),
            "{:?}: {:?}",
            mode,
            errors
        );
        assert_eq!(output.trim(), "false", "{:?}", mode);
    }
}

#[test]
fn e2e_top_tier_promotion_and_retier_match_interpreter() {
    // Low thresholds so every shape reaches the top tier inside the
    // test: methods promoted through their entry count, the module
    // body's loops transferred mid-loop from baseline code, boxed and
    // float arithmetic, remainders, math intrinsics, fields, module
    // variables, closures, lists, maps and strings. Output is compared
    // with the interpreter.
    let src = r#"
class Acc {
  construct new() {
    _sum = 0
    _n = 0
  }
  add(x) {
    _sum = _sum + x
    _n = _n + 1
    return this
  }
  mean { _n == 0 ? 0 : _sum / _n }
  sum { _sum }
}
class Body {
  construct new(x, v) {
    _x = x
    _v = v
  }
  x { _x }
  v { _v }
  step(dt) {
    _x = _x + _v * dt
    if (_x > 10) _v = -_v
    if (_x < -10) _v = -_v
  }
}
var total = 0
var acc = Acc.new()
var bodies = []
for (i in 0...8) bodies.add(Body.new(i * 1.5 - 4, (i % 3) - 1.25))
for (i in 0...20000) {
  var b = bodies[i % 8]
  b.step(0.01)
  acc.add(b.x)
  total = total + (i % 7) * 0.5 - (i & 3)
  if (i % 4096 == 0) total = total + (i.sqrt + i.sin.abs).floor
}
var names = {}
var words = ["a", "b", "c", "d"]
var text = ""
for (i in 0...3000) {
  var w = words[i % 4]
  names[w] = (names[w] || 0) + 1
  if (i % 1000 == 0) text = text + w + i.toString
}
var mul = Fn.new { |a, b| a * b }
var fsum = 0
for (i in 0...5000) fsum = fsum + mul.call(i, 2) % 9
System.print(total)
System.print(acc.sum.truncate)
System.print((acc.mean * 1000).round / 1000)
System.print(bodies.map { |b| (b.x * 100).round / 100 }.toList)
System.print(names["a"] + names["d"])
System.print(text)
System.print(fsum)
"#;
    let run = |mode: ExecutionMode| {
        use std::sync::Arc;
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        let config = VMConfig {
            error_fn: Some(Box::new(move |_, _, _, msg| {
                sink.lock().unwrap().push(msg.to_string());
            })),
            execution_mode: mode,
            jit_threshold: 20,
            opt_threshold: 40,
            ..VMConfig::default()
        };
        let (result, output, _) = run_with_config(src, config);
        let errors = errors.lock().unwrap().clone();
        (result, output, errors)
    };
    let (r0, expected, e0) = run(ExecutionMode::Interpreter);
    assert!(matches!(r0, InterpretResult::Success), "{:?}", e0);
    // Once with the tiers warm in the same process and once in a fresh
    // engine so the promotion path runs at least twice.
    for _ in 0..2 {
        let (r1, output, e1) = run(ExecutionMode::Tiered);
        assert!(matches!(r1, InterpretResult::Success), "{:?}", e1);
        assert_eq!(output, expected);
    }
}

#[test]
fn e2e_top_tier_result_speculation_deopts_mid_body() {
    // The top tier guards call results the baseline only ever saw as
    // Num. When a getter later returns a string, the guard fails after
    // this iteration's store to `_a` has happened, so the interpreter
    // must resume just past the call rather than re-run the method.
    // The warm-up is long enough for the top-tier compile to land.
    let src = r#"
class Cell {
  construct new(v) { _v = v }
  v { _v }
  v=(x) { _v = x }
}
class Acc {
  construct new() {
    _a = Cell.new(1)
    _b = Cell.new(2)
  }
  step(n) {
    var total = 0
    for (i in 0...n) {
      _a.v = _a.v + i
      var x = _b.v
      total = total + x.toString.count
    }
    return total
  }
  switchB(v) { _b.v = v }
  a { _a.v }
}
var acc = Acc.new()
var sum = 0
for (k in 0...40000) sum = sum + acc.step(20)
acc.switchB("str")
var r = acc.step(3)
System.print("%(sum) %(r) %(acc.a)")
"#;
    let run = |mode: ExecutionMode| {
        let config = VMConfig {
            execution_mode: mode,
            jit_threshold: 20,
            opt_threshold: 40,
            ..VMConfig::default()
        };
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", src);
        let output = vm.take_output();
        (result, output, vm.engine.deopt_exits)
    };
    let (r0, expected, _) = run(ExecutionMode::Interpreter);
    assert!(matches!(r0, InterpretResult::Success));
    assert_eq!(expected.trim(), "800000 9 7600004");
    let (r1, output, deopts) = run(ExecutionMode::Tiered);
    assert!(matches!(r1, InterpretResult::Success));
    assert_eq!(output, expected);
    if wren_lift::codegen::top_tier() != wren_lift::codegen::TopTier::Off {
        assert_eq!(deopts, 1, "the guard on `_b.v` fires exactly once");
    }
}

#[test]
fn e2e_top_tier_class_miss_on_guarded_call_deopts() {
    // A guarded getter's class check has no slow path: when the
    // receiver is a different class, even one whose getter also
    // returns a Num, the interpreter redoes the call itself.
    let src = r#"
class Cell {
  construct new(v) { _v = v }
  v { _v }
  v=(x) { _v = x }
}
class Twice {
  construct new(v) { _v = v }
  v { _v * 2 }
}
class Acc {
  construct new() {
    _a = Cell.new(1)
    _b = Cell.new(2)
  }
  step(n) {
    var total = 0
    for (i in 0...n) {
      _a.v = _a.v + i
      total = total + _b.v
    }
    return total
  }
  switchB(v) { _b = v }
  a { _a.v }
}
var acc = Acc.new()
var sum = 0
for (k in 0...400000) sum = sum + acc.step(20)
acc.switchB(Twice.new(3))
var r = acc.step(3)
System.print("%(sum) %(r) %(acc.a)")
"#;
    // The warm-up is long enough for the top-tier compile to land and
    // too long to also run interpreted here; the expected line is the
    // interpreter's.
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    let deopts = vm.engine.deopt_exits;
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output.trim(), "16000000 18 76000004");
    // The top tier lands on another thread: when it does in time, the
    // class miss on `_b.v` fires exactly once, never per call.
    if wren_lift::codegen::top_tier() == wren_lift::codegen::TopTier::Llvm {
        assert!(deopts <= 1, "the class miss on `_b.v` fired {deopts} times");
    }
}

#[test]
fn e2e_top_tier_equality_honours_a_custom_operator() {
    // `==` and `!=` on a non-Num compare by identity only when the left
    // operand's class does not define them; the flag lives on the
    // class, and a null or Num left operand has no class to read.
    let src = r#"
class P {
  construct new(x) { _x = x }
  x { _x }
  ==(other) { other is P && _x == other.x }
  !=(other) { !(this == other) }
}
class H {
  construct new(p) { _p = p }
  count(q, n) {
    var acc = 0
    for (i in 0...n) {
      if (_p == q) acc = acc + 1
      if (_p != q) acc = acc + 10
      if (null == q) acc = acc + 100
      if (q != 1) acc = acc + 1000
    }
    return acc
  }
}
var h = H.new(P.new(7))
System.print(h.count(P.new(7), 3000000))
"#;
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output.trim(), "3003000000");
}

#[test]
fn e2e_top_tier_inlined_constructor_keeps_field_semantics() {
    // A constructor call on a module-level class is inlined into a hot
    // caller: fields the initialiser stores first are not pre-nulled,
    // every other field starts null even when read before its own
    // store, and the call's value is the instance.
    let src = r#"
class Pair {
  construct new(a, b) {
    _a = a
    _b = b
    _seenC = _c
    _c = a + b
  }
  a { _a }
  b { _b }
  c { _c }
  seenC { _seenC }
}
class Maker {
  static build(n) {
    var total = 0
    var nulls = 0
    var last = null
    for (i in 0...n) {
      var p = Pair.new(i, 1)
      total = total + p.a + p.b + p.c
      if (p.seenC == null) nulls = nulls + 1
      last = p
    }
    return "%(total) %(nulls) %(last is Pair) %(last.c)"
  }
}
System.print(Maker.build(2000000))
"#;
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output.trim(), "4000002000000 2000000 true 2000000");
}

#[test]
fn e2e_inlined_sites_in_a_loop_with_a_merge_compile() {
    // Two inlinable constructor calls in the arms of a conditional
    // inside a loop: the loop is versioned, both sites enter the
    // generic copy after their own calls, and the copy's merge block
    // reads values only the copy defines. Every tier must compile it.
    let src = r#"
class P {
  construct new(i) { _i = i }
  i { _i }
}
class M {
  construct new(i) { _i = i * 2 }
  i { _i }
}
class B {
  static build(n) {
    var world = []
    for (i in 0...n) world.add(i % 2 == 0 ? P.new(i) : M.new(i))
    var sum = 0
    for (e in world) sum = sum + e.i
    return sum
  }
}
var total = 0
for (k in 0...2000) total = total + B.build(40)
System.print(total)
"#;
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output.trim(), "2360000");
    let build = (0..vm.engine.function_count() as u32)
        .map(wren_lift::runtime::engine::FuncId)
        .find(|id| {
            vm.engine
                .get_mir(*id)
                .is_some_and(|m| vm.interner.resolve(m.name) == "build(_)")
        })
        .expect("build(_) is registered");
    // The compile runs on another thread; give it time to land, then
    // a failed compile is the only way the body stays interpreted.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while vm.engine.tier_state(build) == wren_lift::runtime::engine::TierState::Interpreted
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(1));
        vm.engine.poll_compilations();
    }
    assert_ne!(
        vm.engine.tier_state(build),
        wren_lift::runtime::engine::TierState::Interpreted,
        "build(_) never left the interpreter"
    );
}

#[test]
fn e2e_closure_captures_for_loop_variable_per_iteration() {
    // A for loop binds a fresh variable each iteration; a closure that
    // captures it, alongside a body local, keeps that iteration's
    // value, and the loop's own code still reads it as a plain value.
    let src = r#"
var keep = []
for (round in 0...3) {
  var n = [round * 10]
  keep.add(Fn.new { "%(round):%(n[0])" })
  if (round == 1) keep.add(Fn.new { round })
}
var out = []
for (f in keep) out.add(f.call())
System.print(out.join(" "))
"#;
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tiered] {
        let mut vm = VM::new(VMConfig {
            execution_mode: mode,
            ..VMConfig::default()
        });
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", src);
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output.trim(), "0:0 1:10 1 2:20");
    }
}

#[test]
fn e2e_calls_after_a_super_call_keep_their_inline_caches() {
    // A super call takes no inline-cache slot in the bytecode; the
    // calls after it in the same body must still resolve to their own
    // entries, or a getter is devirtualised to its neighbour's field.
    let src = r#"
class Base {
  construct new(a, b) {
    _a = a
    _b = b
  }
  a { _a }
  b { _b }
  tag() { 0 }
}
class Derived is Base {
  construct new(a, b) { super(a, b) }
  probe() {
    var t = super.tag()
    var x = a
    var y = b
    return t * 100 + x * 10 + y
  }
  step(n) {
    var s = 0
    for (i in 0...n) s = s + probe()
    return s
  }
}
System.print(Derived.new(3, 7).step(100000))
"#;
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(output.trim(), "3700000");
}

#[test]
fn e2e_native_operator_and_call_errors_surface_from_compiled_code() {
    // A missing operator or method reached from compiled code raises
    // the interpreter's error instead of yielding null and carrying on,
    // and a speculatively typed parameter given another type is handed
    // back to the interpreter, which raises the same error.
    let src = r#"
class K {
  static twice(x) {
    var a = x + x
    var b = a * 2 - x
    var c = b / 4 + a
    return c - b + x * 3
  }
  static poke(o) { o.nothing }
}
var s = 0
for (i in 0...3000) s = s + K.twice(i)
System.print(s)
var caught = Fiber.new { K.twice("ab") }.try()
System.print(caught)
var caught2 = Fiber.new { K.poke(3) }.try()
System.print(caught2)
System.print(K.twice(1.5))
K.twice("zz")
System.print("not reached")
"#;
    let run = |mode: ExecutionMode| {
        use std::sync::Arc;
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        let config = VMConfig {
            error_fn: Some(Box::new(move |_, _, _, msg| {
                sink.lock().unwrap().push(msg.to_string());
            })),
            execution_mode: mode,
            jit_threshold: 20,
            opt_threshold: 40,
            ..VMConfig::default()
        };
        let (result, output, _) = run_with_config(src, config);
        let errors = errors.lock().unwrap().clone();
        (result, output, errors)
    };
    let (r0, expected, e0) = run(ExecutionMode::Interpreter);
    assert!(matches!(r0, InterpretResult::RuntimeError), "{:?}", e0);
    assert!(
        expected.contains("String does not implement '-(_)'"),
        "{expected}"
    );
    assert!(
        expected.contains("Num does not implement 'nothing'"),
        "{expected}"
    );
    assert!(!expected.contains("not reached"));
    let (r1, output, e1) = run(ExecutionMode::Tiered);
    assert!(matches!(r1, InterpretResult::RuntimeError), "{:?}", e1);
    assert_eq!(output, expected);
    assert_eq!(e1.len(), 1, "{:?}", e1);
    assert!(
        e1[0].contains("String does not implement '-(_)'"),
        "{:?}",
        e1
    );
}

/// Run `warm` as module "main", wait for `name` to reach the top tier,
/// then run `then` as another module importing it; returns the second
/// run's output and the deopts it caused, or nothing in a build
/// without a top tier.
fn run_after_top_tier(warm: &str, name: &str, then: &str) -> Option<(String, u32)> {
    use wren_lift::runtime::engine::{FuncId, TierState};
    if wren_lift::codegen::top_tier() == wren_lift::codegen::TopTier::Off {
        return None;
    }
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", warm);
    assert!(
        matches!(result, InterpretResult::Success),
        "{}",
        vm.take_output()
    );
    let id = (0..vm.engine.function_count() as u32)
        .map(FuncId)
        .find(|id| {
            vm.engine
                .get_mir(*id)
                .is_some_and(|m| vm.interner.resolve(m.name) == name)
        })
        .unwrap_or_else(|| panic!("{name} is registered"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while vm.engine.tier_state(id) != TierState::OptimizedNative
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(1));
        vm.engine.poll_compilations();
        let _ = vm.interpret("tick", "var x = 1\n");
    }
    assert_eq!(
        vm.engine.tier_state(id),
        TierState::OptimizedNative,
        "{name} never reached the top tier: {:?}",
        vm.engine.tier_state(id)
    );
    vm.output_buffer = Some(String::new());
    let before = vm.engine.deopt_exits;
    let result = vm.interpret("then", then);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success), "{output}");
    Some((output, vm.engine.deopt_exits - before))
}

const PROMOTED_ACC: &str = r#"
class Acc {
  construct new(v) { _v = v }
  bump(x) {
    _v = _v + x
    return _v
  }
  v { _v }
  v=(x) { _v = x }
}
"#;

#[test]
fn e2e_promoted_instance_is_rebuilt_at_a_deopt() {
    // The top tier keeps `a` as field values; when the guard on
    // `probe`'s result fails, the interpreter resumes with an instance
    // rebuilt from them and finishes the loop and the calls after it.
    let warm = format!(
        "{PROMOTED_ACC}
class K {{
  static probe(i, bad) {{
    var k = 0
    while (k < 1) k = k + 1
    return bad && i == 7 ? \"s\" : 1
  }}
  static run(n, bad) {{
    var a = Acc.new(10)
    var s = 0
    for (i in 0...n) {{
      a.bump(1)
      var m = K.probe(i, bad)
      if (m is Num) s = s + m
    }}
    a.bump(100)
    return \"%(a.v) %(s)\"
  }}
}}
for (k in 0...3000) K.run(20, false)
"
    );
    let then =
        "import \"main\" for K\nSystem.print(K.run(20, true))\nSystem.print(K.run(20, false))\n";
    let Some((output, deopts)) = run_after_top_tier(&warm, "run(_,_)", then) else {
        return;
    };
    assert_eq!(output.trim(), "130 19\n130 20");
    assert!(deopts >= 1, "the result guard never fired");
}

#[test]
fn e2e_promoted_instance_survives_a_class_reassignment() {
    // The constructor site's guard reads the module variable; once it
    // holds another class the compiled body hands the call back to the
    // interpreter, which makes the other class's instance.
    let warm = format!(
        "{PROMOTED_ACC}
class Other {{
  construct new(v) {{ _v = v * 10 }}
  bump(x) {{
    _v = _v + x * 10
    return _v
  }}
  v {{ _v }}
}}
var Ctor = Acc
class K {{
  static build(n) {{
    var s = 0
    for (i in 0...n) {{
      var a = Ctor.new(i)
      a.bump(1)
      s = s + a.v
    }}
    return s
  }}
  static swap() {{ Ctor = Other }}
}}
for (k in 0...3000) K.build(30)
"
    );
    let then = "import \"main\" for K\nSystem.print(K.build(30))\nK.swap()\nSystem.print(K.build(30))\nSystem.print(K.build(30))\n";
    let Some((output, _)) = run_after_top_tier(&warm, "build(_)", then) else {
        return;
    };
    assert_eq!(output.trim(), "465\n4650\n4650");
}

#[test]
fn e2e_promoted_instance_fields_merge_across_branches() {
    // Fields stored on different paths meet where the paths join, and
    // one instance's field feeds another's; the top tier lands mid-run.
    let src = format!(
        "{PROMOTED_ACC}
class K {{
  static branchy(n) {{
    var a = Acc.new(0)
    var t = 0
    for (i in 0...n) {{
      if (i % 2 == 0) a.v = i else a.bump(2)
      t = t + a.v
    }}
    return \"%(a.v) %(t)\"
  }}
  static nested(n) {{
    var a = Acc.new(1)
    var b = Acc.new(2)
    var s = 0
    for (i in 0...n) {{
      a.bump(b.v)
      b.v = a.v % 7
      s = s + b.v
    }}
    return \"%(a.v) %(b.v) %(s)\"
  }}
}}
System.print(K.branchy(300000))
System.print(K.nested(300000))
var x = \"\"
for (r in 0...3) x = K.branchy(50000) + \" \" + K.nested(50000)
System.print(x)
"
    );
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 20,
        opt_threshold: 40,
        ..VMConfig::default()
    };
    let (result, output, _) = run_with_config(&src, config);
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(
        output.trim(),
        "300000 45000000000\n1399998 5 1400000\n50000 1250000000 233330 6 233333"
    );
}

/// The scheduler runs tasks on their own stacks, so it is absent
/// under `WLIFT_KRIO_FIBER=0`.
fn scheduler_available() -> bool {
    VM::new_default().krio_fiber_active
}

#[test]
fn e2e_scheduler_runs_tasks_by_turns_timers_and_wakes() {
    if !scheduler_available() {
        return;
    }
    // Tasks run a turn each per tick; a sleep is a timer park that
    // lets the others run, a park resolves on the first wake only,
    // and a fiber a task calls parks the task with it.
    let src = r#"
var log = []
Fiber.spawn {
  log.add("a1")
  Fiber.yield()
  log.add("a2")
  Fiber.sleep(150)
  log.add("a3")
}
Fiber.spawn {
  log.add("b1")
  Fiber.sleep(50)
  log.add("b2")
}
var w = Fiber.waiter
var c = Fiber.spawn {
  log.add("c %(Fiber.park(w, 5000))")
}
Fiber.spawn {
  Fiber.yield()
  log.add("wake %(Fiber.wake(w))")
  log.add("again %(Fiber.wake(w))")
}
System.print("live %(Fiber.live)")
while (Fiber.tick(0)) Fiber.idle(100)
System.print(log.join(","))
System.print("live %(Fiber.live) done %(c.isDone)")

var order = []
Fiber.spawn {
  var f = Fiber.new {
    order.add("n1")
    Fiber.sleep(100)
    order.add("n2")
    "ret"
  }
  order.add("task %(f.call())")
}
Fiber.spawn {
  Fiber.sleep(5)
  order.add("other")
}
while (Fiber.tick(0)) Fiber.idle(100)
System.print(order.join(","))

var bad = Fiber.spawn { Fiber.abort("boom") }
Fiber.tick(0)
System.print("bad: %(bad.error) %(bad.isDone)")

var pre = Fiber.waiter
System.print("%(Fiber.wake(pre)) %(Fiber.park(pre, 0)) %(Fiber.wake(pre))")
var t = Fiber.spawn { Fiber.tick(0) }
Fiber.tick(0)
System.print(t.error)
"#;
    let expected = [
        "live 4",
        "a1,b1,a2,wake true,again false,c true,b2,a3",
        "live 0 done true",
        "n1,other,n2,task ret",
        "bad: boom true",
        "true true false",
        "Fiber.tick: a task cannot drive the scheduler.",
    ]
    .join("\n");
    for (jit_threshold, opt_threshold) in [(u32::MAX, u32::MAX), (1, 4)] {
        let config = VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold,
            opt_threshold,
            ..VMConfig::default()
        };
        let (result, output, _) = run_with_config(src, config);
        assert!(matches!(result, InterpretResult::Success), "{output}");
        assert_eq!(output.trim(), expected);
    }
}

#[test]
fn e2e_scheduler_tasks_survive_collections() {
    if !scheduler_available() {
        return;
    }
    // Task fibers are roots of the world that holds them (the suite
    // runs under WLIFT_GC_STRESS=1 too).
    let src = r#"
var total = 0
for (i in 0...200) {
  Fiber.spawn {
    var acc = []
    for (j in 0...50) {
      acc.add("s%(j)" * 3)
      if (j % 10 == 0) Fiber.yield()
      if (j == 25) Fiber.sleep(1)
    }
    total = total + acc.count
  }
}
while (Fiber.tick(0)) Fiber.idle(50)
System.print(total)
"#;
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 1,
        opt_threshold: 4,
        ..VMConfig::default()
    };
    let (result, output, _) = run_with_config(src, config);
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(output.trim(), "10000");
}

#[test]
fn e2e_scheduler_wake_from_another_thread() {
    if !scheduler_available() {
        return;
    }
    // A waiter's token can be woken by any thread; the driver parked
    // on it returns as soon as the wake lands.
    let mut vm = VM::new_default();
    vm.output_buffer = Some(String::new());
    let token = vm.sched.get_or_insert_with(Default::default).new_waiter();
    let waker = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(20));
        wren_lift::runtime::sched::wake(token)
    });
    let start = Instant::now();
    let result = vm.interpret("main", &format!("System.print(Fiber.park({token}, 5000))"));
    assert!(matches!(result, InterpretResult::Success));
    assert_eq!(vm.take_output().trim(), "true");
    assert!(start.elapsed() < std::time::Duration::from_secs(4));
    assert!(waker.join().unwrap());
}

/// A VM whose isolates can import `worker` and `ticker` from memory.
fn vm_with_isolate_modules() -> VM {
    fn module_source(name: &str) -> Option<String> {
        match name {
            "worker" => Some(
                r#"
import "isolate" for Isolate
var arg = Isolate.arg
var sum = 0
for (i in 0...arg["n"]) sum = sum + i
arg["reply"].send({"who": arg["who"], "sum": sum, "list": [1, "two", null, true]})
"#
                .to_string(),
            ),
            "ticker" => Some(
                r#"
import "isolate" for Isolate
var a = Isolate.arg
for (k in 0...2) {
  Fiber.spawn {
    for (i in 0...3) {
      Fiber.sleep(2)
      a["out"].send("%(a["name"])-%(k)-%(i)")
    }
  }
}
while (Fiber.tick(0)) Fiber.idle(50)
a["out"].send("done")
"#
                .to_string(),
            ),
            _ => None,
        }
    }
    fn make() -> VM {
        let config = VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            opt_threshold: 4,
            load_module_fn: Some(Box::new(|name: &str, _from: &str| module_source(name))),
            ..VMConfig::default()
        };
        VM::new(config)
    }
    let mut vm = make();
    vm.isolate_factory = Some(std::sync::Arc::new(make));
    vm
}

#[test]
fn e2e_isolates_run_on_threads_and_pass_values_by_copy() {
    if !scheduler_available() {
        return;
    }
    let src = r#"
import "isolate" for Isolate, Channel
var reply = Channel.new()
var workers = []
for (k in 0...4) workers.add(Isolate.spawn("worker", {"who": k, "n": 100000, "reply": reply}))
var got = []
for (k in 0...4) {
  var r = reply.receive(5000)
  got.add(r["who"])
  if (r["who"] == 0) System.print("sum %(r["sum"]) list %(r["list"])")
}
got.sort()
System.print(got)
for (w in workers) System.print("join %(w.join()) done %(w.isDone) err %(w.error)")
var bad = Isolate.spawn("nope")
bad.join(5000)
System.print("bad: %(bad.error)")
System.print(reply.receive(10))
reply.close()
System.print("%(reply.send(1)) %(reply.receive()) %(reply.isClosed) %(reply.count)")
var f = Fiber.new { Channel.new().send(Fiber.current) }
f.try()
System.print(f.error)
var ch = Channel.new()
var w = Isolate.spawn("worker", {"who": 9, "n": 10, "reply": ch})
ch.send(w)
System.print(ch.receive() is Isolate)
System.print(ch.receive()["who"])
"#;
    let mut vm = vm_with_isolate_modules();
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success), "{output}");
    let expected = [
        "sum 4999950000 list [1, two, null, true]",
        "[0, 1, 2, 3]",
        "join true done true err null",
        "join true done true err null",
        "join true done true err null",
        "join true done true err null",
        "bad: compile error in \"nope\"",
        "null",
        "false null true 0",
        "Channel.send: cannot send a Fiber across isolates.",
        "true",
        "9",
    ]
    .join("\n");
    assert_eq!(output.trim(), expected);
}

#[test]
fn e2e_isolate_receive_parks_a_task() {
    // A task parked on a channel is woken by a send from another
    // isolate's thread while the world keeps running its other tasks.
    if !scheduler_available() {
        return;
    }
    let src = r#"
import "isolate" for Isolate, Channel
var out = Channel.new()
var tickers = []
for (name in ["x", "y"]) tickers.add(Isolate.spawn("ticker", {"name": name, "out": out}))
var got = []
var ticks = 0
Fiber.spawn {
  var dones = 0
  while (dones < 2) {
    var r = out.receive(5000)
    if (r == null) break
    if (r == "done") dones = dones + 1 else got.add(r)
  }
}
Fiber.spawn {
  while (Fiber.live > 1) {
    ticks = ticks + 1
    Fiber.sleep(1)
  }
}
while (Fiber.tick(0)) Fiber.idle(100)
System.print("%(got.count) %(ticks > 0)")
for (t in tickers) System.print(t.join(5000))
"#;
    let mut vm = vm_with_isolate_modules();
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(output.trim(), "12 true\ntrue\ntrue");
}

#[test]
fn e2e_two_threads_allocate_and_collect_one_heap() {
    // Two views of one program allocate on two threads; a collection
    // on either stops the other at its safepoint and keeps what it
    // holds. Both threads touch only the runtime API, so nothing
    // Wren-level runs in parallel.
    use wren_lift::runtime::object::NativeContext;
    use wren_lift::runtime::vm::Spill;
    let mut a = VM::new_default();
    fn churn(vm: &mut VM, tag: &str) -> Vec<String> {
        vm.leave_safe();
        let mut kept: Vec<String> = Vec::new();
        for i in 0..200_000 {
            let text = format!("{tag}-{i}");
            let v = vm.alloc_string(text.clone());
            if i % 97 == 0 {
                vm.api_stack.push(v);
                kept.push(text);
            }
            if i % 1000 == 0 {
                vm.poll_gc();
            }
            if i % 50_000 == 0 {
                vm.collect_garbage();
            }
        }
        let out: Vec<String> = vm.api_stack[vm.api_stack.len() - kept.len()..]
            .iter()
            .map(|&v| wren_lift::runtime::core::as_string(v).to_owned())
            .collect();
        assert_eq!(out, kept);
        // Done with the heap: safe from here, so the other thread's
        // collections need not wait for this one.
        let mut spill = Spill::new();
        vm.enter_safe(&mut spill);
        std::hint::black_box(&spill);
        kept
    }
    let worker = a.spawn_thread(|b| churn(b, "b"));
    churn(&mut a, "a");
    let kept_b = worker.join().unwrap();
    assert_eq!(kept_b.len(), 200_000 / 97 + 1);
    // Both threads asked for collections; each stopped the other.
    assert!(
        a.gc.stats().major_collections >= 8,
        "{}",
        a.gc.stats().major_collections
    );
}

#[test]
fn e2e_two_threads_run_wren_on_one_heap() {
    // Two views run Wren at once: both allocate, call a shared class,
    // tier up and load modules while the other runs.
    if !scheduler_available() {
        return;
    }
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 1,
        opt_threshold: 4,
        ..VMConfig::default()
    };
    let mut a = VM::new(config);
    a.output_buffer = Some(String::new());
    let result = a.interpret(
        "main",
        "class Acc {\n  static sum(n) {\n    var s = 0\n    for (i in 0...n) s = s + i\n    return s\n  }\n}\n",
    );
    assert!(matches!(result, InterpretResult::Success));
    let body = |tag: &str| {
        format!(
            r#"import "main" for Acc
class Box_{tag} {{
  construct new(v) {{ _v = v }}
  v {{ _v }}
  double {{ Box_{tag}.new(_v * 2) }}
}}
var t = 0
var seen = {{}}
for (k in 0...300) {{
  var l = []
  for (i in 0...400) l.add("{tag}%(i)")
  var gen = Fiber.new {{
    for (x in l) Fiber.yield(x)
  }}
  var n = 0
  while (!gen.isDone) {{
    var x = gen.call()
    if (x != null) n = n + 1
  }}
  var f = Fn.new {{|a| Box_{tag}.new(a).double.v }}
  seen["k%(k % 7)"] = f.call(k)
  t = t + Acc.sum(1000) + l.count + n
}}
System.print("{tag} %(t) %(seen.count)")
"#
        )
    };
    let worker_src = body("w");
    let worker = a.spawn_thread(move |b| b.interpret("worker", &worker_src));
    let result = a.interpret("main2", &body("m"));
    assert!(matches!(result, InterpretResult::Success));
    assert!(matches!(worker.join().unwrap(), InterpretResult::Success));
    let mut lines: Vec<String> = a.take_output().lines().map(String::from).collect();
    lines.sort();
    assert_eq!(
        lines,
        vec!["m 150090000 7".to_string(), "w 150090000 7".to_string()]
    );
}

#[test]
fn e2e_threads_share_one_heap_through_a_deque_mutex_and_lock() {
    // Tasks on the worker threads produce into a Deque and consume
    // from it under a Mutex, signal a Lock, wait with a timeout, and
    // run compiled code in parallel.
    if !scheduler_available() {
        return;
    }
    let src = r#"
import "thread" for Thread, Mutex, Lock, Deque
var q = Deque.new()
var produced = Lock.new()
var done = Lock.new()
var m = Mutex.new()
var total = 0
var log = []
for (p in 0...4) {
  Thread.create {
    for (i in 0...250) {
      q.add([p, i])
      if (i % 50 == 0) Thread.yield()
    }
    produced.release()
  }
}
for (c in 0...3) {
  Thread.create {
    while (true) {
      var item = q.pop(true)
      if (item == null) break
      m.acquire()
      total = total + item[1]
      m.release()
    }
    m.acquire()
    log.add("end%(c)")
    m.release()
    done.release()
  }
}
for (i in 0...4) produced.wait()
for (i in 0...3) q.add(null)
for (i in 0...3) done.wait()
System.print("total %(total) leftover %(q.count) log %(log.count)")
var l = Lock.new()
System.print(l.wait(20))
Thread.create {
  Fiber.sleep(10)
  l.release()
}
System.print(l.wait(5000))
var held = Mutex.new()
held.acquire()
System.print(held.tryAcquire())
held.release()
System.print(held.tryAcquire())
class F {
  static fib(n) {
    if (n < 2) return n
    return fib(n - 1) + fib(n - 2)
  }
}
var sums = Deque.new()
for (i in 0...8) {
  Thread.create { sums.add(F.fib(22)) }
}
var s = 0
for (i in 0...8) s = s + sums.pop(true)
System.print("fib %(s) workers>1 %(Thread.count > 1)")
"#;
    let (jit_threshold, opt_threshold) = if std::env::var_os("E2E_THREADS_HOT").is_some() {
        (1, 4)
    } else {
        (u32::MAX, u32::MAX)
    };
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold,
        opt_threshold,
        ..VMConfig::default()
    };
    let (result, output, _) = run_with_config(src, config);
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(
        output.trim(),
        [
            "total 124500 leftover 0 log 3",
            "false",
            "true",
            "false",
            "true",
            "fib 141688 workers>1 true",
        ]
        .join("\n")
    );
}

#[test]
fn e2e_threads_fibers_and_isolates_compose() {
    // A task spawns scheduler fibers on its worker and waits on them;
    // a main-thread fiber parks on a Lock a thread releases; an
    // isolate runs threads that drive fibers; a task receives from
    // the isolate's channel.
    if !scheduler_available() {
        return;
    }
    fn module_source(name: &str) -> Option<String> {
        (name == "worker").then(|| {
            r#"
import "isolate" for Isolate
import "thread" for Thread, Lock, Deque
var arg = Isolate.arg
var parts = Deque.new()
var done = Lock.new()
for (t in 0...3) {
  Thread.create {
    var gen = Fiber.new {
      for (i in 0...4) Fiber.yield(t * 10 + i)
    }
    var s = 0
    while (!gen.isDone) {
      var v = gen.call()
      if (v != null) s = s + v
    }
    parts.add(s)
    done.release()
  }
}
for (t in 0...3) done.wait()
var total = 0
while (parts.count > 0) total = total + parts.pop(false)
arg["reply"].send(total)
"#
            .to_string()
        })
    }
    fn make() -> VM {
        VM::new(VMConfig {
            execution_mode: ExecutionMode::Tiered,
            jit_threshold: 1,
            opt_threshold: 4,
            load_module_fn: Some(Box::new(|name: &str, _from: &str| module_source(name))),
            ..VMConfig::default()
        })
    }
    let src = r#"
import "thread" for Thread, Lock, Deque
import "isolate" for Isolate, Channel
var log = Deque.new()
var done = Lock.new()
Thread.create {
  var got = Lock.new()
  var acc = []
  for (i in 0...3) {
    Fiber.spawn {
      Fiber.sleep(5 * i)
      acc.add(i)
      got.release()
    }
  }
  for (i in 0...3) got.wait()
  log.add("task fibers %(acc)")
  done.release()
}
var handoff = Lock.new()
Fiber.spawn {
  handoff.wait()
  log.add("main fiber woke")
  done.release()
}
Thread.create {
  Fiber.sleep(10)
  handoff.release()
}
var reply = Channel.new()
var iso = Isolate.spawn("worker", {"reply": reply})
Thread.create {
  log.add("isolate said %(reply.receive())")
  done.release()
}
for (i in 0...3) done.wait()
iso.join()
var lines = []
while (log.count > 0) lines.add(log.pop(false))
lines.sort {|a, b| a.count < b.count }
for (l in lines) System.print(l)
"#;
    let mut vm = make();
    vm.isolate_factory = Some(std::sync::Arc::new(make));
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(
        output.trim(),
        "main fiber woke\nisolate said 138\ntask fibers [0, 1, 2]"
    );
}

#[test]
fn e2e_closure_in_a_method_reaches_the_class_static_fields() {
    // A function or fiber made inside a method names the class's
    // static fields, at any nesting; two reads of different fields
    // in one expression stay distinct.
    let src = r#"class A {
  static go() {
    __x = 1
    __y = 2
    System.print("direct %(__x) %(__y)")
    var f = Fiber.new { System.print("in fiber x=%(__x)") }
    f.call()
    var g = Fn.new { System.print("in fn x=%(__x) y=%(__y)") }
    g.call()
    var h = Fn.new { __x = 5 }
    h.call()
    System.print("after h x=%(__x)")
    var nested = Fn.new { Fn.new { System.print("nested x=%(__x)") } }
    nested.call().call()
    return g
  }
  construct new() {}
  go2() {
    var t = Fiber.spawn { System.print("task x=%(__x)") }
    Fiber.tick(0)
  }
}
var g = A.go()
g.call()
A.new().go2()
"#;
    // The task needs the scheduler; without it, the fiber alone.
    let (src, expected) = if scheduler_available() {
        (
            src.to_string(),
            "direct 1 2\nin fiber x=1\nin fn x=1 y=2\nafter h x=5\nnested x=5\nin fn x=5 y=2\ntask x=5",
        )
    } else {
        (
            src.replace("A.new().go2()\n", ""),
            "direct 1 2\nin fiber x=1\nin fn x=1 y=2\nafter h x=5\nnested x=5\nin fn x=5 y=2",
        )
    };
    for (mode, threshold) in [
        (ExecutionMode::Interpreter, 0),
        (ExecutionMode::Tiered, 1),
        (ExecutionMode::Tiered, 100),
    ] {
        let mut vm = VM::new(VMConfig {
            execution_mode: mode,
            jit_threshold: threshold,
            opt_threshold: 4,
            ..VMConfig::default()
        });
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", &src);
        let output = vm.take_output();
        assert!(
            matches!(result, InterpretResult::Success),
            "{mode:?}: {output}"
        );
        assert_eq!(output.trim(), expected, "{mode:?} threshold {threshold}");
    }
}

#[test]
fn e2e_thread_create_returns_a_handle_to_join() {
    // The handle joins from the main thread and from a task, reports
    // the abort a task ended with, and times out on one still running.
    if !scheduler_available() {
        return;
    }
    let src = r#"
import "thread" for Thread, Lock
class F {
  static fib(n) {
    if (n < 2) return n
    return fib(n - 1) + fib(n - 2)
  }
}
var out = []
var t = Thread.create {
  out.add(F.fib(20))
  out.add(Thread.current != null)
}
System.print(Thread.current)
System.print(t.join())
System.print(t.isDone)
System.print(out)
var bad = Thread.create { Fiber.abort("boom") }
bad.join()
System.print(bad.error)
var slow = Thread.create { Fiber.sleep(200) }
System.print(slow.join(10))
System.print(slow.isDone)
System.print(slow.join())
System.print(slow.join(1))
var ts = []
for (i in 0...4) ts.add(Thread.create { Fiber.sleep(5 * i) })
var done = Lock.new()
Thread.create {
  for (x in ts) x.join()
  System.print("all joined")
  done.release()
}
done.wait()
"#;
    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 1,
        opt_threshold: 4,
        ..VMConfig::default()
    });
    vm.output_buffer = Some(String::new());
    let result = vm.interpret("main", src);
    let output = vm.take_output();
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(
        output.trim(),
        "null\ntrue\ntrue\n[6765, true]\nboom\nfalse\nfalse\ntrue\ntrue\nall joined"
    );
}

#[test]
fn e2e_compiled_loop_answers_a_collector_on_another_thread() {
    // A task spinning in a compiled loop that allocates nothing
    // reaches the safepoint at its loop header, so collections
    // driven by the main thread's churn complete.
    if !scheduler_available() {
        return;
    }
    let src = r#"
import "thread" for Thread, Mutex, Lock
var gate = Mutex.new()
gate.acquire()
var done = Lock.new()
var spun = 0
Thread.create {
  var i = 0
  while (!gate.tryAcquire()) i = i + 1
  spun = i
  done.release()
}
var junk = []
for (k in 0...300000) {
  junk.add("x%(k)")
  if (junk.count > 1000) junk = []
}
gate.release()
done.wait()
System.print("spun>0 %(spun > 0)")
"#;
    // A hang here is the failure; do not let it hold the suite.
    let watchdog = std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(600));
        eprintln!("compiled loop never answered the collector");
        std::process::abort();
    });
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 1,
        opt_threshold: 4,
        ..VMConfig::default()
    };
    let (result, output, _) = run_with_config(src, config);
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(output.trim(), "spun>0 true");
    drop(watchdog);
}

#[test]
fn e2e_compiled_bodies_yield_from_fiber_stacks() {
    // With fibers on stacks of their own, a body that yields is
    // compiled like any other. Two fibers run the same closure code
    // with different upvalues and yield in turn from inside its
    // loop; each must resume with its own closure's upvalues.
    let src = r#"
class Source {
  static reader(tag, n) {
    var i = 0
    return Fn.new {
      var out = null
      while (out == null) {
        i = i + 1
        if (i % 3 == 0) {
          out = i > n ? "done" : "%(tag)%(i)"
        } else {
          Fiber.yield()
        }
      }
      return out
    }
  }
  static drain(tag, n) {
    var read = Source.reader(tag, n)
    return Fiber.new {
      var s = ""
      while (true) {
        var chunk = read.call()
        if (chunk == "done") break
        s = s + chunk + ","
      }
      return s
    }
  }
}
var a = Source.drain("a", 9)
var b = Source.drain("b", 6)
var ra = null
var rb = null
while (!a.isDone || !b.isDone) {
  if (!a.isDone) ra = a.call()
  if (!b.isDone) rb = b.call()
}
System.print("%(ra) %(rb)")
var total = 0
for (k in 0...200) {
  var f = Source.drain("x", 30)
  var r = null
  while (!f.isDone) r = f.call()
  total = total + r.count
}
System.print(total)
"#;
    let config = VMConfig {
        execution_mode: ExecutionMode::Tiered,
        jit_threshold: 1,
        opt_threshold: 4,
        ..VMConfig::default()
    };
    let (result, output, _) = run_with_config(src, config);
    assert!(matches!(result, InterpretResult::Success), "{output}");
    assert_eq!(output.trim(), "a3,a6,a9, b3,b6,\n7400");
}
