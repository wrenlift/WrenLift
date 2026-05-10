//! Proof-of-concept tests: krio-fiber handles the multi-yield-in-loop
//! pattern that breaks the AOT stackless state-machine transform on
//! `proc::closure_50`, `events::method_0_13`, and io's `Reader.withTryFn`.
//!
//! These tests don't go through Wren — they're pure Rust harnesses
//! that mirror the *shape* of the broken Wren patterns, run on
//! krio-fiber stackful coroutines, and verify the program produces
//! the expected output. If these pass, the stackful runtime model is
//! sufficient to fix the AOT yield bugs once we route AOT-compiled
//! fiber bodies through krio-fiber instead of the SM transform.
//!
//! Each test corresponds to a real failing case:
//!   - `proc_closure_50_shape`        ← `proc::closure_50`
//!   - `io_reader_with_try_fn_shape`  ← `io::Reader.withTryFn`
//!   - `events_multi_yield_loop`      ← `events::method_0_13` (simplified)

#![cfg(feature = "host")]

use krio_fiber::{Fiber, FiberStep, yield_value};

/// Mirror of `hatch-proc::proc.spec::closure_50`. Tight loop, two
/// suspending `_fn.call()` sites per iteration, value-carrying yield.
/// Original Wren shape (paraphrased):
///
/// ```wren
/// while (...) {
///     if (cond1) v44 = a.call() // YIELD 1
///     if (cond2) v50 = b.call() // YIELD 2
///     // use v44, v50, loop-carried v22..v25
/// }
/// ```
///
/// The SM transform produces save/load tables across the two yields
/// and tail-duplicates the loop body; the cloning logic gets the
/// load scopes wrong for blocks reachable from both resumes (see
/// `project_aot_multipath_load_scope.md`). Stackful fibers don't
/// have that class of bug — the loop body's locals just live on the
/// fiber stack across the suspension.
#[test]
fn proc_closure_50_shape() {
    let mut a_count = 0;
    let mut b_count = 0;
    let mut iter_log: Vec<u32> = Vec::new();

    // The fiber pumps values out via `yield_value` — host pumps them
    // back in via `resume_with`. Each loop iteration yields twice.
    let mut fiber = Fiber::new(|| {
        // Loop-carried values that the SM transform would have
        // tracked through `yield_saves`. Here they're just stack
        // locals — yield preserves them automatically.
        let mut acc_a = 0u32;
        let mut acc_b = 0u32;
        for iter in 0..3u32 {
            // YIELD 1 (yields a tag, host responds with the value
            // for `_fn_a.call()` to bind).
            let a_val: u32 = yield_value(("call_a", iter)).unwrap_or(0);
            acc_a += a_val;
            // YIELD 2 (the dual). The point is that `acc_a` —
            // defined before YIELD 2 — is read AFTER YIELD 2.
            let b_val: u32 = yield_value(("call_b", iter)).unwrap_or(0);
            acc_b += b_val;
            // Live-across-yield: iter, acc_a, acc_b. Loop-back
            // re-uses them next iteration.
        }
        // Final yield pushes both accumulators out so the host can
        // assert their values.
        let _ack: i32 = yield_value(("done", acc_a, acc_b)).unwrap_or(0);
    });

    // Drive the fiber. The first `resume()` runs to the first yield;
    // every subsequent `resume_with(v)` feeds a value in and runs to
    // the next yield (or to completion). We never double-call resume.
    let mut acc_a_final = 0u32;
    let mut acc_b_final = 0u32;
    let mut step = fiber.resume();
    while matches!(step, FiberStep::Yielded) {
        if let Some((tag, iter)) = fiber.take_yield_value::<(&str, u32)>() {
            iter_log.push(iter);
            step = match tag {
                "call_a" => {
                    a_count += 1;
                    fiber.resume_with::<u32>(a_count)
                }
                "call_b" => {
                    b_count += 1;
                    fiber.resume_with::<u32>(b_count)
                }
                _ => panic!("unexpected 2-tuple tag {tag:?}"),
            };
        } else if let Some((tag, a, b)) = fiber.take_yield_value::<(&str, u32, u32)>() {
            assert_eq!(tag, "done");
            acc_a_final = a;
            acc_b_final = b;
            step = fiber.resume_with::<i32>(0);
        } else {
            panic!("fiber yielded an unexpected value type");
        }
    }

    // Loop ran 3 times → 3 a-calls + 3 b-calls.
    assert_eq!(a_count, 3);
    assert_eq!(b_count, 3);
    // Accumulators are 1+2+3 = 6 each (sequential resume values).
    assert_eq!(acc_a_final, 6);
    assert_eq!(acc_b_final, 6);
    // `iter` was visible at every yield, in the right order.
    assert_eq!(iter_log, vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(fiber.state(), krio_fiber::FiberState::Done);
}

/// Mirror of `hatch-io::Reader.withTryFn > readLine yields until
/// newline arrives` and `> skips WouldBlock results by yielding`.
/// A retry loop where each iteration may suspend; the loop's
/// "I have my answer" exit is data-dependent.
#[test]
fn io_reader_with_try_fn_shape() {
    // Simulate a byte stream where read() sometimes returns
    // WouldBlock — the fiber yields "WouldBlock" until the host
    // resumes with a real byte. Repeat until newline.
    let mut fiber = Fiber::new(|| {
        let mut accumulated: Vec<u8> = Vec::new();
        loop {
            // Each "read attempt" is a yield. Host either pumps
            // back a byte (Some(b)) or `None` to mean "still
            // WouldBlock, try again".
            let maybe_byte: Option<u8> = yield_value("read").unwrap_or(None);
            match maybe_byte {
                Some(b) => {
                    accumulated.push(b);
                    if b == b'\n' {
                        break;
                    }
                }
                None => {
                    // WouldBlock — loop and yield again.
                    continue;
                }
            }
        }
        // Final yield ships the line out.
        let _ack: i32 = yield_value(("line", accumulated)).unwrap_or(0);
    });

    // Drive: simulate a stream that yields WouldBlock 3×, then
    // "h", "i", "\n" in sequence. The fiber must accumulate
    // through the WouldBlock retries — the bug pattern in io's
    // `Reader.withTryFn` is that AOT lost the accumulator
    // across the retry suspension.
    let pattern: Vec<Option<u8>> = vec![None, None, None, Some(b'h'), Some(b'i'), Some(b'\n')];
    let mut pattern_iter = pattern.into_iter();
    let mut got_line: Option<Vec<u8>> = None;
    let mut step = fiber.resume();
    while matches!(step, FiberStep::Yielded) {
        if let Some(_tag) = fiber.take_yield_value::<&str>() {
            let next = pattern_iter.next().expect("pattern exhausted");
            step = fiber.resume_with::<Option<u8>>(next);
        } else if let Some((tag, line)) = fiber.take_yield_value::<(&str, Vec<u8>)>() {
            assert_eq!(tag, "line");
            got_line = Some(line);
            step = fiber.resume_with::<i32>(0);
        } else {
            panic!("fiber yielded an unexpected value type");
        }
    }
    assert_eq!(got_line, Some(b"hi\n".to_vec()));
}

/// Mirror of `events::method_0_13` — 30+ states, 100+ yielding blocks
/// in the broken AOT shape. We don't replicate the full breadth, but
/// we exercise the property that breaks: many yields with deep
/// nesting and live values flowing across all of them.
#[test]
fn events_multi_yield_loop() {
    // Emit N events through a yielding pump; each event carries a
    // sequence number AND a running checksum that depends on all
    // prior events (so a save/load drop on the checksum would
    // surface as a bad final value).
    const N: u32 = 12;
    let mut fiber = Fiber::new(|| {
        let mut checksum: u64 = 0xCAFEBABE_DEADBEEF;
        for i in 0..N {
            // Multiple yields per iteration, each carrying live
            // values across the suspension boundary.
            let _ack1: i32 = yield_value(("pre", i, checksum)).unwrap_or(0);
            checksum = checksum.wrapping_mul(0x100000001B3).wrapping_add(i as u64);
            let _ack2: i32 = yield_value(("mid", i, checksum)).unwrap_or(0);
            checksum ^= checksum.rotate_left(13);
            let _ack3: i32 = yield_value(("post", i, checksum)).unwrap_or(0);
        }
        let _ack: i32 = yield_value(("done", N, checksum)).unwrap_or(0);
    });

    // Replicate the same checksum on the host side; if at any
    // point the values disagree, a save/load was dropped.
    let mut expected_checksum: u64 = 0xCAFEBABE_DEADBEEF;
    let mut final_seen: Option<u64> = None;
    let mut step = fiber.resume();
    while matches!(step, FiberStep::Yielded) {
        let (tag, i, cs) = fiber
            .take_yield_value::<(&str, u32, u64)>()
            .expect("fiber must yield (&str, u32, u64) tuples");
        match tag {
            "pre" => {
                assert_eq!(cs, expected_checksum, "pre i={i}");
                expected_checksum = expected_checksum
                    .wrapping_mul(0x100000001B3)
                    .wrapping_add(i as u64);
            }
            "mid" => {
                assert_eq!(cs, expected_checksum, "mid i={i}");
                expected_checksum ^= expected_checksum.rotate_left(13);
            }
            "post" => {
                assert_eq!(cs, expected_checksum, "post i={i}");
            }
            "done" => {
                final_seen = Some(cs);
            }
            _ => panic!("unexpected tag {tag:?}"),
        }
        step = fiber.resume_with::<i32>(0);
    }
    assert_eq!(final_seen, Some(expected_checksum));
    assert_eq!(fiber.state(), krio_fiber::FiberState::Done);
}

/// Sanity: `Fiber::take_yield_value` on a `Done` fiber returns None
/// rather than panicking — useful when integrating with WrenLift's
/// host loop that polls fibers without knowing their state.
#[test]
fn done_fiber_yield_value_is_none() {
    let mut fiber = Fiber::new(|| {
        let _: Option<()> = yield_value(42i64);
    });
    fiber.resume();
    let v: Option<i64> = fiber.take_yield_value();
    assert_eq!(v, Some(42));
    fiber.resume();
    assert_eq!(fiber.state(), krio_fiber::FiberState::Done);
    let after: Option<i64> = fiber.take_yield_value();
    assert_eq!(after, None);
}
