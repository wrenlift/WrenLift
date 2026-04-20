//! Optional `time` module — current time, monotonic clock, sleep,
//! and UTC decomposition. `@hatch:time` layers ergonomic Time
//! values and formatters on top.
//!
//! No timezone handling here; localtime belongs in a later pass
//! once we've decided whether to lean on OS tzdata or ship our
//! own. UTC covers logs and unix-epoch math, which is 95% of what
//! libraries actually want.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::runtime::object::NativeContext;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

// --- Clocks ----------------------------------------------------

/// Time.unix → f64 seconds since the Unix epoch.
fn time_unix(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Value::num(d.as_secs_f64())
}

/// Time.mono → f64 seconds from an unspecified monotonic origin.
/// Only meaningful as a difference between two calls — never as
/// an absolute time. Survives clock adjustments (NTP, DST) that
/// would perturb `unix`.
fn time_mono(_ctx: &mut dyn NativeContext, _args: &[Value]) -> Value {
    // Origin: first call. Stashed behind a OnceLock so repeat
    // calls return monotonically increasing seconds without
    // pulling in chrono/time_perf crates.
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let origin = *ORIGIN.get_or_init(Instant::now);
    Value::num(origin.elapsed().as_secs_f64())
}

/// Time.sleep(seconds) — blocks the calling thread. Nothing
/// fancier than std::thread::sleep; cooperative yielding via
/// fibers is the caller's job.
fn time_sleep(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let secs = match args[1].as_num() {
        Some(n) => n,
        None => {
            ctx.runtime_error("Time.sleep: seconds must be a number.".to_string());
            return Value::null();
        }
    };
    if secs < 0.0 || !secs.is_finite() {
        ctx.runtime_error(
            "Time.sleep: seconds must be a non-negative finite number.".to_string(),
        );
        return Value::null();
    }
    std::thread::sleep(Duration::from_secs_f64(secs));
    Value::null()
}

// --- UTC decomposition ----------------------------------------

/// Time.utc(seconds) → Map with year / month / day / hour /
/// minute / second / millisecond / weekday, all ints, computed
/// in UTC.
///
/// Uses Howard Hinnant's `civil_from_days` algorithm — 20 lines
/// of integer arithmetic, correct across leap years from at
/// least -100000 CE through +100000 CE.
fn time_utc(ctx: &mut dyn NativeContext, args: &[Value]) -> Value {
    let secs_f = match args[1].as_num() {
        Some(n) => n,
        None => {
            ctx.runtime_error("Time.utc: seconds must be a number.".to_string());
            return Value::null();
        }
    };
    if !secs_f.is_finite() {
        ctx.runtime_error("Time.utc: seconds must be finite.".to_string());
        return Value::null();
    }
    let ms_total = (secs_f * 1000.0) as i64;
    let (days, millis_in_day) = floor_div_mod(ms_total, 86_400_000);
    let (y, mo, d) = civil_from_days(days);
    let hour = millis_in_day / 3_600_000;
    let minute = (millis_in_day / 60_000) % 60;
    let second = (millis_in_day / 1000) % 60;
    let milli = millis_in_day % 1000;
    let weekday = weekday_from_days(days);

    let map = ctx.alloc_map();
    let map_ptr = map.as_object().unwrap() as *mut crate::runtime::object::ObjMap;
    unsafe {
        put_num(ctx, map_ptr, "year", y as f64);
        put_num(ctx, map_ptr, "month", mo as f64);
        put_num(ctx, map_ptr, "day", d as f64);
        put_num(ctx, map_ptr, "hour", hour as f64);
        put_num(ctx, map_ptr, "minute", minute as f64);
        put_num(ctx, map_ptr, "second", second as f64);
        put_num(ctx, map_ptr, "millisecond", milli as f64);
        put_num(ctx, map_ptr, "weekday", weekday as f64);
    }
    map
}

unsafe fn put_num(
    ctx: &mut dyn NativeContext,
    map: *mut crate::runtime::object::ObjMap,
    key: &str,
    value: f64,
) {
    let k = ctx.alloc_string(key.to_string());
    unsafe { (*map).set(k, Value::num(value)) };
}

/// Floor-divide `a` by `b`, returning (quotient, remainder) where
/// the remainder is always in `[0, b)`. Differs from Rust's `/`
/// and `%` on negative inputs — we need the Euclidean flavour so
/// pre-1970 timestamps decompose to the expected day/time.
fn floor_div_mod(a: i64, b: i64) -> (i64, i64) {
    debug_assert!(b > 0);
    let q = a.div_euclid(b);
    let r = a.rem_euclid(b);
    (q, r)
}

/// Given days since the Unix epoch (1970-01-01), return
/// (year, month 1..=12, day 1..=31) per the proleptic Gregorian
/// calendar. Adapted from Hinnant's public-domain algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // z is days since 0000-03-01 (Howard's epoch)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Weekday for days-since-epoch, where 0 = Monday … 6 = Sunday.
/// Choosing ISO numbering so callers can format against a fixed
/// list without branching.
fn weekday_from_days(days: i64) -> u32 {
    // 1970-01-01 was a Thursday (ISO weekday 4). Days from epoch
    // advance weekday by 1.
    let wd = (days + 3).rem_euclid(7); // shift so Monday = 0
    wd as u32
}

// --- Registration ---------------------------------------------

pub fn register(vm: &mut VM) -> *mut crate::runtime::object::ObjClass {
    // Registered under `TimeCore` so consumers can define
    // full-featured `Clock` / `Time` classes (value types,
    // formatting, etc.) without shadowing the runtime
    // primitives. @hatch:time is the canonical consumer.
    let class = vm.make_class("TimeCore", vm.object_class);

    vm.primitive_static(class, "unix", time_unix);
    vm.primitive_static(class, "mono", time_mono);
    vm.primitive_static(class, "sleep(_)", time_sleep);
    vm.primitive_static(class, "utc(_)", time_utc);

    class
}
