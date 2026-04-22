// Benchmark: nbody — solar system integrator.
//
// Based on the Benchmarks Game `nbody` problem. Simulates five
// bodies (Sun + Jupiter/Saturn/Uranus/Neptune) under mutual
// gravity and reports the total energy before and after the
// integration.
//
// Two variants live in this file:
//
//   * `runList`   — positions/velocities as `List<Num>`s. Works
//                   on stock Wren and WrenLift alike; the subscript
//                   work goes through a generic runtime call.
//   * `runTyped`  — same algorithm on `Float64Array`. Available
//                   only on WrenLift. Tiered compilation inlines
//                   the subscript load/store directly against the
//                   backing buffer — no boxing, no call.
//
// The typed variant is a proxy for how games / physics / audio
// code would look in practice; the list variant is the
// compatibility baseline.

var PI = 3.141592653589793
var SOLAR_MASS = 4 * PI * PI
var DAYS_PER_YEAR = 365.24

// ---------------------------------------------------------------
// List-of-lists version (portable, stock-Wren-compatible)
// ---------------------------------------------------------------

class NBodyList {
  // Each body: [x, y, z, vx, vy, vz, mass]
  static initialBodies() {
    var bodies = [
      // Sun
      [0, 0, 0, 0, 0, 0, SOLAR_MASS],
      // Jupiter
      [
        4.84143144246472090,
        -1.16032004402742839,
        -0.103622044471123109,
        0.00166007664274403694  * DAYS_PER_YEAR,
        0.00769901118419740425  * DAYS_PER_YEAR,
        -0.0000690460016972063023 * DAYS_PER_YEAR,
        0.000954791938424326609 * SOLAR_MASS
      ],
      // Saturn
      [
        8.34336671824457987,
        4.12479856412430479,
        -0.403523417114321381,
        -0.00276742510726862411 * DAYS_PER_YEAR,
        0.00499852801234917238  * DAYS_PER_YEAR,
        0.0000230417297573763929 * DAYS_PER_YEAR,
        0.000285885980666130812 * SOLAR_MASS
      ],
      // Uranus
      [
        12.8943695621391310,
        -15.1111514016986312,
        -0.223307578892655734,
        0.00296460137564761618 * DAYS_PER_YEAR,
        0.00237847173959480950 * DAYS_PER_YEAR,
        -0.0000296589568540237556 * DAYS_PER_YEAR,
        0.0000436624404335156298 * SOLAR_MASS
      ],
      // Neptune
      [
        15.3796971148509165,
        -25.9193146099879641,
        0.179258772950371181,
        0.00268067772490389322  * DAYS_PER_YEAR,
        0.00162824170038242295  * DAYS_PER_YEAR,
        -0.0000951592254519715870 * DAYS_PER_YEAR,
        0.0000515138902046611451 * SOLAR_MASS
      ]
    ]
    // Offset Sun's momentum so the system centroid is stationary.
    var px = 0
    var py = 0
    var pz = 0
    var i = 1
    while (i < bodies.count) {
      var b = bodies[i]
      px = px + b[3] * b[6]
      py = py + b[4] * b[6]
      pz = pz + b[5] * b[6]
      i = i + 1
    }
    var sun = bodies[0]
    sun[3] = -px / SOLAR_MASS
    sun[4] = -py / SOLAR_MASS
    sun[5] = -pz / SOLAR_MASS
    return bodies
  }

  static advance(bodies, dt) {
    var n = bodies.count
    var i = 0
    while (i < n) {
      var bi = bodies[i]
      var j = i + 1
      while (j < n) {
        var bj = bodies[j]
        var dx = bi[0] - bj[0]
        var dy = bi[1] - bj[1]
        var dz = bi[2] - bj[2]
        var d2 = dx * dx + dy * dy + dz * dz
        var dist = d2.sqrt
        var mag = dt / (d2 * dist)
        var bim = bi[6] * mag
        var bjm = bj[6] * mag
        bi[3] = bi[3] - dx * bjm
        bi[4] = bi[4] - dy * bjm
        bi[5] = bi[5] - dz * bjm
        bj[3] = bj[3] + dx * bim
        bj[4] = bj[4] + dy * bim
        bj[5] = bj[5] + dz * bim
        j = j + 1
      }
      i = i + 1
    }
    i = 0
    while (i < n) {
      var b = bodies[i]
      b[0] = b[0] + dt * b[3]
      b[1] = b[1] + dt * b[4]
      b[2] = b[2] + dt * b[5]
      i = i + 1
    }
  }

  static energy(bodies) {
    var e = 0
    var n = bodies.count
    var i = 0
    while (i < n) {
      var bi = bodies[i]
      e = e + 0.5 * bi[6] * (bi[3] * bi[3] + bi[4] * bi[4] + bi[5] * bi[5])
      var j = i + 1
      while (j < n) {
        var bj = bodies[j]
        var dx = bi[0] - bj[0]
        var dy = bi[1] - bj[1]
        var dz = bi[2] - bj[2]
        var d = (dx * dx + dy * dy + dz * dz).sqrt
        e = e - (bi[6] * bj[6]) / d
        j = j + 1
      }
      i = i + 1
    }
    return e
  }
}

// ---------------------------------------------------------------
// Float64Array version (WrenLift only; gets the JIT fast path)
// ---------------------------------------------------------------
//
// Layout: one Float64Array of N*7 elements, laid out per-body as
//
//   [x, y, z, vx, vy, vz, mass]
//
// Contiguous storage avoids pointer chasing between inner lists,
// and every subscript read/write hits the inline Cranelift path.

class NBodyTyped {
  static FIELD_COUNT { 7 }

  static initialBodies() {
    var bodies = NBodyList.initialBodies()
    var n = bodies.count
    var buf = Float64Array.new(n * 7)
    var i = 0
    while (i < n) {
      var src = bodies[i]
      var base = i * 7
      buf[base + 0] = src[0]
      buf[base + 1] = src[1]
      buf[base + 2] = src[2]
      buf[base + 3] = src[3]
      buf[base + 4] = src[4]
      buf[base + 5] = src[5]
      buf[base + 6] = src[6]
      i = i + 1
    }
    return buf
  }

  static advance(buf, n, dt) {
    var i = 0
    while (i < n) {
      var bi = i * 7
      var j = i + 1
      while (j < n) {
        var bj = j * 7
        var dx = buf[bi] - buf[bj]
        var dy = buf[bi + 1] - buf[bj + 1]
        var dz = buf[bi + 2] - buf[bj + 2]
        var d2 = dx * dx + dy * dy + dz * dz
        var dist = d2.sqrt
        var mag = dt / (d2 * dist)
        var bim = buf[bi + 6] * mag
        var bjm = buf[bj + 6] * mag
        buf[bi + 3] = buf[bi + 3] - dx * bjm
        buf[bi + 4] = buf[bi + 4] - dy * bjm
        buf[bi + 5] = buf[bi + 5] - dz * bjm
        buf[bj + 3] = buf[bj + 3] + dx * bim
        buf[bj + 4] = buf[bj + 4] + dy * bim
        buf[bj + 5] = buf[bj + 5] + dz * bim
        j = j + 1
      }
      i = i + 1
    }
    i = 0
    while (i < n) {
      var bi = i * 7
      buf[bi]     = buf[bi]     + dt * buf[bi + 3]
      buf[bi + 1] = buf[bi + 1] + dt * buf[bi + 4]
      buf[bi + 2] = buf[bi + 2] + dt * buf[bi + 5]
      i = i + 1
    }
  }

  static energy(buf, n) {
    var e = 0
    var i = 0
    while (i < n) {
      var bi = i * 7
      var vx = buf[bi + 3]
      var vy = buf[bi + 4]
      var vz = buf[bi + 5]
      var m  = buf[bi + 6]
      e = e + 0.5 * m * (vx * vx + vy * vy + vz * vz)
      var j = i + 1
      while (j < n) {
        var bj = j * 7
        var dx = buf[bi]     - buf[bj]
        var dy = buf[bi + 1] - buf[bj + 1]
        var dz = buf[bi + 2] - buf[bj + 2]
        var d  = (dx * dx + dy * dy + dz * dz).sqrt
        e = e - (m * buf[bj + 6]) / d
        j = j + 1
      }
      i = i + 1
    }
    return e
  }
}

// ---------------------------------------------------------------
// Driver
// ---------------------------------------------------------------

var STEPS = 50000
var DT = 0.01

// Warm up both variants so tiered compilation has time to fire.
var warmL = NBodyList.initialBodies()
NBodyList.advance(warmL, DT)
NBodyList.advance(warmL, DT)
var warmT = NBodyTyped.initialBodies()
NBodyTyped.advance(warmT, 5, DT)
NBodyTyped.advance(warmT, 5, DT)

// ---- List<List<Num>> ----
var bodies_l = NBodyList.initialBodies()
var e0_l = NBodyList.energy(bodies_l)
var t0 = System.clock
var s = 0
while (s < STEPS) {
  NBodyList.advance(bodies_l, DT)
  s = s + 1
}
var t1 = System.clock
var e1_l = NBodyList.energy(bodies_l)
var elapsed_l = t1 - t0
System.print("list<num>   e0=%(e0_l) e1=%(e1_l) elapsed: %(elapsed_l)")

// ---- Float64Array ----
var bodies_t = NBodyTyped.initialBodies()
var e0_t = NBodyTyped.energy(bodies_t, 5)
t0 = System.clock
s = 0
while (s < STEPS) {
  NBodyTyped.advance(bodies_t, 5, DT)
  s = s + 1
}
t1 = System.clock
var e1_t = NBodyTyped.energy(bodies_t, 5)
var elapsed_t = t1 - t0
System.print("float64arr  e0=%(e0_t) e1=%(e1_t) elapsed: %(elapsed_t)")

// Match the run.sh harness format — last line gets grep'd.
var speedup = elapsed_l / elapsed_t
System.print("speedup: %(speedup)x")
System.print("elapsed: %(elapsed_t)")
