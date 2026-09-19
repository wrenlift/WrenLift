// Benchmark: N-body
// Measures field access and floating-point arithmetic on a handful
// of long-lived objects: five bodies under mutual gravity, stepped
// forward and then measured by their total energy. Based on the
// same benchmark in Ash's suite, at a step count that fits the
// reference Wren implementation in the workflow's window.

var PI = 3.141592653589793
var SOLAR_MASS = 4 * PI * PI
var DAYS_PER_YEAR = 365.24

class Body {
  construct new(x, y, z, vx, vy, vz, mass) {
    _x = x
    _y = y
    _z = z
    _vx = vx
    _vy = vy
    _vz = vz
    _mass = mass
  }
  x { _x }
  y { _y }
  z { _z }
  vx { _vx }
  vy { _vy }
  vz { _vz }
  mass { _mass }
  x=(v) { _x = v }
  y=(v) { _y = v }
  z=(v) { _z = v }
  vx=(v) { _vx = v }
  vy=(v) { _vy = v }
  vz=(v) { _vz = v }

  offsetMomentum(px, py, pz) {
    _vx = -px / SOLAR_MASS
    _vy = -py / SOLAR_MASS
    _vz = -pz / SOLAR_MASS
  }

  static jupiter() {
    return Body.new(4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01,
      1.66007664274403694e-03 * DAYS_PER_YEAR, 7.69901118419740425e-03 * DAYS_PER_YEAR,
      -6.90460016972063023e-05 * DAYS_PER_YEAR, 9.54791938424326609e-04 * SOLAR_MASS)
  }
  static saturn() {
    return Body.new(8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
      -2.76742510726862411e-03 * DAYS_PER_YEAR, 4.99852801234917238e-03 * DAYS_PER_YEAR,
      2.30417297573763929e-05 * DAYS_PER_YEAR, 2.85885980666130812e-04 * SOLAR_MASS)
  }
  static uranus() {
    return Body.new(1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
      2.96460137564761618e-03 * DAYS_PER_YEAR, 2.37847173959480950e-03 * DAYS_PER_YEAR,
      -2.96589568540237556e-05 * DAYS_PER_YEAR, 4.36624404335156298e-05 * SOLAR_MASS)
  }
  static neptune() {
    return Body.new(1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
      2.68067772490389322e-03 * DAYS_PER_YEAR, 1.62824170038242295e-03 * DAYS_PER_YEAR,
      -9.51592254519715870e-05 * DAYS_PER_YEAR, 5.15138902046611451e-05 * SOLAR_MASS)
  }
  static sun() { Body.new(0, 0, 0, 0, 0, 0, SOLAR_MASS) }
}

class NBody {
  construct new() {
    _bodies = [Body.sun(), Body.jupiter(), Body.saturn(), Body.uranus(), Body.neptune()]
    var px = 0.0
    var py = 0.0
    var pz = 0.0
    for (b in _bodies) {
      px = px + b.vx * b.mass
      py = py + b.vy * b.mass
      pz = pz + b.vz * b.mass
    }
    _bodies[0].offsetMomentum(px, py, pz)
  }

  advance(dt) {
    var size = _bodies.count
    for (i in 0...size) {
      var a = _bodies[i]
      for (j in (i + 1)...size) {
        var b = _bodies[j]
        var dx = a.x - b.x
        var dy = a.y - b.y
        var dz = a.z - b.z
        var distance = (dx * dx + dy * dy + dz * dz).sqrt
        var mag = dt / (distance * distance * distance)
        a.vx = a.vx - dx * b.mass * mag
        a.vy = a.vy - dy * b.mass * mag
        a.vz = a.vz - dz * b.mass * mag
        b.vx = b.vx + dx * a.mass * mag
        b.vy = b.vy + dy * a.mass * mag
        b.vz = b.vz + dz * a.mass * mag
      }
    }
    for (body in _bodies) {
      body.x = body.x + dt * body.vx
      body.y = body.y + dt * body.vy
      body.z = body.z + dt * body.vz
    }
  }

  energy() {
    var e = 0.0
    var n = _bodies.count
    for (i in 0...n) {
      var a = _bodies[i]
      e = e + 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz)
      for (j in (i + 1)...n) {
        var b = _bodies[j]
        var dx = b.x - a.x
        var dy = b.y - a.y
        var dz = b.z - a.z
        e = e - (a.mass * b.mass) / (dx * dx + dy * dy + dz * dz).sqrt
      }
    }
    return e
  }
}

var start = System.clock
var sim = NBody.new()
for (i in 0...500000) sim.advance(0.01)
var energy = (sim.energy() * 1000000).truncate
var elapsed = System.clock - start

// Print the energy so the bench-correctness step can diff our stdout
// against standard Wren.
System.print("energy: %(energy)")
System.print("elapsed: %(elapsed)")
