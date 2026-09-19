PI = 3.141592653589793
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

class Body
  attr_accessor :x, :y, :z, :vx, :vy, :vz, :mass

  def initialize(x, y, z, vx, vy, vz, mass)
    @x = x
    @y = y
    @z = z
    @vx = vx
    @vy = vy
    @vz = vz
    @mass = mass
  end

  def offset_momentum(px, py, pz)
    @vx = -px / SOLAR_MASS
    @vy = -py / SOLAR_MASS
    @vz = -pz / SOLAR_MASS
  end

  def self.jupiter
    Body.new(4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01,
             1.66007664274403694e-03 * DAYS_PER_YEAR, 7.69901118419740425e-03 * DAYS_PER_YEAR,
             -6.90460016972063023e-05 * DAYS_PER_YEAR, 9.54791938424326609e-04 * SOLAR_MASS)
  end

  def self.saturn
    Body.new(8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
             -2.76742510726862411e-03 * DAYS_PER_YEAR, 4.99852801234917238e-03 * DAYS_PER_YEAR,
             2.30417297573763929e-05 * DAYS_PER_YEAR, 2.85885980666130812e-04 * SOLAR_MASS)
  end

  def self.uranus
    Body.new(1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
             2.96460137564761618e-03 * DAYS_PER_YEAR, 2.37847173959480950e-03 * DAYS_PER_YEAR,
             -2.96589568540237556e-05 * DAYS_PER_YEAR, 4.36624404335156298e-05 * SOLAR_MASS)
  end

  def self.neptune
    Body.new(1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
             2.68067772490389322e-03 * DAYS_PER_YEAR, 1.62824170038242295e-03 * DAYS_PER_YEAR,
             -9.51592254519715870e-05 * DAYS_PER_YEAR, 5.15138902046611451e-05 * SOLAR_MASS)
  end

  def self.sun
    Body.new(0, 0, 0, 0, 0, 0, SOLAR_MASS)
  end
end

class NBody
  def initialize
    @bodies = [Body.sun, Body.jupiter, Body.saturn, Body.uranus, Body.neptune]
    px = 0.0
    py = 0.0
    pz = 0.0
    @bodies.each do |b|
      px += b.vx * b.mass
      py += b.vy * b.mass
      pz += b.vz * b.mass
    end
    @bodies[0].offset_momentum(px, py, pz)
  end

  def advance(dt)
    size = @bodies.length
    i = 0
    while i < size
      a = @bodies[i]
      j = i + 1
      while j < size
        b = @bodies[j]
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        distance = Math.sqrt(dx * dx + dy * dy + dz * dz)
        mag = dt / (distance * distance * distance)
        a.vx = a.vx - dx * b.mass * mag
        a.vy = a.vy - dy * b.mass * mag
        a.vz = a.vz - dz * b.mass * mag
        b.vx = b.vx + dx * a.mass * mag
        b.vy = b.vy + dy * a.mass * mag
        b.vz = b.vz + dz * a.mass * mag
        j += 1
      end
      i += 1
    end
    @bodies.each do |body|
      body.x = body.x + dt * body.vx
      body.y = body.y + dt * body.vy
      body.z = body.z + dt * body.vz
    end
  end

  def energy
    e = 0.0
    n = @bodies.length
    i = 0
    while i < n
      a = @bodies[i]
      e += 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz)
      j = i + 1
      while j < n
        b = @bodies[j]
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        e -= (a.mass * b.mass) / Math.sqrt(dx * dx + dy * dy + dz * dz)
        j += 1
      end
      i += 1
    end
    e
  end
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
sim = NBody.new
500000.times { sim.advance(0.01) }
energy = (sim.energy * 1000000).truncate
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "energy: #{energy}"
puts "elapsed: #{elapsed}"
