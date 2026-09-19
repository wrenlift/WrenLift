import math
import time

PI = 3.141592653589793
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24


class Body:
    def __init__(self, x, y, z, vx, vy, vz, mass):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

    def offset_momentum(self, px, py, pz):
        self.vx = -px / SOLAR_MASS
        self.vy = -py / SOLAR_MASS
        self.vz = -pz / SOLAR_MASS


def jupiter():
    return Body(4.84143144246472090e+00, -1.16032004402742839e+00, -1.03622044471123109e-01,
                1.66007664274403694e-03 * DAYS_PER_YEAR, 7.69901118419740425e-03 * DAYS_PER_YEAR,
                -6.90460016972063023e-05 * DAYS_PER_YEAR, 9.54791938424326609e-04 * SOLAR_MASS)


def saturn():
    return Body(8.34336671824457987e+00, 4.12479856412430479e+00, -4.03523417114321381e-01,
                -2.76742510726862411e-03 * DAYS_PER_YEAR, 4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR, 2.85885980666130812e-04 * SOLAR_MASS)


def uranus():
    return Body(1.28943695621391310e+01, -1.51111514016986312e+01, -2.23307578892655734e-01,
                2.96460137564761618e-03 * DAYS_PER_YEAR, 2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR, 4.36624404335156298e-05 * SOLAR_MASS)


def neptune():
    return Body(1.53796971148509165e+01, -2.59193146099879641e+01, 1.79258772950371181e-01,
                2.68067772490389322e-03 * DAYS_PER_YEAR, 1.62824170038242295e-03 * DAYS_PER_YEAR,
                -9.51592254519715870e-05 * DAYS_PER_YEAR, 5.15138902046611451e-05 * SOLAR_MASS)


def sun():
    return Body(0, 0, 0, 0, 0, 0, SOLAR_MASS)


class NBody:
    def __init__(self):
        self.bodies = [sun(), jupiter(), saturn(), uranus(), neptune()]
        px = 0.0
        py = 0.0
        pz = 0.0
        for b in self.bodies:
            px += b.vx * b.mass
            py += b.vy * b.mass
            pz += b.vz * b.mass
        self.bodies[0].offset_momentum(px, py, pz)

    def advance(self, dt):
        bodies = self.bodies
        size = len(bodies)
        for i in range(size):
            a = bodies[i]
            for j in range(i + 1, size):
                b = bodies[j]
                dx = a.x - b.x
                dy = a.y - b.y
                dz = a.z - b.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                mag = dt / (distance * distance * distance)
                a.vx = a.vx - dx * b.mass * mag
                a.vy = a.vy - dy * b.mass * mag
                a.vz = a.vz - dz * b.mass * mag
                b.vx = b.vx + dx * a.mass * mag
                b.vy = b.vy + dy * a.mass * mag
                b.vz = b.vz + dz * a.mass * mag
        for body in bodies:
            body.x = body.x + dt * body.vx
            body.y = body.y + dt * body.vy
            body.z = body.z + dt * body.vz

    def energy(self):
        e = 0.0
        bodies = self.bodies
        n = len(bodies)
        for i in range(n):
            a = bodies[i]
            e += 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz)
            for j in range(i + 1, n):
                b = bodies[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dz = b.z - a.z
                e -= (a.mass * b.mass) / math.sqrt(dx * dx + dy * dy + dz * dz)
        return e


start = time.time()
sim = NBody()
for _ in range(500000):
    sim.advance(0.01)
energy = int(sim.energy() * 1000000)
elapsed = time.time() - start
print("energy: %d" % energy)
print("elapsed: %s" % elapsed)
