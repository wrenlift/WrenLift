import time


class RGB:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


class Complex:
    def __init__(self, i, j):
        self.i = i
        self.j = j


def palette(fraction):
    r = int(fraction * 255)
    g = int((1 - fraction) * 255)
    b = int((0.5 - abs(fraction - 0.5)) * 2 * 255)
    return RGB(r, g, b)


def run():
    size = 25
    max_iterations = 200
    max_rad = 65536
    width = 350
    height = 200
    pal = [palette(i / max_iterations) for i in range(max_iterations + 1)]
    scale = 0.25 / size
    checksum = 0
    for y in range(height):
        for x in range(width):
            iteration = 0
            offset = Complex(x * scale - 2.5, y * scale - 1)
            val = Complex(0.0, 0.0)
            while val.i * val.i + val.j * val.j < max_rad and iteration < max_iterations:
                val = Complex(val.i * val.i - val.j * val.j + offset.i, 2.0 * val.i * val.j + offset.j)
                iteration += 1
            color = pal[iteration]
            checksum += color.r + color.g + color.b
    return checksum


start = time.time()
checksum = run()
elapsed = time.time() - start
print("checksum: %d" % checksum)
print("elapsed: %s" % elapsed)
