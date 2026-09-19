class RGB
  attr_reader :r, :g, :b
  def initialize(r, g, b)
    @r = r
    @g = g
    @b = b
  end
end

class Complex2
  attr_reader :i, :j
  def initialize(i, j)
    @i = i
    @j = j
  end
end

def palette(fraction)
  r = (fraction * 255).truncate
  g = ((1 - fraction) * 255).truncate
  b = ((0.5 - (fraction - 0.5).abs) * 2 * 255).truncate
  RGB.new(r, g, b)
end

def run
  size = 25
  max_iterations = 200
  max_rad = 65536
  width = 350
  height = 200
  pal = (0..max_iterations).map { |i| palette(i.to_f / max_iterations) }
  scale = 0.25 / size
  checksum = 0
  height.times do |y|
    width.times do |x|
      iteration = 0
      offset = Complex2.new(x * scale - 2.5, y * scale - 1)
      val = Complex2.new(0.0, 0.0)
      while val.i * val.i + val.j * val.j < max_rad && iteration < max_iterations
        val = Complex2.new(val.i * val.i - val.j * val.j + offset.i, 2.0 * val.i * val.j + offset.j)
        iteration += 1
      end
      color = pal[iteration]
      checksum += color.r + color.g + color.b
    end
  end
  checksum
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
checksum = run
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "checksum: #{checksum}"
puts "elapsed: #{elapsed}"
