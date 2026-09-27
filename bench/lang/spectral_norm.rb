def denominator(i, j)
  ij = i + j
  ij * (ij + 1) / 2 + i + 1
end

def multiply_av(v, out)
  n = v.size
  n.times do |i|
    sum = 0.0
    n.times { |j| sum += v[j] / denominator(i, j) }
    out[i] = sum
  end
end

def multiply_atv(v, out)
  n = v.size
  n.times do |i|
    sum = 0.0
    n.times { |j| sum += v[j] / denominator(j, i) }
    out[i] = sum
  end
end

def spectral_norm(n)
  u = Array.new(n, 1.0)
  v = Array.new(n, 0.0)
  tmp = Array.new(n, 0.0)
  10.times do
    multiply_av(u, tmp)
    multiply_atv(tmp, v)
    multiply_av(v, tmp)
    multiply_atv(tmp, u)
  end
  vbv = 0.0
  vv = 0.0
  n.times do |i|
    vbv += u[i] * v[i]
    vv += v[i] * v[i]
  end
  Math.sqrt(vbv / vv)
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
result = spectral_norm(500)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts format("result: %.13f", result)
puts "elapsed: #{elapsed}"
