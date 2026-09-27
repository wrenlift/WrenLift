def sequence(n)
  seed = 42
  Array.new(n) do
    seed = (seed * 3877 + 29573) % 139968
    seed * 4 / 139968
  end
end

def frequencies(dna, k)
  counts = Hash.new(0)
  high = 4**(k - 1)
  key = 0
  k.times { |j| key = key * 4 + dna[j] }
  counts[key] = 1
  (k...dna.size).each do |i|
    key = key % high * 4 + dna[i]
    counts[key] += 1
  end
  counts
end

dna = sequence(250000)
start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
checksum = 0
[1, 2, 3, 4, 6, 12, 18].each do |k|
  frequencies(dna, k).each_value { |count| checksum = (checksum + count * k) % 1000000007 }
end
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "length: #{dna.size}"
puts "checksum: #{checksum}"
puts "elapsed: #{elapsed}"
