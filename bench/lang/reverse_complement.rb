def sequence(n)
  seed = 42
  Array.new(n) do
    seed = (seed * 3877 + 29573) % 139968
    seed * 4 / 139968
  end
end

def reverse_complement(dna)
  lo = 0
  hi = dna.size - 1
  while lo <= hi
    left = 3 - dna[hi]
    dna[hi] = 3 - dna[lo]
    dna[lo] = left
    lo += 1
    hi -= 1
  end
end

def checksum_of(dna)
  dna.reduce(0) { |sum, base| (sum * 5 + base) % 1000000007 }
end

dna = sequence(1000000)
start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
reverse_complement(dna)
checksum = checksum_of(dna)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "length: #{dna.size}"
puts "checksum: #{checksum}"
puts "elapsed: #{elapsed}"
