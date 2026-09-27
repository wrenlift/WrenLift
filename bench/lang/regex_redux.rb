A = 1
C = 2
G = 4
T = 8
VARIANTS = [
  [[A, G, G, G, T, A, A, A], [T, T, T, A, C, C, C, T]],
  [[C | G | T, G, G, G, T, A, A, A], [T, T, T, A, C, C, C, A | C | G]],
  [[A, A | C | T, G, G, T, A, A, A], [T, T, T, A, C, C, A | G | T, T]],
  [[A, G, A | C | T, G, T, A, A, A], [T, T, T, A, C, A | G | T, C, T]],
  [[A, G, G, A | C | T, T, A, A, A], [T, T, T, A, A | G | T, C, C, T]],
  [[A, G, G, G, A | C | G, A, A, A], [T, T, T, C | G | T, C, C, C, T]],
  [[A, G, G, G, T, C | G | T, A, A], [T, T, A | C | G, A, C, C, C, T]],
  [[A, G, G, G, T, A, C | G | T, A], [T, A | C | G, T, A, C, C, C, T]],
  [[A, G, G, G, T, A, A, C | G | T], [A | C | G, T, T, A, C, C, C, T]]
].freeze

def sequence(n)
  seed = 42
  Array.new(n) do
    seed = (seed * 3877 + 29573) % 139968
    seed * 4 / 139968
  end
end

def matches_at(dna, i, pattern)
  pattern.each_with_index do |mask, j|
    return false if mask & (1 << dna[i + j]) == 0
  end
  true
end

def count(dna, alternatives)
  n = 0
  (0..dna.size - 8).each do |i|
    alternatives.each { |pattern| n += 1 if matches_at(dna, i, pattern) }
  end
  n
end

dna = sequence(250000)
start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
checksum = 0
VARIANTS.each { |alternatives| checksum = checksum * 31 + count(dna, alternatives) }
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "length: #{dna.size}"
puts "checksum: #{checksum}"
puts "elapsed: #{elapsed}"
