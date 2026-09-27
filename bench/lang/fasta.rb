CODES = "acgtBDHKMNRSVWY".bytes
CUMULATIVE = [0.27, 0.39, 0.51, 0.78, 0.80, 0.82, 0.84, 0.86,
              0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0]

def fasta(n)
  seed = 42
  checksum = 0
  n.times do
    seed = (seed * 3877 + 29573) % 139968
    r = seed / 139968.0
    j = 0
    j += 1 while r >= CUMULATIVE[j]
    checksum = (checksum + CODES[j]) % 1000000007
  end
  checksum
end

n = 500000
start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
checksum = fasta(n)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "length: #{n}"
puts "checksum: #{checksum}"
puts "elapsed: #{elapsed}"
