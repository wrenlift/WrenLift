def fannkuch(n)
  perm = (0...n).to_a
  count = Array.new(n, 0)
  max_flips = 0
  checksum = 0
  sign = 1
  r = n
  loop do
    while r != 1
      count[r - 1] = r
      r -= 1
    end
    if perm[0] != 0
      copy = perm.dup
      flips = 0
      k = copy[0]
      while k != 0
        lo = 0
        hi = k
        while lo < hi
          copy[lo], copy[hi] = copy[hi], copy[lo]
          lo += 1
          hi -= 1
        end
        flips += 1
        k = copy[0]
      end
      max_flips = flips if flips > max_flips
      checksum += sign * flips
    end
    loop do
      return [checksum, max_flips] if r == n
      first = perm[0]
      perm[0, r] = perm[1, r]
      perm[r] = first
      count[r] -= 1
      break if count[r] > 0
      r += 1
    end
    sign = -sign
  end
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
checksum, max_flips = fannkuch(10)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "checksum: #{checksum}"
puts "max flips: #{max_flips}"
puts "elapsed: #{elapsed}"
