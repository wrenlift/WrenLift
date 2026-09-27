def pidigits(digits)
  boxes = digits * 10 / 3 + 1
  a = Array.new(boxes, 2)
  predigit = 0
  nines = 0
  checksum = 0
  produced = 0
  digits.times do
    q = 0
    i = boxes
    while i > 0
      x = 10 * a[i - 1] + q * i
      denominator = 2 * i - 1
      a[i - 1] = x % denominator
      q = x / denominator
      i -= 1
    end
    a[0] = q % 10
    q /= 10
    if q == 9
      nines += 1
    else
      carry = q == 10 ? 1 : 0
      checksum = (checksum * 10 + predigit + carry) % 1000000007
      produced += 1
      nines.times do
        checksum = (checksum * 10 + (carry == 1 ? 0 : 9)) % 1000000007
        produced += 1
      end
      predigit = carry == 1 ? 0 : q
      nines = 0
    end
  end
  checksum = (checksum * 10 + predigit) % 1000000007
  [produced + 1, checksum]
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
produced, checksum = pidigits(2000)
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "digits: #{produced}"
puts "checksum: #{checksum}"
puts "elapsed: #{elapsed}"
