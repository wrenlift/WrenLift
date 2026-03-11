def fib(n)
  return n if n < 2
  fib(n - 1) + fib(n - 2)
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
5.times { fib(28) }
elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "elapsed: #{elapsed}"
