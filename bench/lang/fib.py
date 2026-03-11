import time

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

start = time.time()
for _ in range(5):
    fib(28)
elapsed = time.time() - start
print("elapsed: %s" % elapsed)
