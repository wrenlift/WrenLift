import time

CODES = [ord(c) for c in "acgtBDHKMNRSVWY"]
CUMULATIVE = [0.27, 0.39, 0.51, 0.78, 0.80, 0.82, 0.84, 0.86,
              0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1]

def fasta(n):
    seed = 42
    checksum = 0
    for _ in range(n):
        seed = (seed * 3877 + 29573) % 139968
        r = seed / 139968
        j = 0
        while r >= CUMULATIVE[j]:
            j += 1
        checksum = (checksum + CODES[j]) % 1000000007
    return checksum

n = 500000
start = time.time()
checksum = fasta(n)
elapsed = time.time() - start
print("length: %d" % n)
print("checksum: %d" % checksum)
print("elapsed: %s" % elapsed)
