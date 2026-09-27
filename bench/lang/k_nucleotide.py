import time

def sequence(n):
    dna = [0] * n
    seed = 42
    for i in range(n):
        seed = (seed * 3877 + 29573) % 139968
        dna[i] = seed * 4 // 139968
    return dna

def frequencies(dna, k):
    counts = {}
    high = 4 ** (k - 1)
    key = 0
    for j in range(k):
        key = key * 4 + dna[j]
    counts[key] = 1
    for i in range(k, len(dna)):
        key = key % high * 4 + dna[i]
        counts[key] = counts.get(key, 0) + 1
    return counts

dna = sequence(250000)
start = time.time()
checksum = 0
for k in [1, 2, 3, 4, 6, 12, 18]:
    for count in frequencies(dna, k).values():
        checksum = (checksum + count * k) % 1000000007
elapsed = time.time() - start
print("length: %d" % len(dna))
print("checksum: %d" % checksum)
print("elapsed: %s" % elapsed)
