import time

def sequence(n):
    dna = [0] * n
    seed = 42
    for i in range(n):
        seed = (seed * 3877 + 29573) % 139968
        dna[i] = seed * 4 // 139968
    return dna

def reverse_complement(dna):
    lo = 0
    hi = len(dna) - 1
    while lo <= hi:
        left = 3 - dna[hi]
        dna[hi] = 3 - dna[lo]
        dna[lo] = left
        lo += 1
        hi -= 1

def checksum_of(dna):
    total = 0
    for base in dna:
        total = (total * 5 + base) % 1000000007
    return total

dna = sequence(1000000)
start = time.time()
reverse_complement(dna)
checksum = checksum_of(dna)
elapsed = time.time() - start
print("length: %d" % len(dna))
print("checksum: %d" % checksum)
print("elapsed: %s" % elapsed)
