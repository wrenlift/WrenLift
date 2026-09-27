import time

A, C, G, T = 1, 2, 4, 8
VARIANTS = [
    [[A, G, G, G, T, A, A, A], [T, T, T, A, C, C, C, T]],
    [[C | G | T, G, G, G, T, A, A, A], [T, T, T, A, C, C, C, A | C | G]],
    [[A, A | C | T, G, G, T, A, A, A], [T, T, T, A, C, C, A | G | T, T]],
    [[A, G, A | C | T, G, T, A, A, A], [T, T, T, A, C, A | G | T, C, T]],
    [[A, G, G, A | C | T, T, A, A, A], [T, T, T, A, A | G | T, C, C, T]],
    [[A, G, G, G, A | C | G, A, A, A], [T, T, T, C | G | T, C, C, C, T]],
    [[A, G, G, G, T, C | G | T, A, A], [T, T, A | C | G, A, C, C, C, T]],
    [[A, G, G, G, T, A, C | G | T, A], [T, A | C | G, T, A, C, C, C, T]],
    [[A, G, G, G, T, A, A, C | G | T], [A | C | G, T, T, A, C, C, C, T]],
]

def sequence(n):
    dna = [0] * n
    seed = 42
    for i in range(n):
        seed = (seed * 3877 + 29573) % 139968
        dna[i] = seed * 4 // 139968
    return dna

def matches_at(dna, i, pattern):
    for j in range(len(pattern)):
        if pattern[j] & (1 << dna[i + j]) == 0:
            return False
    return True

def count(dna, alternatives):
    n = 0
    for i in range(len(dna) - 7):
        for pattern in alternatives:
            if matches_at(dna, i, pattern):
                n += 1
    return n

dna = sequence(250000)
start = time.time()
checksum = 0
for alternatives in VARIANTS:
    checksum = checksum * 31 + count(dna, alternatives)
elapsed = time.time() - start
print("length: %d" % len(dna))
print("checksum: %d" % checksum)
print("elapsed: %s" % elapsed)
