import time

def fannkuch(n):
    perm = list(range(n))
    count = [0] * n
    max_flips = 0
    checksum = 0
    sign = 1
    r = n
    while True:
        while r != 1:
            count[r - 1] = r
            r -= 1
        if perm[0] != 0:
            copy = perm[:]
            flips = 0
            k = copy[0]
            while k != 0:
                copy[:k + 1] = copy[k::-1]
                flips += 1
                k = copy[0]
            max_flips = max(max_flips, flips)
            checksum += sign * flips
        while True:
            if r == n:
                return checksum, max_flips
            first = perm[0]
            perm[:r] = perm[1:r + 1]
            perm[r] = first
            count[r] -= 1
            if count[r] > 0:
                break
            r += 1
        sign = -sign

start = time.time()
checksum, max_flips = fannkuch(10)
elapsed = time.time() - start
print("checksum: %d" % checksum)
print("max flips: %d" % max_flips)
print("elapsed: %s" % elapsed)
