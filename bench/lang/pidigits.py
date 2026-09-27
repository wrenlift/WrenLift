import time

def pidigits(digits):
    boxes = digits * 10 // 3 + 1
    a = [2] * boxes
    predigit = 0
    nines = 0
    checksum = 0
    produced = 0
    for _ in range(digits):
        q = 0
        for i in range(boxes, 0, -1):
            x = 10 * a[i - 1] + q * i
            denominator = 2 * i - 1
            a[i - 1] = x % denominator
            q = x // denominator
        a[0] = q % 10
        q = q // 10
        if q == 9:
            nines += 1
        else:
            carry = 1 if q == 10 else 0
            checksum = (checksum * 10 + predigit + carry) % 1000000007
            produced += 1
            for _ in range(nines):
                checksum = (checksum * 10 + (0 if carry else 9)) % 1000000007
                produced += 1
            predigit = 0 if carry else q
            nines = 0
    checksum = (checksum * 10 + predigit) % 1000000007
    return produced + 1, checksum

start = time.time()
produced, checksum = pidigits(2000)
elapsed = time.time() - start
print("digits: %d" % produced)
print("checksum: %d" % checksum)
print("elapsed: %s" % elapsed)
