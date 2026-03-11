import time

class Tree:
    def __init__(self, item, depth):
        self.item = item
        self.left = None
        self.right = None
        if depth > 0:
            item2 = item + item
            depth -= 1
            self.left = Tree(item2 - 1, depth)
            self.right = Tree(item2, depth)

    def check(self):
        if self.left is None:
            return self.item
        return self.item + self.left.check() - self.right.check()

start = time.time()

min_depth = 4
max_depth = 14
stretch_depth = max_depth + 1

print("stretch tree of depth %d check: %d" % (stretch_depth, Tree(0, stretch_depth).check()))

long_lived_tree = Tree(0, max_depth)

iterations = 1
d = 0
while d < max_depth:
    iterations *= 2
    d += 1

depth = min_depth
while depth <= max_depth:
    check = 0
    i = 1
    while i <= iterations:
        check += Tree(i, depth).check() + Tree(-i, depth).check()
        i += 1
    print("%d trees of depth %d check: %d" % (iterations * 2, depth, check))
    iterations //= 4
    depth += 2

print("long lived tree of depth %d check: %d" % (max_depth, long_lived_tree.check()))

elapsed = time.time() - start
print("elapsed: %s" % elapsed)
