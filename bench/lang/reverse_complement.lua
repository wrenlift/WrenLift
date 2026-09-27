local function sequence(n)
  local dna, seed = {}, 42
  for i = 1, n do
    seed = (seed * 3877 + 29573) % 139968
    dna[i] = math.floor(seed * 4 / 139968)
  end
  return dna
end

local function reverse_complement(dna)
  local lo, hi = 1, #dna
  while lo <= hi do
    local left = 3 - dna[hi]
    dna[hi] = 3 - dna[lo]
    dna[lo] = left
    lo = lo + 1
    hi = hi - 1
  end
end

local function checksum_of(dna)
  local sum = 0
  for i = 1, #dna do sum = (sum * 5 + dna[i]) % 1000000007 end
  return sum
end

local dna = sequence(1000000)
local start = os.clock()
reverse_complement(dna)
local checksum = checksum_of(dna)
local elapsed = os.clock() - start
print(string.format("length: %d", #dna))
print(string.format("checksum: %d", checksum))
print(string.format("elapsed: %s", elapsed))
