local function sequence(n)
  local dna, seed = {}, 42
  for i = 1, n do
    seed = (seed * 3877 + 29573) % 139968
    dna[i] = math.floor(seed * 4 / 139968)
  end
  return dna
end

local function frequencies(dna, k)
  local counts = {}
  local high = 4 ^ (k - 1)
  local key = 0
  for j = 1, k do key = key * 4 + dna[j] end
  counts[key] = 1
  for i = k + 1, #dna do
    key = key % high * 4 + dna[i]
    counts[key] = (counts[key] or 0) + 1
  end
  return counts
end

local dna = sequence(250000)
local start = os.clock()
local checksum = 0
for _, k in ipairs({ 1, 2, 3, 4, 6, 12, 18 }) do
  for _, count in pairs(frequencies(dna, k)) do
    checksum = (checksum + count * k) % 1000000007
  end
end
local elapsed = os.clock() - start
print(string.format("length: %d", #dna))
print(string.format("checksum: %d", checksum))
print(string.format("elapsed: %s", elapsed))
