local bit = require("bit")
local band, lshift = bit.band, bit.lshift

local A, C, G, T = 1, 2, 4, 8
local variants = {
  { { A, G, G, G, T, A, A, A }, { T, T, T, A, C, C, C, T } },
  { { C + G + T, G, G, G, T, A, A, A }, { T, T, T, A, C, C, C, A + C + G } },
  { { A, A + C + T, G, G, T, A, A, A }, { T, T, T, A, C, C, A + G + T, T } },
  { { A, G, A + C + T, G, T, A, A, A }, { T, T, T, A, C, A + G + T, C, T } },
  { { A, G, G, A + C + T, T, A, A, A }, { T, T, T, A, A + G + T, C, C, T } },
  { { A, G, G, G, A + C + G, A, A, A }, { T, T, T, C + G + T, C, C, C, T } },
  { { A, G, G, G, T, C + G + T, A, A }, { T, T, A + C + G, A, C, C, C, T } },
  { { A, G, G, G, T, A, C + G + T, A }, { T, A + C + G, T, A, C, C, C, T } },
  { { A, G, G, G, T, A, A, C + G + T }, { A + C + G, T, T, A, C, C, C, T } },
}

local function sequence(n)
  local dna, seed = {}, 42
  for i = 1, n do
    seed = (seed * 3877 + 29573) % 139968
    dna[i] = math.floor(seed * 4 / 139968)
  end
  return dna
end

local function matches_at(dna, i, pattern)
  for j = 1, #pattern do
    if band(pattern[j], lshift(1, dna[i + j - 1])) == 0 then return false end
  end
  return true
end

local function count(dna, alternatives)
  local n = 0
  for i = 1, #dna - 7 do
    for _, pattern in ipairs(alternatives) do
      if matches_at(dna, i, pattern) then n = n + 1 end
    end
  end
  return n
end

local dna = sequence(250000)
local start = os.clock()
local checksum = 0
for _, alternatives in ipairs(variants) do
  checksum = checksum * 31 + count(dna, alternatives)
end
local elapsed = os.clock() - start
print(string.format("length: %d", #dna))
print(string.format("checksum: %d", checksum))
print(string.format("elapsed: %s", elapsed))
