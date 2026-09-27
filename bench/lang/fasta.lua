local codes = { string.byte("acgtBDHKMNRSVWY", 1, -1) }
local cumulative = { 0.27, 0.39, 0.51, 0.78, 0.80, 0.82, 0.84, 0.86,
  0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1 }

local function fasta(n)
  local seed, checksum = 42, 0
  for _ = 1, n do
    seed = (seed * 3877 + 29573) % 139968
    local r = seed / 139968
    local j = 1
    while r >= cumulative[j] do j = j + 1 end
    checksum = (checksum + codes[j]) % 1000000007
  end
  return checksum
end

local n = 500000
local start = os.clock()
local checksum = fasta(n)
local elapsed = os.clock() - start
print(string.format("length: %d", n))
print(string.format("checksum: %d", checksum))
print(string.format("elapsed: %s", elapsed))
