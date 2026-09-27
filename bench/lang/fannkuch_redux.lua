local function fannkuch(n)
  local perm, count, copy = {}, {}, {}
  for i = 1, n do
    perm[i] = i - 1
    count[i] = 0
  end
  local max_flips, checksum, sign, r = 0, 0, 1, n
  while true do
    while r ~= 1 do
      count[r] = r
      r = r - 1
    end
    if perm[1] ~= 0 then
      for i = 1, n do copy[i] = perm[i] end
      local flips = 0
      local k = copy[1]
      while k ~= 0 do
        local lo, hi = 1, k + 1
        while lo < hi do
          copy[lo], copy[hi] = copy[hi], copy[lo]
          lo = lo + 1
          hi = hi - 1
        end
        flips = flips + 1
        k = copy[1]
      end
      if flips > max_flips then max_flips = flips end
      checksum = checksum + sign * flips
    end
    while true do
      if r == n then return checksum, max_flips end
      local first = perm[1]
      for i = 1, r do perm[i] = perm[i + 1] end
      perm[r + 1] = first
      count[r + 1] = count[r + 1] - 1
      if count[r + 1] > 0 then break end
      r = r + 1
    end
    sign = -sign
  end
end

local start = os.clock()
local checksum, max_flips = fannkuch(10)
local elapsed = os.clock() - start
print(string.format("checksum: %d", checksum))
print(string.format("max flips: %d", max_flips))
print(string.format("elapsed: %s", elapsed))
