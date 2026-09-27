local function pidigits(digits)
  local boxes = math.floor(digits * 10 / 3) + 1
  local a = {}
  for i = 1, boxes do a[i] = 2 end
  local predigit, nines, checksum, produced = 0, 0, 0, 0
  for _ = 1, digits do
    local q = 0
    for i = boxes, 1, -1 do
      local x = 10 * a[i] + q * i
      local denominator = 2 * i - 1
      a[i] = x % denominator
      q = math.floor(x / denominator)
    end
    a[1] = q % 10
    q = math.floor(q / 10)
    if q == 9 then
      nines = nines + 1
    else
      local carry = q == 10 and 1 or 0
      checksum = (checksum * 10 + predigit + carry) % 1000000007
      produced = produced + 1
      for _ = 1, nines do
        checksum = (checksum * 10 + (carry == 1 and 0 or 9)) % 1000000007
        produced = produced + 1
      end
      predigit = carry == 1 and 0 or q
      nines = 0
    end
  end
  checksum = (checksum * 10 + predigit) % 1000000007
  return produced + 1, checksum
end

local start = os.clock()
local produced, checksum = pidigits(2000)
local elapsed = os.clock() - start
print(string.format("digits: %d", produced))
print(string.format("checksum: %d", checksum))
print(string.format("elapsed: %s", elapsed))
