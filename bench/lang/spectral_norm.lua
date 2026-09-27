local function denominator(i, j)
  local ij = i + j
  return ij * (ij + 1) / 2 + i + 1
end

-- Rows and columns are counted from 0, as in a(i, j); the vectors are
-- 1-based.
local function multiply_av(v, out, n)
  for i = 0, n - 1 do
    local sum = 0
    for j = 0, n - 1 do sum = sum + v[j + 1] / denominator(i, j) end
    out[i + 1] = sum
  end
end

local function multiply_atv(v, out, n)
  for i = 0, n - 1 do
    local sum = 0
    for j = 0, n - 1 do sum = sum + v[j + 1] / denominator(j, i) end
    out[i + 1] = sum
  end
end

local function spectral_norm(n)
  local u, v, tmp = {}, {}, {}
  for i = 1, n do
    u[i] = 1
    v[i] = 0
    tmp[i] = 0
  end
  for _ = 1, 10 do
    multiply_av(u, tmp, n)
    multiply_atv(tmp, v, n)
    multiply_av(v, tmp, n)
    multiply_atv(tmp, u, n)
  end
  local vbv, vv = 0, 0
  for i = 1, n do
    vbv = vbv + u[i] * v[i]
    vv = vv + v[i] * v[i]
  end
  return math.sqrt(vbv / vv)
end

local start = os.clock()
local result = spectral_norm(500)
local elapsed = os.clock() - start
print(string.format("result: %.13f", result))
print(string.format("elapsed: %s", elapsed))
