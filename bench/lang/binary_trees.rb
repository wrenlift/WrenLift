class Tree
  attr_reader :item, :left, :right

  def initialize(item, depth)
    @item = item
    if depth > 0
      item2 = item + item
      depth -= 1
      @left = Tree.new(item2 - 1, depth)
      @right = Tree.new(item2, depth)
    end
  end

  def check
    return @item if @left.nil?
    @item + @left.check - @right.check
  end
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)

min_depth = 4
max_depth = 14
stretch_depth = max_depth + 1

puts "stretch tree of depth #{stretch_depth} check: #{Tree.new(0, stretch_depth).check}"

long_lived_tree = Tree.new(0, max_depth)

iterations = 1
d = 0
while d < max_depth
  iterations *= 2
  d += 1
end

depth = min_depth
while depth <= max_depth
  check = 0
  i = 1
  while i <= iterations
    check += Tree.new(i, depth).check + Tree.new(-i, depth).check
    i += 1
  end
  puts "#{iterations * 2} trees of depth #{depth} check: #{check}"
  iterations /= 4
  depth += 2
end

puts "long lived tree of depth #{max_depth} check: #{long_lived_tree.check}"

elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "elapsed: #{elapsed}"
