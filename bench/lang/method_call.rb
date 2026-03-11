class Toggle
  attr_reader :value

  def initialize(start_state)
    @value = start_state
  end

  def activate
    @value = !@value
    self
  end
end

class NthToggle < Toggle
  def initialize(start_state, max_counter)
    super(start_state)
    @count_max = max_counter
    @count = 0
  end

  def activate
    @count += 1
    if @count >= @count_max
      @value = !@value
      @count = 0
    end
    self
  end
end

start = Process.clock_gettime(Process::CLOCK_MONOTONIC)

n = 100000
val = true
toggle = Toggle.new(val)

n.times do
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
  val = toggle.activate.value
end

ntoggle = NthToggle.new(val, 3)

n.times do
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
  val = ntoggle.activate.value
end

elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
puts "elapsed: #{elapsed}"
