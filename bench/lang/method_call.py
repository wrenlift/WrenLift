import time

class Toggle:
    def __init__(self, start_state):
        self.state = start_state

    @property
    def value(self):
        return self.state

    def activate(self):
        self.state = not self.state
        return self

class NthToggle(Toggle):
    def __init__(self, start_state, max_counter):
        super().__init__(start_state)
        self.count_max = max_counter
        self.count = 0

    def activate(self):
        self.count += 1
        if self.count >= self.count_max:
            self.state = not self.state
            self.count = 0
        return self

start = time.time()

n = 100000
val = True
toggle = Toggle(val)

for _ in range(n):
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value
    val = toggle.activate().value

ntoggle = NthToggle(val, 3)

for _ in range(n):
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value
    val = ntoggle.activate().value

elapsed = time.time() - start
print("elapsed: %s" % elapsed)
