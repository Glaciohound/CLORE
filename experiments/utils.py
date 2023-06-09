class RunningMean:
    def __init__(self, p=0.9999):
        self.value = 0
        self.count = -1
        self.p = p

    def update(self, values):
        for _v in values:
            self.update_one(_v)

    def update_one(self, value):
        self.count += 1
        gamma = (1-self.p) / (1 - self.p ** (self.count + 1))
        self.value = self.value * (1 - gamma) + value * gamma

    def __str__(self):
        return f"{self.value:.4f}"

    def __repr__(self):
        return str(self)

    def __format__(self, spec):
        return format(self.value, spec)


class Mean:
    def __init__(self):
        self.value = 0
        self.count = -1

    def update(self, values):
        for _v in values:
            self.update_one(_v)

    def update_one(self, value):
        self.count += 1
        gamma = 1 / (1 + self.count)
        self.value = self.value * (1 - gamma) + value * gamma

    def __str__(self):
        return f"{self.value:.4f}"

    def __repr__(self):
        return str(self)

    def __format__(self, spec):
        return format(self.value, spec)
