import numpy as np


class _stirling_memoizer(object):
    """ Computes the Stirling Numbers of the first kind and stores the
    result in a customly defined memoizer. Stirling numbers of the first
    kind are defined by the recurrence relation
    s(0, 0) = s(1, 1) = 0, s(a, 0) = 0 for a > 0, s(a, b) = 0 for b > a,
    s(a + 1, b) = s(a, b - 1) + a * s(a, b)
    """

    def __init__(self):
        self.memo = [np.array([0])]
        self.max_a = 0

    def __call__(self, a, b):
        if b == 0:
            return 0

        if b > a:
            return 0

        if a > len(self.memo):
            self.add_rows(a)

        return self.memo[a][b]

    def add_rows(self, new_max_a):
        for a in range(self.max_a + 1, new_max_a + 1):
            self.memo.append(np.zeros(a + 1))
            self.memo[a][0] = 0
            for b in range(1, a):
                res = (a - 1) * self.memo[a - 1][b] + self.memo[a - 1][b - 1]
                self.memo[a][b] = res

            self.memo[a][a] = 1

        self.max_a = new_max_a


stirling = _stirling_memoizer()
