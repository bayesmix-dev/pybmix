import numpy as np


class TriangularMemoizer(object):
    """Specific class for triangular recurrence sequences of the kind
    s(0, 0) = s(1, 1) = 0, s(a, 0) = 0 for a > 0, s(a, b) = 0 for b > a,
    s(a, b) = f(a, b) * s(a - 1, b) + g(a, b) s(a - 1, b - 1).

    For instance, Stirling numbers of the first kind are recovered setting
    f(a, b) = (a - 1) and g(a, b) = 1
    """

    def __init__(self, first_term_multiplier, second_term_multiplier):
        self.memo = [np.array([1])]
        self.max_a = 0
        self.first_term_multiplier = first_term_multiplier
        self.second_term_multipier = second_term_multiplier

    def __call__(self, a, b):
        if a == b == 0:
            return 1

        if b == 0:
            return 0

        if b > a:
            return 0

        if a > self.max_a:
            self.add_rows(a)

        return self.memo[a][b]

    def add_rows(self, new_max_a):
        old_max_a = self.max_a
        self.max_a = new_max_a
        for a in range(old_max_a + 1, new_max_a + 1):
            self.memo.append(np.zeros(a + 1))
            self.memo[a][0] = 0
            for b in range(1, a + 1):
                res = self.first_term_multiplier(a, b) * self.__call__(a - 1, b) + \
                      self.second_term_multipier(a, b) * self.__call__(a - 1, b - 1)
                self.memo[a][b] = res


stirling = TriangularMemoizer(lambda a, b: a - 1, lambda a, b: 1)


class generalized_factorial_memoizer(TriangularMemoizer):
    """Computes the Generalized Factorial numbers and stores the results
    in a memoizer.
    The generalized factorial with parameter sigma in [0, 1) satisfy the
    triangular recurrence relation
    s(n, k) = (n - 1 - sigma * k) s(n - 1, k) + sigma * s(n - 1, k-1)
    """

    def __init__(self, sigma):
        def first_arg(n, k): return n - 1 - sigma * k

        def second_arg(n, k): return sigma

        super().__init__(first_arg, second_arg)
