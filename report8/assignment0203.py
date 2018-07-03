# coding: UTF-8
import numpy as np
from scipy import special
from numerical_integral import mc_integration_1d, trapezoidal_rule_1d

np.random.seed(1729)


def beta(pi, a, b):
    return pi**(a - 1) * (1 - pi)**(b - 1) / special.beta(a, b)


def assignment02(integrate):
    def beta52(pi): return beta(pi, 5, 2)
    print("prier: Beta(1, 1)")
    print("\tp(pi>=0.5|data) = ", integrate(beta52, 0.5, 1.0))
    print("\tp(pi>=0.8|data) = ", integrate(beta52, 0.8, 1.0))

    def beta41_11(pi): return beta(pi, 4.1, 1.1)
    print("prier: Beta(0.1, 0.1)")
    print("\tp(pi>=0.5|data) = ", integrate(beta41_11, 0.5, 1.0))
    print("\tp(pi>=0.8|data) = ", integrate(beta41_11, 0.8, 1.0))

    def beta96(pi): return beta(pi, 9, 6)
    print("prier: Beta(5, 5)")
    print("\tp(pi>=0.5|data) = ", integrate(beta96, 0.5, 1.0))
    print("\tp(pi>=0.8|data) = ", integrate(beta96, 0.8, 1.0))


def assignment03(integrate):
    print("prier: Beta(1, 1)")

    def expect6_16(pi): return (2 * (1 - pi) / pi) * beta(pi, 6, 16)
    print("E[x|data] = ", integrate(expect6_16, 1e-8, 1))

    def pr6_16(pi): return beta(pi, 6, 16)
    print("Pr(pi<=0.1|data) = ", integrate(pr6_16, 1e-8, 0.1))

    print("prier: Beta(0.1, 0.1)")

    def expect51_151(pi): return (2 * (1 - pi) / pi) * beta(pi, 5.1, 15.1)
    print("E[x|data] = ", integrate(expect51_151, 1e-8, 1))

    def pr51_151(pi): return beta(pi, 5.1, 15.1)
    print("Pr(pi<=0.1|data) = ", integrate(pr51_151, 1e-8, 0.1))

    print("prier: Beta(5, 5)")

    def expect10_20(pi): return (2 * (1 - pi) / pi) * beta(pi, 10, 20)
    print("E[x|data] = ", integrate(expect10_20, 1e-8, 1))

    def pr10_20(pi): return beta(pi, 10, 20)
    print("Pr(pi<=0.1|data) = ", integrate(pr10_20, 1e-8, 0.1))


if __name__ == '__main__':
    assignment03(trapezoidal_rule_1d)
