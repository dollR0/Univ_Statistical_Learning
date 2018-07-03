# coding: UTF-8
import numpy as np
from scipy import special


def mc_integration_1d(func, a, b):
    N = 1000000
    max_y = func(a)
    min_y = func(a)

    for x in np.arange(a, b, 0.01):
        y = func(x)
        if y > max_y:
            max_y = y
    if max_y <= 1e-8:
        return 0.0

    Xs = np.random.uniform(a, b, N)
    Ys = np.random.uniform(0.0, max_y, N)
    count = 0
    for x, y in zip(Xs, Ys):
        if func(x) >= y:
            count += 1

    return (count / N) * (b - a) * max_y


def trapezoidal_rule_1d(func, a, b):
    N = 100000
    Xs = np.linspace(a, b, N, endpoint=True)
    dx = (b - a) / (N - 1)
    y_prev = func(a)

    area = 0
    for x in Xs[1:]:
        y = func(x)
        area += (y + y_prev) * dx / 2
        y_prev = y

    return area
