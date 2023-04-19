import numpy as np
from joblib import Parallel, delayed
from scipy.special import gammaln

round = np.round


def binomial(n, k):
    return np.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def euler1785(l, n, d, t, precision=4):
    delta = binomial(n, t) ** d
    if delta == 0:
        return 1.0

    min_delta = delta / 10 ** (precision + 1)
    total = 0

    for k in range(1, n - l + 1):
        product = (-1) ** k
        product *= binomial(l + k - 1, l)
        product *= binomial(n, l + k)
        product *= binomial(n - l - k, t) ** d

        total += product

        if abs(product) <= min_delta:
            break

    return abs(round((delta + total) / delta, precision))


def euler1785_v(l, n, d, precision=4):
    delta = np.prod([binomial(n, t) for t in d])
    if delta == 0:
        return 1.0

    min_delta = delta / 10 ** (precision + 1)
    total = 0

    for k in range(1, n - l + 1):
        product = (-1) ** k
        product *= binomial(l + k - 1, l)
        product *= binomial(n, l + k)
        product *= np.prod([binomial(n - l - k, t) for t in d])

        total += product

        if abs(product) <= min_delta:
            break

    return abs(round((delta + total) / delta, precision))


def analysis(k, s):
    threshold = 0.99
    n = (2 * k) ** 2
    g = k * (3 * k - 2)
    lam = n - g

    start = 1
    end = 1
    c = start
    found = False

    while not found:
        p = euler1785(lam, n, c, s)

        if p > threshold:
            found = True
        else:
            start = c
            c *= 2
            end = c

    while end - start > 100:
        middle = (start + end) // 2
        c = middle

        p = euler1785(lam, n, c, s)

        if p > 0.99:
            end = middle
        else:
            start = middle

    for c in range(start, end):
        p = euler1785(lam, n, c, s)

        if p > 0.99:
            return f'k={k}, s={s}, c={c}'


def main_analysis(k, s_values):
    results = Parallel(n_jobs=-1)(
        delayed(analysis)(k_value, s_value) for k_value in k for s_value in s_values
    )
    for result in results:
        print(result)


if __name__ == '__main__':
    k_values = [128, 256]
    s_values = [10, 16, 20]

    for k in k_values:
        main_analysis(k_values, s_values)
        print()