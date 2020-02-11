import numpy as np
import random
from math import floor


def halton_sample_random_base(size, dim):
    gen = gen_prime()
    possible_bases = [next(gen) for _ in range(4 * dim)]
    bases = random.choices(possible_bases, k=dim)
    skip = max(bases)
    result = np.zeros((size, dim))
    for i in range(dim):
        base = bases[i]
        temp = halton_1d(base, size + skip)
        result[:, i] = temp[skip:]
    return result


def halton_sample(size, dim):
    gen = gen_prime()
    bases = [next(gen) for _ in range(dim)]
    skip = max(bases)
    result = np.zeros((size, dim))
    for i in range(dim):
        base = bases[i]
        temp = halton_1d(base, size + skip)
        result[:, i] = temp[skip:]
    return result[np.random.permutation(result.shape[0]), :]


def is_prime(num):
    if num == 2:
        return True
    if num % 2 != 1:
        return False
    for i in range(3, floor(num ** 0.5) + 1, 2):
        if num % i == 0:
            return False
    return True


def gen_prime():
    yield 2
    num = 3
    while True:
        if is_prime(num):
            yield num
        num += 2


def halton_single(i, base):
    f = 1
    result = 0
    while i > 0:
        f /= base
        result += f * (i % base)
        i = floor(i / base)
    return result


def halton_generator(base):
    i = 1
    while True:
        yield halton_single(i, base)
        i += 1


def halton_1d(base, size):
    gen = halton_generator(base)
    return np.array([next(gen) for _ in range(1, size + 1)])


def logn(x, n):
    return np.log(x) / np.log(n)


def generate_halton_denominator(size, base):
    return base ** np.ceil(logn(np.arange(1, size + 1), base))
