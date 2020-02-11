from math import floor, ceil


def is_prime(num):
    if num == 2:
        return True
    if num % 2 != 1:
        return False
    for i in range(3, floor(num ** 0.5) + 1, 2):
        if num % i == 0:
            return False
    return True




def gen_prime_list(start=31, max_prime=1e+7):
    prime_list = []
    if (start % 2) != 1:
        start = ceil(start)
    if (start % 2) == 0:
        start += 1
    start = int(start)
    max_prime = floor(max_prime)

    for num in range(start, max_prime, 2):
        for i in range(3, 1 + floor(num ** 0.5), 2):
            if (num % i) == 0:
                break
        else:
            prime_list.append(num)
    return prime_list


def gen_selected_prime_list(start=31, max_prime=1e+7, ratio=1.5):
    prime_list = gen_prime_list(start, max_prime)
    selected_prime_list = []
    current_prime = prime_list[0]
    current_location = 0
    selected_prime_list.append(current_prime)
    while ratio * current_prime < prime_list[-1]:
        next_prime_approx = ratio * current_prime
        for i in range(current_location + 1, len(prime_list)):
            if prime_list[i] > next_prime_approx:
                if (prime_list[i] - next_prime_approx) > (next_prime_approx - prime_list[i - 1]):
                    current_prime = prime_list[i - 1]
                    current_location = i - 1
                else:
                    current_prime = prime_list[i]
                    current_location = i
                selected_prime_list.append(current_prime)
                break
    return selected_prime_list
