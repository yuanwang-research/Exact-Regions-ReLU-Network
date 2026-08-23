import math
from functools import lru_cache

def T(m, n):
    return sum(math.comb(m, i) for i in range(min(m + 1, n + 1)))


def G_orig(lN, lK):
    """Count regions for nested normal-subspace restrictions.

    Implemented by the inductive rules (writing the arguments as columns
    (A_i; B_i), i = 0..l):

    1. l = 0:  G(A_0; B_0) = T(A_0, B_0); in particular G(A_0; 0) = 1.
    2. l >= 1: if B_{i+1} >= B_i for some i < l, the pair (A_{i+1}, B_{i+1})
       can be removed without changing the value.
    3. l >= 1: if A_0 = A_1 and B_0 >= B_1, the pair (A_0, B_0) can be removed.
    4. l >= 1, A_0 > A_1 and B_0 > B_1:
       G(A_0, ...; B_0, ...) = G(A_0 - 1, ...; B_0, ...)
                             + G(A_0 - 1, ...; B_0 - 1, ...).

    After rule 2 is exhausted B is strictly decreasing, so exactly one of
    rules 3 and 4 applies (lN is non-increasing, and rule 4 never lowers
    A_0 below A_1 nor B_0 below B_1); rule 4 terminates because A_0 + B_0
    strictly decreases.

    Args:
        lN: Non-increasing list [n0, n1, ..., nl], where ni hyperplanes are
            restricted at level i (nested).
        lK: List [k0, k1, ..., kl] of the same length as ``lN``.

    Returns:
        Number of regions induced by the arrangement.
    """
    if len(lN) != len(lK):
        raise ValueError("lN and lK must have the same length")
    if len(lN) == 0:
        return 1
    if any((not isinstance(x, int)) or x < 0 for x in lN):
        raise ValueError("lN must contain non-negative integers")
    if any((not isinstance(x, int)) or x < 0 for x in lK):
        raise ValueError("lK must contain non-negative integers")
    if any(lN[i] < lN[i + 1] for i in range(len(lN) - 1)):
        raise ValueError("lN must be non-increasing")

    @lru_cache(maxsize=None)
    def g(A, B):
        l = len(A) - 1
        if l == 0:                                   # rule 1
            return T(A[0], B[0])
        for i in range(l):                           # rule 2
            if B[i + 1] >= B[i]:
                return g(A[:i + 1] + A[i + 2:], B[:i + 1] + B[i + 2:])
        if A[0] == A[1]:                             # rule 3 (B_0 > B_1 here)
            return g(A[1:], B[1:])
        return (g((A[0] - 1,) + A[1:], B)            # rule 4
                + g((A[0] - 1,) + A[1:], (B[0] - 1,) + B[1:]))

    return g(tuple(lN), tuple(lK))


def G(ln, lk):
    assert len(ln) == len(lk) + 2
    sum, list_sum = 0, []
    for i in range(len(ln) - 1, 0, -1):
        sum += ln[i]
        list_sum.append(sum)
    list_sum.reverse() # 1, 2,..., l
    for i in range(1, len(list_sum)):
        list_sum[i] += lk[i-1] # (S_1, k_1 + S_2, ..., k_l-1 + S_l)
    lk.insert(0, ln[0]) # (n_0, k_1, ..., k_l-1)
    return G_orig(list_sum, lk)


def G_lin_plus(ln, lk):
    assert len(ln) == len(lk) + 2
    sum, list_sum = 0, []
    for i in range(len(ln) - 1, 0, -1):
        sum += ln[i]
        list_sum.append(sum)
    list_sum.reverse() # 1, 2,..., l
    for i in range(1, len(list_sum)):
        list_sum[i] += lk[i-1] # (S_1, k_1 + S_2, ..., k_l-1 + S_l)
    res_g = G_orig(list_sum, [ln[0]] + lk) # (n_0, k_1, ..., k_l-1)
    res_t = T(ln[-1], min(ln[0], min(lk)) if lk else ln[0])
    res = ((2 ** ln[-1] - 1) * res_g + res_t) / 2 ** ln[-1]
    return res


def G_lin_minus(ln, lk):
    l, t = len(ln) - 1, len(lk)
    tmp = 1.0
    for i in range(t + 1, l + 1):
        tmp *= (1 - 1 / 2 ** ln[i])
    tmp_min = min(ln[0], min(lk[:-1])) if len(lk) > 1 else ln[0]
    tmp_t = T(ln[len(lk)], tmp_min)
    res = tmp * tmp_t + (1 - tmp)
    return res
    

def E_indct(ln, lk, g, from_zero = True): # max_len_k is the t in the paper
    """
    ln: (n_0, n_1, ..., n_l), where n_i is the number of neurons in layer i (including input layer).
    lk: (k_1, ..., k_l-1), where k_i is the temporary values
    This is different from the lN and lK in G_orig
    """
    l = len(ln) - 1
    if len(lk) == l - 1:
        tmp = 1.0
        for i in range(1, l):
            tmp *= math.comb(ln[i], lk[i-1])
        g_val = g(ln, lk)
        return tmp * g_val
    ind = len(lk) + 1
    res = 0.0
    ind_start = 0 if from_zero else 1
    for i in range(ind_start, ln[ind] + 1):
        res += E_indct(ln, lk + [i], g, from_zero)
    return res


def E_indct_minus(ln, lk, max_len_k): # max_len_k is the t in the paper
    """
    ln: (n_0, n_1, ..., n_l), where n_i is the number of neurons in layer i (including input layer).
    lk: (k_1, ..., k_l-1), where k_i is the temporary values
    This is different from the lN and lK in G_orig
    """
    l = len(ln) - 1
    if len(lk) == max_len_k:
        tmp = 1.0
        for i in range(1, max_len_k + 1):
            tmp *= math.comb(ln[i], lk[i-1])
        g_val = G_lin_minus(ln, lk + [0])
        return tmp * g_val
    ind = len(lk) + 1
    res = 0.0
    for i in range(1, ln[ind] + 1):
        res += E_indct_minus(ln, lk + [i], max_len_k)
    return res


def E_act(ln):
    l, res, lk = len(ln) - 1, 0, []
    res = E_indct(ln, lk, G)
    return res / 2**(sum(ln[1:-1]))


def E_lin_fake(ln):
    l = len(ln) - 1
    lk = []
    res_plus = E_indct(ln, lk, G_lin_plus, from_zero=False)
    res_plus /= 2**(sum(ln[1:-1]))
    sum_res_minus = 0.0
    for i in range(0, l - 1):
        lk = []
        tmp_e = E_indct_minus(ln, lk, i)
        sum_res_minus += tmp_e / 2**(sum(ln[1:i+2]))
    return res_plus + sum_res_minus


if __name__ == "__main__":
    ln = [2, 2, 3] # (n_0, n_1, ..., n_{l-1})
    print(f"Expected number of activation regions: {E_act(ln)}")
    print(f"(Fake) expected number of linear regions: {E_lin_fake(ln)}")