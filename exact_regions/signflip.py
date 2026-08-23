"""Exact averages of region statistics over all sign patterns of a parameter vector.

For magnitudes |theta| (all non-zero) the *sign-flip average* of a statistic S is

    (1 / 2^P)  sum_{eps in {+1,-1}^P}  S(eps * |theta|).

If the expectation E[S] under independent parameters with even densities were
independent of the densities, this average would have to be the same for
(almost) every |theta|.  Conversely, two magnitude vectors with different
sign-flip averages prove density dependence.

Symmetries used to shrink the enumeration (they leave every statistic invariant):
  * negating the whole output layer (f -> -f): the output bias sign is fixed;
  * when ``fix_first_layer`` is set, the first hidden layer is not flipped at all.
    This is valid when input dim >= first width: an affine change of input
    coordinates turns any full-rank first layer into [I | 0], 0, so the region
    structure does not depend on the first layer at all.
  * (without ``fix_first_layer``) reflecting input coordinate j == flipping
    column j of W_1, so the signs of W_1[0][j] are fixed.
"""
import itertools
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

from .network import ReLUNetwork
from .stats import region_summary, STAT_NAMES


def flip_positions(net, fix_first_layer):
    """Parameters whose sign is enumerated, as (kind, layer, row, col)."""
    pos = []
    L = len(net.weights)
    for k in range(L):
        if fix_first_layer and k == 0:
            continue
        for i, row in enumerate(net.weights[k]):
            for j in range(len(row)):
                if k == 0 and i == 0:
                    continue                 # input reflections
                pos.append(("w", k, i, j))
        if k == L - 1:
            continue                         # output negation fixes the output bias
        for i in range(len(net.biases[k])):
            pos.append(("b", k, i, 0))
    return pos


def with_signs(net, pos, signs):
    W = [[list(r) for r in Wk] for Wk in net.weights]
    b = [list(bk) for bk in net.biases]
    for (kind, k, i, j), s in zip(pos, signs):
        if s:
            if kind == "w":
                W[k][i][j] = -W[k][i][j]
            else:
                b[k][i] = -b[k][i]
    return ReLUNetwork(W, b)


_G = {}


def _init(net, pos, adjacency, generic, extra):
    _G.update(net=net, pos=pos, adjacency=adjacency, generic=generic, extra=extra)


def _work(chunk):
    out = []
    for signs in chunk:
        n2 = with_signs(_G["net"], _G["pos"], signs)
        s = region_summary(n2, _G["adjacency"], _G["generic"])
        if _G["extra"] is not None:
            s.update(_G["extra"](n2))
        out.append((signs, s))
    return out


def magnitudes(net):
    return ReLUNetwork([[[abs(v) for v in r] for r in Wk] for Wk in net.weights],
                       [[abs(v) for v in bk] for bk in net.biases])


def sign_flip_averages(net, fix_first_layer=False, adjacency="facet", generic=False,
                       jobs=1, extra=None, progress=None, chunk=32):
    """Exact sign-flip averages of all statistics of ``region_summary`` (plus ``extra``).

    Returns (averages: dict name -> Fraction, per_pattern: list of (signs, stats), P).
    ``extra(net) -> dict`` may add further statistics (e.g. an independent verification).
    """
    net = magnitudes(net)
    pos = flip_positions(net, fix_first_layer)
    for (kind, k, i, j) in pos:
        v = net.weights[k][i][j] if kind == "w" else net.biases[k][i]
        if v == 0:
            raise ValueError("flipped parameter %s[%d][%d][%d] is zero; all flipped parameters must be non-zero"
                             % (kind, k, i, j))
    P = len(pos)
    configs = list(itertools.product((0, 1), repeat=P))
    chunks = [configs[i:i + chunk] for i in range(0, len(configs), chunk)]
    res = []
    t0 = time.time()
    if jobs > 1:
        with Pool(jobs, initializer=_init, initargs=(net, pos, adjacency, generic, extra)) as pool:
            for r in pool.imap_unordered(_work, chunks):
                res.extend(r)
                if progress:
                    progress(len(res), len(configs), time.time() - t0)
    else:
        _init(net, pos, adjacency, generic, extra)
        for ch in chunks:
            res.extend(_work(ch))
            if progress:
                progress(len(res), len(configs), time.time() - t0)
    N = len(res)
    names = list(res[0][1].keys())
    avg = {n: Fraction(sum(s[n] for _, s in res), N) for n in names}
    return avg, res, P
