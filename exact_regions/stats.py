"""Summary statistics of the region structure of one network."""
from .regions import RegionCounter

STAT_NAMES = ("activation", "linear", "nonconst_linear", "const_linear",
              "const_activation", "distinct_maps", "distinct_const_maps")


def _is_constant(affine):
    a, _ = affine
    return all(v == 0 for v in a)


def region_summary(net, adjacency="facet", generic=False):
    """Return a dict of exact integer statistics:

    activation           number of (full-dimensional) activation regions
    linear               number of linear regions
    nonconst_linear      linear regions on which the network is non-constant
    const_linear         linear regions on which the network is constant
    const_activation     activation regions on which the network is constant
    distinct_maps        number of distinct affine maps realised
    distinct_const_maps  number of distinct constant values realised
    """
    c = RegionCounter(net)
    acts = c.activation_regions()
    lins = c.linear_regions(adjacency, generic)
    maps = {r.output for r in acts}
    const_lin = sum(1 for L in lins if _is_constant(L[0].output))
    return {
        "activation": len(acts),
        "linear": len(lins),
        "nonconst_linear": len(lins) - const_lin,
        "const_linear": const_lin,
        "const_activation": sum(1 for r in acts if _is_constant(r.output)),
        "distinct_maps": len(maps),
        "distinct_const_maps": sum(1 for m in maps if _is_constant(m)),
    }
