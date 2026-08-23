"""Exact (rational-arithmetic) counting of activation and linear regions of ReLU networks."""
from .numbers import PrecisionOverflowError, to_fraction, set_max_bits
from .lp import LPError
from .network import ReLUNetwork
from .regions import RegionCounter, ActivationRegion, TooManyRegionsError
from .stats import region_summary, STAT_NAMES
from .signflip import sign_flip_averages

__all__ = [
    "PrecisionOverflowError", "to_fraction", "set_max_bits", "LPError",
    "ReLUNetwork", "RegionCounter", "ActivationRegion", "TooManyRegionsError",
    "region_summary", "STAT_NAMES", "sign_flip_averages",
]
