"""SPSA for layer-wise training of deep networks (Wulff, Schuecker & Bauckhage, ICANN 2018)."""
from . import datasets, gd, layerwise, mlp, spsa
from .device import get_device
from .layerwise import ORDERS, SPSAConfig, TrainConfig

__all__ = ["datasets", "gd", "layerwise", "mlp", "spsa", "get_device",
           "ORDERS", "SPSAConfig", "TrainConfig"]
__version__ = "1.0.0"
__author__ = "Ananth Shyam, Rama Narasimha, Mahan Bhimireddy"
