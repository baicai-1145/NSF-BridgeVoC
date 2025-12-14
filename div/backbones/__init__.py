from .shared import BackboneRegistry
from .bcd import BCD
try:
    from .bcd_nsf_bridge import NsfBcdBridge
except ModuleNotFoundError as e:
    # Allow importing lightweight backbones (e.g. BCD) without training-only deps.
    # NsfBcdBridge requires pytorch_lightning via div.data_module.
    if getattr(e, "name", None) != "pytorch_lightning":
        raise
    NsfBcdBridge = None

__all__ = [
    'BackboneRegistry',
    'BCD',
]

if NsfBcdBridge is not None:
    __all__.append('NsfBcdBridge')
