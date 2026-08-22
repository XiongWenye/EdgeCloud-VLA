"""LeRobot policy plugin registration for CloudEdge pi0.5."""

from .configuration_cloudedge_pi05 import CloudEdgePI05Config
from .modeling_cloudedge_pi05 import CloudEdgePI05Policy
from .processor_cloudedge_pi05 import SelectCurrentStateProcessorStep

__all__ = [
    "CloudEdgePI05Config",
    "CloudEdgePI05Policy",
    "SelectCurrentStateProcessorStep",
]

