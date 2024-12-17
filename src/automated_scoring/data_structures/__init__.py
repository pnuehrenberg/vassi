"""
Data structures for trajectories.
"""

from .collection import InstanceCollection
from .timestamped_collection import TimestampedInstanceCollection
from .trajectory import Trajectory

__all__ = [
    "InstanceCollection",
    "TimestampedInstanceCollection",
    "Trajectory",
]
