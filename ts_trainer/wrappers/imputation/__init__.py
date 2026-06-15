"""Imputation wrappers."""

from .point import PointImputationModule
from .probabilistic import ProbImputationModule

__all__ = [
    "PointImputationModule",
    "ProbImputationModule",
]
