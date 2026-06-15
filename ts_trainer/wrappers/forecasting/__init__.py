"""Forecasting wrappers."""

from .point import PointForecastModule
from .probabilistic import ProbForecastModule

__all__ = [
    "PointForecastModule",
    "ProbForecastModule",
]
