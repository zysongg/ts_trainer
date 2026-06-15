"""Lightning module wrappers for ts_model models.

Structure:
    wrappers/
    ├── forecasting/
    │   ├── point.py          # PointForecastModule
    │   └── probabilistic.py  # ProbForecastModule
    └── imputation/
        ├── point.py          # PointImputationModule
        └── probabilistic.py  # ProbImputationModule
"""

from .forecasting import PointForecastModule, ProbForecastModule
from .imputation import PointImputationModule, ProbImputationModule

__all__ = [
    # Forecasting
    "PointForecastModule",
    "ProbForecastModule",
    # Imputation
    "PointImputationModule",
    "ProbImputationModule",
]
