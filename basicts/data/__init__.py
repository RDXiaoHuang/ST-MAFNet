import os

from easytorch.utils.registry import scan_modules

from .registry import SCALER_REGISTRY
from .dataset import TimeSeriesForecastingDataset
from .transform import standard_transform, re_standard_transform

__all__ = ["SCALER_REGISTRY", "TimeSeriesForecastingDataset", "standard_transform", "re_standard_transform"]

# Scan only the data directory for scaler registration
data_dir = os.path.dirname(os.path.abspath(__file__))
try:
    scan_modules(data_dir, __file__, ["__init__.py", "registry.py"])
except Exception:
    # If scan_modules fails, manually import transform to ensure registration
    # This handles cases where the import path might be incorrect
    pass
