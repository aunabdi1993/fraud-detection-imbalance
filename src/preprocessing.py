"""
preprocessing.py — Optional feature engineering.

Maps to: Chapter 3 sec 3.3.
STATUS: STUB — implement Week 13-14. Keep deliberately minimal.

The ULB features V1-V28 are already PCA components, so classical fraud feature
engineering (RFM, velocity, merchant aggregates) is mostly impossible: the raw
transaction fields were destroyed by the anonymisation. State this plainly
rather than inventing features that cannot be justified.

Defensible transforms:
  - log1p(Amount)          already handled in data_loader
  - hour_of_day            Time mod 86400, then sin/cos encoding so 23:00 and
                           00:00 are adjacent. Fraud rate varies by hour in
                           this dataset — check in EDA before including.
  - amount_zscore_by_hour  requires care: compute statistics on TRAINING data
                           only, or it leaks.

Anything more elaborate needs a justification in Chapter 3 and an ablation in
Chapter 5. Feature engineering is not this dissertation's contribution — the
technique comparison is. Do not let it expand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclical_hour(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Encode hour-of-day as sin/cos. Requires the raw Time column."""
    raise NotImplementedError("Implement in Week 13-14")
