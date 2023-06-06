from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from datetime import date
import numpy as np


@dataclass
class VaRCalculationInput:

    parameters_dictionary: Dict[str, Dict[str, Union[str, date, int, float]]]
    market_rate_matrix: np.ndarray
    horizon: int
    risk_type: str
    spot_portfolio_value_list: List[float] = None
