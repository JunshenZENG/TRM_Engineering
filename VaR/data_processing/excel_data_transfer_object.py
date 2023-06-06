from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from datetime import date
import numpy as np


@dataclass
class VaRCalculationInput:

    parameters_dictionary: Dict[str, Dict[str, Union[str, date, int, float, np.ndarray]]]
    market_rate_matrix: np.ndarray


