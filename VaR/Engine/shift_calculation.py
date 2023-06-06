import numpy as np
from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Tuple, Union


class ShiftCalculation:
    """ Base class for shift calculation. """

    def __init__(self, market_rate_matrix: np.ndarray, horizon: List[int]):
        self._market_rate_matrix = market_rate_matrix
        self._horizon = np.array(horizon)

        self._shifted_market_rate_matrix = None
        self._scenario_value_matrix = None

    @abstractmethod
    def get_shifted_market_rate(self):
        pass

    @abstractmethod
    def get_scenario_value(self):
        pass

    def get_shifted_market_rates_matrix(self):
        if self._shifted_market_rate_matrix is None:
            self.get_shifted_market_rate()
        return self._shifted_market_rate_matrix

    def get_scenario_value_matrix(self):
        if self._scenario_value_matrix is None:
            self.get_scenario_value()
        return self._scenario_value_matrix

    def set_shifted_market_rates_matrix(self, shifted_market_rate: np.ndarray):
        self._shifted_market_rate_matrix = shifted_market_rate

    def set_scenario_value_matrix(self, scenario_value: np.ndarray):
        self._scenario_value_matrix = scenario_value


class AbsoluteShiftCalculation(ShiftCalculation):
    """ Class for absolute shift calculation. """

    def __init__(self, market_rate_matrix: np.ndarray, horizon: List[int]):
        super().__init__(market_rate_matrix=market_rate_matrix, horizon=horizon)

    def get_shifted_market_rate(self):
        # calculate the absolute return of all the asset
        original_market_rate = deepcopy(self._market_rate_matrix)
        original_horizon = deepcopy(self._horizon)
        shifted_market_rate = original_market_rate[: -1, :] - original_market_rate[1:, :]
        shifted_market_rate = shifted_market_rate * np.sqrt(original_horizon)

        # set the shifted market rate ndarray
        self.set_shifted_market_rates_matrix(shifted_market_rate)

    def get_scenario_value(self):
        original_market_rate = deepcopy(self._market_rate_matrix)
        original_shifted_market_rate = deepcopy(self.get_shifted_market_rates_matrix())
        scenario_value_matrix = original_market_rate[1:, :] + original_shifted_market_rate
        self.set_scenario_value_matrix(scenario_value_matrix)


class LogarithmicShiftCalculation(ShiftCalculation):
    """ Class for logarithmic shift calculation. """

    def __init__(self, market_rate_matrix: np.ndarray, horizon: List[int]):
        super().__init__(market_rate_matrix=market_rate_matrix, horizon=horizon)

    def get_shifted_market_rate(self):
        # calculated the logarithmic return of the market rate
        original_market_rate = deepcopy(self._market_rate_matrix)
        original_horizon = deepcopy(self._horizon)
        shifted_market_rate = np.log(original_market_rate[: -1, :] / original_market_rate[1:, :])
        shifted_market_rate = shifted_market_rate * np.sqrt(original_horizon)

        # set the shifted market rate ndarray
        self.set_shifted_market_rates_matrix(shifted_market_rate)

    def get_scenario_value(self):
        original_market_rate = deepcopy(self._market_rate_matrix)
        original_shifted_market_rate = deepcopy(self.get_shifted_market_rates_matrix())
        scenario_value_matrix = original_market_rate[1:, :] * np.exp(original_shifted_market_rate)
        self.set_scenario_value_matrix(scenario_value_matrix)


class RelativeShiftCalculation(ShiftCalculation):
    """ Class for relative shift calculation. """

    def __init__(self, market_rate_matrix: np.ndarray, horizon: List[int]):
        super().__init__(market_rate_matrix=market_rate_matrix, horizon=horizon)

    def get_shifted_market_rate(self):
        # calculated the relative return of the market rate
        original_market_rate = deepcopy(self._market_rate_matrix)
        original_horizon = deepcopy(self._horizon)
        shifted_market_rate = original_market_rate[: -1, :] / original_market_rate[1:, :] - 1
        shifted_market_rate = shifted_market_rate * np.sqrt(original_horizon)

        # set the shifted market rate ndarray
        self.set_shifted_market_rates_matrix(shifted_market_rate)

    def get_scenario_value(self):
        original_market_rate = deepcopy(self._market_rate_matrix)
        original_shifted_market_rate = deepcopy(self.get_shifted_market_rates_matrix())
        scenario_value_matrix = original_market_rate[1:, :] * (1 + original_shifted_market_rate)
        self.set_scenario_value_matrix(scenario_value_matrix)