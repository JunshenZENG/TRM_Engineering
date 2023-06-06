import numpy as np
from abc import abstractmethod
from copy import deepcopy
from typing import List, Union

from VaR.Engine.shift_calculation import ShiftCalculation


class ScenarioPnLCalculation:
    """ Base class for scenario PnL calculation. """

    def __init__(self, shift_calculation_object: ShiftCalculation, sensitivity_vector: Union[float, List[float]]):
        self._shift_calculation_object = shift_calculation_object
        self._sensitivity_vector = np.array(sensitivity_vector)

        self._scenario_pnl_matrix = None

    @abstractmethod
    def get_scenario_pnl(self):
        pass

    def get_scenario_pnl_matrix(self):
        if self._scenario_pnl_matrix is None:
            self.get_scenario_pnl()
        return self._scenario_pnl_matrix

    def set_scenario_pnl_matrix(self, scenario_pnl_matrix: np.ndarray):
        self._scenario_pnl_matrix = scenario_pnl_matrix


class FXScenarioPnLCalculation(ScenarioPnLCalculation):
    """ Class for FX scenario PnL calculation. """

    def __init__(self, shift_calculation_object: ShiftCalculation, sensitivity_vector: Union[float, List[float]]):
        super().__init__(shift_calculation_object=shift_calculation_object, sensitivity_vector=sensitivity_vector)

    def get_scenario_pnl(self):
        # calculate the scenario PnL
        original_shifted_market_rate = deepcopy(self._shift_calculation_object.get_shifted_market_rates_matrix())
        # element-wise multiplication
        scenario_pnl_matrix = (np.exp(original_shifted_market_rate) - 1) * self._sensitivity_vector

        # set the scenario PnL ndarray
        self.set_scenario_pnl_matrix(scenario_pnl_matrix)


class IRScenarioPnLCalculation(ScenarioPnLCalculation):
    """ Class for IR scenario PnL calculation. """

    def __init__(self, shift_calculation_object: ShiftCalculation, sensitivity_vector: Union[float, List[float]]):
        super().__init__(shift_calculation_object=shift_calculation_object, sensitivity_vector=sensitivity_vector)

    def get_scenario_pnl(self):
        # calculate the scenario PnL
        original_shifted_market_rate = deepcopy(self._shift_calculation_object.get_shifted_market_rates_matrix())
        # element-wise multiplication
        scenario_pnl_matrix = original_shifted_market_rate * 100 * self._sensitivity_vector

        # set the scenario PnL ndarray
        self.set_scenario_pnl_matrix(scenario_pnl_matrix)
