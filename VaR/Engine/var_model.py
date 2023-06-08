import numpy as np
from abc import abstractmethod
from copy import deepcopy
from typing import List, Union

from VaR.Engine.scenario_pnl_calculation import ScenarioPnLCalculation, \
    FXScenarioPnLCalculation, IRScenarioPnLCalculation


class HistoricalVaRModel:

    def __init__(self, scenario_pnl_calculation_object_list: List[ScenarioPnLCalculation]):

        self._scenario_pnl_calculation_object_list = scenario_pnl_calculation_object_list
        self._total_asset_scenario_pnl_vector = None
        self._var = None

    def get_total_asset_scenario_pnl(self):
        scenario_pnl_matrix = None
        for scenario_pnl_calculation_object in self._scenario_pnl_calculation_object_list:
            if scenario_pnl_matrix is None:
                scenario_pnl_matrix = deepcopy(scenario_pnl_calculation_object.get_scenario_pnl_matrix())
            else:
                # concatenate the scenario PnL matrix to the right
                scenario_pnl_matrix = np.concatenate((scenario_pnl_matrix,
                                                      scenario_pnl_calculation_object.get_scenario_pnl_matrix()),
                                                     axis=1)

        # for each row, sum up the elements
        total_asset_scenario_pnl = np.sum(scenario_pnl_matrix, axis=1)
        self._total_asset_scenario_pnl_vector = total_asset_scenario_pnl

    def calculated_var(self):
        if self._total_asset_scenario_pnl_vector is None:
            self.get_total_asset_scenario_pnl()

        # sort the total asset scenario PnL vector from smallest to largest
        sorted_total_asset_scenario_pnl_vector = np.sort(self._total_asset_scenario_pnl_vector)

        # calculate the VaR by 0.4 * second-smallest element + 0.6 * third-smallest element
        self._var = 0.4 * sorted_total_asset_scenario_pnl_vector[1] + 0.6 * sorted_total_asset_scenario_pnl_vector[2]

        return self._var
