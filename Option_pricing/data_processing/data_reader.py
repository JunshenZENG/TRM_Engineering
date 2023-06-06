import pandas as pd
import numpy as np
import os
from copy import deepcopy
from typing import List, Dict, Tuple, Union
from pathlib import Path
from Option_pricing.data_processing.input_data_validdator import InputDataValidator


class OptionPricingExcelReader:
    """
    This class reads the input data from an excel file.
    """
    def __init__(self, input_folder: str = "input", file_name: str = "option_pricing_input.xlsx"):

        full_path = os.path.join(input_folder, file_name)
        self._input_data = pd.read_excel(full_path, sheet_name="parameters")

        validator = InputDataValidator(input_data=self._input_data)
        validator.validate()

        self.input_data_dict = None
        self.forward_pricing = None
        self.div_yield_cont = None

    def get_input_data(self) -> Dict[str, Union[List, float]]:
        """
        This function returns the input data for normal BSM model.
        """
        input_data = deepcopy(self._input_data)
        # convert the column names to lower case
        input_data.columns = [col.lower() for col in input_data.columns]

        input_data_dict = dict()
        input_data_dict['S0'] = input_data['s0'].values[0]
        input_data_dict['K'] = input_data['k'].values[0]
        input_data_dict['T'] = input_data['time_to_maturity'].values[0]
        input_data_dict['r'] = input_data['r_cont'].values[0]
        input_data_dict['sigma'] = input_data['vol'].values[0]
        input_data_dict['forward_pricing'] = input_data['forward_pricing'].values[0]
        input_data_dict['div_yield_cont'] = input_data['div_yield_cont'].values[0]

        self.input_data_dict = input_data_dict
        self.forward_pricing = input_data_dict['forward_pricing']
        self.div_yield_cont = input_data_dict['div_yield_cont']

        return input_data_dict







