import pandas as pd
import numpy as np
import os
from copy import deepcopy
from typing import List, Dict, Tuple, Union
from pathlib import Path
from datetime import date
from VaR.data_processing.excel_data_transfer_object import VaRCalculationInput


class VaRExcelReader:
    """
    This class reads the input data from an excel file.
    """
    def __init__(self, input_folder: str = "VaR/input", file_name: str = "var_input_data.xlsx"):

        full_path = os.path.join(input_folder, file_name)
        self._input_data = pd.read_excel(full_path, sheet_name=None)

        self.var_calculation_data = None
        self.market_rate_array = None

    def get_var_calculation_input(self, valuation_date: date, portfolio_name: str):
        # load the parameters for all asset_id in the portfolio:
        asset_parameters = self._read_parameters(valuation_date=valuation_date, portfolio_name=portfolio_name)
        # new a 2d-array to store the market rate for all asset_id in the portfolio:
        market_rate_ndarray = None
        # load the historical market rate for all asset_id in the portfolio and transfer the dataframe to np.ndarray:
        for asset_id, asset_id_parameters in asset_parameters.items():
            market_rate_df = self._read_historical_market_rate(valuation_date=valuation_date, asset_id=asset_id)
            asset_parameters[asset_id]['market_rate'] = market_rate_df['market_rate'].to_numpy()
            if market_rate_ndarray is None:
                market_rate_ndarray = market_rate_df['market_rate'].to_numpy()
            else:
                market_rate_ndarray = np.vstack((market_rate_ndarray, market_rate_df['market_rate'].to_numpy())).T

        self.var_calculation_data = asset_parameters
        self.market_rate_array = market_rate_ndarray

        return VaRCalculationInput(parameters_dictionary=asset_parameters, market_rate_matrix=market_rate_ndarray)

    def _read_parameters(self, valuation_date: date, portfolio_name: str):
        var_parameters_df = self._input_data['var_parameter']
        # transfer the element in 'asset_id' and 'portfolio' to lower case
        var_parameters_df['asset_id'] = var_parameters_df['asset_id'].str.lower()
        var_parameters_df['portfolio'] = var_parameters_df['portfolio'].str.lower()
        # transfer the element in 'valuation_date' to date
        var_parameters_df['valuation_date'] = pd.to_datetime(var_parameters_df['valuation_date']).dt.date
        # filter the data
        var_parameters_entry = var_parameters_df[(var_parameters_df['valuation_date'] == valuation_date) &
                                                 (var_parameters_df['portfolio'] == portfolio_name.lower())]

        # check if the data is valid
        if var_parameters_entry.shape[0] == 0:
            raise ValueError(f"The input data for {portfolio_name} is not valid!")
        # get the list of asset_id
        asset_id_list = var_parameters_entry['asset_id'].unique().tolist()
        # get the parameters for each asset_id
        asset_parameters_dict = dict()
        for asset_id in asset_id_list:
            temp_parameter = var_parameters_entry[var_parameters_entry['asset_id'] == asset_id]
            asset_parameters_dict[asset_id] = {
                'portfolio': portfolio_name.lower(),
                'risk_type': temp_parameter['risk_type'].values[0],
                'valuation_date': valuation_date,
                'spot_portfolio_value': temp_parameter['spot_portfolio_value'].values[0],
                'horizon': temp_parameter['horizon'].values[0]
            }
        return asset_parameters_dict

    def _read_historical_market_rate(self, valuation_date: date, asset_id: str):
        var_historical_market_rate_df = self._input_data['market_rate_historical_data']
        # transfer the element in 'asset_id' to lower case
        var_historical_market_rate_df['asset_id'] = var_historical_market_rate_df['asset_id'].str.lower()
        # transfer the element in 'valuation_date' to date
        var_historical_market_rate_df['date'] = \
            pd.to_datetime(var_historical_market_rate_df['date']).dt.date
        # filter the data
        var_historical_market_rate_data = \
            var_historical_market_rate_df[(var_historical_market_rate_df['date'] <= valuation_date) &
                                          (var_historical_market_rate_df['asset_id'] == asset_id.lower())]
        # check if the data is valid
        if var_historical_market_rate_data.shape[0] == 0:
            raise ValueError(f"The input data for {asset_id} is not valid!")
        # sort the data by valuation_date from the latest to the oldest
        var_historical_market_rate_data = \
            var_historical_market_rate_data.sort_values(by='date', ascending=False)
        return var_historical_market_rate_data
