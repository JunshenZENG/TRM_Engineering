from Option_pricing.Engine.black_scholes_model import BlackScholesModel
import pandas as pd
import numpy as np
import os
from pandas import ExcelWriter
import datetime
from typing import List, Dict, Tuple, Union


class BSUnitTest:
    def __init__(self, input_folder: str = "test/Data", file_name: str = "bs_benchmark_results.xlsx"):
        self.file_name = file_name
        self.input_folder = input_folder

        self._benchmark_results = self.read_bm_data_excel_input()

    def read_bm_data_excel_input(self):
        full_path = os.path.join(self.input_folder, self.file_name)
        df = pd.read_excel(full_path, sheet_name="spot_price")
        df = df.set_index('test_id')
        benchmark_results = df.to_dict(orient='index')
        return benchmark_results

    def run_unit_test(self):
        for test_id, test_data in self._benchmark_results.items():
            bs_model = BlackScholesModel(parameters=test_data)
            bs_model.run()

            # test d1
            assert np.isclose(bs_model.d1_value, test_data['d1'], rtol=1e-3), \
                f"test_id: {test_id}, d1: {bs_model.d1_value}, expected: {test_data['d1']}"

            # test d2
            assert np.isclose(bs_model.d2_value, test_data['d2'], rtol=1e-3), \
                f"test_id: {test_id}, d2: {bs_model.d2_value}, expected: {test_data['d2']}"

            # test call price
            assert np.isclose(bs_model.call_option, test_data['call_price'], rtol=1e-3), \
                f"test_id: {test_id}, call_price: {bs_model.call_option}, expected: {test_data['call_price']}"

            # test put price
            assert np.isclose(bs_model.put_option, test_data['put_price'], rtol=1e-3), \
                f"test_id: {test_id}, put_price: {bs_model.put_option}, expected: {test_data['put_price']}"

        print("All unit tests passed.")
