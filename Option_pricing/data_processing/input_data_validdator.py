import pandas as pd


class InputDataValidator:
    def __init__(self, input_data: pd.DataFrame):
        self._input_data = input_data
        self._allow_column = ['trade_date', 'expiry', 's0', 'k', 'time_to_maturity', 'r', 'r_cont',
                              'div_yield_cont', 'vol', 'forward_pricing']

    def validate(self):
        for col in self._input_data.columns:
            if col.lower() not in self._allow_column:
                raise ValueError(f"column {col} is not allowed")
            elif col.lower() in ['s0', 'k', 'time_to_maturity', 'r_cont', 'vol', 'forward_pricing']:
                assert self._input_data[col.lower()].isnull().sum() == 0, f"column {col} has null value"