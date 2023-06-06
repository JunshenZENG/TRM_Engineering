import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Union


class BlackScholesModel:
    def __init__(self, parameters: Dict[str, float]):
        """ Initialize Black-Scholes model

        Args:
            parameters: A dictionary of parameters or an instance of ReadExcelInput
            If parameters is a dictionary, it should contain the following keys:
                'stock_price': float
                'strike_price': float
                'risk_free_rate': float
                'time_to_maturity': float
                'volatility': float

        """
        self.parameters = parameters

        self._stock_price = self.parameters['S0']
        self._strike_price = self.parameters['K']
        self._risk_free_rate = self.parameters['r']
        self._time_to_maturity = self.parameters['T']
        self._volatility = self.parameters['sigma']

        # Initialize d1 and d2
        self.d1_value = None
        self.d2_value = None

        # Initialize call and put option prices
        self.call_option = None
        self.put_option = None

        # Greeks
        self.call_delta_value = None
        self.put_delta_value = None
        self.gamma_value = None
        self.vega_value = None
        self.call_theta_value = None
        self.put_theta_value = None
        self.call_rho_value = None
        self.put_rho_value = None

    @staticmethod
    def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate d1
        return (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate d2
        return (np.log(S/K) + (r - sigma**2/2)*T)/(sigma*np.sqrt(T))

    @staticmethod
    def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate call option price
        return S*norm.cdf(BlackScholesModel.d1(S, K, r, sigma, T)) - \
               K*np.exp(-r*T)*norm.cdf(BlackScholesModel.d2(S, K, r, sigma, T))

    @staticmethod
    def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate put option price
        return -S*norm.cdf(-BlackScholesModel.d1(S, K, r, sigma, T)) + \
               K*np.exp(-r*T)*norm.cdf(-BlackScholesModel.d2(S, K, r, sigma, T))

    @staticmethod
    def call_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate call delta
        return norm.cdf(BlackScholesModel.d1(S, K, r, sigma, T))

    @staticmethod
    def put_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate put delta
        return norm.cdf(BlackScholesModel.d1(S, K, r, sigma, T)) - 1

    @staticmethod
    def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate gamma
        return norm.pdf(BlackScholesModel.d1(S, K, r, sigma, T))/(S*sigma*np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate vega
        return S*np.sqrt(T)*norm.pdf(BlackScholesModel.d1(S, K, r, sigma, T))

    @staticmethod
    def call_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate call theta
        return -(S*norm.pdf(BlackScholesModel.d1(S, K, r, sigma, T))*sigma)/(2*np.sqrt(T)) - \
               r*K*np.exp(-r*T)*norm.cdf(BlackScholesModel.d2(S, K, r, sigma, T))

    @staticmethod
    def put_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate put theta
        return -(S*norm.pdf(BlackScholesModel.d1(S, K, r, sigma, T))*sigma)/(2*np.sqrt(T)) + \
               r*K*np.exp(-r*T)*norm.cdf(-BlackScholesModel.d2(S, K, r, sigma, T))

    @staticmethod
    def call_rho(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate call rho
        return K*T*np.exp(-r*T)*norm.cdf(BlackScholesModel.d2(S, K, r, sigma, T))

    @staticmethod
    def put_rho(S: float, K: float, r: float, sigma: float, T: float) -> float:
        # Calculate put rho
        return -K*T*np.exp(-r*T)*norm.cdf(-BlackScholesModel.d2(S, K, r, sigma, T))

    def run(self):
        """
        Run the Black-Scholes model
        :return: None
        """
        # Calculate d1 and d2
        self.d1_value = self.d1(self._stock_price, self._strike_price, self._risk_free_rate,
                                self._volatility, self._time_to_maturity)
        self.d2_value = self.d2(self._stock_price, self._strike_price, self._risk_free_rate,
                                self._volatility, self._time_to_maturity)

        # Calculate option prices
        self.call_option = self.call_price(self._stock_price, self._strike_price, self._risk_free_rate,
                                           self._volatility, self._time_to_maturity)

        self.put_option = self.put_price(self._stock_price, self._strike_price, self._risk_free_rate,
                                         self._volatility, self._time_to_maturity)

        # Calculate Greeks
        self.call_delta_value = self.call_delta(self._stock_price, self._strike_price, self._risk_free_rate,
                                                self._volatility, self._time_to_maturity)
        self.put_delta_value = self.put_delta(self._stock_price, self._strike_price, self._risk_free_rate,
                                              self._volatility, self._time_to_maturity)
        self.gamma_value = self.gamma(self._stock_price, self._strike_price, self._risk_free_rate,
                                      self._volatility, self._time_to_maturity)
        self.vega_value = self.vega(self._stock_price, self._strike_price, self._risk_free_rate,
                                    self._volatility, self._time_to_maturity)
        self.call_theta_value = self.call_theta(self._stock_price, self._strike_price, self._risk_free_rate,
                                                self._volatility, self._time_to_maturity)
        self.put_theta_value = self.put_theta(self._stock_price, self._strike_price, self._risk_free_rate,
                                              self._volatility, self._time_to_maturity)
        self.call_rho_value = self.call_rho(self._stock_price, self._strike_price, self._risk_free_rate,
                                            self._volatility, self._time_to_maturity)
        self.put_rho_value = self.put_rho(self._stock_price, self._strike_price, self._risk_free_rate,
                                          self._volatility, self._time_to_maturity)

    # ------------------------------ Getters ------------------------------ #
    def get_stock_price(self) -> float:
        return self._stock_price

    def get_strike_price(self) -> float:
        return self._strike_price

    def get_risk_free_rate(self) -> float:
        return self._risk_free_rate

    def get_volatility(self) -> float:
        return self._volatility

    def get_time_to_maturity(self) -> float:
        return self._time_to_maturity

    def get_call_option(self) -> float:
        if self.call_option is None:
            self.run()
        return self.call_option

    def get_put_option(self) -> float:
        if self.put_option is None:
            self.run()
        return self.put_option

    # ------------------------------ Setters ------------------------------ #
    def set_stock_price(self, stock_price: float) -> None:
        self._stock_price = stock_price
        self.d1_value = None
        self.d2_value = None
        self.call_option = None
        self.put_option = None
        self.call_delta_value = None
        self.put_delta_value = None
        self.gamma_value = None
        self.vega_value = None
        self.call_theta_value = None
        self.put_theta_value = None
        self.call_rho_value = None
        self.put_rho_value = None

    def set_strike_price(self, strike_price: float) -> None:
        self._strike_price = strike_price
        self.d1_value = None
        self.d2_value = None
        self.call_option = None
        self.put_option = None
        self.call_delta_value = None
        self.put_delta_value = None
        self.gamma_value = None
        self.vega_value = None
        self.call_theta_value = None
        self.put_theta_value = None
        self.call_rho_value = None
        self.put_rho_value = None

    def set_risk_free_rate(self, risk_free_rate: float) -> None:
        self._risk_free_rate = risk_free_rate
        self.d1_value = None
        self.d2_value = None
        self.call_option = None
        self.put_option = None
        self.call_delta_value = None
        self.put_delta_value = None
        self.gamma_value = None
        self.vega_value = None
        self.call_theta_value = None
        self.put_theta_value = None
        self.call_rho_value = None
        self.put_rho_value = None

    def set_volatility(self, volatility: float) -> None:
        self._volatility = volatility
        self.d1_value = None
        self.d2_value = None
        self.call_option = None
        self.put_option = None
        self.call_delta_value = None
        self.put_delta_value = None
        self.gamma_value = None
        self.vega_value = None
        self.call_theta_value = None
        self.put_theta_value = None
        self.call_rho_value = None
        self.put_rho_value = None

    def set_time_to_maturity(self, time_to_maturity: float) -> None:
        self._time_to_maturity = time_to_maturity
        self.d1_value = None
        self.d2_value = None
        self.call_option = None
        self.put_option = None
        self.call_delta_value = None
        self.put_delta_value = None
        self.gamma_value = None
        self.vega_value = None
        self.call_theta_value = None
        self.put_theta_value = None
        self.call_rho_value = None
        self.put_rho_value = None

    # ------------------------------ Printers ------------------------------ #
    def print_BS_results(self) -> pd.DataFrame:
        """
        Print the results of the Black-Scholes model to pandas DataFrame
        :return: None
        """

        # Create pandas DataFrame
        df = pd.DataFrame(columns=['Call', 'Put'],
                          index=['d1', 'd2', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])

        # Fill DataFrame
        df.loc['d1', 'Call'] = self.d1_value
        df.loc['d1', 'Put'] = self.d1_value
        df.loc['d2', 'Call'] = self.d2_value
        df.loc['d2', 'Put'] = self.d2_value
        df.loc['Price', 'Call'] = self.call_option
        df.loc['Price', 'Put'] = self.put_option
        df.loc['Delta', 'Call'] = self.call_delta_value
        df.loc['Delta', 'Put'] = self.put_delta_value
        df.loc['Gamma', 'Call'] = self.gamma_value
        df.loc['Gamma', 'Put'] = self.gamma_value
        df.loc['Vega', 'Call'] = self.vega_value
        df.loc['Vega', 'Put'] = self.vega_value
        df.loc['Theta', 'Call'] = self.call_theta_value
        df.loc['Theta', 'Put'] = self.put_theta_value
        df.loc['Rho', 'Call'] = self.call_rho_value
        df.loc['Rho', 'Put'] = self.put_rho_value

        # Return DataFrame
        return df

