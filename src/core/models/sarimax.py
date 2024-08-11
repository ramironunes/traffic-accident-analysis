# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:13:54
# @Info:   Implementation of SarimaxModel for predicting traffic accidents
# ============================================================================


import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple


class SarimaxModel:
    """
    A class to encapsulate the SARIMAX model for predicting traffic accidents.

    This class provides functionality to train a SARIMAX model on preprocessed
    traffic accident data with an exogenous variable (traffic volume) and generate forecasts.

    Attributes:
    -----------
    data : pd.DataFrame
        The DataFrame containing the preprocessed data for training.
    exog : pd.Series
        The exogenous data to be used in the SARIMAX model.
    model : SARIMAX
        The SARIMAX model instance.
    sarimax_model : SARIMAXResultsWrapper or None
        The fitted SARIMAX model after training.
    test_data : pd.Series or None
        The testing portion of the data used for forecasting.

    Methods:
    --------
    train(order: Tuple[int, int, int] = (1, 1, 1),
          seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52)):
        Trains the SARIMAX model using the specified parameters.

    forecast(steps: int = None, exog: pd.Series = None) -> Tuple[pd.Series, pd.DataFrame]:
        Generates forecasted values and confidence intervals based on the
        trained SARIMAX model.
    """

    def __init__(self, data: pd.DataFrame, exog: pd.Series):
        """
        Initialize the model with processed data and exogenous variable.

        :param data: DataFrame containing preprocessed data.
        :param exog: Series containing the exogenous variable (e.g., traffic volume).
        """
        self.data = data
        self.exog = exog
        self.model = None
        self.sarimax_model = None
        self.test_data = None

    def train(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52),
    ):
        """
        Train the SARIMAX model on the processed data.

        :param order: The (p, d, q) order of the model for the number of AR
                      parameters, differences, and MA parameters.
        :param seasonal_order: The (P, D, Q, m) order of the seasonal
                               component.
        :return: The trained SARIMAX model.
        """
        # Split the data into training and testing sets
        train_data = self.data["accidents"][: int(0.8 * len(self.data))]
        self.test_data = self.data["accidents"][int(0.8 * len(self.data)) :]

        train_exog = self.exog[: int(0.8 * len(self.exog))]

        # Manually specify SARIMAX parameters
        self.model = SARIMAX(
            train_data,
            exog=train_exog,
            order=order,  # ARIMA parameters (p, d, q)
            seasonal_order=seasonal_order,  # Seasonal parameters (P, D, Q, m)
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        # Train the SARIMAX model
        self.sarimax_model = self.model.fit(disp=False)

        return self.sarimax_model

    def forecast(
        self, steps: int = None, exog: pd.Series = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Forecast future values using the trained SARIMAX model.

        :param steps: Number of steps to forecast. Defaults to the length
                      of test data.
        :param exog: Exogenous data to use in the forecast (e.g., traffic volume for future periods).
        :return: A tuple containing forecasted values and confidence
                 intervals.
        """
        if steps is None:
            steps = len(self.test_data)

        predictions = self.sarimax_model.get_forecast(steps=steps, exog=exog)
        pred_conf_int = predictions.conf_int()

        return predictions.predicted_mean, pred_conf_int
