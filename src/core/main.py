# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Main script to train and forecast using SARIMAX model on traffic data
# ============================================================================


import os
import sys
import pandas as pd
import numpy as np

from data_loader import get_file_paths, load_preprocessed_data
from plot_utils import plot_comparison_chart, export_to_excel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates both accident and traffic data to a monthly basis.

    :param data: DataFrame containing the traffic and accident data.
    :return: DataFrame aggregated on a monthly basis.
    """
    print("-" * 50)
    print(df.columns)
    df.set_index("year_month", inplace=True)
    df["accidents"] = pd.to_numeric(df["accidents"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")

    # Aggregate both accidents and traffic volume on a monthly basis
    monthly_data = (
        df.groupby(["br", pd.Grouper(freq="M")])
        .agg(
            {
                "accidents": "sum",  # Sum the accidents for each month
                "traffic_volume": "mean",  # Take the mean traffic volume for each month
            }
        )
        .reset_index()
    )

    return monthly_data


def load_data(base_dir: str) -> pd.DataFrame:
    """
    Load and preprocess accident and toll data, then merge them.

    :param base_dir: Base directory where the data is stored.
    :return: Merged DataFrame containing accident and toll data.
    """
    accident_preprocessed_dir = os.path.join(
        base_dir,
        "traffic-accident-analysis/data/traffic_accidents/all_causes_and_types/preprocessed",
    )
    accident_datasets = [
        "acidentes2018_todas_causas_tipos_mg",
        "acidentes2019_todas_causas_tipos_mg",
        "acidentes2020_todas_causas_tipos_mg",
        "acidentes2021_todas_causas_tipos_mg",
        "acidentes2022_todas_causas_tipos_mg",
        "acidentes2023_todas_causas_tipos_mg",
    ]
    accident_files = get_file_paths(accident_preprocessed_dir, accident_datasets)

    toll_preprocessed_dir = os.path.join(
        base_dir, "traffic-accident-analysis/data/toll_stations/preprocessed"
    )
    toll_datasets = [
        "volume-trafego-praca-pedagio-2018_mg",
        "volume-trafego-praca-pedagio-2019_mg",
        "volume-trafego-praca-pedagio-2020_mg",
        "volume-trafego-praca-pedagio-2021_mg",
        "volume-trafego-praca-pedagio-2022_mg",
        "volume-trafego-praca-pedagio-2023_mg",
    ]
    toll_files = get_file_paths(toll_preprocessed_dir, toll_datasets)

    merged_data = load_preprocessed_data(accident_files, toll_files)
    return merged_data


def calculate_metrics(
    real_accidents: np.ndarray,
    predicted_accidents: np.ndarray,
) -> dict[str, float]:
    """
    Calculate evaluation metrics for model performance.

    :param real_accidents: Array of actual accident numbers.
    :param predicted_accidents: Array of predicted accident numbers.
    :return: Dictionary containing MAE, MAPE and RMSE metrics.
    """
    metrics = {
        "MAE": mean_absolute_error(real_accidents, predicted_accidents),
        "MAPE": np.mean(np.abs((real_accidents - predicted_accidents) / real_accidents))
        * 100,
        "RMSE": np.sqrt(mean_squared_error(real_accidents, predicted_accidents)),
    }
    return metrics


def export_training_testing_data(
    br: str, br_data: pd.DataFrame, forecast_data: pd.DataFrame, output_dir: str
) -> None:
    """
    Export the training and testing data to an Excel file.

    :param br: The BR identifier.
    :param br_data: DataFrame containing the BR's training data.
    :param forecast_data: DataFrame containing the BR's testing data.
    :param output_dir: Directory where the Excel file will be saved.
    """
    output_path = os.path.join(output_dir, f"BR_{br}_training_testing_data.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        br_data.to_excel(writer, sheet_name="Training Data", index=False)
        forecast_data.to_excel(writer, sheet_name="Testing Data", index=False)


def export_forecast_results(
    br: str,
    forecast_data: pd.DataFrame,
    predicted_accidents: pd.Series,
    output_dir: str,
    metrics: dict[str, float],
) -> None:
    """
    Export the actual and predicted data, along with evaluation metrics, to an Excel file.

    :param br: The BR identifier.
    :param forecast_data: DataFrame containing the BR's testing data.
    :param predicted_accidents: Series of predicted accident numbers.
    :param output_dir: Directory where the Excel file will be saved.
    :param metrics: Dictionary containing the evaluation metrics.
    """
    output_path = os.path.join(output_dir, f"BR_{br}_forecast_results_2023.xlsx")
    results_df = forecast_data.copy()
    results_df["Predicted Accidents"] = predicted_accidents
    results_df["Error"] = results_df["accidents"] - results_df["Predicted Accidents"]

    metrics_df = pd.DataFrame([metrics])

    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name="Forecast Results", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)


def train_sarimax_model(br_data: pd.DataFrame, config: dict[str, tuple]) -> SARIMAX:
    """
    Train a SARIMAX model on the provided data.

    :param br_data: DataFrame containing the BR's training data.
    :param config: Dictionary containing the 'order' and 'seasonal_order' configuration.
    :return: Trained SARIMAX model.
    """
    exog_data = br_data["traffic_volume"]

    br_data.loc[:, "accidents"] = pd.to_numeric(br_data["accidents"], errors="coerce")
    exog_data = pd.to_numeric(exog_data, errors="coerce")

    br_data = br_data.loc[br_data["accidents"].notna()]
    exog_data.fillna(0, inplace=True)

    if len(br_data) < 3:
        raise ValueError("Not enough data to train SARIMAX model.")

    sarimax_model = SARIMAX(
        br_data["accidents"].astype(float),
        exog=exog_data.astype(float),
        order=config["order"],
        seasonal_order=config["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return sarimax_model.fit(disp=False)


def generate_forecast(
    br: str, sarimax_result: SARIMAX, forecast_data: pd.DataFrame, output_img_dir: str
) -> None:
    """
    Generate and save the forecast for the specified BR and year 2023.

    :param br: The BR identifier.
    :param sarimax_result: Trained SARIMAX model result.
    :param forecast_data: DataFrame containing the BR's testing data.
    :param output_img_dir: Directory where the output images will be saved.
    """
    exog_forecast = pd.to_numeric(forecast_data["traffic_volume"], errors="coerce")
    exog_forecast.fillna(0, inplace=True)

    try:
        predictions = sarimax_result.get_forecast(
            steps=len(forecast_data), exog=exog_forecast.astype(float)
        )
        predicted_accidents = predictions.predicted_mean

        min_length = min(
            len(forecast_data["accidents"].values), len(predicted_accidents)
        )
        real_accidents = forecast_data["accidents"].values[:min_length]
        predicted_accidents = predicted_accidents[:min_length]

        metrics = calculate_metrics(real_accidents, predicted_accidents)
        print(f"Evaluation Metrics for BR-{br}: {metrics}")

        export_forecast_results(
            br, forecast_data, predicted_accidents, output_img_dir, metrics
        )
        plot_comparison_chart(forecast_data, predicted_accidents, 2023, output_img_dir)
        export_to_excel(forecast_data, predicted_accidents, 2023, output_img_dir)

    except ValueError as e:
        print(f"Failed to forecast for BR-{br} in 2023: {e}")


def perform_rolling_window_validation(
    data: pd.DataFrame,
    initial_train_size: int,
    window_size: int,
    config: dict[str, tuple],
):
    """
    Perform rolling window validation on SARIMAX model.

    :param data: DataFrame containing the time series data.
    :param initial_train_size: Initial size of the training dataset.
    :param window_size: Size of the validation window.
    :param config: Dictionary containing the 'order' and 'seasonal_order' configuration.
    """
    n_splits = (len(data) - initial_train_size) // window_size

    for i in range(n_splits):
        train_data = data.iloc[: initial_train_size + i * window_size]
        test_data = data.iloc[
            initial_train_size + i * window_size : initial_train_size
            + (i + 1) * window_size
        ]

        print("-" * 50)
        print(f"Training on window {i+1}/{n_splits}...")

        try:
            sarimax_result = train_sarimax_model(train_data, config)
            predictions = sarimax_result.get_forecast(
                steps=len(test_data), exog=test_data["traffic_volume"]
            )
            predicted_accidents = predictions.predicted_mean

            metrics = calculate_metrics(
                test_data["accidents"].values, predicted_accidents.values
            )
            print(f"Metrics for window {i+1}: {metrics}")

        except ValueError as e:
            print(f"Failed to train SARIMAX model on window {i+1}: {e}")


def filter_by_km_range(
    data: pd.DataFrame, km_start: float, km_end: float
) -> pd.DataFrame:
    """
    Filter the data to include only records within the specified kilometer range.

    :param data: DataFrame containing the traffic or accident data.
    :param km_start: Start of the kilometer range.
    :param km_end: End of the kilometer range.
    :return: Filtered DataFrame within the specified kilometer range.
    """
    if "km" not in data.columns:
        raise KeyError("The 'km' column is missing from the data.")

    return data[(data["km"] >= km_start) & (data["km"] <= km_end)]


def train_multiple_sarimax_models(
    br_data: pd.DataFrame,
    config_list: list[dict[str, tuple]],
) -> list[SARIMAX]:
    """
    Train SARIMAX models using different configurations and return the results.

    :param br_data: DataFrame containing the BR's training data.
    :param config_list: List of configurations to train the model with.
    :return: List of trained SARIMAX model results.
    """
    results = []
    for config in config_list:
        try:
            result = train_sarimax_model(br_data, config)
            results.append(result)
        except ValueError as e:
            print(f"Failed to train SARIMAX model with config {config}: {e}")
            continue
    return results


def process_br_data(
    monthly_data: pd.DataFrame,
    output_img_dir: str,
    config_list: list[dict[str, tuple]],
) -> None:
    """
    Process each BR's data, train the SARIMAX model, and generate forecasts.

    :param monthly_data: DataFrame containing the monthly traffic data.
    :param output_img_dir: Directory where the output images will be saved.
    :param config_list: List of configurations to train the model with.
    """
    br_list = monthly_data["br"].unique()

    for br in br_list:
        print("-" * 50)
        print(f"Processing BR-{br}...")
        br_data = monthly_data[monthly_data["br"] == br]
        print(f"Number of records for BR-{br}: {len(br_data)}")

        if len(br_data) < 3:
            print(f"Skipping BR-{br} due to insufficient data.")
            continue

        try:
            br_data_filtered = filter_by_km_range(br_data, 100.0, 115.0)
            forecast_data = monthly_data[
                (monthly_data["br"] == br) & (monthly_data["data"].dt.year == 2023)
            ]
            export_training_testing_data(
                br, br_data_filtered, forecast_data, output_img_dir
            )

            for config in config_list:
                sarimax_result = train_sarimax_model(br_data_filtered, config)
                print(f"Model training completed for BR-{br} with config {config}.")
                print("-" * 50)

                if len(forecast_data) == 0:
                    print(
                        f"No data available to forecast for BR-{br} in 2023. Skipping..."
                    )
                    continue

                generate_forecast(br, sarimax_result, forecast_data, output_img_dir)

        except ValueError as e:
            print(f"Failed to train SARIMAX model for BR-{br}: {e}")
            continue


def get_sarimax_configs():
    """
    Parameters:
    order (tuple[int, int, int]): A tuple representing the non-seasonal components (p, d, q).
        - p (int): Number of autoregressive terms. This determines the lagged values
          used as input to the model.
        - d (int): Number of non-seasonal differences required to make the series stationary.
          A higher value indicates more differencing to remove trends.
        - q (int): Number of moving average terms. This models the relationship between
          the observed values and the previous forecast errors.

    seasonal_order (tuple[int, int, int, int]): A tuple representing the seasonal components (P, D, Q, m).
        - P (int): Number of seasonal autoregressive terms. Similar to p but applied to the
          seasonal component of the series.
        - D (int): Number of seasonal differences. Similar to d but applied to the seasonal
          component of the series.
        - Q (int): Number of seasonal moving average terms. Similar to q but applied to the
          seasonal component of the series.
        - m (int): Number of periods in each season. This defines the length of the seasonal cycle,
          e.g., 12 for monthly data with annual seasonality.
    """
    return [
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 52)},
        {"order": (2, 1, 2), "seasonal_order": (2, 1, 2, 52)},
    ]


def main() -> None:
    """
    Main function to load data, train SARIMAX models, and generate forecasts.
    """
    print("-" * 50)
    print("Loading preprocessed data...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    merged_data = load_data(base_dir)

    print("Data loaded and merged successfully.")
    print("Aggregating data on a monthly basis...")

    # monthly_data = aggregate_monthly(merged_data)
    # output_img_dir = os.path.join(base_dir, "traffic-accident-analysis/out/img/core")

    # print("-" * 50)
    # print("Training SARIMAX model on all data...")
    # sarimax_configs = get_sarimax_configs()

    # process_br_data(
    #     monthly_data,
    #     output_img_dir,
    #     sarimax_configs,
    # )

    # print("SARIMAX model training and forecasting completed.")


if __name__ == "__main__":
    main()
