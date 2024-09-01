# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Main script to train and forecast using SARIMAX model on traffic data
# ============================================================================

import pandas as pd
import numpy as np
import os
import sys
import re


from data_loader import get_file_paths, load_preprocessed_data
from plot_utils import plot_comparison_chart, export_to_excel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


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


def aggregate_traffic_volume(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate accident data by BR, praca, and month.

    :param data: DataFrame containing preprocessed accident and toll data.
    :return: Aggregated DataFrame with accident counts and traffic volume.
    """
    required_columns = ["traffic_volume", "accidents", "praca"]

    # Check if the required columns are present in the DataFrame
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' does not exist in the DataFrame.")

    # Group by 'br', 'praca', and 'year_month', counting the accidents
    aggregated_data = (
        data.groupby(["br", "praca", "year_month"])
        .agg({"accidents": "count", "traffic_volume": "first"})
        .reset_index()
    )

    return aggregated_data


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


def sanitize_filename(name: str) -> str:
    """
    Replace spaces, commas, and other unsafe characters in filenames with underscores.
    """
    return re.sub(r"[^\w\-_.]", "_", name)


def export_training_testing_data(
    br: str,
    praca: str,
    br_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Export the training and testing data to an Excel file.

    :param br: The BR identifier.
    :param praca: The praca identifier.
    :param br_data: DataFrame containing the BR's training data.
    :param forecast_data: DataFrame containing the BR's testing data.
    :param output_dir: Directory where the Excel file will be saved.
    """
    sanitized_praca = sanitize_filename(praca)
    output_path = os.path.join(
        output_dir, f"BR_{br}_PRACA_{sanitized_praca}_training_testing_data.xlsx"
    )

    os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        br_data.to_excel(writer, sheet_name="Training Data", index=False)
        forecast_data.to_excel(writer, sheet_name="Testing Data", index=False)


def export_forecast_results(
    br: str,
    praca: str,
    year: int,
    forecast_data: pd.DataFrame,
    predicted_accidents: pd.Series,
    output_dir: str,
    metrics: dict[str, float],
) -> None:
    """
    Export the actual and predicted data, along with evaluation metrics, to an Excel file.

    :param br: The BR identifier.
    :param praca: The praca identifier.
    :param year: The year of the forecast.
    :param forecast_data: DataFrame containing the BR's testing data.
    :param predicted_accidents: Series of predicted accident numbers.
    :param output_dir: Directory where the Excel file will be saved.
    :param metrics: Dictionary containing the evaluation metrics.
    """
    sanitized_praca = sanitize_filename(praca)
    output_path = os.path.join(
        output_dir,
        f"BR_{br}_PRACA_{sanitized_praca}_forecast_results_{year}.xlsx",
    )

    os.makedirs(output_dir, exist_ok=True)

    results_df = forecast_data.copy()
    results_df["Predicted Accidents"] = predicted_accidents
    results_df["Error"] = results_df["accidents"] - results_df["Predicted Accidents"]

    metrics_df = pd.DataFrame([metrics])

    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name="Forecast Results", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)


def train_sarimax_model(
    br_data: pd.DataFrame,
    config: dict[str, tuple],
) -> SARIMAX:
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
    br: str,
    praca: str,
    year: int,
    sarimax_result: SARIMAX,
    forecast_data: pd.DataFrame,
    output_img_dir: str,
) -> None:
    """
    Generate and save the forecast for the specified BR and year 2023.

    :param br: The BR identifier.
    :param praca: The praca identifier.
    :param year: The year of the forecast.
    :param sarimax_result: Trained SARIMAX model result.
    :param forecast_data: DataFrame containing the BR's testing data.
    :param output_img_dir: Directory where the output images will be saved.
    """
    exog_forecast = pd.to_numeric(forecast_data["traffic_volume"], errors="coerce")
    exog_forecast.fillna(0, inplace=True)

    try:
        predictions = sarimax_result.get_forecast(
            steps=len(forecast_data),
            exog=exog_forecast.astype(float),
        )
        predicted_accidents = predictions.predicted_mean

        min_length = min(
            len(forecast_data["accidents"].values), len(predicted_accidents)
        )
        real_accidents = forecast_data["accidents"].values[:min_length]
        predicted_accidents = predicted_accidents[:min_length]

        metrics = calculate_metrics(real_accidents, predicted_accidents)
        print(f"Evaluation Metrics for BR-{br}, PRACA-{praca}: {metrics}")

        export_forecast_results(
            br,
            praca,
            year,
            forecast_data,
            predicted_accidents,
            output_img_dir,
            metrics,
        )
        export_to_excel(
            sanitize_filename(praca),
            year,
            forecast_data,
            predicted_accidents,
            output_img_dir,
        )
        plot_comparison_chart(
            sanitize_filename(praca),
            year,
            forecast_data,
            predicted_accidents,
            output_img_dir,
        )

    except ValueError as e:
        print(f"Failed to forecast for BR-{br}, PRACA-{praca} in {year}: {e}")


def process_praca_data(
    merged_data: pd.DataFrame,
    output_dir: str,
    config_list: list[dict[str, tuple]],
) -> None:
    """
    Process each praca's data within each BR, train the SARIMAX model, and generate forecasts.

    :param merged_data: DataFrame containing the merged traffic and accident data.
    :param output_dir: Directory where the output will be saved.
    :param config_list: List of configurations to train the model with.
    """
    br_list = merged_data["br"].unique()
    test_years = ["2022", "2023"]

    for br in br_list:
        print("-" * 50)
        print(f"Processing BR-{br}...")
        br_data = merged_data[merged_data["br"] == br]
        praca_list = br_data["praca"].unique()

        for praca in praca_list:
            print(f"Processing PRACA-{praca} within BR-{br}...")
            praca_data = br_data[br_data["praca"] == praca]
            print(f"Number of records for PRACA-{praca} in BR-{br}: {len(praca_data)}")

            if len(praca_data) < 3:
                print(
                    f"Skipping PRACA-{praca} within BR-{br} due to insufficient data."
                )
                continue

            for year in test_years:
                try:
                    train_data = praca_data[praca_data["year_month"] < f"{year}-01"]
                    test_data = praca_data[
                        praca_data["year_month"].str.startswith(year)
                    ]

                    if len(test_data) == 0:
                        print(
                            f"No data available to forecast for PRACA-{praca} in BR-{br} in {year}. Skipping..."
                        )
                        continue

                    export_training_testing_data(
                        br,
                        praca,
                        train_data,
                        test_data,
                        output_dir + f"/{year}",
                    )

                    for config in config_list:
                        sarimax_result = train_sarimax_model(train_data, config)
                        print(
                            f"Model training completed for PRACA-{praca} within BR-{br} with config {config} for year {year}."
                        )
                        print("-" * 50)

                        generate_forecast(
                            br,
                            praca,
                            year,
                            sarimax_result,
                            test_data,
                            output_dir + f"/{year}",
                        )

                except ValueError as e:
                    print(
                        f"Failed to train SARIMAX model for PRACA-{praca} within BR-{br} in year {year}: {e}"
                    )
                    continue


def get_sarimax_configs() -> list[dict[str, tuple]]:
    """
    Return the list of SARIMAX configurations to be tested.

    :return: List of configurations with 'order' and 'seasonal_order'.
    """
    return [
        # {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
        {"order": (2, 1, 2), "seasonal_order": (2, 1, 2, 12)},
    ]


def main() -> None:
    """
    Main function to load data, train SARIMAX models, and generate forecasts.
    """
    print("-" * 50)
    print("Loading preprocessed data...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    preprocessed_data = load_data(base_dir)

    # Aggregate traffic volume and accidents by BR, praca, and month
    aggregated_data = aggregate_traffic_volume(preprocessed_data)

    output_dir = os.path.join(base_dir, "traffic-accident-analysis/out/core/praca")

    print("-" * 50)
    print("Training SARIMAX model on all data...")
    sarimax_configs = get_sarimax_configs()

    process_praca_data(
        aggregated_data,
        output_dir,
        sarimax_configs,
    )

    print("SARIMAX model training and forecasting completed.")


if __name__ == "__main__":
    main()
