# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Main script to train and forecast using SARIMAX model on traffic data
# ============================================================================

import os
import sys
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from data_loader import get_file_paths, load_preprocessed_data
from plot_utils import plot_comparison_chart, export_to_excel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def aggregate_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates data to a weekly basis.
    """
    data["data"] = pd.to_datetime(data["data"])
    data.set_index("data", inplace=True)
    weekly_data = data.groupby(["br", pd.Grouper(freq="W")]).sum().reset_index()
    return weekly_data


def load_data(base_dir: str) -> pd.DataFrame:
    """
    Load and preprocess accident and toll data, then merge them.
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


def train_sarimax_model(br_data: pd.DataFrame) -> SARIMAX:
    """
    Train a SARIMAX model on the provided data.
    """
    exog_data = br_data["traffic_volume"]

    # Ensure all data is numeric
    br_data["accidents"] = pd.to_numeric(br_data["accidents"], errors="coerce")
    exog_data = pd.to_numeric(exog_data, errors="coerce")

    # Remove or fill NaNs if needed
    br_data.dropna(subset=["accidents"], inplace=True)
    exog_data.fillna(0, inplace=True)

    # Ensure that the dataset has enough observations
    if len(br_data) < 3:
        raise ValueError("Not enough data to train SARIMAX model.")

    # Train the SARIMAX model
    sarimax_model = SARIMAX(
        br_data["accidents"].astype(float),
        exog=exog_data.astype(float),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return sarimax_model.fit(disp=False)


def generate_forecast(
    br: str, sarimax_result: SARIMAX, forecast_data: pd.DataFrame, output_img_dir: str
) -> None:
    """
    Generate and save the forecast for the specified BR and year 2023.
    """
    exog_forecast = pd.to_numeric(forecast_data["traffic_volume"], errors="coerce")
    exog_forecast.fillna(0, inplace=True)

    try:
        predictions = sarimax_result.get_forecast(
            steps=len(forecast_data), exog=exog_forecast.astype(float)
        )
        predicted_accidents = predictions.predicted_mean

        # Ensure that both arrays have the same length
        min_length = min(
            len(forecast_data["accidents"].values), len(predicted_accidents)
        )
        real_accidents = forecast_data["accidents"].values[:min_length]
        predicted_accidents = predicted_accidents[:min_length]

        # Generate comparison chart for 2023
        plot_comparison_chart(forecast_data, predicted_accidents, 2023, output_img_dir)
        export_to_excel(forecast_data, predicted_accidents, 2023, output_img_dir)

    except ValueError as e:
        print(f"Failed to forecast for BR-{br} in 2023: {e}")


def process_br_data(weekly_data: pd.DataFrame, output_img_dir: str) -> None:
    """
    Process each BR's data, train the SARIMAX model, and generate forecasts.
    """
    br_list = weekly_data["br"].unique()

    for br in br_list:
        print(f"Training SARIMAX model for BR-{br}...")
        br_data = weekly_data[weekly_data["br"] == br]

        try:
            sarimax_result = train_sarimax_model(br_data)
            print(f"Model training completed for BR-{br}.")
            print("-" * 50)

            # Generate predictions for the year 2023
            forecast_data = weekly_data[
                (weekly_data["br"] == br) & (weekly_data["data"].dt.year == 2023)
            ]

            if len(forecast_data) == 0:
                print(f"No data available to forecast for BR-{br} in 2023. Skipping...")
                continue

            generate_forecast(br, sarimax_result, forecast_data, output_img_dir)

        except ValueError as e:
            print(f"Failed to train SARIMAX model for BR-{br}: {e}")
            continue


def main() -> None:
    print("-" * 50)
    print("Loading preprocessed data...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    merged_data = load_data(base_dir)

    print("Data loaded and merged successfully.")
    print("Aggregating data on a weekly basis...")

    weekly_data = aggregate_weekly(merged_data)
    output_img_dir = os.path.join(base_dir, "traffic-accident-analysis/out/img/core")

    print("-" * 50)
    print("Training SARIMAX model on all data...")
    process_br_data(weekly_data, output_img_dir)

    print("SARIMAX model training and forecasting completed.")


if __name__ == "__main__":
    main()
