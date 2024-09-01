# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Functions to load and preprocess traffic data
# ============================================================================


import matplotlib.pyplot as plt
import os
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose


def get_file_paths(base_dir: str, datasets: list[str]) -> list[str]:
    """
    Get the full file paths for the datasets located in a base directory.

    :param base_dir: The base directory where datasets are stored.
    :param datasets: A list of dataset filenames.
    :return: List of full file paths for each dataset.
    """
    return [os.path.join(base_dir, f"{dataset}.csv") for dataset in datasets]


def load_preprocessed_data(
    accident_files: list[str],
    toll_files: list[str],
) -> pd.DataFrame:
    """
    Load, preprocess, and merge accident and toll data.

    :param accident_files: List of paths to the preprocessed accident data files.
    :param toll_files: List of paths to the preprocessed toll data files.
    :return: Merged DataFrame with accident and toll data aggregated monthly.
    """
    accident_data = pd.concat(
        [
            pd.read_csv(file, encoding="latin1", sep=";", on_bad_lines="skip")
            for file in accident_files
        ],
        ignore_index=True,
    )
    toll_data = pd.concat(
        [
            pd.read_csv(file, encoding="latin1", sep=",", on_bad_lines="skip")
            for file in toll_files
        ],
        ignore_index=True,
    )

    # Monthly aggregation of accident data and traffic volume
    accident_data_grouped = (
        accident_data.groupby(["br", "km", "year_month"])
        .agg({"id": "count"})
        .reset_index()
    )

    preprocessed_data = merge_datasets_on_km_range(
        accident_data_grouped,
        process_toll_data(toll_data),
    )

    # Save the processed data to a CSV file
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../..",
        )
    )
    output_dir = os.path.join(
        base_dir,
        "traffic-accident-analysis/out/data",
    )
    output_file_path = os.path.join(output_dir, "preprocessed_data.csv")
    preprocessed_data.to_csv(output_file_path, index=False, encoding="utf-8")

    # print("Preprocessed data head:\n", preprocessed_data.head())
    # monthly_data = preprocessed_data.groupby("year_month")["volume_total"].sum()

    # fig, ax = plt.subplots()
    # plot_acf(monthly_data, ax=ax)
    # plt.savefig(os.path.join(output_dir, "autocorrelation_plot.png"))
    # plt.close(fig)

    # decomposition = seasonal_decompose(monthly_data, model="additive", period=12)
    # fig = decomposition.plot()
    # plt.savefig(os.path.join(output_dir, "seasonal_decomposition_plot.png"))
    # plt.close(fig)

    # Renaming columns for clarity
    preprocessed_data.rename(
        columns={"id": "accidents", "volume_total": "traffic_volume"},
        inplace=True,
    )

    return preprocessed_data


def merge_datasets_on_km_range(
    accident_data: pd.DataFrame,
    toll_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge accident data with toll data based on km range.

    :param accident_data: DataFrame containing accident data.
    :param toll_data: DataFrame containing toll data with distance_to_prev and distance_to_next.
    :return: Merged DataFrame with accidents matched to toll stations based on km range.
    """
    # Expand the toll_data to account for the km range
    toll_data_expanded = toll_data.copy()
    toll_data_expanded["km_start"] = (
        toll_data_expanded["km"] - toll_data_expanded["distance_to_prev"]
    )
    toll_data_expanded["km_end"] = (
        toll_data_expanded["km"] + toll_data_expanded["distance_to_next"]
    )

    # Perform the merge based on the km range
    merged_data = pd.merge(
        accident_data,
        toll_data_expanded,
        how="inner",
        left_on=["br", "year_month"],
        right_on=["br", "year_month"],
    )

    # Filter rows where the accident km falls within the toll km range
    merged_data = merged_data[
        (merged_data["km_x"] >= merged_data["km_start"])
        & (merged_data["km_x"] <= merged_data["km_end"])
    ]

    # Drop auxiliary columns and rename as needed
    merged_data.drop(columns=["km_start", "km_end"], inplace=True)
    merged_data.rename(
        columns={"km_x": "accident_km", "km_y": "toll_km"},
        inplace=True,
    )

    return merged_data


def process_toll_data(toll_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process toll station data to calculate distances between toll stations and
    adjust distance values for the first and last toll stations in each group.

    :param toll_data: DataFrame containing the toll station data.
    :return: Processed DataFrame with adjusted distance columns.
    """
    toll_data = toll_data.groupby(["br", "km", "year_month"], as_index=False).agg(
        {
            "volume_total": "first",
            "praca": "first",
        }
    )

    # Sort toll stations by 'br' and 'km'
    toll_data = toll_data.sort_values(by=["br", "km"]).reset_index(drop=True)

    # Calculate the distance to the next and previous toll stations
    toll_data["next_km"] = toll_data.groupby("br")["km"].shift(-1)
    toll_data["prev_km"] = toll_data.groupby("br")["km"].shift(1)

    # Handle cases with duplicate 'km' values
    def adjust_km(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) > 1:
            next_km = group["next_km"].iloc[-1]
            prev_km = group["prev_km"].iloc[0]
            group["next_km"] = next_km
            group["prev_km"] = prev_km

        return group

    toll_data = toll_data.groupby(["br", "km"]).apply(adjust_km)

    # Fill NaN values for first and last toll stations
    toll_data["next_km"] = toll_data["next_km"].fillna(toll_data["km"].max())
    toll_data["prev_km"] = toll_data["prev_km"].fillna(toll_data["km"].min())

    # Reset the index to avoid ambiguity with 'br'
    toll_data = toll_data.reset_index(drop=True)

    # Calculate the distances to the next and previous toll stations
    toll_data["distance_to_next"] = (toll_data["next_km"] - toll_data["km"]) / 2
    toll_data["distance_to_prev"] = (toll_data["km"] - toll_data["prev_km"]) / 2

    # Adjust all distance_to_prev for the first km and distance_to_next for
    # the last km in each 'br' group
    first_km_indices = (
        toll_data.groupby("br")
        .apply(lambda x: x[x["km"] == x["km"].min()].index)
        .explode()
        .values
    )
    last_km_indices = (
        toll_data.groupby("br")
        .apply(lambda x: x[x["km"] == x["km"].max()].index)
        .explode()
        .values
    )

    toll_data.loc[first_km_indices, "distance_to_prev"] = toll_data.loc[
        first_km_indices, "distance_to_next"
    ]
    toll_data.loc[last_km_indices, "distance_to_next"] = toll_data.loc[
        last_km_indices, "distance_to_prev"
    ]

    # Drop the auxiliary columns
    toll_data.drop(columns=["next_km", "prev_km"], inplace=True)

    return toll_data
