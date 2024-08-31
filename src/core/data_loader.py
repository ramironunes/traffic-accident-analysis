# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Functions to load and preprocess traffic data
# ============================================================================


import os
import pandas as pd


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

    processed_toll_data = process_toll_data(toll_data)

    # Save the processed data to a CSV file
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_file_path = os.path.join(base_dir, "processed_toll_data.csv")
    processed_toll_data.to_csv(output_file_path, index=False, encoding="utf-8")

    # Confirm that the data was saved
    print(f"Processed toll data saved to {output_file_path}")

    # Monthly aggregation of accident data and traffic volume
    accident_data_grouped = (
        accident_data.groupby(["br", "km", "year_month"])
        .agg({"id": "count"})  # Counting the number of accidents per month
        .reset_index()
    )
    toll_data_grouped = (
        toll_data.groupby(["br", "km", "year_month"])
        .agg({"volume_total": "sum"})  # Summing the traffic volume per month
        .reset_index()
    )

    # Merging the monthly aggregated data
    merged_data = pd.merge(
        accident_data_grouped,
        toll_data_grouped,
        on=["br", "km", "year_month"],
        how="inner",
    )

    # Renaming columns for clarity
    merged_data.rename(
        columns={"id": "accidents", "volume_total": "traffic_volume"},
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
            group["next_km"] = next_km  # Replicate the next_km value for all rows
            group["prev_km"] = prev_km  # Replicate the prev_km value for all rows
        return group

    toll_data = toll_data.groupby(["br", "km"]).apply(adjust_km)

    # Fill NaN values for first and last toll stations
    toll_data["next_km"] = toll_data["next_km"].fillna(toll_data["km"].max())
    toll_data["prev_km"] = toll_data["prev_km"].fillna(toll_data["km"].min())

    # Reset the index to avoid ambiguity with 'br'
    toll_data = toll_data.reset_index(drop=True)

    # Calculate the distances to the next and previous toll stations
    toll_data["distance_to_next"] = toll_data["next_km"] - toll_data["km"]
    toll_data["distance_to_prev"] = toll_data["km"] - toll_data["prev_km"]

    # Adjust all distance_to_prev for the first km and distance_to_next for the last km in each 'br' group
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
