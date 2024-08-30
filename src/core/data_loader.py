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

    # Converting 'year_month' to datetime for compatibility with SARIMAX
    merged_data["data"] = merged_data["year_month"].dt.to_timestamp()

    # Renaming columns for clarity
    merged_data.rename(
        columns={"id": "accidents", "volume_total": "traffic_volume"}, inplace=True
    )

    # Dropping the 'year_month' column as it is no longer needed
    merged_data.drop(columns=["year_month"], inplace=True)

    return merged_data
