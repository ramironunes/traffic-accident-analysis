# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Functions to load and preprocess traffic data
# ============================================================================


import os
import pandas as pd

from typing import List


def get_file_paths(base_dir: str, datasets: List[str]) -> List[str]:
    """
    Get the full file paths for the datasets located in a base directory.

    Args:
        base_dir (str): The base directory where datasets are stored.
        datasets (List[str]): A list of dataset filenames.

    Returns:
        List[str]: List of full file paths for each dataset.
    """
    return [os.path.join(base_dir, f"{dataset}.csv") for dataset in datasets]


def load_preprocessed_data(
    accident_files: List[str], toll_files: List[str]
) -> pd.DataFrame:
    """
    Load and merge preprocessed accident and toll data.

    Args:
        accident_files (List[str]): List of paths to the preprocessed accident data files.
        toll_files (List[str]): List of paths to the preprocessed toll data files.

    Returns:
        pd.DataFrame: Merged DataFrame with accident and toll data.
    """
    print("Loading accident data...")
    accident_data = pd.concat(
        [
            pd.read_csv(file, encoding="latin1", sep=";", on_bad_lines="skip")
            for file in accident_files
        ],
        ignore_index=True,
    )

    print("Loading toll data...")
    toll_data = pd.concat(
        [
            pd.read_csv(file, encoding="latin1", sep=",", on_bad_lines="skip")
            for file in toll_files
        ],
        ignore_index=True,
    )

    print("Normalizing column names to lowercase...")
    accident_data.columns = accident_data.columns.str.lower()
    toll_data.columns = toll_data.columns.str.lower()

    print("Grouping and aggregating accident and toll data...")
    accident_data = (
        accident_data.groupby(["br", "data"]).agg({"id": "count"}).reset_index()
    )
    toll_data = (
        toll_data.groupby(["br", "data"]).agg({"volume_total": "sum"}).reset_index()
    )

    print("Converting 'data' columns to datetime format...")
    accident_data["data"] = pd.to_datetime(accident_data["data"], format="%d/%m/%Y")
    toll_data["data"] = pd.to_datetime(toll_data["data"], format="%d/%m/%Y")

    print("Merging accident and toll data on 'BR' and 'data' columns...")
    merged_data = pd.merge(accident_data, toll_data, on=["br", "data"], how="inner")
    merged_data.rename(
        columns={"id": "accidents", "volume_total": "traffic_volume"}, inplace=True
    )

    print(
        f"Merged dataset contains {len(merged_data)} records and {len(merged_data.columns)} columns."
    )
    return merged_data
