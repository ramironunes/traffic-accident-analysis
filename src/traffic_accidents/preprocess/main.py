# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 05:28:05
# @Info:   Script to run the preprocessing for multiple datasets
# ============================================================================


import os

from preprocess_data import preprocess_data


def main() -> None:
    """
    Main function to preprocess multiple datasets.

    This function iterates over a list of predefined dataset names,
    reads each dataset, applies the preprocessing function to filter
    data in Minas Gerais (MG), and saves the filtered data to
    the specified directory.

    :return: None
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    raw_dir = os.path.join(base_dir, "data/traffic_accidents/all_causes_and_types/raw")
    preprocessed_dir = os.path.join(
        base_dir, "data/traffic_accidents/all_causes_and_types/preprocessed"
    )

    datasets = [
        "acidentes2017_todas_causas_tipos",
        "acidentes2018_todas_causas_tipos",
        "acidentes2019_todas_causas_tipos",
        "acidentes2020_todas_causas_tipos",
        "acidentes2021_todas_causas_tipos",
        "acidentes2022_todas_causas_tipos",
        "acidentes2023_todas_causas_tipos",
        "acidentes2024_todas_causas_tipos",
    ]

    for dataset_name in datasets:
        input_file_path = os.path.join(raw_dir, f"{dataset_name}.csv")
        output_file_path = os.path.join(preprocessed_dir, f"{dataset_name}_mg.csv")

        preprocess_data(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
