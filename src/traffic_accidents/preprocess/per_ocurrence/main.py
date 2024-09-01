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
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

    raw_dir = os.path.join(
        base_dir,
        "data/traffic_accidents/per_occurrence/raw",
    )
    preprocessed_dir = os.path.join(
        base_dir,
        "data/traffic_accidents/per_occurrence/preprocessed",
    )

    datasets = [
        "datatran2018",
        "datatran2019",
        "datatran2020",
        "datatran2021",
        "datatran2022",
        "datatran2023",
    ]

    for dataset_name in datasets:
        input_file_path = os.path.join(raw_dir, f"{dataset_name}.csv")
        output_file_path = os.path.join(preprocessed_dir, f"{dataset_name}_mg.csv")

        preprocess_data(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
