# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 05:22:09
# @Info:   Script to preprocess traffic accident data
# ============================================================================


import pandas as pd


def preprocess_data(input_file_path: str, output_file_path: str) -> None:
    """
    Preprocess the data to filter only accidents in Minas Gerais (MG).

    This function reads the traffic accident data from a CSV file,
    filters the data to include only accidents that occurred in the
    state of Minas Gerais (MG), and saves the filtered data to a new
    CSV file.

    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to save the preprocessed CSV file.
    
    :return: None
    """
    data = pd.read_csv(input_file_path, encoding='latin1', delimiter=';')
    data_mg = data[data['uf'] == 'MG']
    data_mg.to_csv(output_file_path, index=False, encoding='latin1', sep=';')
