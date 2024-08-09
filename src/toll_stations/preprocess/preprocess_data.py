# -*- coding: utf-8 -*-
# @Author: Jean Mira
# @Date:   2024-08-09 20:08:10
# @Last Modified by:   Jean Mira
# @Last Modified time: 2024-08-09 20:46:29


import pandas as pd
import unidecode


def preprocess_toll_data(input_file_path: str, output_file_path: str) -> None:
    """
    Preprocess the data to filter only toll stations in Minas Gerais (MG) and remove accents.

    This function reads the toll station data from a CSV file with 'latin1' encoding,
    normalizes the data by removing accents, filters the data to include only toll
    stations located in Minas Gerais (MG), and saves the filtered data to a new CSV
    file with 'utf-8' encoding.

    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to save the preprocessed CSV file.
    :return: None
    """
    try:
        data = pd.read_csv(input_file_path, encoding='latin1', delimiter=';')

        data.columns = [unidecode.unidecode(col) for col in data.columns]
        data['praca'] = data['praca'].apply(unidecode.unidecode)

        data_mg = data[data['praca'].str.contains('MG')]

        data_mg.to_csv(output_file_path, index=False,
                       encoding='utf-8', sep=',')
    except FileNotFoundError:
        print(f"File not found: {input_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
