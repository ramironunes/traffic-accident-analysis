# -*- coding: utf-8 -*-
# @Author: Jean Mira
# @Date:   2024-08-09 20:10:21
# @Last Modified by:   Jean Mira
# @Last Modified time: 2024-08-09 20:46:14


import os
from preprocess_data import preprocess_toll_data


def main() -> None:
    base_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../..'))

    raw_dir = os.path.join(base_dir, 'data/toll_stations/raw')
    preprocessed_dir = os.path.join(
        base_dir, 'data/toll_stations/preprocessed')

    datasets = [
        'volume-trafego-praca-pedagio-2017',
        'volume-trafego-praca-pedagio-2018',
        'volume-trafego-praca-pedagio-2019',
        'volume-trafego-praca-pedagio-2020',
        'volume-trafego-praca-pedagio-2021',
        'volume-trafego-praca-pedagio-2022',
        'volume-trafego-praca-pedagio-2023',
        'volume-trafego-praca-pedagio-2024',
    ]

    for dataset_name in datasets:
        input_file_path = os.path.join(raw_dir, f'{dataset_name}.csv')
        output_file_path = os.path.join(
            preprocessed_dir, f'{dataset_name}_mg.csv')

        if not os.path.exists(input_file_path):
            print(f"Input file does not exist: {input_file_path}")
        else:
            preprocess_toll_data(input_file_path, output_file_path)


if __name__ == "__main__":
    main()
