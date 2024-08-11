# -*- coding: utf-8 -*-
# @Author: Jean Mira
# @Date:   2024-08-09 20:08:10
# @Last Modified by:   Ramiro Luiz Nunes
# @Last Modified time: 2024-08-10 21:46:57


import pandas as pd
import re
import unidecode


def extract_br_and_km(praca: str) -> tuple:
    """
    Extract BR and KM information from the 'praca' field in toll data.

    This function handles various formats of the 'praca' field, including:
    - "BR-XXX/MG km: XXX,XX"
    - "BR-XXX/MG km XXX,XX"
    - "BR-XXX/MG km XXX"

    :param praca: The 'praca' field containing information about BR and KM.
    :return: A tuple (BR, KM) or (None, None) if not found.
    """
    # Normalize the string by removing accents and converting to lowercase
    praca_normalized = unidecode.unidecode(praca).lower()

    # Regular expression to match 'BR' and 'KM'
    br_match = re.search(r"br-(\d+)", praca_normalized)
    km_match = re.search(r"km[: ]\s*(\d+(?:,\d+)?)", praca_normalized)

    # Extract BR if found, returning only the number
    br = br_match.group(1) if br_match else None

    # Extract KM if found, converting to float and normalizing
    km = float(km_match.group(1).replace(",", ".")) if km_match else None

    return br, km


def preprocess_toll_data(input_file_path: str, output_file_path: str) -> None:
    """
    Preprocess the data to filter only toll stations in Minas Gerais (MG),
    remove accents, and extract BR and KM columns.

    This function reads the toll station data from a CSV file with 'latin1' encoding,
    normalizes the data by removing accents, filters the data to include only toll
    stations located in Minas Gerais (MG), extracts the BR and KM information,
    renames the 'mes_ano' column to 'data', and saves the filtered data to a new CSV file.

    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to save the preprocessed CSV file.
    :return: None
    """
    try:
        data = pd.read_csv(input_file_path, encoding="latin1", delimiter=";")

        # Remove accents from columns and 'praca' field
        data.columns = [unidecode.unidecode(col) for col in data.columns]
        data["praca"] = data["praca"].apply(unidecode.unidecode)

        # Rename 'mes_ano' column to 'data'
        data.rename(columns={"mes_ano": "data"}, inplace=True)

        # Filter for Minas Gerais toll stations
        data_mg = data[data["praca"].str.contains("MG")]

        # Extract BR and KM and add them as new columns
        data_mg[["br", "km"]] = data_mg["praca"].apply(
            lambda x: pd.Series(extract_br_and_km(x))
        )

        # Convert 'km' to float
        data_mg["km"] = data_mg["km"].astype(float)

        # Save the preprocessed data with only numeric BR and KM
        data_mg.to_csv(output_file_path, index=False, encoding="utf-8", sep=",")

        print(f"Data preprocessed and saved to {output_file_path}")

    except FileNotFoundError:
        print(f"File not found: {input_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
