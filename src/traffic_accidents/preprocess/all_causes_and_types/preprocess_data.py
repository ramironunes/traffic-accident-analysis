# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 05:22:09
# @Info:   Script to preprocess dataset data
# ============================================================================


import pandas as pd


def load_data(input_file_path: str) -> pd.DataFrame:
    """
    Load the data from the input CSV file.

    :param input_file_path: Path to the input CSV file.
    :return: Loaded DataFrame.
    """
    df = pd.read_csv(input_file_path, encoding="latin1", delimiter=";")
    df.columns = df.columns.str.lower()
    return df


def filter_mg_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data to include only records that occurred in Minas Gerais (MG).

    :param data: DataFrame containing the dataset data.
    :return: Filtered DataFrame with only MG records.
    """
    return df[df["uf"] == "MG"]


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that do not contain valid 'br' and 'km' columns.

    :param data: DataFrame containing the dataset data.
    :return: DataFrame with rows containing valid 'br' and 'km' columns.
    """
    # Ensure the columns 'br' and 'km' exist, if not, create them with NaN values
    if "br" not in df.columns:
        df["br"] = pd.NA
    if "km" not in df.columns:
        df["km"] = pd.NA

    # Convert 'br' to integer, handling errors
    df["br"] = pd.to_numeric(df["br"], errors="coerce").astype("Int64")

    # Convert 'km' to float, handle errors, and remove NaNs
    df["km"] = pd.to_numeric(
        df["km"].astype(str).str.replace(",", "."), errors="coerce"
    )

    # Remove rows where 'br' or 'km' is NaN or empty
    data_cleaned = df.dropna(subset=["br", "km"])

    return data_cleaned


def remove_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on the 'id' column, keeping only the first occurrence.

    :param data: DataFrame containing the dataset data.
    :return: DataFrame with unique 'id' values.
    """
    return df.drop_duplicates(subset=["id"])


def format_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the 'data_inversa' column to 'data' and format it to 'dd-mm-yyyy'.

    :param data: DataFrame containing the dataset data.
    :return: DataFrame with the 'data' column formatted.
    """
    df.rename(columns={"data_inversa": "data"}, inplace=True)

    # Convert 'data' column to datetime
    df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d")

    # Create 'year_month' before formatting 'data' as a string
    df["year_month"] = df["data"].dt.to_period("M")

    # Format the 'data' column to 'dd/mm/yyyy' as string
    df["data"] = df["data"].dt.strftime("%d/%m/%Y")

    return df


def save_data(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Save the processed data to the output CSV file.

    :param data: DataFrame to save.
    :param output_file_path: Path to save the preprocessed CSV file.
    :return: None
    """
    df.to_csv(output_file_path, index=False, encoding="latin1", sep=";")
    print(f"Data processed and saved successfully to {output_file_path}")


def preprocess_data(input_file_path: str, output_file_path: str) -> None:
    """
    Execute the full preprocessing pipeline: load, filter, remove invalid rows,
    remove duplicates, format the date column, and save.

    :param input_file_path: Path to the input CSV file.
    :param output_file_path: Path to save the preprocessed CSV file.
    :return: None
    """
    data = load_data(input_file_path)
    data_mg = filter_mg_records(data)
    data_cleaned = remove_invalid_rows(data_mg)
    data_unique = remove_duplicate_ids(data_cleaned)
    data_formatted = format_date_column(data_unique)
    save_data(data_formatted, output_file_path)
