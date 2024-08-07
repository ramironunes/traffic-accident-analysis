# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 04:37:35
# @Info:   Script to load and prepare data for spatial analysis
# ============================================================================


import pandas as pd


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare the traffic accident data for spatial analysis.

    This function reads the traffic accident data from a CSV file,
    replaces commas with dots in the latitude and longitude columns,
    and renames the columns to ensure they are within the 10-character
    limit for shapefiles.

    :param file_path: Path to the CSV file to load.
    :return: DataFrame with the prepared data.
    """
    data = pd.read_csv(file_path, encoding='latin1', delimiter=';')
    
    # Replace commas with dots in latitude and longitude columns
    data['latitude'] = data['latitude'].str.replace(',', '.').astype(float)
    data['longitude'] = data['longitude'].str.replace(',', '.').astype(float)
    
    # Rename columns to ensure they are within the 10-character limit for shapefiles
    data = data.rename(columns={
        'data_inversa': 'data_inv',
        'causa_principal': 'causa_princ',
        'causa_acidente': 'causa_acid',
        'ordem_tipo_acidente': 'ordem_tipo',
        'tipo_acidente': 'tipo_acid',
        'classificacao_acidente': 'classific',
        'fase_dia': 'fase_dia',
        'sentido_via': 'sentido_vi',
        'condicao_metereologica': 'cond_met',
        'tracado_via': 'tracado_vi',
        'tipo_veiculo': 'tipo_veicu',
        'ano_fabricacao_veiculo': 'ano_fab',
        'tipo_envolvido': 'tipo_env',
        'estado_fisico': 'estado_fis',
        'feridos_leves': 'feridos_le',
        'feridos_graves': 'feridos_gr',
        'mortos': 'mortos',
        'ilesos': 'ilesos'
    })

    return data
