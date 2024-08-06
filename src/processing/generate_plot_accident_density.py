# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 04:00:23
# @Info:   Script to perform spatial analysis on traffic accident data
# ============================================================================

import os

from create_shapefile import create_shapefile
from load_data import load_and_prepare_data
from plot_graphs import (
    plot_accident_cause,
    plot_accident_density,
    plot_weather_condition,
    plot_vehicle_type,
    plot_static_map
)


def main() -> None:
    """
    Main function to perform spatial analysis on multiple datasets.

    This function iterates over a list of predefined dataset names,
    loads and prepares the preprocessed data, creates shapefiles,
    and generates various plots for each dataset.
    
    :return: None
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.join(base_dir, '../../data/traffic_accidents/all_causes_and_types/preprocessed')
    shapefile_dir = os.path.join(base_dir, '../../data/traffic_accidents/all_causes_and_types/shapefile')
    output_img_dir = os.path.join(base_dir, '../../out/img')
    output_html_dir = os.path.join(base_dir, '../../out/html')
    
    datasets = [
        'acidentes2017_todas_causas_tipos',
        'acidentes2018_todas_causas_tipos',
        'acidentes2019_todas_causas_tipos',
        'acidentes2020_todas_causas_tipos',
        'acidentes2021_todas_causas_tipos',
        'acidentes2022_todas_causas_tipos',
        'acidentes2023_todas_causas_tipos',
        'acidentes2024_todas_causas_tipos'
    ]
    
    for dataset_name in datasets:
        preprocessed_file_path = os.path.join(preprocessed_dir, f'{dataset_name}_mg.csv')
        shapefile_output_path = os.path.join(shapefile_dir, f'{dataset_name}_mg.shp')
        static_map_output_path = os.path.join(output_img_dir, f'{dataset_name}_mg_static_map.png')
        interactive_map_output_path = os.path.join(output_html_dir, f'{dataset_name}_mg.html')
        
        # Load and prepare data
        data = load_and_prepare_data(preprocessed_file_path)
        
        # Create shapefile
        create_shapefile(data, shapefile_output_path)
        
        # Plot static map
        plot_static_map(shapefile_output_path, static_map_output_path)
        
        # Generate additional plots
        plot_accident_cause(data, dataset_name)
        plot_weather_condition(data, dataset_name)
        plot_vehicle_type(data, dataset_name)
        plot_accident_density(data, dataset_name)


if __name__ == "__main__":
    main()
