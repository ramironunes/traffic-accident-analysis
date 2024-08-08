# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 04:00:23
# @Info:   Script to perform analysis on traffic accident data
# ============================================================================

"""
Script to perform analysis on traffic accident data.

This script provides functionality to generate various types of graphs
for spatial analysis on multiple traffic accident datasets. It supports
the following types of graphs:
    - accident_cause: Generates a bar chart showing the distribution of accident causes.
    - accident_density: Generates a density plot of accidents on a map.
    - static_map: Generates a static map from a shapefile.
    - vehicle_type: Generates a bar chart showing the distribution of vehicle types involved in accidents.
    - weather_condition: Generates a bar chart showing the distribution of weather conditions during accidents.

Usage:
    python script.py <graph_type>

Arguments:
    graph_type (str): The type of graph to generate. Options are:
        accident_cause, accident_density, static_map, vehicle_type, weather_condition

Example:
    python script.py accident_cause

This example will generate bar charts showing the distribution of accident causes for each dataset.
"""


import argparse
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../..'
        )
    )
)

from create_shapefile import create_shapefile
from load_data import load_and_prepare_data
from src.utils.plotter import DataPlotter


def main(graph_type: str) -> None:
    """
    Main function to perform spatial analysis on multiple datasets.

    This function iterates over a list of predefined dataset names,
    loads and prepares the preprocessed data, creates shapefiles,
    and generates the specified plot for each dataset.

    :param graph_type: The type of graph to generate.
    :return: None
    """
    base_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../..',
        )
    )
    preprocessed_dir = os.path.join(
        base_dir,
        'data/traffic_accidents/all_causes_and_types/preprocessed',
    )
    shapefile_dir = os.path.join(
        base_dir,
        'data/traffic_accidents/all_causes_and_types/shapefile',
    )
    output_html_dir = os.path.join(base_dir, 'out/html/traffic_accidents')
    output_img_dir = os.path.join(base_dir, 'out/img/traffic_accidents')

    datasets = [
        'acidentes2017_todas_causas_tipos',
        'acidentes2018_todas_causas_tipos',
        'acidentes2019_todas_causas_tipos',
        'acidentes2020_todas_causas_tipos',
        'acidentes2021_todas_causas_tipos',
        'acidentes2022_todas_causas_tipos',
        'acidentes2023_todas_causas_tipos',
        'acidentes2024_todas_causas_tipos',
    ]

    # Ensure the graph type directory exists using the DataPlotter method
    plotter = DataPlotter(output_img_dir)
    graph_output_dir = os.path.join(output_img_dir, graph_type)
    plotter.ensure_directory_exists(graph_output_dir)
    plotter.output_dir = graph_output_dir  # Update the output directory for the plotter

    for dataset_name in datasets:
        preprocessed_file_path = os.path.join(
            preprocessed_dir,
            f'{dataset_name}_mg.csv',
        )
        shapefile_output_path = os.path.join(
            shapefile_dir,
            f'{dataset_name}_mg.shp',
        )

        # Load and prepare data
        data = load_and_prepare_data(preprocessed_file_path)

        # Create shapefile
        create_shapefile(data, shapefile_output_path)

        # Generate the specified plot
        if graph_type == 'accident_cause':
            plotter.plot_bar_chart(
                title=f'Accident Cause Distribution for {dataset_name}',
                data=data,
                x='causa_acid',
                y='count',
                dataset_name=dataset_name,
                palette='viridis'
            )
        elif graph_type == 'accident_density':
            plotter.plot_spatial_density(
                data=data,
                dataset_name=dataset_name
            )
        elif graph_type == 'static_map':
            plotter.plot_spatial_data(
                title=f'Static Map for {dataset_name}',
                shapefile_path=shapefile_output_path,
                dataset_name=dataset_name
            )
        elif graph_type == 'vehicle_type':
            plotter.plot_count_distribution(
                title=f'Vehicle Type Distribution for {dataset_name}',
                data=data,
                column='tipo_veicu',
                dataset_name=dataset_name,
                palette='viridis'
            )
        elif graph_type == 'weather_condition':
            plotter.plot_count_distribution(
                title=f'Weather Condition Distribution for {dataset_name}',
                data=data,
                column='cond_met',
                dataset_name=dataset_name,
                palette='viridis'
            )
        else:
            print(f"Unknown graph type: {graph_type}")


if __name__ == "__main__":
    text_description: str = \
        "Generate specific type of traffic accident analysis graph."
    text_help: str = \
        "Type of graph to generate (accident_cause, accident_density, " + \
        "static_map, vehicle_type, weather_condition)"

    parser = argparse.ArgumentParser(description=text_description)
    parser.add_argument('graph_type', type=str, help=text_help)
    args = parser.parse_args()

    main(args.graph_type)
