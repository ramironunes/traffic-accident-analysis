# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 04:37:52
# @Info:   Functions to plot graphs for data analysis
# ============================================================================


import os

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shapely.geometry import Point
from sklearn.neighbors import KernelDensity


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure the directory exists, if not, create it.

    :param directory: Path to the directory.
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_accident_cause(data: pd.DataFrame, dataset_name: str) -> None:
    """
    Plot the distribution of accident causes.

    :param data: DataFrame containing the traffic accident data.
    :param dataset_name: Name of the dataset for file naming.
    :return: None
    """
    plt.figure(figsize=(12, 8))
    sns.countplot(data=data, y='causa_acid', order=data['causa_acid'].value_counts().index)
    plt.title('Distribution of Accident Causes')
    plt.xlabel('Count')
    plt.ylabel('Cause')
    plt.tight_layout()
    
    output_path = f'../img/{dataset_name}_accident_cause_distribution.png'
    ensure_directory_exists(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.show()


def plot_accident_density(data: pd.DataFrame, dataset_name: str) -> None:
    """
    Plot the density of accidents on a map.

    :param data: DataFrame containing the traffic accident data.
    :param dataset_name: Name of the dataset for file naming.
    :return: None
    """
    data = data.dropna(subset=['latitude', 'longitude'])
    xy = np.vstack([data['longitude'], data['latitude']]).T

    if xy.size == 0:
        print(f"No valid data for {dataset_name}. Skipping density plot.")
        return

    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(np.radians(xy))

    x, y = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), 100),
                       np.linspace(xy[:, 1].min(), xy[:, 1].max(), 100))
    xy_sample = np.vstack([x.ravel(), y.ravel()]).T
    z = np.exp(kde.score_samples(np.radians(xy_sample)))
    z = z.reshape(x.shape)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='hot', alpha=0.6)
    ax.scatter(data['longitude'], data['latitude'], s=1, color='blue')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.title('Density of Traffic Accidents')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    output_path = f'../img/{dataset_name}_accident_density_map.png'
    ensure_directory_exists(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def plot_static_map(shapefile_path: str, output_image_path: str) -> None:
    """
    Plot a static map with accident data.

    :param shapefile_path: Path to the shapefile.
    :param output_image_path: Path to save the output image.
    :return: None
    """
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='red', markersize=5, alpha=0.5)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.title('Traffic Accidents in Minas Gerais')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    ensure_directory_exists(os.path.dirname(output_image_path))
    plt.savefig(output_image_path)
    plt.close()


def plot_vehicle_type(data: pd.DataFrame, dataset_name: str) -> None:
    """
    Plot the distribution of vehicle types involved in accidents.

    :param data: DataFrame containing the traffic accident data.
    :param dataset_name: Name of the dataset for file naming.
    :return: None
    """
    plt.figure(figsize=(12, 8))
    sns.countplot(y='tipo_veicu', data=data, order=data['tipo_veicu'].value_counts().index)
    plt.title('Distribution of Vehicle Types in Accidents')
    plt.xlabel('Count')
    plt.ylabel('Vehicle Type')

    output_path = f'../img/{dataset_name}_vehicle_type_distribution.png'
    ensure_directory_exists(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()


def plot_weather_condition(data: pd.DataFrame, dataset_name: str) -> None:
    """
    Plot the distribution of weather conditions during accidents.

    :param data: DataFrame containing the traffic accident data.
    :param dataset_name: Name of the dataset for file naming.
    :return: None
    """
    plt.figure(figsize=(12, 8))
    sns.countplot(y='cond_met', data=data, order=data['cond_met'].value_counts().index)
    plt.title('Distribution of Weather Conditions during Accidents')
    plt.xlabel('Count')
    plt.ylabel('Weather Condition')

    output_path = f'../img/{dataset_name}_weather_condition_distribution.png'
    ensure_directory_exists(os.path.dirname(output_path))
    plt.savefig(output_path)
    plt.close()
