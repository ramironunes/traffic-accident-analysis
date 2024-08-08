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

from sklearn.neighbors import KernelDensity


class DataPlotter:
    """
    A class used to plot various types of graphs for data analysis.

    This class provides methods to plot various types of graphs such as 
    bar charts, pie charts, spatial analyses, and more. Each plotting 
    function allows for customization through various parameters.

    Attributes:
    ----------
    output_dir : str
        The directory where plots will be saved.

    Methods:
    -------
    ensure_directory_exists(directory: str) -> None
        Ensure the directory exists, if not, create it.

    plot_bar_chart(
        title: str,
        data: pd.DataFrame,
        x: str,
        y: str, 
        dataset_name: str,
        **kwargs
    ) -> None
        Plot a bar chart with customizable settings.

    plot_pie_chart(
        title: str,
        data: pd.DataFrame,
        labels: str, 
        values: str,
        dataset_name: str,
        **kwargs
    ) -> None
        Plot a pie chart with customizable settings.

    plot_spatial_density(
        data: pd.DataFrame,
        dataset_name: str, 
        **kwargs
    ) -> None
        Plot the density of data points on a map with customizable settings.

    plot_spatial_data(
        title: str,
        shapefile_path: str,
        dataset_name: str,
        **kwargs
    ) -> None
        Plot a static map using a shapefile with customizable settings.

    plot_count_distribution(
        title: str,
        data: pd.DataFrame,
        column: str, 
        dataset_name: str,
        **kwargs
    ) -> None
        Plot the count distribution of a column with customizable settings.

    Example:
    -------
    >>> plotter = DataPlotter(output_dir='plots')
    >>> plotter.plot_bar_chart(
    ...     title='Bar Chart Example',
    ...     data=df,
    ...     x='category',
    ...     y='value',
    ...     dataset_name='example',
    ...     palette='coolwarm'
    ... )
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.ensure_directory_exists(output_dir)

    @staticmethod
    def ensure_directory_exists(directory: str) -> None:
        """
        Ensure the directory exists, if not, create it.

        :param directory: Path to the directory.
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_bar_chart(
        self,
        title: str,
        data: pd.DataFrame,
        x: str,
        y: str,
        dataset_name: str,
        figsize: tuple = (12, 8),
        xlabel: str = '',
        ylabel: str = '',
        palette: str = 'viridis',
        save_as: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> None:
        """
        Plot a bar chart with customizable settings.

        :param title: Title of the plot.
        :param data: DataFrame containing the data.
        :param x: Column name for the x-axis.
        :param y: Column name for the y-axis.
        :param dataset_name: Name of the dataset for file naming.
        :param figsize: Size of the figure.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param palette: Color palette for the plot.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param kwargs: Additional keyword arguments for sns.barplot.
        :return: None
        """
        plt.figure(figsize=figsize)
        sns.barplot(data=data, x=x, y=y, palette=palette, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir, f'{dataset_name}_bar_chart.{save_as}'
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def plot_pie_chart(
        self,
        title: str,
        data: pd.DataFrame,
        labels: str,
        values: str,
        dataset_name: str,
        figsize: tuple = (8, 8),
        save_as: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> None:
        """
        Plot a pie chart with customizable settings.

        :param title: Title of the plot.
        :param data: DataFrame containing the data.
        :param labels: Column name for the pie labels.
        :param values: Column name for the pie values.
        :param dataset_name: Name of the dataset for file naming.
        :param figsize: Size of the figure.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param kwargs: Additional keyword arguments for plt.pie.
        :return: None
        """
        plt.figure(figsize=figsize)
        plt.pie(data[values], labels=data[labels], **kwargs)
        plt.title(title)
        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir, f'{dataset_name}_pie_chart.{save_as}'
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def plot_spatial_density(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        figsize: tuple = (12, 8),
        cmap: str = 'hot',
        scatter_color: str = 'blue',
        alpha: float = 0.6,
        save_as: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> None:
        """
        Plot the density of data points on a map with customizable settings.

        :param data: DataFrame containing the data with 'latitude' and 'longitude' columns.
        :param dataset_name: Name of the dataset for file naming.
        :param figsize: Size of the figure.
        :param cmap: Color map for the density plot.
        :param scatter_color: Color for the scatter points.
        :param alpha: Transparency level for the density overlay.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param kwargs: Additional keyword arguments for kde and plotting.
        :return: None
        """
        data = data.dropna(subset=['latitude', 'longitude'])
        xy = np.vstack([data['longitude'], data['latitude']]).T

        if xy.size == 0:
            print(f"No valid data for {dataset_name}. Skipping density plot.")
            return

        kde = KernelDensity(bandwidth=0.01, metric='haversine')
        kde.fit(np.radians(xy))

        x, y = np.meshgrid(
            np.linspace(xy[:, 0].min(), xy[:, 0].max(), 100),
            np.linspace(xy[:, 1].min(), xy[:, 1].max(), 100),
        )
        xy_sample = np.vstack([x.ravel(), y.ravel()]).T
        z = np.exp(kde.score_samples(np.radians(xy_sample)))
        z = z.reshape(x.shape)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            z, extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower', cmap=cmap, alpha=alpha,
        )
        ax.scatter(data['longitude'], data['latitude'], s=1, color=scatter_color)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        plt.title('Spatial Density')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        output_path = os.path.join(
            self.output_dir, f'{dataset_name}_spatial_density.{save_as}'
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def plot_spatial_data(
        self,
        title: str,
        shapefile_path: str,
        dataset_name: str,
        figsize: tuple = (12, 8),
        color: str = 'red',
        markersize: int = 5,
        alpha: float = 0.5,
        save_as: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> None:
        """
        Plot a static map using a shapefile with customizable settings.

        :param title: Title of the plot.
        :param shapefile_path: Path to the shapefile.
        :param dataset_name: Name of the dataset for file naming.
        :param figsize: Size of the figure.
        :param color: Color for the plot.
        :param markersize: Size of the markers.
        :param alpha: Transparency level for the markers.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param kwargs: Additional keyword arguments for gpd.plot.
        :return: None
        """
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=figsize)
        gdf.plot(ax=ax, color=color, markersize=markersize, alpha=alpha, **kwargs)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        output_path = os.path.join(
            self.output_dir, f'{dataset_name}_spatial_data.{save_as}'
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def plot_count_distribution(
        self,
        title: str,
        data: pd.DataFrame,
        column: str,
        dataset_name: str,
        figsize: tuple = (12, 8),
        xlabel: str = 'Count',
        ylabel: str = '',
        palette: str = 'viridis',
        save_as: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> None:
        """
        Plot the count distribution of a column with customizable settings.

        :param title: Title of the plot.
        :param data: DataFrame containing the data.
        :param column: Column name for the count distribution.
        :param dataset_name: Name of the dataset for file naming.
        :param figsize: Size of the figure.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param palette: Color palette for the plot.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param kwargs: Additional keyword arguments for sns.countplot.
        :return: None
        """
        plt.figure(figsize=figsize)
        sns.countplot(
            data=data,
            y=column,
            order=data[column].value_counts().index,
            palette=palette,
            **kwargs
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or column)
        plt.tight_layout()

        output_path = os.path.join(
            self.output_dir, f'{dataset_name}_count_distribution.{save_as}'
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()
