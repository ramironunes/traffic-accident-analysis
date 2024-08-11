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
import textwrap

from graphviz import Digraph
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
        graph_type: str,
        figsize: tuple = (12, 8),
        xlabel: str = "",
        ylabel: str = "",
        palette: str = "viridis",
        save_as: str = "png",
        dpi: int = 100,
        orient: str = "vertical",
        exclude_zero: bool = False,
        limit: int = None,
        **kwargs,
    ) -> None:
        """
        Plot a bar chart with customizable settings.

        :param title: Title of the plot.
        :param data: DataFrame containing the data.
        :param x: Column name for the x-axis.
        :param y: Column name for the y-axis.
        :param dataset_name: Name of the dataset for file naming.
        :param graph_type: Type of the graph for directory structure.
        :param figsize: Size of the figure.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param palette: Color palette for the plot.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param orient: Orientation of the bar chart ('vertical' or 'horizontal').
        :param exclude_zero: Flag to exclude categories with zero counts.
        :param limit: Limit the number of data points displayed.
        :param kwargs: Additional keyword arguments for sns.barplot.
        :return: None
        """
        plt.figure(figsize=figsize)

        if orient == "horizontal":
            # Check if the column y exists in the DataFrame
            if y not in data.columns:
                raise KeyError(f"Column '{y}' does not exist in the DataFrame")

            # Count the values in the y column and create a new DataFrame with counts
            count_data = data[y].value_counts().reset_index()
            count_data.columns = [y, x]

            # Exclude zero counts if the flag is set
            if exclude_zero:
                count_data = count_data[count_data[x] > 0]

            # Limit the number of data points if the limit is set
            if limit is not None:
                count_data = count_data.head(limit)

            # Plot horizontal bar chart
            sns.barplot(data=count_data, x=x, y=y, palette=palette, **kwargs)
            plt.xlabel(xlabel or x)
            plt.ylabel(ylabel or y)
        else:
            # Check if the column x exists in the DataFrame
            if x not in data.columns:
                raise KeyError(f"Column '{x}' does not exist in the DataFrame")

            # Count the values in the x column and create a new DataFrame with counts
            count_data = data[x].value_counts().reset_index()
            count_data.columns = [x, y]

            # Exclude zero counts if the flag is set
            if exclude_zero:
                count_data = count_data[count_data[y] > 0]

            # Limit the number of data points if the limit is set
            if limit is not None:
                count_data = count_data.head(limit)

            # Plot vertical bar chart
            sns.barplot(data=count_data, x=x, y=y, palette=palette, **kwargs)
            plt.xlabel(xlabel or x)
            plt.ylabel(ylabel or y)

        plt.title(title)
        plt.tight_layout()

        # Update output_path to include the type of plot
        output_path = os.path.join(
            self.output_dir, graph_type, f"{dataset_name}_bar_chart.{save_as}"
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def plot_pie_chart(
        self,
        title: str,
        data: pd.DataFrame,
        labels_col: str,
        values_col: str,
        dataset_name: str,
        graph_type: str,
        figsize: tuple = (8, 8),
        save_as: str = "png",
        dpi: int = 100,
        legend: bool = False,
        limit: int = None,
        colors: list = None,
        wrap_length: int = 50,
        **kwargs,
    ) -> None:
        """
        Plot a pie chart with labels showing values and percentages.

        :param title: Title of the plot.
        :param data: DataFrame containing the data.
        :param labels_col: Column name for the labels.
        :param values_col: Column name for the values.
        :param dataset_name: Name of the dataset for file naming.
        :param graph_type: Type of the graph for directory structure.
        :param figsize: Size of the figure.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param legend: Flag to show legend instead of labels on the pie chart.
        :param limit: Limit the number of data points displayed.
        :param colors: List of colors for the pie chart.
        :param wrap_length: Maximum length of label lines before wrapping.
        :param kwargs: Additional keyword arguments for plt.pie.
        :return: None
        """
        plt.figure(figsize=figsize)

        # Get the labels and values for the pie chart
        labels = data[labels_col]
        values = data[values_col]

        # Limit the number of data points if the limit is set
        if limit is not None:
            labels = labels.head(limit)
            values = values.head(limit)

        # Wrap labels to specified length
        labels = [textwrap.fill(label, wrap_length) for label in labels]

        # Define a color palette for color blindness if colors are not provided
        if colors is None:
            colors = sns.color_palette("colorblind", len(labels))

        # Create the pie chart
        wedges, texts, autotexts = plt.pie(
            values,
            labels=None if legend else labels,
            autopct="%1.1f%%",
            colors=colors,
            **kwargs,
        )

        if legend:
            # Display a legend with the labels and colors below the pie chart
            plt.legend(
                wedges,
                labels,
                title=labels_col,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.1),
                ncol=2,
            )

        # Customize the font properties for the labels and percentages
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color("white")

        plt.title(title)
        plt.tight_layout()

        print("output_dir", self.output_dir)

        # Update output_path to include the type of plot
        output_path = os.path.join(
            self.output_dir, graph_type, f"{dataset_name}_pie_chart.{save_as}"
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def plot_spatial_density(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        graph_type: str,
        figsize: tuple = (12, 8),
        cmap: str = "viridis",
        scatter_color: str = "blue",
        alpha: float = 0.6,
        save_as: str = "png",
        dpi: int = 100,
        title: str = "Spatial Density",
        xlabel: str = "Longitude",
        ylabel: str = "Latitude",
        add_colorbar: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot the density of data points on a map with customizable settings.

        :param data: DataFrame containing the data with 'latitude' and 'longitude' columns.
        :param dataset_name: Name of the dataset for file naming.
        :param graph_type: Type of the graph for directory structure.
        :param figsize: Size of the figure.
        :param cmap: Color map for the density plot.
        :param scatter_color: Color for the scatter points.
        :param alpha: Transparency level for the density overlay.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param add_colorbar: Whether to add a colorbar to the plot.
        :param kwargs: Additional keyword arguments for kde and plotting.
        :return: None
        """
        data = data.dropna(subset=["latitude", "longitude"])
        xy = np.vstack([data["longitude"], data["latitude"]]).T

        if xy.size == 0:
            print(f"No valid data for {dataset_name}. Skipping density plot.")
            return

        kde = KernelDensity(bandwidth=0.01, metric="haversine")
        kde.fit(np.radians(xy))

        x, y = np.meshgrid(
            np.linspace(xy[:, 0].min(), xy[:, 0].max(), 100),
            np.linspace(xy[:, 1].min(), xy[:, 1].max(), 100),
        )
        xy_sample = np.vstack([x.ravel(), y.ravel()]).T
        z = np.exp(kde.score_samples(np.radians(xy_sample)))
        z = z.reshape(x.shape)

        fig, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(
            z,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            interpolation="nearest",
        )
        ax.scatter(data["longitude"], data["latitude"], s=1, color=scatter_color)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if add_colorbar:
            cbar = plt.colorbar(img, ax=ax, orientation="vertical")
            cbar.set_label("Density")

        # Update output_path to include the type of plot
        output_path = os.path.join(
            self.output_dir, graph_type, f"{dataset_name}_spatial_density.{save_as}"
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
        color: str = "red",
        markersize: int = 5,
        alpha: float = 0.5,
        save_as: str = "png",
        dpi: int = 100,
        **kwargs,
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
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        output_path = os.path.join(
            self.output_dir, f"{dataset_name}_spatial_data.{save_as}"
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
        graph_type: str,
        figsize: tuple = (12, 8),
        xlabel: str = "Count",
        ylabel: str = "",
        palette: str = "viridis",
        save_as: str = "png",
        dpi: int = 100,
        orient: str = "vertical",
        exclude_zero: bool = False,
        limit: int = None,
        **kwargs,
    ) -> None:
        """
        Plot the count distribution of a column with customizable settings.

        :param title: Title of the plot.
        :param data: DataFrame containing the data.
        :param column: Column name for the count distribution.
        :param dataset_name: Name of the dataset for file naming.
        :param graph_type: Type of the graph for directory structure.
        :param figsize: Size of the figure.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param palette: Color palette for the plot.
        :param save_as: File format to save the plot.
        :param dpi: Resolution of the saved plot.
        :param orient: Orientation of the bar chart ('vertical' or 'horizontal').
        :param exclude_zero: Flag to exclude categories with zero counts.
        :param limit: Limit the number of categories displayed.
        :param kwargs: Additional keyword arguments for sns.countplot.
        :return: None
        """
        plt.figure(figsize=figsize)

        # Order the data by count
        ordered_data = data[column].value_counts()
        if exclude_zero:
            ordered_data = ordered_data[ordered_data > 0]
        if limit is not None:
            ordered_data = ordered_data.head(limit)

        ordered_data = ordered_data.reset_index()
        ordered_data.columns = [column, "count"]

        if orient == "horizontal":
            sns.barplot(
                data=ordered_data, x="count", y=column, palette=palette, **kwargs
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel or column)
        else:
            sns.barplot(
                data=ordered_data, x=column, y="count", palette=palette, **kwargs
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel or "Count")

        plt.title(title)
        plt.tight_layout()

        # Update output_path to include the type of plot
        output_path = os.path.join(
            self.output_dir, graph_type, f"{dataset_name}_count_distribution.{save_as}"
        )
        self.ensure_directory_exists(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=dpi)
        plt.close()

    def create_improved_sarimax_diagram(output_path: str) -> None:
        """
        Creates an improved diagram illustrating the components of the SARIMAX model.
        """
        # Create a Digraph object
        dot = Digraph(comment="Improved SARIMAX Model Components")

        # Add nodes with different shapes and colors
        dot.node(
            "A",
            "AR (AutoRegressive)",
            shape="ellipse",
            style="filled",
            color="lightblue",
        )
        dot.node(
            "I", "I (Integrated)", shape="ellipse", style="filled", color="lightyellow"
        )
        dot.node(
            "M",
            "MA (Moving Average)",
            shape="ellipse",
            style="filled",
            color="lightgreen",
        )
        dot.node("S", "Seasonality", shape="ellipse", style="filled", color="lightpink")
        dot.node(
            "X",
            "Exogenous Variables\n(Traffic Volume)",
            shape="ellipse",
            style="filled",
            color="lightcoral",
        )
        dot.node(
            "SARIMAX", "SARIMAX", shape="ellipse", style="filled", color="lightgrey"
        )

        # Define the edges (connections)
        dot.edge("A", "SARIMAX")
        dot.edge("I", "SARIMAX")
        dot.edge("M", "SARIMAX")
        dot.edge("S", "SARIMAX")
        dot.edge("X", "SARIMAX")

        # Save the diagram to a file
        dot.render(output_path, format="png", cleanup=True)
        print(f"Improved diagram saved to {output_path}.png")

    def create_sarimax_data_flow_diagram(output_path: str) -> None:
        """
        Creates a data flow diagram illustrating how traffic volume data is used
        in the SARIMAX model to predict traffic accidents.
        """
        # Create a Digraph object
        dot = Digraph(comment="SARIMAX Data Flow Diagram")

        # Define nodes for the data flow
        dot.node(
            "Data", "Traffic Data", shape="box", style="filled", color="lightyellow"
        )
        dot.node(
            "Preprocessing",
            "Preprocessing",
            shape="box",
            style="filled",
            color="lightblue",
        )
        dot.node(
            "Exogenous",
            "Exogenous Variables\n(Traffic Volume)",
            shape="box",
            style="filled",
            color="lightcoral",
        )
        dot.node(
            "SARIMAX",
            "SARIMAX Model",
            shape="ellipse",
            style="filled",
            color="lightgrey",
        )
        dot.node(
            "Forecast",
            "Predicted Accidents",
            shape="box",
            style="filled",
            color="lightgreen",
        )

        # Define edges to show data flow
        dot.edge("Data", "Preprocessing", label="Load & Merge Data")
        dot.edge("Preprocessing", "Exogenous", label="Extract Exogenous Variables")
        dot.edge("Preprocessing", "SARIMAX", label="Pass to SARIMAX Model")
        dot.edge("Exogenous", "SARIMAX", label="Use as Exogenous Variables")
        dot.edge("SARIMAX", "Forecast", label="Generate Predictions")

        # Save the diagram to a file
        dot.render(output_path, format="png", cleanup=True)
        print(f"Data flow diagram saved to {output_path}.png")
