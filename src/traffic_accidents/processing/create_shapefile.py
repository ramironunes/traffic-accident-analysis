# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 04:37:43
# @Info:   Create shapefile
# ============================================================================


import geopandas as gpd
import pandas as pd

from shapely.geometry import Point


def create_shapefile(data: pd.DataFrame, output_path: str) -> None:
    """
    Create a shapefile from the traffic accident data.

    :param data: DataFrame containing the traffic accident data.
    :param output_path: Path to save the shapefile.
    :return: None
    """
    # data['latitude'] = data['latitude'].apply(lambda x: round(x, 5))
    # data['longitude'] = data['longitude'].apply(lambda x: round(x, 5))

    # geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    # gdf = gpd.GeoDataFrame(data, geometry=geometry)
    # gdf.set_crs(epsg=4326, inplace=True)
    # gdf.to_file(output_path, driver='ESRI Shapefile')
    # Convert latitude and longitude to geospatial points
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry)
    
    # Ensure values are within acceptable ranges
    gdf['latitude'] = gdf['latitude'].apply(lambda x: x if abs(x) < 90 else None)
    gdf['longitude'] = gdf['longitude'].apply(lambda x: x if abs(x) < 180 else None)
    
    # Drop rows with invalid geometry
    gdf = gdf.dropna(subset=['geometry'])
    
    # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Save GeoDataFrame to shapefile
    gdf.to_file(output_path, driver='ESRI Shapefile')
