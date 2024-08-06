# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-05 04:38:06
# @Info:   Create interactive map
# ============================================================================


import folium
import geopandas as gpd


def create_interactive_map(shapefile_path: str, output_html_path: str) -> None:
    # Load the GeoDataFrame from the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Create an interactive map using folium
    m = folium.Map(location=[-19.9173, -43.9346], zoom_start=7)
    
    # Add accident data points to the interactive map
    for _, row in gdf.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=(
                f"Date: {row['data_inv']}\nType: {row['tipo_acid']}\n"
                f"Severity: {row['classific']}"
            )
        ).add_to(m)
    
    # Save the interactive map to an HTML file
    m.save(output_html_path)
