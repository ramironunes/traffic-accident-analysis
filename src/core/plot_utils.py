# -*- coding: utf-8 -*-
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-08-10 17:05:27
# @Info:   Utility functions for plotting and exporting SARIMAX model results
# ============================================================================


import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def plot_comparison_chart(
    forecast_data: pd.DataFrame,
    predicted_accidents: pd.Series,
    year: int,
    output_img_dir: str,
) -> None:
    """
    Plot and save a comparison chart of actual vs predicted accidents.

    :param forecast_data: DataFrame containing the BR's testing data.
    :param predicted_accidents: Series of predicted accident numbers.
    :param year: The year for which the forecast was made.
    :param output_img_dir: Directory where the output image will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_data["year_month"], forecast_data["accidents"], label="Actual")
    plt.plot(
        forecast_data["year_month"],
        predicted_accidents,
        label="Predicted",
        linestyle="--",
    )
    plt.title(f"Actual vs Predicted Accidents for {year}")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(output_img_dir, f"BR_forecast_comparison_{year}.png")
    plt.savefig(output_path)
    plt.close()


def export_to_excel(
    forecast_data: pd.DataFrame,
    predictions: pd.Series,
    year: int,
    output_dir: str,
) -> None:
    """
    Export the actual and forecasted data along with performance metrics to an Excel file.

    Args:
        forecast_data (pd.DataFrame): DataFrame containing the actual data.
        predictions (pd.Series): Series containing the forecasted data.
        year (int): The year for which the export is being made.
        output_dir (str): The directory to save the output Excel file.
    """
    mape = mean_absolute_percentage_error(forecast_data["accidents"], predictions)
    rmse = mean_squared_error(forecast_data["accidents"], predictions, squared=False)
    forecast_data["Previsão SARIMA"] = predictions
    forecast_data["Erro (%)"] = (
        abs((forecast_data["accidents"] - predictions) / forecast_data["accidents"])
        * 100
    )
    forecast_data["Erro Absoluto"] = abs(forecast_data["accidents"] - predictions)

    # Save to Excel
    excel_path = os.path.join(output_dir, f"performance_sarima_{year}.xlsx")
    forecast_data.to_excel(excel_path, index=False)

    # Add summary sheet with metrics
    with pd.ExcelWriter(excel_path, mode="a") as writer:
        summary_df = pd.DataFrame({"Métrica": ["MAPE", "RMSE"], "Valor": [mape, rmse]})
        summary_df.to_excel(writer, sheet_name="Resumo de Performance", index=False)

    print(f"Performance data exported to {excel_path}")
