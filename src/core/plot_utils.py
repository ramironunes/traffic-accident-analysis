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
    forecast_data: pd.DataFrame, predictions: pd.Series, year: int, output_img_dir: str
) -> None:
    """
    Plot a comparison chart between the actual and forecasted data for a specific year.

    Args:
        forecast_data (pd.DataFrame): DataFrame containing the actual data.
        predictions (pd.Series): Series containing the forecasted data.
        year (int): The year for which the comparison is being made.
        output_img_dir (str): The directory to save the output image.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(
        forecast_data["data"],
        forecast_data["accidents"],
        label="Acidentes Reais",
        marker="o",
    )
    plt.plot(forecast_data["data"], predictions, label="Previsão SARIMA", marker="x")
    plt.title(f"Comparação entre Acidentes Reais e Previstos - {year}", fontsize=16)
    plt.xlabel("Mês", fontsize=14)
    plt.ylabel("Número de Acidentes", fontsize=14)
    plt.xticks(
        forecast_data["data"], forecast_data["data"].dt.strftime("%b"), rotation=45
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, f"sarima_comparison_{year}.png"), dpi=100)
    plt.close()


def export_to_excel(
    forecast_data: pd.DataFrame, predictions: pd.Series, year: int, output_dir: str
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
