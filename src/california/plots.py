import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lightgbm import plot_importance
from matplotlib.ticker import EngFormatter
from sklearn.base import RegressorMixin
from sklearn.metrics import PredictionErrorDisplay
from sklearn.pipeline import Pipeline

from typing import Optional

from .models import RANDOM_STATE

SCATTER_ALPHA = 0.2


def plot_county_map(
    gdf_counties: gpd.GeoDataFrame,
    gdf_points: Optional[pd.DataFrame] = None,
    show_scatter: bool = False,
    show_abbrev: bool = False,
    scatter_alpha: float = 0.2,
    fig_title: str = "",
    gdf_counties_plot_kw: Optional[dict] = None,
    axis: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot a county map using GeoPandas with optional scatter points and county abbreviations.

    Parameters
    ----------
    gdf_counties : gpd.GeoDataFrame
        GeoDataFrame containing county geometries. Must have 'centroid' and 'abbrev' columns.
    gdf_points : Optional[pd.DataFrame], optional
        DataFrame containing 'longitude' and 'latitude' of data points to scatter,
        optionally including 'centroid' and 'abbrev' for reference (default is None).
    show_scatter : bool, optional
        Whether to show the red scatter points (default is False).
    show_abbrev : bool, optional
        Whether to annotate counties with abbreviations at their centroids (default is False).
    scatter_alpha : float, optional
        Transparency level for the scatter plot (default is 0.2).

    Returns
    -------
    Optional[plt.Axes]
        Axes object if `axis` is provided, otherwise displays the plot.
    """

    if not axis:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # Plot county shapes
    gdf_counties.plot(
        ax=ax,
        edgecolor="black",
        color="lightgrey" if gdf_counties_plot_kw is None else None,
        **(gdf_counties_plot_kw if gdf_counties_plot_kw else {}),
    )

    # Optionally add scatter points
    if show_scatter and gdf_points is not None:
        ax.scatter(
            gdf_points["longitude"],
            gdf_points["latitude"],
            color="C1",
            s=1,
            alpha=scatter_alpha,
        )

    # Optionally add county abbreviations
    if show_abbrev:
        for x, y, abbrev in zip(
            gdf_counties["centroid"].x,
            gdf_counties["centroid"].y,
            gdf_counties["abbrev"],
        ):
            ax.text(
                x,
                y,
                abbrev,
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
            )

    if fig_title and not axis:
        fig.suptitle(fig_title)
    else:
        ax.set_title(fig_title)

    if axis is None:
        plt.show()
    else:
        return ax


def plot_coefficients(
    df_coefs: "pd.DataFrame", title: str = "Coefficients"
) -> None:
    """
    Plot model coefficients as a horizontal bar chart.

    Parameters
    ----------
    df_coefs : DataFrame
        DataFrame containing model coefficients.
    title : str, optional
        Title for the plot (default is "Coefficients").

    Returns
    -------
    None
    """
    df_coefs.plot.barh()
    plt.title(title)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient Value")
    plt.gca().get_legend().remove()
    plt.show()


def plot_residuals(y_true: "pd.Series", y_pred: "pd.Series") -> None:
    """
    Plot residual diagnostics given true and predicted values.

    Parameters
    ----------
    y_true : Series
        True target values.
    y_pred : Series
        Predicted target values.

    Returns
    -------
    None
    """
    residuals = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residuals, kde=True, ax=axs[0])

    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    plt.tight_layout()
    plt.show()


def plot_residuals_from_estimator(
    estimator: "RegressorMixin",
    X: "pd.DataFrame",
    y: "pd.Series",
    eng_formatter: bool = False,
    sample_fraction: float = 0.25,
    figsize: tuple = (12, 6),
) -> None:
    """
    Plot residual diagnostics directly from a fitted estimator.

    Parameters
    ----------
    estimator : RegressorMixin
        Trained regression model.
    X : DataFrame
        Feature matrix.
    y : Series
        Target vector.
    eng_formatter : bool, optional
        Whether to apply engineering notation to axes (default is False).
    sample_fraction : float, optional
        Fraction of points to sample for scatter plots (default is 0.25).
    figsize : tuple, optional
        Size of the figure (default is (12, 6)).

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    error_display_residual = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=sample_fraction,
    )

    error_display_actual = PredictionErrorDisplay.from_estimator(  # NoQA
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=sample_fraction,
    )

    residuals = error_display_residual.y_true - error_display_residual.y_pred
    sns.histplot(residuals, kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter(places=1))
            ax.xaxis.set_major_formatter(EngFormatter(places=1))

    # change labels fontsize
    for ax in axs:
        ax.tick_params(axis="x", labelsize="x-small")
        ax.tick_params(axis="y", labelsize="x-small")

    plt.tight_layout()
    plt.show()


def plot_compare_model_metrics(
    df_results: pd.DataFrame, figsize=(8, 8)
) -> None:
    """
    Plot boxplots to compare model evaluation metrics.

    Parameters
    ----------
    df_results : DataFrame
        DataFrame containing model evaluation results.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)

    metrics = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    metric_names = [
        "Time (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metric, name in zip(axs.flatten(), metrics, metric_names):
        sns.boxplot(
            x="model",
            y=metric,
            data=df_results,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()


def plot_importance_lgbm(
    grid_search_estimator: Pipeline,
    importance_type: str = "gain",
    precision: float = 1,
) -> None:
    """
    Plot feature importance for a LightGBM regressor.

    Parameters
    ----------
    regressor : RegressorMixin
        Trained LightGBM regression model.
    importance_type : str, optional
        Type of importance to plot (default is "gain").
    precision : float, optional
        Precision for formatting the importance values (default is 1).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()

    regressor = grid_search_estimator["reg"]
    preprocessor = grid_search_estimator["preprocessor"]

    importances = regressor.booster_.feature_importance(importance_type)

    all_feature_names = preprocessor.get_feature_names_out()

    valid_indices = np.where(importances > 0)[0]
    sorted_valid_indices = valid_indices[np.argsort(importances[valid_indices])]

    feature_names = np.array(all_feature_names)[sorted_valid_indices]

    title = f"Feature importance - {importance_type}"

    plot_importance(
        regressor,
        importance_type=importance_type,
        title=title,
        precision=precision,
        ax=ax,
    )

    ax.set_yticklabels(feature_names)

    # Reformat bar labels to scientific notation
    for text in ax.texts:
        try:
            value = float(text.get_text())
            text.set_text(f"{value:.2e}")  # scientific format, e.g., 1.2e+03
        except ValueError:
            pass  # skip if it's not a number

    plt.show()
