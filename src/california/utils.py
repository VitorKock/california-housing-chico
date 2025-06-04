import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import statsmodels.stats.api as sms

from shapely.geometry import Point
from scipy.stats import (
    kstest,
    norm,
    rv_continuous,
    shapiro,
    zscore,
)

from typing import List, NamedTuple, Optional, Tuple, Union


CURRENT_COLOR_CYCLER = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class VIFReductionResult(NamedTuple):
    original_vif: pd.DataFrame
    drop_history: pd.DataFrame
    final_features: List[str]
    final_vif: pd.DataFrame


def create_coefficients_dataframe(
    coefs: Union[np.ndarray, list], features: list
) -> pd.DataFrame:
    """
    Create a sorted DataFrame of model coefficients.

    Parameters
    ----------
    coefs : ndarray or list
        List or array of model coefficients.
    columns : list
        Feature names corresponding to the coefficients.

    Returns
    -------
    df_coefs : DataFrame
        DataFrame with coefficients sorted by value.
    """
    df_coefs = pd.DataFrame(
        data=coefs, index=features, columns=["coefficient"]
    ).sort_values(by="coefficient")

    return df_coefs


def best_grid_shape(n: int) -> tuple[int, int]:
    """
    Compute the best (rows, cols) arrangement for a given number of plots.

    The goal is to create a nearly square grid that efficiently fits all plots.

    Parameters
    ----------
    n : int
        Number of plots.

    Returns
    -------
    tuple[int, int]
        Optimal (rows, cols) grid size.
    """
    if n == 1:
        return (1, 1)

    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    return rows, cols


def nearest_county(row: pd.Series, geodf: gpd.GeoDataFrame) -> pd.Series:
    """
    Find the nearest county to a given geometry point.

    This function computes the distance between a point (from a GeoDataFrame row)
    and the centroids of all counties in the global `gdf_counties` GeoDataFrame,
    returning the name and abbreviation of the closest county.

    Parameters
    ----------
    row : pd.Series
        A row from a GeoDataFrame, expected to contain a 'geometry' field
        with a shapely Point or geometry.
    geodf : gpd.GeoDataFrame
        A GeoDataFrame containing county geometries and their attributes,
        including 'name' and 'abbrev'.

    Returns
    -------
    pd.Series
        A Series containing the 'name' and 'abbrev' of the closest county.
    """
    point: Point = row["geometry"]
    distances = geodf["centroid"].distance(point)
    nearest_county_idx = distances.idxmin()
    nearest_county = geodf.loc[nearest_county_idx]
    return nearest_county[["name", "abbrev"]]


class StatisticalAnalysisRegression:
    """
    Perform statistical analysis on a regression dataset.

    This class provides methods for analyzing distribution shapes, normality,
    detecting outliers, heteroscedasticity, multicollinearity, and correlation
    patterns.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        self.dataframe_isolation_forest = None

    def feature_null_zero_minmax_summary(self) -> pd.DataFrame:
        """
        Summarize key statistics per column:
        - Number and percentage of null values
        - Number and percentage of zero values
        - Number and percentage of minimum value occurrences
        - Number and percentage of maximum value occurrences

        Returns
        -------
        pd.DataFrame
            Profile summary per column, sorted by null counts descending.
        """
        total_rows = len(self.dataframe)
        results = []

        for column in self.dataframe.select_dtypes("number").columns:
            col_data = self.dataframe[column]

            null_cells = col_data.isnull().sum()
            zero_cells = (col_data == 0).sum()

            # Safely skip min/max calculations if the column is entirely null
            if col_data.dropna().empty:
                min_cells = np.nan
                max_cells = np.nan
            else:
                min_val = col_data.min()
                max_val = col_data.max()
                min_cells = (col_data == min_val).sum()
                max_cells = (col_data == max_val).sum()

            results.append(
                {
                    "column": column,
                    "null_count": null_cells,
                    "null_percentage": (null_cells / total_rows) * 100,
                    "zero_count": zero_cells,
                    "zero_percentage": (zero_cells / total_rows) * 100,
                    "min_count": min_cells,
                    "min_percentage": (min_cells / total_rows) * 100
                    if pd.notnull(min_cells)
                    else np.nan,
                    "max_count": max_cells,
                    "max_percentage": (max_cells / total_rows) * 100
                    if pd.notnull(max_cells)
                    else np.nan,
                }
            )

        summary = pd.DataFrame(results)
        summary = summary.sort_values(
            by="null_count", ascending=False
        ).reset_index(drop=True)

        return summary

    def shape_analysis(self, columns: List[str]) -> pd.DataFrame:
        """
        Analyze the shape of the distribution of numerical columns.

        This method calculates skewness and kurtosis for the specified columns
        and classifies them based on their symmetry and tailedness.

        Skewness Classification:
        - "symmetric" if skewness is between -0.05 and 0.05.
        - "right-skewed" if skewness is greater than or equal to 0.05.
        - "left-skewed" if skewness is less than -0.05.

        Kurtosis Classification:
        - "mesokurtic" if kurtosis is between -0.05 and 0.05 (normal distribution).
        - "platykurtic" if kurtosis is less than -0.05 (flatter distribution).
        - "leptokurtic" if kurtosis is greater than 0.05 (more peaked distribution).

        Parameters
        ----------
        columns : list[str]
            List of numerical column names to analyze.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing:
            - "column": Name of the column.
            - "skewness": Skewness value.
            - "skewness_classification": Classification of skewness.
            - "kurtosis": Kurtosis value.
            - "kurtosis_classification": Classification of kurtosis.
        """
        results = []
        for column in columns:
            skewness = self.dataframe[column].skew()
            kurtosis = self.dataframe[column].kurtosis()

            skewness_class = (
                "symmetric"
                if -0.05 < skewness < 0.05
                else ("right-skewed" if skewness > 0.05 else "left-skewed")
            )
            kurtosis_class = (
                "mesokurtic"
                if -0.05 < kurtosis < 0.05
                else ("platykurtic" if kurtosis < -0.05 else "leptokurtic")
            )

            results.append(
                {
                    "column": column,
                    "skewness": skewness,
                    "skewness_class": skewness_class,
                    "kurtosis": kurtosis,
                    "kurtosis_class": kurtosis_class,
                }
            )

        return pd.DataFrame(results)

    def shapiro_wilk(
        self, columns: List[str], alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Perform the Shapiro-Wilk test for normality.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        for column in columns:
            stat, p_value = shapiro(self.dataframe[column], nan_policy="omit")
            results.append(
                {
                    "column": column,
                    "statistic": stat,
                    "p_value": p_value,
                    "normal": p_value > alpha,
                }
            )

        return pd.DataFrame(results)

    def kolmogorov_smirnov(
        self,
        columns: List[str],
        alpha: float = 0.05,
        reference_cdf: Union[rv_continuous, np.ndarray] = norm.cdf,
        mode: str = "auto",
    ) -> pd.DataFrame:
        """
        Perform the Kolmogorov-Smirnov test for normality.

        Parameters
        ----------
        columns : list[str]
            Columns to test.
        alpha : float, optional
            Significance level, by default 0.05.
        reference_cdf : rv_continuous or np.ndarray, optional
            Cumulative distribution function to compare against.
            Defaults to the standard normal distribution (norm.cdf).
        mode : str, optional
            Method to estimate the distribution, by default "auto".

        Returns
        -------
        pd.DataFrame
            DataFrame with the test results.
        """
        results = []
        for column in columns:
            standardized = zscore(
                self.dataframe[column], ddof=1, nan_policy="omit"
            )
            stat, p_value = kstest(
                standardized, reference_cdf, mode=mode, nan_policy="omit"
            )
            results.append(
                {
                    "column": column,
                    "statistic": stat,
                    "p_value": p_value,
                    "normal": p_value > alpha,
                }
            )

        return pd.DataFrame(results)

    def vif_analysis(self, features: List[str]) -> pd.DataFrame:
        """Calculate Variance Inflation Factor (VIF) for the specified features.
        VIF quantifies how much the variance of a regression coefficient is
        inflated due to multicollinearity with other features. A VIF above 5
        typically indicates high multicollinearity.

        Parameters
        ----------
        features : list of str
            List of numeric feature names to evaluate for multicollinearity.

        Returns
        -------
        pd.DataFrame
            DataFrame with VIF scores for each feature.
            Columns: "feature", "VIF".
        """

        X = self.dataframe[features]
        X = sm.add_constant(X)

        vif_data = {"feature": [], "VIF": []}

        for i in range(1, X.shape[1]):
            vif = (
                sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1))
                .fit()
                .rsquared
            )
            vif_score = 1 / (1 - vif)
            vif_data["feature"].append(X.columns[i])
            vif_data["VIF"].append(vif_score)

        return pd.DataFrame(vif_data)

    def reduce_vif(
        self, features: List[str], threshold: float = 5.0, verbose: bool = True
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Iteratively remove features with the highest VIF until all are below the
        threshold.  Also returns a history of which features were dropped and
        their VIF values.

        Parameters
        ----------
        features : list of str
            List of numeric feature names to evaluate for multicollinearity.
        threshold : float, optional
            VIF threshold (default is 5.0).
        verbose : bool, optional
            Whether to print dropped features during execution (default is True).

        Returns
        -------
        Tuple[List[str], pd.DataFrame]
            - List of remaining features below the VIF threshold.
            - DataFrame with VIF scores of dropped features at each step.
        """

        current_features = features.copy()
        drop_history = []

        while True:
            X = self.dataframe[current_features].dropna()
            X = sm.add_constant(X)

            vif_scores = pd.Series(index=current_features, dtype=float)

            for i, feature in enumerate(current_features):
                other_cols = [
                    col for col in X.columns if col not in ["const", feature]
                ]
                model = sm.OLS(X[feature], X[["const"] + other_cols]).fit()
                r2 = model.rsquared
                vif_scores[feature] = 1 / (1 - r2) if r2 < 1 else np.inf

            max_vif = vif_scores.max()
            to_drop = vif_scores.idxmax()

            if max_vif < threshold:
                break

            # Drop and record
            current_features.remove(to_drop)
            drop_history.append({"feature": to_drop, "vif": max_vif})

            if verbose:
                print(f"Dropped '{to_drop}' (VIF = {max_vif:.2f})")

        history_df = pd.DataFrame(drop_history)

        return current_features, history_df

    def vif_reduce_summary(
        self, features: List[str], threshold: float = 5.0, verbose: bool = True
    ) -> VIFReductionResult:
        """
        Performs VIF analysis and iteratively removes features with high VIF.
        Returns original VIFs, drop history, final features, and final VIFs.

        Parameters
        ----------
        features : list of str
            Initial features to analyze for multicollinearity.
        threshold : float, optional
            Maximum allowed VIF (default is 5.0).
        verbose : bool, optional
            Whether to print dropped features during execution.

        Returns
        -------
        VIFReductionResult : NamedTuple
            - original_vif: VIFs of all initial features.
            - drop_history: VIFs of features removed during reduction.
            - final_features: List of retained features after reduction.
            - final_vif: Final VIFs of retained features.
        """

        def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
            X = sm.add_constant(X)
            vif_data = {"feature": [], "VIF": []}
            for i in range(1, X.shape[1]):
                r2 = (
                    sm.OLS(X.iloc[:, i], X.drop(X.columns[i], axis=1))
                    .fit()
                    .rsquared
                )
                vif_score = 1 / (1 - r2) if r2 < 1 else np.inf
                vif_data["feature"].append(X.columns[i])
                vif_data["VIF"].append(vif_score)
            return pd.DataFrame(vif_data)

        current_features = features.copy()
        drop_history = []

        # Step 1: Compute original VIF
        X_initial = self.dataframe[current_features].dropna()
        original_vif = compute_vif(X_initial)

        # Step 2: Iteratively remove high-VIF features
        while True:
            X_current = self.dataframe[current_features].dropna()
            vif_df = compute_vif(X_current)

            max_vif = vif_df["VIF"].max()
            to_drop = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]

            if max_vif < threshold:
                break

            current_features.remove(to_drop)
            drop_history.append({"feature": to_drop, "VIF": max_vif})

            if verbose:
                print(f"Dropped '{to_drop}' (VIF = {max_vif:.2f})")

        drop_history_df = pd.DataFrame(drop_history)
        final_vif = compute_vif(self.dataframe[current_features].dropna())

        return VIFReductionResult(
            original_vif=original_vif,
            drop_history=drop_history_df,
            final_features=current_features,
            final_vif=final_vif,
        )

    def breusch_pagan_test(
        self, model: sm.regression.linear_model.RegressionResultsWrapper
    ) -> dict:
        """
        Perform the Breusch-Pagan test for heteroscedasticity.  This test checks
        if the variance of the residuals is constant across all levels of the
        independent variables.  The null hypothesis is that the variance of the
        residuals is constant (homoscedasticity).  The alternative hypothesis is
        that the variance of the residuals is not constant (heteroscedasticity).
        The test returns the Lagrange Multiplier statistic and its p-value.  The
        test is performed using the statsmodels library.

        Parameters
        ----------
        model : sm.regression.linear_model.RegressionResultsWrapper
            Fitted regression model.
            The model should be fitted using the statsmodels library.
            The model should have a residuals attribute and a model attribute
            with an exogenous variable.

        Returns
        -------
        dict
            A dictionary containing the test statistic, p-value, and a boolean
            indicating whether heteroscedasticity is present.
            The keys are:
            - "lm_statistic": Lagrange Multiplier statistic.
            - "lm_pvalue": p-value for the Lagrange Multiplier statistic.
            - "f_statistic": F-statistic for the test.
            - "f_pvalue": p-value for the F-statistic.
            - "heteroscedasticity_present": boolean indicating if heteroscedasticity
              is present (True) or not (False).
        """

        lm_stat, lm_pvalue, f_stat, f_pvalue = sms.het_breuschpagan(
            model.resid, model.model.exog
        )

        return {
            "lm_statistic": lm_stat,
            "lm_pvalue": lm_pvalue,
            "f_statistic": f_stat,
            "f_pvalue": f_pvalue,
            "heteroscedasticity_present": lm_pvalue < 0.05,
        }

    def durbin_watson_test(
        self, model: sm.regression.linear_model.RegressionResultsWrapper
    ) -> float:
        """
        Perform the Durbin-Watson test for autocorrelation in the residuals.
        The test statistic ranges from 0 to 4, where:
        - 2 indicates no autocorrelation.
        - <2 indicates positive autocorrelation.
        - >2 indicates negative autocorrelation.
        The test is performed using the statsmodels library.

        Parameters
        ----------
        model : sm.regression.linear_model.RegressionResultsWrapper
            Fitted regression model.
            The model should be fitted using the statsmodels library.
            The model should have a residuals attribute.
            The model should have a model attribute with an exogenous variable.

        Returns
        -------
        float
            The Durbin-Watson test statistic.
            The value ranges from 0 to 4, where:
            - 2 indicates no autocorrelation.
            - <2 indicates positive autocorrelation.
            - >2 indicates negative autocorrelation.
        """
        return sms.durbin_watson(model.resid)

    def detect_outliers_zscore(
        self, columns: List[str], threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.

        Parameters
        ----------
        columns : list of str
            List of columns to analyze.
        threshold : float
            Z-score threshold to consider an observation an outlier (default 3.0).

        Returns
        -------
        pd.DataFrame
            DataFrame with boolean indicators for outliers.
        """
        outlier_flags = pd.DataFrame(index=self.dataframe.index)

        for column in columns:
            z_scores = zscore(self.dataframe[column], ddof=1, nan_policy="omit")
            outlier_flags[column + "_outlier_zscore"] = (
                np.abs(z_scores) > threshold
            )

        return outlier_flags

    def detect_outliers_modified_zscore(
        self, columns: List[str], threshold: float = 3.5
    ) -> pd.DataFrame:
        """
        Detect outliers using the Modified Z-Score method (based on MAD).

        Parameters
        ----------
        columns : list of str
            List of columns to analyze.
        threshold : float, optional
            Modified Z-Score threshold to consider an observation an outlier (default 3.5).

        Returns
        -------
        pd.DataFrame
            DataFrame with boolean indicators for outliers.
        """
        outlier_flags = pd.DataFrame(index=self.dataframe.index)

        for column in columns:
            col_data = self.dataframe[column]
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))

            if mad == 0:
                mad = 1e-9  # avoid division by zero

            modified_z_scores = 0.6745 * (col_data - median) / mad

            outlier_flags[column + "_outlier_modified_zscore"] = (
                np.abs(modified_z_scores) > threshold
            )

        return outlier_flags

    def detect_outliers_iqr(
        self, columns: List[str], factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using the IQR method.

        Parameters
        ----------
        columns : list of str
            List of columns to analyze.
        factor : float
            Multiplier for IQR to determine outlier boundaries (default 1.5).

        Returns
        -------
        pd.DataFrame
            DataFrame with boolean indicators for outliers.
        """
        outlier_flags = pd.DataFrame(index=self.dataframe.index)

        for column in columns:
            q1 = self.dataframe[column].quantile(0.25)
            q3 = self.dataframe[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            outlier_flags[column + "_outlier_iqr"] = (
                self.dataframe[column] < lower_bound
            ) | (self.dataframe[column] > upper_bound)

        return outlier_flags

    def remove_outliers_quantile(
        self,
        columns: List[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        inplace: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Remove rows where any specified column has values outside the given
        quantile range.

        Parameters
        ----------
        columns : list of str
            List of columns to check for outliers.
        lower_quantile : float, optional
            Lower quantile threshold (default 0.01).
        upper_quantile : float, optional
            Upper quantile threshold (default 0.99).
        inplace : bool, optional
            Whether to modify the original dataframe in place (default False).

        Returns
        -------
        pd.DataFrame or None
            Cleaned DataFrame if inplace=False, otherwise None.
        """
        if not (0 <= lower_quantile < upper_quantile <= 1):
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")

        mask = pd.Series(
            [True] * len(self.dataframe), index=self.dataframe.index
        )

        inclusive = {
            "neither": lower_quantile != 0 and upper_quantile != 1,
            "both": lower_quantile == 0 and upper_quantile == 1,
            "left": lower_quantile == 0,
            "right": upper_quantile == 1,
        }

        priority_order = ["both", "left", "right", "neither"]

        inclusive_key = next(
            (key for key in priority_order if inclusive.get(key, False)), None
        )

        for column in columns:
            lower_bound = self.dataframe[column].quantile(lower_quantile)
            upper_bound = self.dataframe[column].quantile(upper_quantile)
            mask &= self.dataframe[column].between(
                lower_bound, upper_bound, inclusive=inclusive_key
            )

        if inplace:
            self.dataframe = self.dataframe.loc[mask]
            return None
        else:
            return self.dataframe.loc[mask]

    def detect_outliers_isolation_forest(
        self,
        columns: List[str],
        preprocessor: sklearn.base.TransformerMixin = None,
        contamination: float = 0.01,
        random_state: Optional[int] = 42,
        iforest_kw: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Detect outliers using the Isolation Forest algorithm.

        Parameters
        ----------
        columns : list of str
            List of columns to use for building the model.
        contamination : float, optional
            Expected proportion of outliers in the data (default 0.05).
        random_state : int, optional
            Random state for reproducibility (default 42).

        Returns
        -------
        pd.DataFrame
            DataFrame with boolean indicator for outliers.
        """

        X = self.dataframe[columns].dropna()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    IsolationForest(
                        contamination=contamination,
                        random_state=random_state,
                        **(iforest_kw if iforest_kw else {}),
                    ),
                ),
            ]
        )

        pipeline.fit(X)

        preds = pipeline.predict(X)
        score = pipeline.decision_function(X)

        # return dataframe with original columns plus preds and score columns
        df_outliers = X.copy()
        df_outliers["outlier_isolation_forest"] = preds
        df_outliers["score_isolation_forest"] = score

        self.dataframe_isolation_forest = df_outliers

        return df_outliers

    def plot_qqplots(self, figsize=(12, 12)) -> None:
        """
        Plot the QQ plot for the specified columns.
        The comparison is made against a standard normal distribution.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 12).
        """
        columns = self.dataframe.select_dtypes("number").columns
        grid_shape = best_grid_shape(len(columns))

        fig, axs = plt.subplots(
            *grid_shape,
            figsize=figsize,
            constrained_layout=True,
        )
        for ax, column in zip(axs.flat, columns):
            sm.qqplot(
                self.dataframe[column].dropna(), line="s", ax=ax, fit=True
            )
            ax.set_title(f"{column}", fontsize="medium", loc="center")
            ax.set_xlabel("Theoretical", fontsize="small")
            ax.set_ylabel("Observed", fontsize="small")

        fig.suptitle("Q-Q Plot for Selected Columns")

        plt.show()

    def plot_boxplots(
        self,
        figsize=(12, 12),
        title: str = "Boxplot for Numeric Columns",
    ) -> None:
        """
        Plot boxplots for the specified columns.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 12).
        title : str, optional
            Title of the plot, by default "Boxplot for Numeric Columns".
        """

        columns = self.dataframe.select_dtypes("number").columns

        grid_shape = best_grid_shape(len(columns))

        fig, axs = plt.subplots(
            *grid_shape,
            figsize=figsize,
            constrained_layout=True,
        )
        for ax, column in zip(axs.flat, columns):
            sns.boxplot(data=self.dataframe, x=column, ax=ax, showmeans=True)
            ax.set_title(f"{column}", fontsize="medium", loc="center")
            ax.set_xlabel("Value", fontsize="small")
            ax.set_ylabel("Frequency", fontsize="small")

        fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    def plot_histograms(
        self,
        figsize=(12, 12),
        title: str = "Histogram for Numeric Columns",
    ) -> None:
        """
        Plot histograms for the specified columns.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 12).
        title : str, optional
            Title of the plot, by default "Histogram for Numeric Columns".
        bins : int, optional
            Number of bins for the histogram, by default 30.
        """
        columns = self.dataframe.select_dtypes("number").columns

        grid_shape = best_grid_shape(len(columns))

        fig, axs = plt.subplots(
            *grid_shape,
            figsize=figsize,
            constrained_layout=True,
        )
        for ax, column in zip(axs.flat, columns):
            sns.histplot(
                data=self.dataframe[column],
                kde=True,
                ax=ax,
            )
            ax.set_title(f"{column}", fontsize="medium", loc="center")
            ax.set_xlabel("Value", fontsize="small")
            ax.set_ylabel("Density", fontsize="small")

        fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(
        self,
        figsize=(12, 12),
        title: str = "Correlation Matrix",
        palette: str = "coolwarm",
    ) -> None:
        """
        Plot the correlation matrix for the specified columns.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 12).
        title : str, optional
            Title of the plot, by default "Correlation Matrix".
        palette : str, optional
            Color palette for the heatmap, by default "coolwarm".
        """
        corr_values = self.dataframe.select_dtypes("number").corr()
        matrix = np.triu(corr_values)

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            corr_values,
            mask=matrix,
            annot=True,
            fmt=".2f",
            ax=ax,
            cmap=palette,
        )

        fig.suptitle(title)

        plt.show()

    def plot_pairplot(
        self,
        title: str = "Pairplot for Numeric Columns",
        scatter_alpha: float = 0.2,
    ) -> None:
        """
        Plot pairplots for the numeric columns.

        Parameters
        ----------
        title : str, optional
            Title of the plot, by default "Pairplot for Numeric Columns".
        scatter_alpha : float, optional
            Transparency level for scatter points, by default 0.2.
        """
        g = sns.pairplot(
            self.dataframe,
            diag_kind="kde",
            corner=True,
            plot_kws={"alpha": scatter_alpha},
        )

        g.figure.suptitle(title, y=1.02)

        plt.show()

    def plot_pairplot_outliers(
        self,
        title: str = "Pairplot for Outlier Detection using Isolation Forest",
        sample: float = 1.0,
        scatter_alpha: float = 0.2,
        common_norm: bool = False,
    ) -> None:
        """
        Plot pairplots for the numeric columns.

        Parameters
        ----------
        title : str, optional
            Title of the plot, by default "Pairplot for Numeric Columns".
        sample : float, optional
            Fraction of data to sample for the plot (default 1.0).
        scatter_alpha : float, optional
            Transparency level for scatter points (default 0.2).
        common_norm : bool, optional
            Whether to normalize the diagonal histograms together (default
            False).
        """

        df_outliers = self.dataframe_isolation_forest

        palette = CURRENT_COLOR_CYCLER[
            : len(df_outliers["outlier_isolation_forest"].unique())
        ]

        g = sns.pairplot(
            df_outliers.sample(frac=sample),
            diag_kind="kde",
            hue="outlier_isolation_forest",
            corner=True,
            plot_kws={"alpha": scatter_alpha},
            diag_kws={"common_norm": common_norm},
            palette=palette[::-1],
        )

        g.figure.suptitle(title, y=1.02)

        # change legend. -1 = outlier, 1 = inlier
        legend = g._legend
        legend.set_title("Isolation Forest")
        legend.texts[0].set_text("Outlier")
        legend.texts[1].set_text("Inlier")

        plt.show()
