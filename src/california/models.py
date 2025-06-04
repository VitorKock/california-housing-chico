import pandas as pd
from typing import Optional, Dict, Any, Union

from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def build_regression_pipeline(
    regressor: RegressorMixin,
    preprocessor: Optional[TransformerMixin] = None,
    target_transformer: Optional[TransformerMixin] = None,
) -> Union[Pipeline, TransformedTargetRegressor]:
    """
    Build a regression model pipeline, optionally including preprocessing and
    target transformation.

    Parameters
    ----------
    regressor : RegressorMixin
        Regression model to fit.
    preprocessor : TransformerMixin, optional
        Transformer for preprocessing features (default is None).
    target_transformer : TransformerMixin, optional
        Transformer for target variable (default is None).

    Returns
    -------
    model : Pipeline or TransformedTargetRegressor
        Constructed model pipeline.
    """
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    steps.append(("reg", regressor))

    pipeline = Pipeline(steps)

    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline

    return model


def cross_validate_regression_model(
    X: pd.DataFrame,
    y: pd.Series,
    regressor: RegressorMixin,
    preprocessor: Optional[TransformerMixin] = None,
    target_transformer: Optional[TransformerMixin] = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Perform cross-validation on a regression pipeline.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Target vector.
    regressor : RegressorMixin
        Regression model.
    preprocessor : TransformerMixin, optional
        Preprocessing transformer (default is None).
    target_transformer : TransformerMixin, optional
        Target transformer (default is None).
    n_splits : int, optional
        Number of KFold splits (default is 5).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    scores : dict
        Dictionary with cross-validation scores (R², negative MAE, negative
        RMSE).
    """
    model = build_regression_pipeline(
        regressor, preprocessor, target_transformer
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def grid_search_regression_model(
    regressor: RegressorMixin,
    param_grid: Dict[str, Any],
    preprocessor: Optional[TransformerMixin] = None,
    target_transformer: Optional[TransformerMixin] = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    return_train_score: bool = False,
) -> GridSearchCV:
    """
    Perform grid search with cross-validation for a regression pipeline.

    Parameters
    ----------
    regressor : RegressorMixin
        Regression model.
    param_grid : dict
        Dictionary with parameter names mapped to lists of settings to try.
    preprocessor : TransformerMixin, optional
        Preprocessing transformer (default is None).
    target_transformer : TransformerMixin, optional
        Target transformer (default is None).
    n_splits : int, optional
        Number of KFold splits (default is 5).
    random_state : int, optional
        Random seed for reproducibility (default is 42).
    return_train_score : bool, optional
        Whether to include training scores in the results (default is False).

    Returns
    -------
    grid_search : GridSearchCV
        Fitted GridSearchCV object ready for best model selection.
    """
    model = build_regression_pipeline(
        regressor, preprocessor, target_transformer
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organize_cv_results(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Organize cross-validation results from multiple models into a tidy DataFrame.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their evaluation results.

    Returns
    -------
    df_results_expanded : DataFrame
        Expanded and numeric DataFrame with model evaluation scores and timings.
    """
    for key, value in results.items():
        value["time_seconds"] = value["fit_time"] + value["score_time"]

    df_results = (
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    )

    df_results_expanded = df_results.explode(
        df_results.columns.difference(["model"]).tolist()
    ).reset_index(drop=True)

    try:
        df_results_expanded = df_results_expanded.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_results_expanded
