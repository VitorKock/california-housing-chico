[![author](https://img.shields.io/badge/Author-Francisco&nbsp;Bustamante-red.svg)](https://www.linkedin.com/in/flsbustamante/)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

# California Housing Prices Prediction

> Predicting California housing prices — data-driven modeling enriched with geospatial intelligence

This project explores the use of machine learning to predict housing prices in
California based on demographic, geographic, and real estate-related features.
By combining regression models with geospatial analysis, the goal is to generate
accurate, interpretable, and actionable predictions.

![map_visual](reports/figures/map_house_income.png)

The dataset comes from the 1990 California census and includes information about
population, housing age, income levels, location coordinates, and proximity to
the ocean. Although the original data is dated, it offers a rich context for
building a complete data science workflow—from preprocessing to deployment.

The model is deployed in a simple interactive Streamlit application that allows
users to select a county, set demographic and housing parameters, and get
real-time price estimates powered by the trained model.

The original dataset can be found on [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data).

## Main results

The project is divided into four main stages, each one documented in its own notebook in the [`notebooks`](notebooks/) folder:

1. **Data cleaning and feature engineering**: Raw data was cleaned and new
features such as `rooms_per_household` and `bedrooms_per_room` were created to
improve model performance. [Notebook 1](notebooks/01-flsb-data_cleansing.ipynb)

2. **Geospatial enrichment**: County geometries were merged with the housing
dataset using geographic coordinates, enabling both visualization and
location-aware predictions. [Notebook 2](notebooks/02-flsb-geo.ipynb)

3. **Linear modeling**: Several linear regressors were trained and evaluated,
including OLS, Ridge, Lasso, and ElasticNet. Model coefficients and residuals
were analyzed to assess fit and interpretability. [Notebook
3](notebooks/03-flsb-linear_models.ipynb)

4. **Model comparison and final selection**: Multiple regression models (linear,
tree-based, SVR and KNN) were compared using cross-validation. LightGBM emerged
as the best model and was fine-tuned using GridSearchCV. [Notebook
4](notebooks/04-flsb-all_models.ipynb)

The final model’s performance was evaluated using RMSE and MAE. Below is a
comparison of model metrics:

![metrics](reports/figures/metrics.png)

The trained model was then exported and integrated into a Streamlit web app,
where predictions can be visualized spatially on an interactive map.

## Demo: Streamlit Application

![streamlit](reports/figures/streamlit.png)

The final model is available for interaction in a Streamlit app, where users can:

[![streamlit app](https://img.shields.io/badge/-Streamlit%20app-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://california-housing-prices.streamlit.app/)

* Select a county from the map
* Adjust inputs like house age or median income
* View predicted median house values instantly

This allows for intuitive exploration of how price varies by geography and demographics.

## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `california` with the help of [conda]:

   ```
   conda env create -f environment.yml
   ```

2. activate the new environment with:
   ```
   conda activate california
   ```

3. install the package in editable mode with:
   ```
   pip install -e .
   ```
   This will install the package in editable mode, so you can modify the source code and
   directly import the package in Python.

> **_NOTE:_**  The conda environment will have california installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `notebooks` folder.

### MLFlow setup

The notebooks have all the consolidated results of the models, with the
numerical results and visualizations. If you want even more detailed results,
you can use MLFlow to track the experiments. To set up MLFlow, you need to:

1. Open the `.env` file present in the `src/california` folder.
2. Set the `MLFLOW_ON` variable to True.
3. Run the code of the regression notebook.
4. A folder called `mlruns` will be created in the `notebooks` folder.
5. To see the results, you need to run the following command in the terminal:
   ```bash
   mlflow ui
   ```

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in
`environment.yml` and eventually in `setup.cfg` if you want to ship and install
your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n california -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── data
│   ├── geo                 <- Geospatial data, e.g. county geometries.
│   ├── interim             <- Data that has been transformed.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── pages                   <- Streamlit pages' content.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── california          <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.6 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
