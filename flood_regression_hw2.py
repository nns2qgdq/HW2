"""Multiple linear regression analysis for the Kaggle Flood Prediction dataset.

Feature selection and model comparison use training-only samples. The untouched
holdout split is used once for final metrics and prediction interval plots.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results" / "flood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_FILE = DATA_DIR / "train.csv"
TARGET = "FloodProbability"
ID_COLUMN = "id"
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.20
SELECTION_SAMPLE_SIZE = 40_000
VALIDATION_SAMPLE_SIZE = 12_000
SELECTED_FEATURE_COUNT = 14
CV_SPLITS = 5
FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors",
]


def validate_data(frame: pd.DataFrame) -> None:
    required = [ID_COLUMN, *FEATURES, TARGET]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"train.csv 缺少必要欄位：{missing}")
    if frame[required].isna().any().any():
        raise ValueError("必要欄位含缺值，請先處理資料後再分析。")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in [*FEATURES, TARGET]):
        raise ValueError("特徵與目標欄必須是數值欄位。")
    if frame[TARGET].nunique() < 2:
        raise ValueError("目標欄需要至少兩個不同值。")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def choose_features(selection: pd.DataFrame) -> dict[str, list[str]]:
    X, y = selection[FEATURES], selection[TARGET]
    k = min(SELECTED_FEATURE_COUNT, len(FEATURES))
    chosen: dict[str, list[str]] = {}

    kbest = SelectKBest(score_func=f_regression, k=k).fit(X, y)
    chosen["SelectKBest"] = list(X.columns[kbest.get_support()])

    rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X, y)
    chosen["RFE"] = list(X.columns[rfe.get_support()])

    sfs = SequentialFeatureSelector(
        LinearRegression(), n_features_to_select=k, direction="forward",
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    ).fit(X, y)
    chosen["Sequential Forward Selection"] = list(X.columns[sfs.get_support()])

    lasso = make_pipeline(
        StandardScaler(), Lasso(alpha=0.0005, max_iter=20_000, tol=1e-5, selection="cyclic")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso.fit(X, y)
    coefficients = lasso.named_steps["lasso"].coef_
    lasso_order = np.argsort(np.abs(coefficients))[::-1]
    lasso_features = [FEATURES[i] for i in lasso_order if abs(coefficients[i]) > 1e-10]
    chosen["Lasso"] = lasso_features or [FEATURES[i] for i in lasso_order[:k]]

    X_fit, X_perm, y_fit, y_perm = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    permutation_model = LinearRegression().fit(X_fit, y_fit)
    permutation = permutation_importance(
        permutation_model, X_perm, y_perm, scoring="neg_root_mean_squared_error",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    permutation_order = np.argsort(permutation.importances_mean)[::-1]
    chosen["Permutation Importance"] = [FEATURES[i] for i in permutation_order[:k]]
    return chosen


def cv_summary(X: pd.DataFrame, y: pd.Series, estimator) -> tuple[float, float]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1)
    scores = cross_validate(
        estimator, X, y, cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1, return_train_score=False,
    )
    return float(-scores["test_rmse"].mean()), float(scores["test_r2"].mean())


def save_eda_plots(training: pd.DataFrame, features: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(training[TARGET], bins=35, color="#3b82f6", edgecolor="white")
    axes[0].set(title="FloodProbability distribution", xlabel=TARGET, ylabel="Count")
    correlations = training[features].corrwith(training[TARGET]).sort_values()
    order = correlations.index
    axes[1].barh(order, correlations.values, color="#10b981")
    axes[1].set(title="Feature correlation with target", xlabel="Pearson correlation")
    axes[1].axvline(0, color="#64748b", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_target_and_correlations.png", dpi=180)
    plt.close(fig)


def save_prediction_plot(actual: np.ndarray, predicted: np.ndarray,
                         lower: np.ndarray, upper: np.ndarray) -> None:
    order = np.argsort(predicted)
    shown = order[: min(250, len(order))]
    x = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(x, lower[shown], upper[shown], color="#93c5fd", alpha=0.45,
                    label="95% prediction interval")
    ax.plot(x, predicted[shown], color="#1d4ed8", linewidth=1.8, label="Predicted")
    ax.scatter(x, actual[shown], s=13, color="#dc2626", alpha=0.7, label="Actual")
    ax.set(title="OLS holdout predictions with 95% prediction intervals",
           xlabel="Holdout observations sorted by predicted value", ylabel=TARGET)
    ax.legend(frameon=False, ncols=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_prediction_intervals.png", dpi=180)
    plt.close(fig)


def diagnostics(fit, X_test: pd.DataFrame, y_test: pd.Series,
                predictions: np.ndarray, model_features: list[str]) -> pd.DataFrame:
    # Use a deterministic 5,000-row sample for diagnostic tests and plots.
    predictions = np.asarray(predictions)
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=min(5_000, len(X_test)), replace=False)
    X_diag = X_test.iloc[idx][model_features]
    y_diag = y_test.iloc[idx]
    pred_diag = predictions[idx]
    residuals = y_diag.to_numpy() - pred_diag
    diag_model = sm.OLS(y_diag, sm.add_constant(X_diag, has_constant="add")).fit()

    reset = linear_reset(diag_model, power=2, use_f=True)
    bp = het_breuschpagan(diag_model.resid, diag_model.model.exog)
    jb = jarque_bera(residuals)
    sorted_idx = np.argsort(X_test.iloc[idx][ID_COLUMN].to_numpy())
    dw = float(durbin_watson(residuals[sorted_idx]))
    vif_values = [variance_inflation_factor(sm.add_constant(X_diag, has_constant="add").to_numpy(), i + 1)
                  for i in range(len(model_features))]

    rows = [
        {"Assumption": "Linearity (Ramsey RESET, power=2)", "Statistic": float(reset.fvalue),
         "p_value": float(reset.pvalue), "Interpretation": "p<0.05 suggests possible nonlinearity; inspect residual plot."},
        {"Assumption": "Homoscedasticity (Breusch-Pagan)", "Statistic": float(bp[0]),
         "p_value": float(bp[1]), "Interpretation": "p<0.05 suggests non-constant residual variance."},
        {"Assumption": "Residual normality (Jarque-Bera)", "Statistic": float(jb.statistic),
         "p_value": float(jb.pvalue), "Interpretation": "Large samples make this test sensitive; also inspect Q-Q plot."},
        {"Assumption": "Error independence (Durbin-Watson by id order)", "Statistic": dw,
         "p_value": np.nan, "Interpretation": "Values near 2 indicate little first-order autocorrelation; row order is a proxy."},
        {"Assumption": "Multicollinearity (maximum VIF)", "Statistic": float(np.nanmax(vif_values)),
         "p_value": np.nan, "Interpretation": "VIF above 5 warrants review; report feature-level values separately."},
    ]
    pd.DataFrame({"Feature": model_features, "VIF": vif_values}).to_csv(
        OUTPUT_DIR / "vif_by_feature.csv", index=False
    )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "regression_assumptions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(pred_diag, residuals, s=8, alpha=0.3, color="#2563eb")
    axes[0, 0].axhline(0, color="#dc2626", linestyle="--")
    axes[0, 0].set(title="Residuals vs fitted", xlabel="Fitted value", ylabel="Residual")
    sm.qqplot(residuals, line="45", ax=axes[0, 1], markerfacecolor="#2563eb", alpha=0.35)
    axes[0, 1].set_title("Residual Q-Q plot")
    axes[1, 0].scatter(pred_diag, np.sqrt(np.abs(residuals)), s=8, alpha=0.3, color="#059669")
    axes[1, 0].set(title="Scale-location", xlabel="Fitted value", ylabel="sqrt(|residual|)")
    axes[1, 1].hist(residuals, bins=40, color="#8b5cf6", edgecolor="white")
    axes[1, 1].set(title="Residual distribution", xlabel="Residual", ylabel="Count")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_regression_diagnostics.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"找不到資料檔：{TRAIN_FILE}")
    print(f"Loading {TRAIN_FILE.name} ...", flush=True)
    data = pd.read_csv(TRAIN_FILE)
    validate_data(data)
    X = data[FEATURES]
    y = data[TARGET]
    ids = data[ID_COLUMN]
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE
    )
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    selection = train.sample(n=min(SELECTION_SAMPLE_SIZE, len(train)), random_state=RANDOM_STATE)
    unused = train.drop(index=selection.index)
    evaluation = unused.sample(n=min(VALIDATION_SAMPLE_SIZE, len(unused)), random_state=RANDOM_STATE + 1)
    print(f"Rows: {len(data):,}; train: {len(train):,}; final holdout: {len(test):,}", flush=True)
    print(f"Feature selection sample: {len(selection):,}; independent CV sample: {len(evaluation):,}", flush=True)

    selected_sets = choose_features(selection)
    # Include the complete 20-feature model as the reference.
    selected_sets["All 20 features"] = FEATURES.copy()
    feature_rows = []
    for method, cols in selected_sets.items():
        score, r2 = cv_summary(evaluation[cols], evaluation[TARGET], LinearRegression())
        feature_rows.append({"Method": method, "Number of Features": len(cols),
                             "Selected Features": ", ".join(cols), "CV RMSE": score, "CV R2": r2})
        print(f"Feature set {method}: {len(cols)} features, CV RMSE={score:.6f}", flush=True)
    feature_results = pd.DataFrame(feature_rows).sort_values("CV RMSE").reset_index(drop=True)
    feature_results.to_csv(OUTPUT_DIR / "feature_selection_comparison.csv", index=False)
    best_method = str(feature_results.iloc[0]["Method"])
    best_features = selected_sets[best_method]

    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso (alpha=0.0005)": make_pipeline(StandardScaler(), Lasso(alpha=0.0005, max_iter=20_000, tol=1e-5)),
    }
    model_rows = []
    for name, estimator in estimators.items():
        score, r2 = cv_summary(evaluation[best_features], evaluation[TARGET], estimator)
        model_rows.append({"Model": name, "Feature Selection": best_method,
                           "Number of Features": len(best_features), "Features": ", ".join(best_features),
                           "CV RMSE": score, "CV R2": r2})
        print(f"Model {name}: CV RMSE={score:.6f}, CV R2={r2:.6f}", flush=True)
    model_results = pd.DataFrame(model_rows).sort_values("CV RMSE").reset_index(drop=True)
    model_results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_cv_rmse = float(model_results.iloc[0]["CV RMSE"])
    near_best = model_results[model_results["CV RMSE"] <= best_cv_rmse * 1.001]
    # Prefer the simplest model when CV scores differ by less than 0.1%.
    linear_row = near_best[near_best["Model"] == "Linear Regression"]
    best_model_name = str(linear_row.iloc[0]["Model"] if not linear_row.empty else model_results.iloc[0]["Model"])
    final_estimator = estimators[best_model_name]
    final_estimator.fit(train[best_features], train[TARGET])
    test_predictions = final_estimator.predict(test[best_features])

    best_metrics = {
        "Model": best_model_name,
        "Feature Selection": best_method,
        "Number of Features": len(best_features),
        "Features": best_features,
        "Test MAE": mean_absolute_error(test[TARGET], test_predictions),
        "Test MSE": mean_squared_error(test[TARGET], test_predictions),
        "Test RMSE": rmse(test[TARGET], test_predictions),
        "Test R2": r2_score(test[TARGET], test_predictions),
        "Test Adjusted R2": adjusted_r2(r2_score(test[TARGET], test_predictions), len(test), len(best_features)),
        "Training Rows": len(train),
        "Test Rows": len(test),
        "Seed": RANDOM_STATE,
        "Model Selection Rule": "Choose lowest-CV-RMSE model; prefer Linear Regression if within 0.1% of the minimum.",
    }
    (OUTPUT_DIR / "best_model_summary.json").write_text(
        json.dumps(best_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    predictions_frame = pd.DataFrame({"id": test[ID_COLUMN], "actual": test[TARGET], "predicted": test_predictions})

    # OLS intervals and diagnostics provide interpretable inferential output.
    ols = sm.OLS(train[TARGET], sm.add_constant(train[best_features], has_constant="add")).fit()
    interval_sample = test.sample(n=min(5_000, len(test)), random_state=RANDOM_STATE)
    interval_result = ols.get_prediction(sm.add_constant(interval_sample[best_features], has_constant="add")).summary_frame(alpha=0.05)
    interval_chart_frame = pd.DataFrame({
        "id": interval_sample[ID_COLUMN].to_numpy(),
        "actual": interval_sample[TARGET].to_numpy(),
        "ols_mean": interval_result["mean"].to_numpy(),
        "mean_ci_lower": interval_result["mean_ci_lower"].to_numpy(),
        "mean_ci_upper": interval_result["mean_ci_upper"].to_numpy(),
        "obs_ci_lower": interval_result["obs_ci_lower"].to_numpy(),
        "obs_ci_upper": interval_result["obs_ci_upper"].to_numpy(),
    })
    interval_chart_frame["residual"] = interval_chart_frame["actual"] - interval_chart_frame["ols_mean"]
    interval_chart_frame.to_csv(OUTPUT_DIR / "interactive_diagnostics_sample.csv", index=False)
    predictions_frame = predictions_frame.merge(
        pd.DataFrame({"id": interval_sample[ID_COLUMN], "ols_mean": interval_result["mean"].to_numpy(),
                      "mean_ci_lower": interval_result["mean_ci_lower"].to_numpy(),
                      "mean_ci_upper": interval_result["mean_ci_upper"].to_numpy(),
                      "obs_ci_lower": interval_result["obs_ci_lower"].to_numpy(),
                      "obs_ci_upper": interval_result["obs_ci_upper"].to_numpy()}),
        on="id", how="left",
    )
    predictions_frame.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    save_eda_plots(train, FEATURES)
    # Plot the OLS baseline and its prediction interval, even if a regularized model
    # wins CV; the CSV includes both confidence and prediction bounds for auditability.
    save_prediction_plot(
        interval_sample[TARGET].to_numpy(), interval_result["mean"].to_numpy(),
        interval_result["obs_ci_lower"].to_numpy(), interval_result["obs_ci_upper"].to_numpy(),
    )
    diagnostics(ols, test, test[TARGET], ols.predict(sm.add_constant(test[best_features], has_constant="add")), best_features)
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps({
        "dataset": "Kaggle Flood Prediction Dataset",
        "source": "https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset",
        "target": TARGET,
        "excluded_identifier": ID_COLUMN,
        "candidate_features": FEATURES,
        "feature_selection_sample_size": len(selection),
        "independent_cv_sample_size": len(evaluation),
        "selection_cv_folds": 3,
        "evaluation_cv_folds": CV_SPLITS,
        "final_holdout_fraction": HOLDOUT_FRACTION,
        "seed": RANDOM_STATE,
        "metrics": best_metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nFinal holdout metrics:", json.dumps(best_metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"\nOutputs saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
