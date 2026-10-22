"""Compare feature-selection strategies for the 50 Startups dataset.

The final method comparison uses nested CV: inner folds select features and
outer folds estimate performance. The feature-count plots show exploratory
4-fold CV results on the training split, not unbiased nested-CV estimates.
The held-out test set is used after each method is configured on training data.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "50_Startups.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
OUTER_CV = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
INNER_CV = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE + 1)
LASSO_ALPHAS = np.logspace(-4, 6, 160)


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def cv_metrics(X, y):
    """Return pooled out-of-fold RMSE and R-squared for a fixed feature set."""
    predictions = cross_val_predict(LinearRegression(), X, y, cv=INNER_CV)
    return rmse(y, predictions), float(r2_score(y, predictions))


def select_features(method, X, y):
    """Select a feature subset using only the data passed to this function."""
    names = list(X.columns)
    if method == "EFS":
        candidates = (
            list(subset)
            for k in range(1, len(names) + 1)
            for subset in itertools.combinations(names, k)
        )
        return min(candidates, key=lambda subset: cv_metrics(X[subset], y)[0])

    if method == "RFE":
        # Each inner fold fits its own RFE selector before scoring the model.
        best_k = min(
            range(1, len(names) + 1),
            key=lambda k: rmse(
                y,
                cross_val_predict(
                    make_pipeline(
                        RFE(LinearRegression(), n_features_to_select=k),
                        LinearRegression(),
                    ),
                    X,
                    y,
                    cv=INNER_CV,
                ),
            ),
        )
        selector = RFE(LinearRegression(), n_features_to_select=best_k)
        selector.fit(X, y)
        return list(X.columns[selector.support_])

    if method == "Lasso":
        best_features = None
        best_score = float("inf")
        for alpha in LASSO_ALPHAS:
            model = make_pipeline(
                StandardScaler(),
                Lasso(alpha=alpha, max_iter=100000, random_state=RANDOM_STATE),
            )
            # Each inner fold fits its own scaler, preventing leakage.
            predictions = cross_val_predict(model, X, y, cv=INNER_CV)
            score = rmse(y, predictions)
            model.fit(X, y)
            coefficients = model.named_steps["lasso"].coef_
            selected = [name for name, coef in zip(names, coefficients) if abs(coef) > 1e-8]
            if selected and score < best_score:
                best_score, best_features = score, selected
        return best_features or names

    if method == "Backward":
        remaining = names.copy()
        path = [remaining.copy()]
        while len(remaining) > 1:
            model = sm.OLS(y, sm.add_constant(X[remaining], has_constant="add")).fit()
            p_values = model.pvalues.drop("const", errors="ignore")
            remaining.remove(p_values.idxmax())
            path.append(remaining.copy())
        return min(path, key=lambda subset: cv_metrics(X[subset], y)[0])

    if method == "SFS":
        best_features = None
        best_score = float("inf")
        for k in range(1, len(names) + 1):
            if k == len(names):
                selected = names.copy()
            elif k == 1:
                selected = [min(names, key=lambda name: cv_metrics(X[[name]], y)[0])]
            else:
                selector = SequentialFeatureSelector(
                    LinearRegression(), n_features_to_select=k,
                    direction="forward", scoring="neg_mean_squared_error", cv=INNER_CV,
                )
                selector.fit(X, y)
                selected = list(X.columns[selector.get_support()])
            score = cv_metrics(X[selected], y)[0]
            if score < best_score:
                best_score, best_features = score, selected
        return best_features

    raise ValueError(f"Unknown method: {method}")


def nested_cv_evaluate(method, X, y):
    """Estimate selection procedure performance on held-out outer folds."""
    predictions = np.empty(len(y), dtype=float)
    fold_features = []
    for train_idx, valid_idx in OUTER_CV.split(X):
        X_fold, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_fold = y.iloc[train_idx]
        selected = select_features(method, X_fold, y_fold)
        model = LinearRegression().fit(X_fold[selected], y_fold)
        predictions[valid_idx] = model.predict(X_valid[selected])
        fold_features.append(selected)
    return rmse(y, predictions), float(r2_score(y, predictions)), fold_features


def feature_count_results(method, X, y):
    """Return the best feature subset at each size using inner CV only."""
    names = list(X.columns)
    rows = []

    if method == "EFS":
        for k in range(1, len(names) + 1):
            candidates = itertools.combinations(names, k)
            selected = min(candidates, key=lambda subset: cv_metrics(X[list(subset)], y)[0])
            score, r2 = cv_metrics(X[list(selected)], y)
            rows.append({"Number of Features": k, "Selected Features": list(selected), "RMSE": score, "R-squared": r2})
        return rows

    if method == "RFE":
        for k in range(1, len(names) + 1):
            predictions = cross_val_predict(
                make_pipeline(RFE(LinearRegression(), n_features_to_select=k), LinearRegression()),
                X, y, cv=INNER_CV,
            )
            selector = RFE(LinearRegression(), n_features_to_select=k).fit(X, y)
            selected = list(X.columns[selector.support_])
            rows.append({"Number of Features": k, "Selected Features": selected,
                         "RMSE": rmse(y, predictions), "R-squared": float(r2_score(y, predictions))})
        return rows

    if method == "Lasso":
        best_by_count = {}
        for alpha in LASSO_ALPHAS:
            model = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=100000, random_state=RANDOM_STATE))
            predictions = cross_val_predict(model, X, y, cv=INNER_CV)
            model.fit(X, y)
            coefficients = model.named_steps["lasso"].coef_
            selected = [name for name, coef in zip(names, coefficients) if abs(coef) > 1e-8]
            k = len(selected)
            if k and (k not in best_by_count or rmse(y, predictions) < best_by_count[k]["RMSE"]):
                best_by_count[k] = {"Number of Features": k, "Selected Features": selected,
                                    "RMSE": rmse(y, predictions), "R-squared": float(r2_score(y, predictions))}
        return [best_by_count[k] for k in sorted(best_by_count)]

    if method == "Backward":
        remaining = names.copy()
        path = {}
        while remaining:
            path[len(remaining)] = remaining.copy()
            if len(remaining) == 1:
                break
            fitted = sm.OLS(y, sm.add_constant(X[remaining], has_constant="add")).fit()
            remaining.remove(fitted.pvalues.drop("const", errors="ignore").idxmax())
        for k in sorted(path):
            selected = path[k]
            score, r2 = cv_metrics(X[selected], y)
            rows.append({"Number of Features": k, "Selected Features": selected, "RMSE": score, "R-squared": r2})
        return rows

    if method == "SFS":
        for k in range(1, len(names) + 1):
            if k == len(names):
                selected = names.copy()
            elif k == 1:
                selected = [min(names, key=lambda name: cv_metrics(X[[name]], y)[0])]
            else:
                selector = SequentialFeatureSelector(
                    LinearRegression(), n_features_to_select=k, direction="forward",
                    scoring="neg_mean_squared_error", cv=INNER_CV,
                ).fit(X, y)
                selected = list(X.columns[selector.get_support()])
            score, r2 = cv_metrics(X[selected], y)
            rows.append({"Number of Features": k, "Selected Features": selected, "RMSE": score, "R-squared": r2})
        return rows

    raise ValueError(f"Unknown method: {method}")


def plot_results(results, method_name, filename, n_features):
    results_df = pd.DataFrame(results).sort_values("Number of Features")
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.5], hspace=0.4, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_df["Number of Features"], results_df["RMSE"], marker="o", linewidth=2)
    ax1.set_title("4-Fold CV RMSE by Number of Features (Training Split)")
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("RMSE")
    ax1.set_xticks(range(1, n_features + 1))
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results_df["Number of Features"], results_df["R-squared"], marker="o", linewidth=2)
    ax2.set_title("4-Fold CV R-squared by Number of Features (Training Split)")
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("R-squared")
    ax2.set_xticks(range(1, n_features + 1))
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_data = [
        [int(row["Number of Features"]), "[" + ", ".join(row["Selected Features"]) + "]",
         f'{row["RMSE"]:.6f}', f'{row["R-squared"]:.6f}']
        for _, row in results_df.iterrows()
    ]
    table = ax3.table(
        cellText=table_data,
        colLabels=["Number of Features", "Selected Features", "RMSE", "R-squared"],
        cellLoc="left", colLoc="left", loc="upper center", colWidths=[0.15, 0.59, 0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 else "#ffffff")
    ax3.set_title("Feature Selection Results (inner CV)", fontsize=14, pad=15)
    fig.suptitle(method_name, fontsize=18, fontweight="bold")
    fig.savefig(OUTPUT_DIR / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== Startup Profit Regression ===")
    df = pd.read_csv(DATASET)
    required = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df[required].isnull().any().any():
        raise ValueError("Missing values detected in required columns")
    if len(df) <= OUTER_SPLITS:
        raise ValueError(f"Need more than {OUTER_SPLITS} rows for nested cross-validation")

    # Encode from fixed known categories so future web inputs use the same columns.
    df_encoded = pd.get_dummies(df[required], columns=["State"], drop_first=True, dtype=int)
    X = df_encoded.drop(columns="Profit")
    y = df_encoded["Profit"].astype(float)
    if not all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
        raise ValueError("All predictor columns must be numeric after encoding")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    methods = ["EFS", "RFE", "Lasso", "Backward", "SFS"]
    rows = []
    for method in methods:
        print(f"Evaluating {method} with nested CV...")
        nested_rmse, nested_r2, fold_features = nested_cv_evaluate(method, X_train, y_train)

        curve_results = feature_count_results(method, X_train, y_train)
        plot_results(curve_results, {
            "EFS": "Exhaustive Feature Selection",
            "RFE": "Recursive Feature Elimination (RFE)",
            "Lasso": "Lasso Feature Selection",
            "Backward": "Backward Elimination",
            "SFS": "Sequential Forward Selection",
        }[method], {
            "EFS": "01_exhaustive_selection.png",
            "RFE": "02_rfecv.png",
            "Lasso": "03_lasso.png",
            "Backward": "04_backward_elimination.png",
            "SFS": "05_forward_selection.png",
        }[method], len(X_train.columns))

        # Select the final feature set from training data only; test set is untouched until now.
        selected = select_features(method, X_train, y_train)
        final_model = LinearRegression().fit(X_train[selected], y_train)
        test_predictions = final_model.predict(X_test[selected])
        rows.append({
            "Method": method,
            "Number of Features": len(selected),
            "Selected Features": selected,
            "Nested CV RMSE": nested_rmse,
            "Nested CV R2": nested_r2,
            "Test RMSE": rmse(y_test, test_predictions),
            "Test R2": float(r2_score(y_test, test_predictions)),
            "Outer Fold Features": fold_features,
        })

    summary = pd.DataFrame(rows).sort_values("Nested CV RMSE").reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "final_comparison.csv", index=False)
    print("\n=== Final Model Comparison ===")
    print(summary.drop(columns="Outer Fold Features").to_string(index=False))
    best = summary.iloc[0]
    print(f"\nLowest nested-CV RMSE: {best['Method']} ({best['Nested CV RMSE']:.2f})")
    print("Test metrics are reported for reference and must not be used to choose the method.")


if __name__ == "__main__":
    main()
