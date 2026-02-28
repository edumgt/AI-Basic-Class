"""특성 공학 실습 파일"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import joblib


def run() -> dict:
    np.random.seed(42)
    result = {"chapter": "chapter12", "topic": "특성 공학"}

    if "feature_engineering" == "numpy":
        arr = np.array([1, 2, 3, 4, 5])
        result["mean"] = float(arr.mean())
        result["std"] = float(arr.std())

    elif "feature_engineering" == "pandas":
        df = pd.DataFrame({"feature": [1, 2, np.nan, 4], "target": [10, 15, 14, 20]})
        result["missing_before"] = int(df["feature"].isna().sum())
        df["feature"] = df["feature"].fillna(df["feature"].mean())
        result["missing_after"] = int(df["feature"].isna().sum())

    elif "feature_engineering" == "probability":
        toss = np.random.binomial(1, 0.6, size=1000)
        result["estimated_p"] = float(toss.mean())

    elif "feature_engineering" == "linear_algebra":
        A = np.array([[2, 1], [1, 3]])
        b = np.array([1, 2])
        x = np.linalg.solve(A, b)
        result["solution"] = x.round(4).tolist()

    elif "feature_engineering" == "linear_regression":
        X, y = make_regression(n_samples=120, n_features=3, noise=7, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)
        result["mse"] = float(mean_squared_error(y_test, pred))

    elif "feature_engineering" == "logistic_regression":
        X, y = make_classification(n_samples=180, n_features=5, n_informative=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=500).fit(X_train, y_train)
        pred = model.predict(X_test)
        result["accuracy"] = float(accuracy_score(y_test, pred))

    elif "feature_engineering" == "decision_tree":
        X, y = make_classification(n_samples=180, n_features=6, n_informative=4, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)
        result["feature_importance_sum"] = float(model.feature_importances_.sum())

    elif "feature_engineering" == "random_forest":
        X, y = make_classification(n_samples=220, n_features=6, n_informative=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=120, random_state=42).fit(X_train, y_train)
        pred = model.predict(X_test)
        result["f1"] = float(f1_score(y_test, pred))

    elif "feature_engineering" == "kmeans":
        X = np.vstack([np.random.normal(0, 1, (40, 2)), np.random.normal(4, 1, (40, 2))])
        model = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X)
        result["cluster_count"] = int(len(np.unique(model.labels_)))

    elif "feature_engineering" == "metrics":
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.95])
        result["precision"] = float(precision_score(y_true, y_pred))
        result["recall"] = float(recall_score(y_true, y_pred))
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    elif "feature_engineering" == "validation":
        X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=42)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=cv)
        result["cv_mean"] = float(scores.mean())

    elif "feature_engineering" == "feature_engineering":
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "salary": [3000, 4000, 5000, 6000],
            "city": ["Seoul", "Busan", "Seoul", "Daegu"],
        })
        pre = ColumnTransformer([
            ("num", StandardScaler(), ["age", "salary"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ])
        transformed = pre.fit_transform(df)
        result["transformed_shape"] = list(transformed.shape)

    elif "feature_engineering" == "regularization":
        X, y = make_regression(n_samples=100, n_features=10, noise=15, random_state=42)
        ridge = Ridge(alpha=1.0).fit(X, y)
        lasso = Lasso(alpha=0.1).fit(X, y)
        result["ridge_coef_norm"] = float(np.linalg.norm(ridge.coef_))
        result["lasso_non_zero"] = int(np.count_nonzero(lasso.coef_))

    elif "feature_engineering" == "dimensionality_reduction":
        X = np.random.randn(150, 10)
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(X)
        result["reduced_shape"] = list(reduced.shape)
        result["explained"] = float(pca.explained_variance_ratio_.sum())

    elif "feature_engineering" == "outlier_detection":
        X = np.vstack([np.random.normal(0, 1, (100, 2)), np.array([[8, 8], [9, 9], [10, 10]])])
        model = IsolationForest(contamination=0.03, random_state=42).fit(X)
        pred = model.predict(X)
        result["outliers_detected"] = int((pred == -1).sum())

    elif "feature_engineering" == "time_series":
        series = pd.Series(np.sin(np.linspace(0, 6, 60)) + np.random.normal(0, 0.1, 60))
        result["ma_5_last"] = float(series.rolling(5).mean().iloc[-1])
        result["lag_1_last"] = float(series.shift(1).iloc[-1])

    elif "feature_engineering" == "pipeline":
        X, y = make_classification(n_samples=180, n_features=6, n_informative=4, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        result["accuracy"] = float(accuracy_score(y_test, pred))

    elif "feature_engineering" == "model_persistence":
        X, y = make_regression(n_samples=100, n_features=4, noise=3, random_state=42)
        model = LinearRegression().fit(X, y)
        model_path = Path(__file__).with_name("linear_model.joblib")
        joblib.dump(model, model_path)
        loaded = joblib.load(model_path)
        result["coef_match"] = bool(np.allclose(model.coef_, loaded.coef_))

    elif "feature_engineering" == "serving":
        result["message"] = "FastAPI 백엔드(app/main.py)의 /api/chapters 엔드포인트를 확인하세요."

    elif "feature_engineering" == "mini_project":
        X, y = make_classification(n_samples=250, n_features=7, n_informative=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=150, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        result["final_accuracy"] = float(accuracy_score(y_test, pred))

    return result


if __name__ == "__main__":
    print(run())
