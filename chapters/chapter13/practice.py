# [초등학생 설명 주석 적용됨]
# 설명: 이 파일 설명(문서 문자열)을 적어요.
"""정규화 실습 파일"""
# 설명: 필요한 도구를 가져와요.
from __future__ import annotations

# 설명: 필요한 도구를 가져와요.
from pathlib import Path

# 설명: 필요한 도구를 가져와요.
import numpy as np
# 설명: 필요한 도구를 가져와요.
import pandas as pd
# 설명: 필요한 도구를 가져와요.
from sklearn.datasets import make_classification, make_regression
# 설명: 필요한 도구를 가져와요.
from sklearn.ensemble import IsolationForest, RandomForestClassifier
# 설명: 필요한 도구를 가져와요.
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
# 설명: 필요한 도구를 가져와요.
from sklearn.cluster import KMeans
# 설명: 필요한 도구를 가져와요.
from sklearn.decomposition import PCA
# 설명: 필요한 도구를 가져와요.
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score
# 설명: 필요한 도구를 가져와요.
from sklearn.model_selection import KFold, cross_val_score, train_test_split
# 설명: 필요한 도구를 가져와요.
from sklearn.pipeline import Pipeline
# 설명: 필요한 도구를 가져와요.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# 설명: 필요한 도구를 가져와요.
from sklearn.compose import ColumnTransformer
# 설명: 필요한 도구를 가져와요.
from sklearn.tree import DecisionTreeClassifier
# 설명: 필요한 도구를 가져와요.
import joblib


# 설명: `run` 함수를 만들어요.
def run() -> dict:
    # 설명: 다음 코드를 실행해요.
    np.random.seed(42)
    # 설명: 값을 저장하거나 바꿔요.
    result = {"chapter": "chapter13", "topic": "정규화"}

    # 설명: 조건이 맞는지 확인해요.
    if "regularization" == "numpy":
        # 설명: 값을 저장하거나 바꿔요.
        arr = np.array([1, 2, 3, 4, 5])
        # 설명: 값을 저장하거나 바꿔요.
        result["mean"] = float(arr.mean())
        # 설명: 값을 저장하거나 바꿔요.
        result["std"] = float(arr.std())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "pandas":
        # 설명: 값을 저장하거나 바꿔요.
        df = pd.DataFrame({"feature": [1, 2, np.nan, 4], "target": [10, 15, 14, 20]})
        # 설명: 값을 저장하거나 바꿔요.
        result["missing_before"] = int(df["feature"].isna().sum())
        # 설명: 값을 저장하거나 바꿔요.
        df["feature"] = df["feature"].fillna(df["feature"].mean())
        # 설명: 값을 저장하거나 바꿔요.
        result["missing_after"] = int(df["feature"].isna().sum())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "probability":
        # 설명: 값을 저장하거나 바꿔요.
        toss = np.random.binomial(1, 0.6, size=1000)
        # 설명: 값을 저장하거나 바꿔요.
        result["estimated_p"] = float(toss.mean())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "linear_algebra":
        # 설명: 값을 저장하거나 바꿔요.
        A = np.array([[2, 1], [1, 3]])
        # 설명: 값을 저장하거나 바꿔요.
        b = np.array([1, 2])
        # 설명: 값을 저장하거나 바꿔요.
        x = np.linalg.solve(A, b)
        # 설명: 값을 저장하거나 바꿔요.
        result["solution"] = x.round(4).tolist()

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "linear_regression":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_regression(n_samples=120, n_features=3, noise=7, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        model = LinearRegression().fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = model.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        result["mse"] = float(mean_squared_error(y_test, pred))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "logistic_regression":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_classification(n_samples=180, n_features=5, n_informative=3, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        model = LogisticRegression(max_iter=500).fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = model.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        result["accuracy"] = float(accuracy_score(y_test, pred))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "decision_tree":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_classification(n_samples=180, n_features=6, n_informative=4, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)
        # 설명: 값을 저장하거나 바꿔요.
        result["feature_importance_sum"] = float(model.feature_importances_.sum())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "random_forest":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_classification(n_samples=220, n_features=6, n_informative=4, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        model = RandomForestClassifier(n_estimators=120, random_state=42).fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = model.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        result["f1"] = float(f1_score(y_test, pred))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "kmeans":
        # 설명: 값을 저장하거나 바꿔요.
        X = np.vstack([np.random.normal(0, 1, (40, 2)), np.random.normal(4, 1, (40, 2))])
        # 설명: 값을 저장하거나 바꿔요.
        model = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X)
        # 설명: 값을 저장하거나 바꿔요.
        result["cluster_count"] = int(len(np.unique(model.labels_)))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "metrics":
        # 설명: 값을 저장하거나 바꿔요.
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        # 설명: 값을 저장하거나 바꿔요.
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        # 설명: 값을 저장하거나 바꿔요.
        y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.95])
        # 설명: 값을 저장하거나 바꿔요.
        result["precision"] = float(precision_score(y_true, y_pred))
        # 설명: 값을 저장하거나 바꿔요.
        result["recall"] = float(recall_score(y_true, y_pred))
        # 설명: 값을 저장하거나 바꿔요.
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "validation":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        scores = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=cv)
        # 설명: 값을 저장하거나 바꿔요.
        result["cv_mean"] = float(scores.mean())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "feature_engineering":
        # 설명: 값을 저장하거나 바꿔요.
        df = pd.DataFrame({
            # 설명: 다음 코드를 실행해요.
            "age": [25, 30, 35, 40],
            # 설명: 다음 코드를 실행해요.
            "salary": [3000, 4000, 5000, 6000],
            # 설명: 다음 코드를 실행해요.
            "city": ["Seoul", "Busan", "Seoul", "Daegu"],
        # 설명: 다음 코드를 실행해요.
        })
        # 설명: 값을 저장하거나 바꿔요.
        pre = ColumnTransformer([
            # 설명: 다음 코드를 실행해요.
            ("num", StandardScaler(), ["age", "salary"]),
            # 설명: 값을 저장하거나 바꿔요.
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        # 설명: 다음 코드를 실행해요.
        ])
        # 설명: 값을 저장하거나 바꿔요.
        transformed = pre.fit_transform(df)
        # 설명: 값을 저장하거나 바꿔요.
        result["transformed_shape"] = list(transformed.shape)

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "regularization":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_regression(n_samples=100, n_features=10, noise=15, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        ridge = Ridge(alpha=1.0).fit(X, y)
        # 설명: 값을 저장하거나 바꿔요.
        lasso = Lasso(alpha=0.1).fit(X, y)
        # 설명: 값을 저장하거나 바꿔요.
        result["ridge_coef_norm"] = float(np.linalg.norm(ridge.coef_))
        # 설명: 값을 저장하거나 바꿔요.
        result["lasso_non_zero"] = int(np.count_nonzero(lasso.coef_))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "dimensionality_reduction":
        # 설명: 값을 저장하거나 바꿔요.
        X = np.random.randn(150, 10)
        # 설명: 값을 저장하거나 바꿔요.
        pca = PCA(n_components=2, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        reduced = pca.fit_transform(X)
        # 설명: 값을 저장하거나 바꿔요.
        result["reduced_shape"] = list(reduced.shape)
        # 설명: 값을 저장하거나 바꿔요.
        result["explained"] = float(pca.explained_variance_ratio_.sum())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "outlier_detection":
        # 설명: 값을 저장하거나 바꿔요.
        X = np.vstack([np.random.normal(0, 1, (100, 2)), np.array([[8, 8], [9, 9], [10, 10]])])
        # 설명: 값을 저장하거나 바꿔요.
        model = IsolationForest(contamination=0.03, random_state=42).fit(X)
        # 설명: 값을 저장하거나 바꿔요.
        pred = model.predict(X)
        # 설명: 값을 저장하거나 바꿔요.
        result["outliers_detected"] = int((pred == -1).sum())

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "time_series":
        # 설명: 값을 저장하거나 바꿔요.
        series = pd.Series(np.sin(np.linspace(0, 6, 60)) + np.random.normal(0, 0.1, 60))
        # 설명: 값을 저장하거나 바꿔요.
        result["ma_5_last"] = float(series.rolling(5).mean().iloc[-1])
        # 설명: 값을 저장하거나 바꿔요.
        result["lag_1_last"] = float(series.shift(1).iloc[-1])

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "pipeline":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_classification(n_samples=180, n_features=6, n_informative=4, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        pipe = Pipeline([
            # 설명: 다음 코드를 실행해요.
            ("scaler", StandardScaler()),
            # 설명: 값을 저장하거나 바꿔요.
            ("model", LogisticRegression(max_iter=500, random_state=42)),
        # 설명: 다음 코드를 실행해요.
        ])
        # 설명: 값을 저장하거나 바꿔요.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 설명: 다음 코드를 실행해요.
        pipe.fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = pipe.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        result["accuracy"] = float(accuracy_score(y_test, pred))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "model_persistence":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_regression(n_samples=100, n_features=4, noise=3, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        model = LinearRegression().fit(X, y)
        # 설명: 값을 저장하거나 바꿔요.
        model_path = Path(__file__).with_name("linear_model.joblib")
        # 설명: 다음 코드를 실행해요.
        joblib.dump(model, model_path)
        # 설명: 값을 저장하거나 바꿔요.
        loaded = joblib.load(model_path)
        # 설명: 값을 저장하거나 바꿔요.
        result["coef_match"] = bool(np.allclose(model.coef_, loaded.coef_))

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "serving":
        # 설명: 값을 저장하거나 바꿔요.
        result["message"] = "FastAPI 백엔드(app/main.py)의 /api/chapters 엔드포인트를 확인하세요."

    # 설명: 앞 조건이 아니면 다른 조건을 확인해요.
    elif "regularization" == "mini_project":
        # 설명: 값을 저장하거나 바꿔요.
        X, y = make_classification(n_samples=250, n_features=7, n_informative=4, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 설명: 값을 저장하거나 바꿔요.
        pipe = Pipeline([
            # 설명: 다음 코드를 실행해요.
            ("scaler", StandardScaler()),
            # 설명: 값을 저장하거나 바꿔요.
            ("model", RandomForestClassifier(n_estimators=150, random_state=42)),
        # 설명: 다음 코드를 실행해요.
        ])
        # 설명: 다음 코드를 실행해요.
        pipe.fit(X_train, y_train)
        # 설명: 값을 저장하거나 바꿔요.
        pred = pipe.predict(X_test)
        # 설명: 값을 저장하거나 바꿔요.
        result["final_accuracy"] = float(accuracy_score(y_test, pred))

    # 설명: 함수 결과를 돌려줘요.
    return result


# 설명: 조건이 맞는지 확인해요.
if __name__ == "__main__":
    # 설명: 다음 코드를 실행해요.
    print(run())
