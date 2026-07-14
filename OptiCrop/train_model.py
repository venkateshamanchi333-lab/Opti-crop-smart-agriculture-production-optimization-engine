import json
import os
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_and_clean_dataset():
    df = pd.read_csv(DATASET_PATH)
    required_columns = FEATURES + ["label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    df = df.dropna()
    df = df.drop_duplicates()

    for column in FEATURES:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        df[column] = df[column].clip(lower, upper)

    return df


def build_models():
    return {
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=3)),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    df = load_and_clean_dataset()
    print(df.head())
    print(df.shape)
    print(df.columns)


    encoder = LabelEncoder()
    X = df[FEATURES]
    y = encoder.fit_transform(df["label"])

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    results = {}
    trained_models = {}
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[model_name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4),
        }
        trained_models[model_name] = model

    cluster_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=min(5, df["label"].nunique()), random_state=42, n_init=10)),
    ])
    cluster_model.fit(X)

    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    artifact = {
        "model": trained_models[best_model_name],
        "encoder": encoder,
        "features": FEATURES,
        "best_model_name": best_model_name,
        "model_results": results,
        "cluster_model": cluster_model,
        "crop_profiles": df.groupby("label")[FEATURES].mean().round(2).to_dict(orient="index"),
    }

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(artifact, model_file)

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump({"best_model": best_model_name, "results": results}, metrics_file, indent=2)

    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train()
