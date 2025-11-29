# Predicting tumor diagnosis from new patient measurements

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def preprocessing_dataframe(df: pd.DataFrame):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df = df.dropna(axis=1, how='all')  # Dropping Nan columns

    df['label']=df['diagnosis'].map({'M':1, 'B':0})
    assert df['label'].isna().sum()==0, "Unexpected Values in diagnosis column"

    X = df.drop(columns=['diagnosis', 'label'])
    y = df['label']

    return X, y

def split_data(X,y,test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_and_tune_knn(X_train, y_train):
    pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:")
    print(grid_search.best_params_)
    print(f"Best Cross-validation F1 score: {grid_search.best_score_:.4f}\n")

    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc: .4f}")
    print("\nClassification Report(0=benign, 1=malignant):")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

def predict_new_patient(model, feature_names, values):
    X_new = pd.DataFrame([values], columns=feature_names)
    proba = model.predict_proba(X_new)[0]
    pred_label = model.predict(X_new)[0]

    diagnosis_map = {0: "Bengin", 1: "Malignant"}
    diagnosis_str = diagnosis_map[pred_label]
    prob_malignant = proba[1]

    print("Prediction for new patient:")
    print(f"  Diagnosis: {diagnosis_str}")
    print(f"  P(malignant): {prob_malignant:.4f}")

    return diagnosis_str, prob_malignant

def main():
    csv_path = "dataset/KNNAlgorithmDataset.csv"
    df = load_data(csv_path)
    print("Data loaded. Shape:", df.shape)

    X, y = preprocessing_dataframe(df)
    print("Featur shape:", X.shape, "| Labels Shape", y.shape)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print("Train size:", X_train.shape[0], "| Test Size:", X_test.shape[0])

    best_model = build_and_tune_knn(X_train, y_train)

    evaluate_model(best_model, X_test, y_test)

    example_values = X_test.iloc[0].values
    predict_new_patient(best_model, feature_names, example_values)

    def plot_before_and_after_scaling(model, X_train, y_train):
        f1 = 'radius mean'
        f2 = 'texture mean'

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_train[f1], X_train[f2],
            c=y_train,
            cmap="coolwarm",
            alpha=0.8
        )
        plt.title("Before scaling")
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.colorbar(label="Diagnosis (0=Bengin, 1=Malignant)")
        plt.grid(True)
        plt.savefig("before_scaling.png", dpi=200)
        plt.show()

        scaled = model['scaler'].transform(X_train)
        scaled_df = pd.DataFrame(scaled, columns=X_train.columns)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            scaled_df[f1], scaled_df[f2],
            c=y_train,
            cmap="coolwarm",
            alpha=0.8
        )
        plt.title("After Scaling (StandardScaler Applied)")
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.colorbar(label="Diagnosis (0=Benign, 1=Malignant)")
        plt.grid(True)
        plt.savefig("after_scaling.png", dpi=200)
        plt.show()

if __name__ == "__main__":
    main()