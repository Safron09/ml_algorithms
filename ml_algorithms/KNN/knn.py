# Predicting tumor diagnosis from new patient measurements

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
csv_path = "ml_algorithms\KNN\dataset\KNNAlgorithmDataset.csv"
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def preprocessing_dataframe(df: pd.DataFrame):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

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
    # StandardScaler solves this by normalizing each feature to: z = (x - mean) / std
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {
        'knn_n_neigbors':[3,5,7,9,11,15],
        'knn_weights':['uniform', 'distance'],
        'knn_metrics':['euclidean', 'manhattan']
    }
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5, # 5-fold cross-validation
        scoring='f1',
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("Best parameter:")
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

    