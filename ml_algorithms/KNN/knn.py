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
    