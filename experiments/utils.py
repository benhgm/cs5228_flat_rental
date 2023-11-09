import numpy as np
import pandas as pd

def split_features_and_monthly_rent_label(df: pd.DataFrame):
    if "monthly_rent" not in df.columns:
        # test data
        X = df.copy()
        return X, None
    X = df.drop(columns="monthly_rent")
    y = df["monthly_rent"]
    return X, y