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

def split_features_and_rent_per_sqm_label(df: pd.DataFrame):
    # Also return the floor_area_sqm data for conversion later
    if "monthly_rent" not in df.columns:
        # test data
        X = df.drop(columns="floor_area_sqm")
        floor_area_sqm = df["floor_area_sqm"]
        return X, None, floor_area_sqm
    X = df.drop(columns=["floor_area_sqm", "monthly_rent"])
    y = df["monthly_rent"] / df["floor_area_sqm"]
    floor_area_sqm = df["floor_area_sqm"]
    return X, y, floor_area_sqm

def convert_rent_per_sqm_label_to_monthly_rent_label(y, floor_area_sqm):
    # convert back to original label for final prediction & error calculation
    return np.array(y) * np.array(floor_area_sqm)

def round_to_nearest_price(y, round_interval: float = 50.0):
    # function to round the label to the nearest price (default: nearest 50 sgd)
    return np.round(np.array(y) / round_interval) * round_interval
