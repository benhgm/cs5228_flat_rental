"""
Script for Preprocessing the Data for CS5228 In-Class Kaggle Competition: Predicting HDB Flat Rental Prices
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')

def encode_categorical_labels(df):
    """
    Performs one-hot encoding on the 'flat_type' labels and categorical encoding on flat models

    Args:
        df (pandas.DataFrame): Dataframe object
    """
    df = pd.get_dummies(df, columns=['flat_type'], dtype="int64")
    
    flat_models = df['flat_model'].unique()
    categories_flat_model = {f: idx for idx, f in enumerate(flat_models)}
    df["flat_model_cat"] = df["flat_model"].apply(lambda x: categories_flat_model[x])

    return df

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file)
    
    df = encode_categorical_labels(dataset)
    
    df.to_csv(args.output_file, index=False)