"""
Script for Preprocessing the Data for CS5228 In-Class Kaggle Competition: Predicting HDB Flat Rental Prices
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')

def convert_strings_to_lowercase(df):
    """
    Function to find all columns with string data and standardises the casing to lowercase

    Args:
        df (pandas.DataFrame): Dataframe object
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()
    
    return df

def split_data_to_year_month(df):
    """
    Takes a column indicating the rent approval date and splits it into its month and year components
    and converts the data to integer value

    Args:
        df (pandas.DataFrame): Dataframe object
    """
    df["rent_approval_year"] = df["rent_approval_date"].apply(lambda x: x[:4]).astype(int)
    df["rent_approval_month"] = df["rent_approval_date"].apply(lambda x: x[5:]).astype(int)

    return df

def clean_flat_type_labels(df):
    """
    Function to standardise the flat type labels. 
    There are currently two different labels for each flat type - "x-room" and "x room".
    This function converts all "x room" instances to "x-room"

    Args:
        df (pandas.DataFrame): Dataframe object
    """
    df['flat_type'].replace({'2 room': '2-room',
                           '3 room': '3-room',
                           '4 room': '4-room',
                           '5 room': '5-room'}, inplace=True)
    return df

def drop_data(df, column_names):
    """
    Function to drop data from specified columns

    Args:
        df (pandas.Dataframe): Dataframe object
        column_names (list): List of columns to drop
    """
    df = df.drop(column_names, axis='columns')
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file)
    
    df = convert_strings_to_lowercase(dataset)
    df = split_data_to_year_month(df)
    df = clean_flat_type_labels(df)

    with open('data/data_to_drop.txt', "r") as f:
        data_to_drop = f.readlines()
        data_to_drop = data_to_drop[0].split(',')
    df = drop_data(df, data_to_drop)
    df.to_csv(args.output_file, index=False)
