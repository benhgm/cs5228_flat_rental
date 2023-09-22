"""
Script for Preprocessing the Data for CS5228 In-Class Kaggle Competition: Predicting HDB Flat Rental Prices
"""

import pandas as pd
import argparse
from typing import Optional

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

def std_flat_type(flat_type:str, lease_commence_year:int) -> Optional[float]:
    """
    Convert flat_type column from categorical to numerical column (Capture notion of expected utility of # of bedrooms in HDB flat)

    Source: https://www.teoalida.com/singapore/hdbflattypes/ (Domain expert: Teoalida)

    Possible value conversions:    
    2-room -> 2.0
    3-room -> 3.0
    4-room -> 4.0
    5-room -> 5.0
    executive -> additional logic is required due to historical context
        Before 1988, all Executives had 3 bedrooms and a large room with separate living and dining areas, + additional utility/maid room.
        From 1988 to 2000, all Executives had 4 bedrooms or 3 bedrooms + study, but Executives had larger living room, sometimes separate living and dining areas.
        After 2000, Executive Apartments have 3 bedrooms plus an open study area.

        To simplify:
        Before 1988: Executive = 3 bedrooms + 1 large room + 1 utility room = 3 + 1 + 0.5 = 4.5
        From 1988 to 2000: Executive = average of 3-4 bedrooms + study = 3.5 + 0.5 = 4
        After 2000: Executive = 3 bedrooms plus an open study area = 3 + 0.5 = 3.5
    """
    if ' room' in flat_type:
        output = float(eval(flat_type.replace('-room', '')))
    elif flat_type == 'executive':
        if lease_commence_year < 1988:
            output = 4.5
        elif  1988 <= lease_commence_year <= 2000:
            output = 4
        elif lease_commence_year > 2000:
            output = 3.5
    else:
        output = None

    return output

def std_flat_model(flat_model:str) -> str:
    """ 
    Standardise flat model types based on domain expert (Teoalida)

    Source: https://www.teoalida.com/singapore/hdbflattypes/
    Flat type abbreviation: 
    1) STD = Standard, 
    2) I = Improved, 
    3) NG = New Generation, 
    4) S = Simplified, 
    5) A = Model A
    6) P = Premium (Luxury apartment types: maisonette, adjoined, dbss etc.)
    7) OTH = Others (catch all exceptions, potential outlier types)

    Mapping of dataset values to standardised flat type abbreviations:
    model a                   -> A
    improved                  -> I
    new generation            -> NG
    premium apartment         -> P
    simplified                -> S
    standard                  -> STD
    apartment                 -> OTH
    maisonette                -> P
    model a2                  -> A
    dbss                      -> P
    type s1                   -> STD
    model a-maisonette        -> P
    adjoined flat             -> P
    type s2                   -> STD
    2-room                    -> OTH
    premium apartment loft    -> P
    premium maisonette        -> P
    terrace                   -> P
    3gen                      -> OTH

    """

    flat_model = flat_model.strip().lower()

    flat_model_mapping = {
        'model a': 'A',
        'improved': 'I',
        'new generation': 'NG',
        'premium apartment': 'P',
        'simplified': 'S',
        'standard': 'STD',
        'apartment': 'OTH',
        'maisonette': 'P',
        'model a2': 'A',
        'dbss': 'P',
        'type s1': 'STD',
        'model a-maisonette': 'P',
        'adjoined flat': 'P',
        'type s2': 'STD',
        '2-room': 'OTH',
        'premium apartment loft':'P',
        'premium maisonette': 'P',
        'terrace': 'P',
        '3gen': 'OTH'
    }
    
    try:
        std_flat_model = flat_model_mapping[flat_model]
    except:
        std_flat_model = 'OTH'

    return std_flat_model

def std_remaining_lease(lease_commence_year:int, current_year:int) -> int:
    """ 
    Standardise lease_commence_date by converting lease start year to number of years left to lease (Semantic Scaling of data)
    As all HDB flats have a 99-year lease period, the number of years left can be calculated by:
    years_left = 99 - (current_year - lease_commence_year)
    """
    return int(99 - (current_year - lease_commence_year))

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file)
    
    df = encode_categorical_labels(dataset)

    df['std_flat_type'] = df.apply(lambda x: std_flat_type(x['flat_type'], x['lease_commence_year']), axis=1)
    
    df['std_flat_model'] = df.apply(lambda x: std_flat_model(x['flat_model']), axis=1)

    df['std_remaining_lease'] = df.apply(lambda x: std_remaining_lease(x['lease_commence_year'], x['current_year']), axis=1)
    
    df.to_csv(args.output_file, index=False)