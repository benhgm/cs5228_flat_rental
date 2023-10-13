"""
Script for Preprocessing the Data for CS5228 In-Class Kaggle Competition: Predicting HDB Flat Rental Prices
"""

import pandas as pd
import argparse
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')

def preprocess_town(df):
    """
    Convert town to lowercase
    Do one-hot encoding for town
    """
    df['town'] = df['town'].str.lower()

    one_hot = pd.get_dummies(df['town'], dtype="int64")
    df = df.drop(columns=['town'])
    df = df.join(one_hot)
    return df


def preprocess_street_name(df):
    """
    Convert street name to lowercase
    Remove the number in street_name
    Append the prefix "street_name" to avoid name collision with values of other columns
    Do one-hot encoding for street_name
    """
    df['street_name'] = df['street_name'].str.lower()

    street_name_list = list(df['street_name'])
    for i in range(len(street_name_list)):
        tokenized_street_name = street_name_list[i].split()
        try:
            final_word_to_int = int(tokenized_street_name[-1])
        except:
            continue
        street_name_list[i] = ' '.join(tokenized_street_name[:-1]).strip()
    df['street_name'] = street_name_list


    street_name_list = list(df['street_name'])
    for i in range(len(street_name_list)):
        street_name_list[i] = "street_name " + street_name_list[i]
    df['street_name'] = street_name_list


    one_hot = pd.get_dummies(df['street_name'], dtype="int64")
    df = df.drop(columns=['street_name'])
    df = df.join(one_hot)

    return df

def preprocess_planning_area(df):
    """
    Append the prefix "planning_area" to avoid name collision with values of other columns
    Do the one-hot encoding for planning_area
    """

    planning_area_list = list(df['planning_area'])
    for i in range(len(planning_area_list)):
        planning_area_list[i] = "planning_area " + planning_area_list[i]
    df['planning_area'] = planning_area_list

    one_hot = pd.get_dummies(df['planning_area'], dtype="int64")
    df = df.drop(columns=['planning_area'])
    df = df.join(one_hot)
    return df

def preprocess_subzone(df):
    """
    Append the prefix "subzone" to avoid name collision with values of other columns
    Do the one-hot encoding for subzone
    """

    subzone_list = list(df['subzone'])
    for i in range(len(subzone_list)):
        subzone_list[i] = "subzone " + subzone_list[i]
    df['subzone'] = subzone_list

    one_hot = pd.get_dummies(df['subzone'], dtype="int64")
    df = df.drop(columns=['subzone'])
    df = df.join(one_hot)
    return df

def preprocess_block(block_num):
    """
    Convert block numbers to integer values and removing any letters if any.
    """
    last_char = block_num[-1]
    if not last_char.isnumeric():
        block_num = block_num[:-1]
    return int(block_num)

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
    if '-room' in flat_type:
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


def preprocess_rent_approval_date(df):
    """
    Split date into year and month
    Drop the original rent_approval_date column
    """
    df['rent_approval_year'] = df['rent_approval_date'].apply(lambda x: x[:4]).astype(int)
    df['rent_approval_month'] = df['rent_approval_date'].apply(lambda x: x[5:]).astype(int)

    df = df.drop("rent_approval_date", axis=1)
    return df

def preprocess_flat_model(df):
    """
    Do the one-hot encoding for flat_model
    """
    one_hot = pd.get_dummies(df['std_flat_model'], dtype="int64")
    df = df.drop(columns=['std_flat_model', 'flat_model'])
    df = df.join(one_hot)

    return df


def preprocess_planning_area(df):
    """
    Append the prefix "planning_area" to avoid name collision with values of other columns
    Do the one-hot encoding for planning_area
    """

    planning_area_list = list(df['planning_area'])
    for i in range(len(planning_area_list)):
        planning_area_list[i] = "planning_area_ " + planning_area_list[i]
    df['planning_area'] = planning_area_list

    one_hot = pd.get_dummies(df['planning_area'], dtype="int64")
    df = df.drop(columns=['planning_area'])
    df = df.join(one_hot)
    return df

def preprocess_region(df):
    """
    Append the prefix "region" to avoid name collision with values of other columns
    Do the one-hot encoding for region
    """
    one_hot = pd.get_dummies(df['region'], dtype="int64")
    df = df.drop(columns=['region'])
    df = df.join(one_hot)
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    df['std_flat_type'] = df.apply(lambda x: std_flat_type(x['flat_type'], x['lease_commence_year']), axis=1)
    
    df['std_flat_model'] = df.apply(lambda x: std_flat_model(x['flat_model']), axis=1)

    df['std_remaining_lease'] = df.apply(lambda x: std_remaining_lease(x['lease_commence_year'], x['current_year']), axis=1)
    
    df.to_csv(args.output_file, index=False)