import argparse
import numpy as np
import pandas as pd
from math import radians
from sklearn.metrics.pairwise import haversine_distances

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--ref_file')
parser.add_argument('--output_file')

def calculate_distance(flat_latitude:list, flat_longitude:list, amenity_latitude:list, amenity_longitude:list) -> np.ndarray:
    """
    Calculate approx distance using haversine distance
    """
    flat_latitude = [radians(x) for x in flat_latitude]
    flat_longitude = [radians(x) for x in flat_longitude]
    amenity_latitude = [radians(x) for x in amenity_latitude]
    amenity_longitude = [radians(x) for x in amenity_longitude]
    
    flat_coords = np.array([flat_latitude, flat_longitude]).T
    amenity_coords = np.array([amenity_latitude, amenity_longitude]).T
    
    # 6371: Earth radius
    flat_amenity_distance = haversine_distances(flat_coords, amenity_coords) * 6371 
    return flat_amenity_distance

def count_amenity(target_df:pd.DataFrame, ref_df:pd.DataFrame, threshold:float) -> pd.DataFrame:
    """ 
    Processes distance matrix to find counts of amenities under threshold distance

    Returns target_df wwith additional column 'amenity_count'
    """
    flat_latitude_list = list(target_df['latitude'])
    flat_longitude_list = list(target_df['longitude'])
    amenity_latitude_list = list(ref_df['latitude'])
    amenity_longitude_list = list(ref_df['longitude'])

    flat_amenity_distance = calculate_distance(flat_latitude_list, flat_longitude_list, amenity_latitude_list, amenity_longitude_list)
    
    dist_matrix = np.where(flat_amenity_distance <= threshold)[0] # find idx of pairwise distances within eps <= 0.1

    unique, counts = np.unique(dist_matrix, return_counts=True)

    flat_amenity_count_ref = dict(zip(unique, counts))

    # for instances whereby amenity is 0, pad dict with zero
    for idx in range(len(flat_amenity_count_ref)):
        if idx not in flat_amenity_count_ref.keys(): 
            flat_amenity_count_ref[idx] = 0

    amenity_count_df = pd.DataFrame.from_dict(flat_amenity_count_ref, orient='index').sort_index(ascending=True).rename(columns={0:'amenity_count'})
    
    output_df = pd.merge(target_df, amenity_count_df, how='left', left_index=True, right_index=True)
    output_df["amenity_count"] = output_df["amenity_count"].fillna(value=0.0)

    return output_df

def compute_mean_rental_prices(dataset):
    """
    Function to compute the mean monthly rental prices.

    The steps implemented are as follows:
    - Compute the median rental price for each flat type
    - Compute the mean over the median over all flat types
    """
    flat_types = dataset['std_flat_type'].unique()
    flat_type_data = {}
    for ft in flat_types:
        key = str(ft)
        flat_type_data[key] = dataset.loc[dataset["std_flat_type"] == ft]

    median_rental = {}
    for rm in flat_type_data.keys():
        df = flat_type_data[rm]
        median_rental[rm] = []
        for y in [2021, 2022, 2023]:
            yearly_data = df.loc[df['rent_approval_year']==y]
            for m in range(1, 13):
                if m in yearly_data['rent_approval_month'].unique():
                    monthly_data = yearly_data.loc[yearly_data['rent_approval_month']==m]
                    median_rent = monthly_data['monthly_rent'].median()
                    median_rental[rm].append(median_rent)

    num_types = len(median_rental.keys())
    num_entries = len(median_rental[list(median_rental.keys())[0]])

    mean_rental_prices = [sum([median_data[i] for _, median_data in median_rental.items()])/num_types for i in range(num_entries)]
    return mean_rental_prices

def compute_mean_coe_prices(coe_prices, dataset):
    """
    Function to compute the mean monthly coe prices and add the data to an existing dataset
    """
    months = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12
    }

    coe_prices['month'] = coe_prices['month'].apply(lambda x: months[x])
    coe_prices["date"] = pd.to_datetime(coe_prices[["year", "month"]].assign(DAY=1))
    coe_prices["date"] = coe_prices["date"].dt.strftime('%Y-%m')
    mean_coe_prices = coe_prices.groupby(pd.PeriodIndex(coe_prices['date'], freq="M"), as_index=False)["price"].mean()

    date = ["2021-" + i for i in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]]
    date += ["2022-" + i for i in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]]
    date += ["2023-" + i for i in ["01", "02", "03", "04", "05", "06", "07"]]

    coe_price_hash_table = {}
    for i, p in enumerate(mean_coe_prices.values.tolist()):
        d = date[i]
        coe_price_hash_table[d] = p[0]

    new_column = {"mean_coe_price": []}

    for i, row in dataset.iterrows():
        row_date = str(int(row['rent_approval_year'])) + "-" + str(int(row['rent_approval_month'])).zfill(2)
        new_column["mean_coe_price"].append(coe_price_hash_table[row_date])

    dataset['mean_coe_price'] = new_column['mean_coe_price']
    return mean_coe_prices, dataset


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file)
    ref = pd.read_csv(args.ref_file)

    df = count_amenity(dataset, ref, 0.8)

    df.to_csv(args.output_file, index=False)