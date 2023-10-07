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
    for idx in range(flat_amenity_count_ref.shape[0]):
        if idx not in flat_amenity_count_ref.keys(): 
            flat_amenity_count_ref[idx] = 0

    amenity_count_df = pd.DataFrame.from_dict(flat_amenity_count_ref, orient='index').sort_index(ascending=True).rename(columns={0:'amenity_count'})
    
    output_df = pd.merge(target_df, amenity_count_df, how='left', left_index=True, right_index=True)

    return output_df

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = pd.read_csv(args.input_file)
    ref = pd.read_csv(args.ref_file)

    df = count_amenity(dataset, ref, 0.8)

    df.to_csv(args.output_file, index=False)