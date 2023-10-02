from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from math import radians
import matplotlib.pyplot as plt
import pandas as pd
from geopy.distance import distance


def calculate_distance(flat_latitude, flat_longitude, amenity_latitude, amenity_longitude):
    flat_latitude = [radians(x) for x in flat_latitude]
    flat_longitude = [radians(x) for x in flat_longitude]
    amenity_latitude = [radians(x) for x in amenity_latitude]
    amenity_longitude = [radians(x) for x in amenity_longitude]
    
    flat_coords = np.array([flat_latitude, flat_longitude]).T
    amenity_coords = np.array([amenity_latitude, amenity_longitude]).T
    
    # 6371: Earth radius
    flat_amenity_distance = haversine_distances(flat_coords, amenity_coords) * 6371 
    return flat_amenity_distance


def main():
    data = pd.read_csv("data/train.csv")
    shopping_mall_data = pd.read_csv("./data/auxiliary-data/auxiliary-data/sg-shopping-malls.csv")
    primary_school_data = pd.read_csv("./data/auxiliary-data/auxiliary-data/sg-primary-schools.csv")
    mrt_existing_station_data = pd.read_csv("./data/auxiliary-data/auxiliary-data/sg-mrt-existing-stations.csv")

  
    flat_latitude_list = list(data['latitude'])
    flat_longitude_list = list(data['longitude'])
    shopping_mall_latitude_list = list(shopping_mall_data['latitude'])
    shopping_mall_longitude_list = list(shopping_mall_data['longitude'])
    
    flat_mall_distance = calculate_distance(flat_latitude_list, flat_longitude_list, shopping_mall_latitude_list, shopping_mall_longitude_list)
    
    i = 0
    j = 1
    print("Geopy distance: {}".format(distance((flat_latitude_list[i], flat_longitude_list[i]), (shopping_mall_latitude_list[j], shopping_mall_longitude_list[j])).km))
    print("Haversine-based distance: {}".format(flat_mall_distance[i][j]))    

if __name__ == '__main__':
    main()
