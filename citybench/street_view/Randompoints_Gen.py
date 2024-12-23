import csv
import random
import ast
import pandas as pd


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='SanFrancisco', help='City name')
    args = parser.parse_args()
    city = args.city
    print(city)
    boudary = CITY_BOUNDARY[city]
    #"SanFrancisco": [(-122.5099, 37.8076), (-122.5099, 37.6153), (-122.3630, 37.6153), (-122.3630, 37.8076)]
    
    #latitude and longitude range box
    upper_lat = boudary[0][1]
    lower_lat = boudary[1][1]
    left_lon = boudary[1][0]
    right_lon = boudary[2][0]





    # random generate 10000 points in the box
    all_lon = []
    all_lat = []
    for _ in range(10000):
        lon = random.uniform(left_lon, right_lon)
        lat = random.uniform(lower_lat, upper_lat)
        all_lon.append(lon)
        all_lat.append(lat)

    # split all_lat and all_lon into 10 pieces
    all_lat = [all_lat[i:i + 1000] for i in range(0, len(all_lat), 1000)]
    all_lon = [all_lon[i:i + 1000] for i in range(0, len(all_lon), 1000)]

    # write data to csvv file
    with open(f'./random_points_{city}.csv', 'w', newline='') as csvfile:
        fieldnames = ['Index', 'City', 'NEAR_X', 'NEAR_Y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(10):
            writer.writerow({'Index': i, 'City': city+'_'+str(i), 'NEAR_X': all_lon[i], 'NEAR_Y': all_lat[i]})

    # Transform the string info to list
    csv_file_path = f'./random_points_{city}.csv'
    df = pd.read_csv(csv_file_path)
    df['NEAR_X'] = df['NEAR_X'].apply(lambda x: ast.literal_eval(x))
    df['NEAR_Y'] = df['NEAR_Y'].apply(lambda x: ast.literal_eval(x))
