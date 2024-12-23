import json
import csv
import os
import re
import math
import numpy as np
from config import RESULTS_PATH

def getDist_points(q, v):
    try:
        EARTH_REDIUS = 6378.137
        v=np.array(v).reshape(-1,2)
        q=np.array(q).reshape(-1,2)
        v, q = np.deg2rad(v), np.deg2rad(q)
        
        lat1, lat2 = v[:,1].reshape(-1,1), q[:,1].reshape(-1,1)
        lng1, lng2 = v[:,0].reshape(-1,1), q[:,0].reshape(-1,1)
        sin_lats = np.dot(np.sin(lat1), np.sin(lat2.T))
        cos_lats = np.dot(np.cos(lat1), np.cos(lat2.T))
        cos_lng_diff = np.cos(lng2.reshape(-1) - lng1.reshape(-1,1))
        s = np.arccos(sin_lats + cos_lats*cos_lng_diff)
        d = s * EARTH_REDIUS ###* 1000
        d = d.T # q to v
        return d
    except:
        print('error')

#like above way to compute the distance
def get_two_point_distance(lat1, lon1, lat2, lon2):
    EARTH_REDIUS = 6378.137
    
    #deg2rad
    
    sin_lats = math.sin(math.radians(lat1)) * math.sin(math.radians(lat2))
    cos_lats = math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
    cos_lng_diff = math.cos(math.radians(lon2) - math.radians(lon1))
    s = math.acos(sin_lats + cos_lats * cos_lng_diff)
    d = s * EARTH_REDIUS
    return d

def compute_accuracy_planet(pred_coordinates_list, true_coordinates_list):
    accuracy_planet = {
        "1km": 0,
        "25km": 0,
    }
    for idx in range(len(pred_coordinates_list)):

        distance_km = getDist_points([pred_coordinates_list[idx][1], pred_coordinates_list[idx][0]], [true_coordinates_list[idx][1], true_coordinates_list[idx][0]])
        distance_km=distance_km[0][0]
        if distance_km <= 1:
            accuracy_planet["1km"] += 1
        if distance_km <= 25:
            accuracy_planet["25km"] += 1

    for k in accuracy_planet.keys():
        accuracy_planet[k] /= len(true_coordinates_list)
        accuracy_planet[k] *= 100.0 # percent

    return accuracy_planet


def calculate_acc(results_path, output_file):
    # Initialize the results list
    results = []

    # Calculate accuracy for each city and model
    for root, dirs, files in os.walk(results_path):
        for file in files:
            # 使用正则表达式匹配 {city_name}_{model_name}_geoloc.jsonl 文件
            match = re.match(r"([A-Za-z]+)_(.+)_geoloc.jsonl", file)
            if match:
                city_name = match.group(1)
                model_name = match.group(2)
                city_results = [model_name, city_name]

                try:
                    with open(os.path.join(root, file), "r") as f:
                        img2cityloc = [json.loads(line) for line in f]

                    # Coordinates accuracy calculation
                    if city_name == "Shanghai" or city_name == "Beijing":
                        true_lng = [float(data['image_path'].split("&")[1]) for data in img2cityloc]
                        true_lat = [float(data['image_path'].split("&")[2]) for data in img2cityloc]
                        true_coords = list(zip(true_lat, true_lng))

                    else:
                        true_lat = [float(data['image_path'].split("&")[1]) for data in img2cityloc]
                        true_lng = [float(data['image_path'].split("&")[2]) for data in img2cityloc]
                        true_coords = list(zip(true_lat, true_lng))
                        
                    pred_coords = []
                    for data in img2cityloc:
                        try:
                            pred = data['response'].split("(")[1].split(")")[0]
                            pred_lat = float(pred.split(", ")[0])
                            pred_lng = float(pred.split(", ")[1])
                            pred_coords.append((pred_lat, pred_lng))
                        except:
                            pred_coords.append((0, 0))
                    result = compute_accuracy_planet(pred_coords, true_coords)
                    # City accuracy calculation
                    city_count = 0
                    for data in img2cityloc:
                        if city_name in data['response'] or "S\u00e3o Paulo" in data['response'] or "S\u00e3oPaulo" in data['response']:
                            city_count += 1
                    city_accuracy = city_count / len(img2cityloc) if img2cityloc else 0

                    city_results.append(city_accuracy) 
                    city_results.append(result["1km"])  
                    city_results.append(result["25km"])  

                    results.append(city_results)

                except FileNotFoundError:
                    # In case the file is not found, set accuracy as 'N/A'
                    city_results.append('N/A')
                    city_results.append('N/A')
                    city_results.append('N/A')
                    results.append(city_results)

    # Write results to CSV
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Model_Name', 'City_Name', 'City_Accuracy', 'Acc@1km', 'Acc@25km'])

        # Write the rows
        for row in results:
            writer.writerow(row)

    print("Street View results have been saved!")

if __name__ == '__main__':
    results_path = os.path.join(RESULTS_PATH, "street_view/")
    output_file = os.path.join(results_path, "geoloc_benchmark_results.csv")
    calculate_acc(results_path, output_file)
