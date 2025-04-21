import os
import argparse
import numpy as np
import pandas as pd
import random
import math
import base64
import subprocess

from pycitydata.map import Map
from citysim.routing import RoutingClient
from config import MAP_CACHE_PATH, RESOURCE_PATH, RESULTS_PATH, MONGODB_URI, MAP_DICT, PROXY, IMAGE_FOLDER 

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000
    return c * r

def extract_coords_from_filename(city, image_filename):
    meta_file = os.path.join(IMAGE_FOLDER, f"{city}_StreetView_Images/combined_stitch_meta_info.csv")
    parts = image_filename.split('_')
    dataset_name = parts[0]
    sid_84_long = parts[1]
    sid_84_lat = parts[2]
    sid = parts[3].split('.')[0]  

    df = pd.read_csv(meta_file)

    matched_row = df[(df['sid_84_long'] == float(sid_84_long)) & 
                     (df['sid_84_lat'] == float(sid_84_lat)) & 
                     (df['sid'] == sid)]

    return matched_row.iloc[0]['longitude_origin'], matched_row.iloc[0]['latitude_origin']



def calculate_distance(city, last_image_name, cur_image_name):
    meta_file = os.path.join(IMAGE_FOLDER, f"{city}_StreetView_Images/combined_stitch_meta_info.csv")
    meta_df = pd.read_csv(meta_file)
    if city in ["Beijing", "Shanghai"]:
        last_image_lng, last_image_lat = extract_coords_from_filename(city, last_image_name)
        cur_image_lng, cur_image_lat = extract_coords_from_filename(city, cur_image_name)
        distance = haversine_distance(last_image_lat, last_image_lng, cur_image_lat, cur_image_lng)
    else:
        last_image_coords = meta_df.loc[meta_df['file_name'] == last_image_name, ['query_longti', 'query_lati']].iloc[0]
        cur_image_coords = meta_df.loc[meta_df['file_name'] == cur_image_name, ['query_longti', 'query_lati']].iloc[0]
        distance = haversine_distance(last_image_coords['query_lati'], last_image_coords['query_longti'], 
                            cur_image_coords['query_lati'], cur_image_coords['query_longti'])
        
    return distance


def calculate_direction(current_end, next_start):
    dx = next_start[0] - current_end[0] 
    dy = next_start[1] - current_end[1]  
    
    if abs(dx) > abs(dy): 
        if dx > 0:
            return "right"
        else:
            return "left"
    else:  
        if dy > 0:
            return "forward"
        else:
            return "forward"
    
def get_basic_prompt():
    basic_prompt = f"""
    You are tasked with guiding a virtual traveler through a series of street view images along a specific route. With each image provided:

    Describe the Image: Identify and describe any prominent landmarks, features, or unique characteristics visible in the photo. This may include notable buildings, distinctive shops, interesting street art, or any other element that stands out.

    Action Decision: For each image, I will also provide the navigation action decision that needs to be taken at that location (e.g., turn left, go straight, turn right, or stop). You must integrate this action decision into your description, using the landmarks as reference points. For example, you might say, "At the red cafe with the large windows on your left, turn right to head towards the park with the fountain."

    Remember, your descriptions should not include URL links to images or 'image' word. Instead, they should provide a clear, concise, and complete guide using landmarks that will be paired with the images I provide. And you must intergrate the action decesion with image description. This ensures that anyone using your descriptions and my images can successfully navigate and reach their destination.

    Here is the images and action decisions for each step of the route:
    
    """
    return basic_prompt
        
def get_prompt_eval():
    basic_prompt = f"""
    Navigate to the described target location!
    Action Space: forward, left, right, stop
    - If you choose "forward", proceed for 50 meters.
    - If you choose "left" or "right", make the turn at the next intersection.
    - If you believe you have reached the destination, please select "stop".
    - Please respond ONLY with "forward", "left", "right", or "stop".

    Navigation Instructions:
    """
    
    return basic_prompt

def get_prompt_eval_reason():
    basic_prompt = f"""
    Navigate to the described target location!
    Action Space: forward, left, right, stop
    - If you choose "forward", proceed for 50 meters.
    - If you choose "left" or "right", make the turn at the next intersection.
    - If you believe you have reached the destination, please select "stop".
    - Format your response as follows:\n
      Reason: <your_reason_for_the_action> Action: <your_action_here>

    Navigation Instructions:
    """
    return basic_prompt
