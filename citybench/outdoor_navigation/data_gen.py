import os
import argparse
import numpy as np
import pandas as pd
import random
import jsonlines
import asyncio
import signal
from multiprocessing import Pool 
from tqdm import tqdm

from pycitydata.map import Map
from citysim.routing import RoutingClient
from citysim.player import Player

from collections import Counter
from global_utils import load_map
from config import RESOURCE_PATH, RESULTS_PATH, MAP_CACHE_PATH, ROUTING_PATH, MAP_DICT, IMAGE_FOLDER, STEP, REGION_CODE

def find_image_file(city, image_folder, sid_84_long, sid_84_lat, sid):
    # stitching the picture file name
    dataset_name = REGION_CODE[city]
    file_name = f"{dataset_name}_{sid_84_long}_{sid_84_lat}_{sid}.jpg"
    file_path = os.path.join(image_folder, file_name)
    
    if os.path.exists(file_path):
        return file_name
    else:
        return None
    
# 将图片与道路关联起来
def process_road_id_google(args):
    road_id, city, city_map, meta_info_df = args
    lng = []
    lat = []
    distance_list = []  
    matched_data = []
    
    road_info = city_map.get_road(road_id)
    lane_id = road_info['lane_ids'][0]
    lane_info = city_map.get_lane(lane_id)
    lane_len = lane_info['length']
    start_lng, start_lat = lane_info['shapely_lnglat'].coords[0][0], lane_info['shapely_lnglat'].coords[0][1]
    end_lng, end_lat = lane_info['shapely_lnglat'].coords[-1][0], lane_info['shapely_lnglat'].coords[-1][1]
    # start point
    lng.append(start_lng)
    lat.append(start_lat)
    distance_list.append(0)  
    # end point
    lng.append(end_lng)
    lat.append(end_lat)
    distance_list.append(int(lane_len))  # 终点的距离为 lane_len
    # print(f"s: {start_lng}, {start_lat}, e: {end_lng}, {end_lat}")
    # Intermediate points
    for ds in range(STEP, int(lane_len), STEP):
        xy = lane_info["shapely_xy"].interpolate(ds)
        x, y = xy.coords[0][0], xy.coords[0][1]
        lng_in, lat_in = city_map.xy2lnglat(x, y)
        lng.append(lng_in)
        lat.append(lat_in)
        distance_list.append(ds)  # 中间点的距离为插值距离 ds

    # 检查图片是否存在
    for lng_val, lat_val, dist_val in zip(lng, lat, distance_list):
        rounded_lat_val = np.round(float(lat_val), 4)
        rounded_lng_val = np.round(float(lng_val), 4)
        
        # 遍历 meta_info_df 中的每一行，直接从其中提取数据
        for idx, row in meta_info_df.iterrows():
            query_lati = float(row['query_lati'])
            query_longti = float(row['query_longti'])
            file_name = row['file_name']  
            
            # 比较经纬度是否匹配
            if abs(rounded_lng_val - query_longti) < 1e-4 and abs(rounded_lat_val - query_lati) < 1e-4:
                # print("match")
                matched_data.append({
                    "road_id": road_id,
                    "lane_id": lane_id,
                    "file_name": file_name,
                    "distance": dist_val
                })
                break  # 匹配到一张图片即可，继续下一个坐标
    return matched_data


def process_road_id_baidu(args):
    road_id, city, city_map, meta_info_df = args
    lng = []
    lat = []
    distance_list = []  
    matched_data = []
    mapping_dict = {
    (row['longitude_origin'], row['latitude_origin']): (row['sid_84_long'], row['sid_84_lat'], row['sid'])
    for idx, row in meta_info_df.iterrows()
    } 
    road_info = city_map.get_road(road_id)
    lane_id = road_info['lane_ids'][0]
    lane_info = city_map.get_lane(lane_id)
    lane_len = lane_info['length']
    start_lng, start_lat = lane_info['shapely_lnglat'].coords[0][0], lane_info['shapely_lnglat'].coords[0][1]
    end_lng, end_lat = lane_info['shapely_lnglat'].coords[-1][0], lane_info['shapely_lnglat'].coords[-1][1]
    # start point
    lng.append(start_lng)
    lat.append(start_lat)
    distance_list.append(0)  
    # end point
    lng.append(end_lng)
    lat.append(end_lat)
    distance_list.append(int(lane_len))  # 终点的距离为 lane_len
    # print(f"s: {start_lng}, {start_lat}, e: {end_lng}, {end_lat}")
    # Intermediate point
    for ds in range(STEP, int(lane_len), STEP):
        xy = lane_info["shapely_xy"].interpolate(ds)
        x, y = xy.coords[0][0], xy.coords[0][1]
        lng_in, lat_in = city_map.xy2lnglat(x, y)
        lng.append(lng_in)
        lat.append(lat_in)
        distance_list.append(ds)  # 中间点的距离为插值距离 ds

    # 检查图片是否存在
    for lng_val, lat_val, dist_val in zip(lng, lat, distance_list):
        rounded_lat_val = np.round(float(lat_val), 4)
        rounded_lng_val = np.round(float(lng_val), 4)
        
        if (rounded_lng_val, rounded_lat_val) in mapping_dict:
            sid_84_long, sid_84_lat, sid = mapping_dict[(rounded_lng_val, rounded_lat_val)]
            file_name = find_image_file(city, IMAGE_FOLDER, sid_84_long, sid_84_lat, sid)
            if file_name:
                matched_data.append({
                    "road_id": road_id,
                    "lane_id": lane_id,
                    "file_name": file_name,
                    "distance": dist_val
                })
                print(f"match")
        else:
            print(f"no match for {rounded_lng_val}, {rounded_lat_val}")
            continue

    # print(f"road {road_id} done")
    return matched_data


# 将图片与道路关联起来（并行版本）
def image_connect_road(city, city_map, match_path, num_workers=8):
    # meta_file存储文件名、经纬度等信息
    meta_file = os.path.join(IMAGE_FOLDER, f"{city}_StreetView_Images/combined_stitch_meta_info.csv")
    road_df = pd.read_csv(os.path.join(RESOURCE_PATH, '{}_roads.csv'.format(city)))
    meta_info_df = pd.read_csv(meta_file)
    
    args_list = [(road_id, city, city_map, meta_info_df) for road_id in road_df['road_id']]
    # 并行处理
    matched_data = []
    if city == "Beijing" or city == "Shanghai":
         with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap(process_road_id_baidu, args_list), total=len(args_list)):
                if result:
                    matched_data.extend(result)
    else:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap(process_road_id_google, args_list), total=len(args_list)):
                if result:
                    matched_data.extend(result)
    
    if matched_data:
        matched_df = pd.DataFrame(matched_data)
        matched_df.to_csv(match_path, index=False)
    else:
        print("No matched data found")


async def generate_tasks_nav(city, city_map, routing_client, aoi_file, task_file, match_file):
    search_type = "poi"
    print(f"Generating navigation tasks for {city}")
    match_data_df = pd.read_csv(match_file) 
    road_id_to_count = match_data_df['road_id'].value_counts().to_dict()
    aois_data = pd.read_csv(aoi_file)

    aois_data_info = []
    for aoi_id in aois_data.aoi_id.to_list():
        info = city_map.get_aoi(id=aoi_id)
        aois_data_info.append(info)

    random.shuffle(aois_data_info)
    navigation_task = []
    for aoi in aois_data_info:
        start_aoi_id = aoi["id"]
        # 过滤掉与 start_aoi 相同的aoi
        filtered_candidates = [aoi_id for aoi_id in aois_data.aoi_id.to_list() if aoi_id != start_aoi_id]
        # 保证有足够的候选名称供随机抽样
        for dest_aoi_id in random.sample(filtered_candidates, k=5):
            # 从出发AOI中随机选择POI, 并记录其位置
            player = Player(city_map=city_map, city_routing_client=routing_client, init_aoi_id=start_aoi_id, search_type=search_type)
            try:
                route = await player.get_driving_route(dest_aoi_id)
                # print("route ok")
            except Exception as e:
                print(f"Error: {e}")
                continue
            if route is None:
                continue
            
            # 初始化计数器，记录总共的图片匹配数
            total_match_count = 0
            road_collection = route["road_ids"]  # 路径上的道路 ID 列表

            # 如果路径经过的道路超过4条，则跳过
            if len(road_collection) > 5:
                continue

            # 遍历每条道路，累积匹配的文件数
            for road_id in road_collection:
                total_match_count += road_id_to_count.get(road_id, 0)  # 累积匹配到的图片数

            # 如果累积的匹配文件数大于道路数，则将该路径保存到 navigation_task
            if total_match_count > len(road_collection):
                navigation_task.append({
                    "start_aoi_id": start_aoi_id,
                    "dest_aoi_id": dest_aoi_id,
                    "road_ids": road_collection
                })

    with jsonlines.open(task_file, mode='w') as writer:
        for task in navigation_task:
            writer.write(task)
    print(f"Navigation tasks generated for {city} finished")


async def main(args):
    city_map = MAP_DICT[args.city_name]
    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=args.port)
    aoi_file = os.path.join(RESOURCE_PATH, "{}_aois_visual.csv".format(args.city_name))
    match_file = "citydata/outdoor_navigation_tasks/{}_matched_images.csv".format(args.city_name)
    navigation_task_file = "citydata/outdoor_navigation_tasks/{}_navigation_tasks.jsonl".format(args.city_name)
    os.makedirs(os.path.dirname(navigation_task_file), exist_ok=True)

    # 将图片与道路关联起来
    image_connect_road(args.city_name, m, match_file, num_workers=20)
    # 生成导航任务
    await generate_tasks_nav(
        city=args.city_name,
        city_map=m,
        routing_client=routing_client,
        aoi_file=aoi_file,
        task_file=navigation_task_file,
        match_file=match_file
    )

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="NewYork")
    parser.add_argument("--port", type=int, default=52562)
    args = parser.parse_args()
    asyncio.run(main(args))
