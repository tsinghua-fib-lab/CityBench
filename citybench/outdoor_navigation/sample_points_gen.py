import os
import pandas as pd
import time
import argparse
import signal
import ast

from config import ROUTING_PATH, MAP_DATA_PATH, MAP_CACHE_PATH, RESOURCE_PATH, RESULTS_PATH, MAP_DICT, MONGODB_URI, SAMPLE_POINT_PATH, REGION_CODE
from .utils import load_map


def sample_points(city, city_map):
    road_df = pd.read_csv(os.path.join(RESOURCE_PATH, '{}_roads.csv'.format(city)))
    # 初始化存储经纬度的列表
    coord_x = []
    coord_y = []

    # 对每条lane，以50m为步长，生成样本点
    step = 50
    for road_id in road_df['road_id']:
        road_info = city_map.get_road(road_id)
        lane_id = road_info['lane_ids'][0]
        lane_info = city_map.get_lane(lane_id)
        lane_len = lane_info['length']
        start_lng, start_lat = lane_info['shapely_lnglat'].coords[0][0], lane_info['shapely_lnglat'].coords[0][1]
        end_lng, end_lat = lane_info['shapely_lnglat'].coords[-1][0], lane_info['shapely_lnglat'].coords[-1][1]
        coord_x.append(start_lng)
        coord_y.append(start_lat)
        coord_x.append(end_lng)
        coord_y.append(end_lat)
        for ds in range(step, int(lane_len), step):
            xy = lane_info["shapely_xy"].interpolate(ds)
            x, y = xy.coords[0][0], xy.coords[0][1]
            lng, lat = city_map.xy2lnglat(x, y)
            coord_x.append(lng)
            coord_y.append(lat)

    results_df = pd.DataFrame({
        'coord_x': [coord_x],  
        'coord_y': [coord_y]   
    })
    y_x_value = REGION_CODE[city]

    if args.mode == "Google":
        # 设定要分割成多少行
        num_rows = args.group_num

        expanded_rows = []
        for index, row in results_df.iterrows():
            coord_x_list = ast.literal_eval(row['coord_x'])
            coord_y_list = ast.literal_eval(row['coord_y'])
        
            n = len(coord_x_list) // num_rows
            extra = len(coord_x_list) % num_rows

            for i in range(num_rows):
                start_index = i * n + min(i, extra)
                end_index = start_index + n + (1 if i < extra else 0)
                new_coord_x = coord_x_list[start_index:end_index]
                new_coord_y = coord_y_list[start_index:end_index]
                expanded_rows.append({
                    'y_x': y_x_value,
                    'coord_x': new_coord_x,
                    'coord_y': new_coord_y
                })

                y_x_value += 1  

        expanded_df = pd.DataFrame(expanded_rows)

        expanded_df.to_csv(os.path.join(SAMPLE_POINT_PATH, f'{city}_sampled_points_expand.csv'), index=False)
    else:
        results_df['code'] = y_x_value
        results_df.to_csv(os.path.join(SAMPLE_POINT_PATH, f'{city}_sampled_points.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Shanghai")
    parser.add_argument("--mode", type=str, default="Baidu", choices=["Baidu", "Google"])
    parser.add_argument("--port", type=int, default=52107)
    parser.add_argument("--group_num", type=int, default=30, help="The number of groups to split the sampled points")
    args = parser.parse_args()

    city_map = MAP_DICT[args.city_name]
    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=args.port)

    # 等待地图加载完成
    time.sleep(10)
    sample_points(args.city_name, m, args.group_num)
    
    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
