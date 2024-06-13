from pymongo import MongoClient
from mosstool.util.format_converter import coll2pb
from mosstool.type import Map

# 过滤trip文件
from mosstool.type import Persons
from moss import Map
from utils.utils import whether_road_in_region, get_coords
from typing import List, Tuple


import asyncio
import argparse
import signal
import subprocess
import time
from mosstool.type import Map
from mosstool.util.format_converter import pb2dict
from utils.const import *
from utils.utils import generate_persons, multi_aoi_pos2lane_pos, with_preroute, MAP_DICT

MONGODB_URI = ""

# step0: 转换地图格式
def process_map(city):
    city_map = MAP_DICT[city]
    OUTPUT_PATH ="./EXP_ORIG_DATA/{}/{}.map.pb".format(city, city)
    print(city)
    client = MongoClient(f"{MONGODB_URI}")
    coll = client["llmsim"][city_map]
    pb = Map()
    pb = coll2pb(coll, pb)

    with open(OUTPUT_PATH,"wb") as f:
        f.write(pb.SerializeToString())

# step1: 生成车流
async def gen_trips(city):
    CITY_TO_AGENT_NUM = 10_0000
    CITY_TO_HOST = 54344
    routing_path = "../config/routing_linux_amd64"
    cache_dir = "../data/map_cache/"
    city_map = MAP_DICT[city]

    try:
        print(f"Processing city {city}")
        # map需按照实际修改
        with open(f"./EXP_ORIG_DATA/{city}/{city}.map.pb", "rb") as f:
            m = Map()
            m.ParseFromString(f.read())
        m_dict = pb2dict(m)
        # listen需按照实际修改
        route_command = f"{routing_path} -mongo_uri {MONGODB_URI} -map llmsim.{city_map} -cache {cache_dir} -listen localhost:{CITY_TO_HOST}"
        cmd = route_command.split(" ")
        process = subprocess.Popen(args=cmd, cwd="./")
        persons = generate_persons(
            m, city, agent_num=int(1.1 * CITY_TO_AGENT_NUM)
        )
        persons = multi_aoi_pos2lane_pos(persons=persons, map_dict=m_dict)
        lanes = {d["id"]: d for d in m_dict["lanes"]}
        await with_preroute(
            persons=persons,
            lanes=lanes,
            listen=CITY_TO_HOST,
            output_path=f"./trips/{city}_trip.pb",
            max_num=CITY_TO_AGENT_NUM,
        )
        # await task
        time.sleep(0.1)
        print("send signal")
        process.send_signal(sig=signal.SIGTERM)
        process.wait()
        
    except Exception as e:
        print(f"Error processing city {city}: {e}")


# step2: 过滤车流
def filter_trips(city, FLOW_RATIO=5):
    AGENT_PATH = "./trips/{}_trip.pb".format(city)

    START_TIME, END_TIME = 30000-1000, 30000+2000
    NEW_AGENT_PATH = "./trips/{}_trip_filtered_start_{}_end_{}_extend_{}.pb".format(city, START_TIME, END_TIME, FLOW_RATIO)
    M=Map('./EXP_ORIG_DATA/{}/{}.map.pb'.format(city, city))

    # 划分的交通灯控制区域
    coords = get_coords(city)
    target_road_ids = whether_road_in_region(M, coords)
    print(len(target_road_ids))

    with open(AGENT_PATH, "rb") as f:
        pb = Persons()
        pb.ParseFromString(f.read())
    ok_persons = []
    for p in pb.persons:
        flag = False
        for schedule in p.schedules:
            for trip in schedule.trips:  
                if trip.routes:  
                    for route in trip.routes:
                        if route.HasField("driving"):  
                            road_ids = set(route.driving.road_ids) 
                            if road_ids.intersection(target_road_ids) and (START_TIME<schedule.departure_time<END_TIME):
                                flag = True
                                break
                        else:
                            print("No driving information available.")
                    if flag:
                        break
                else:
                    print("No routes available.")
            if flag:
                break
    
        if flag == 1:
            # 扩大车流量
            pid = p.id
            for i in range(FLOW_RATIO):
                p.id = pid + 100000 + i
                ok_persons.append(p)

    new_pb = Persons(persons=ok_persons)
    with open(NEW_AGENT_PATH, "wb") as f:
        f.write(new_pb.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="shanghai")
    parser.add_argument("--mode", type=str, default="filter_trips", choices=["map_transfer", "gen_trips", "filter_trips", "all"])

    args = parser.parse_args()

    if args.mode in ["all", "map_transfer"]:
        process_map(args.city)
    if args.mode in ["all", "gen_trips"]:
        asyncio.run(gen_trips(args.city))
    if args.mode in ["all", "filter_trips"]:
        filter_trips(args.city)
