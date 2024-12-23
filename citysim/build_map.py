import logging
import os
import pyproj
import datetime
import argparse
import logging
import geojson

from pymongo import MongoClient
from mosstool.map.builder import Builder
from mosstool.type import Map
from mosstool.map.osm import Building, PointOfInterest, RoadNet
from mosstool.util.format_converter import dict2pb, pb2coll

from config import PROXIES, MONGODB_URI

def get_net(args):
    ## 提供经纬度范围
    min_lon, min_lat = args.min_lon, args.min_lat
    max_lon, max_lat = args.max_lon, args.max_lat
    proj_str = f"+proj=tmerc +lat_0={(max_lat+min_lat)/2} +lon_0={(max_lon+min_lon)/2}"
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    )
    ## 路网
    rn = RoadNet(
        max_latitude=max_lat,
        min_latitude=min_lat,
        max_longitude=max_lon,
        min_longitude=min_lon,
        proj_str=proj_str,
        proxies=PROXIES,
    )
    rn.create_road_net(output_path=os.path.join(OUTPUT_PATH, f"roadnet_{args.city_name}.geojson"))
    ## AOI
    building = Building(
        max_latitude=max_lat,
        min_latitude=min_lat,
        max_longitude=max_lon,
        min_longitude=min_lon,
        proj_str=proj_str,
        proxies=PROXIES,
    )
    building.create_building(output_path=os.path.join(OUTPUT_PATH, f"aois_{args.city_name}.geojson"))
    ## POI
    pois = PointOfInterest(
        max_latitude=max_lat,
        min_latitude=min_lat,
        max_longitude=max_lon,
        min_longitude=min_lon,
        proxies=PROXIES,
    )
    pois.create_pois(output_path=os.path.join(OUTPUT_PATH, f"pois_{args.city_name}.geojson"))


def get_map(args):
    workers = args.workers
    date = datetime.datetime.now().strftime("%m%d")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
   
    # 加载配置
    logging.info(f"Generating map of {args.city_name}")
    lat = (args.max_lat + args.min_lat) / 2
    lon = (args.max_lon + args.min_lon) / 2
    try:
        with open(f"citydata/map_construction/roadnet_{args.city_name}.geojson", "r") as f:
            net = geojson.load(f)
        with open(f"citydata/map_construction/aois_{args.city_name}.geojson", "r") as f:
            aois = geojson.load(f)
        with open(f"citydata/map_construction/pois_{args.city_name}.geojson", "r") as f:
            pois = geojson.load(f)
        builder = Builder(
            net=net,
            proj_str=f"+proj=tmerc +lat_0={lat} +lon_0={lon}",
            aois=aois,
            pois=pois,
            gen_sidewalk_speed_limit=50 / 3.6,
            road_expand_mode="M",
            workers=workers,
        )
        m = builder.build(args.city_name)
        pb = dict2pb(m, Map())
        client = MongoClient(MONGODB_URI)
        coll = client["llmsim"][f"map_{args.city_name}_2024{date}"]
        pb2coll(pb, coll, drop=True)
    except Exception as e:
        print(f"{args.city_name} failed!")
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Paris")
    parser.add_argument("--min_lon", type=float, default=2.249)
    parser.add_argument("--max_lon", type=float, default=2.4239)
    parser.add_argument("--min_lat", type=float, default=48.8115)
    parser.add_argument("--max_lat", type=float, default=48.9038)
    parser.add_argument("--workers", "-ww", help="workers for multiprocessing", type=int, default=128)
    args = parser.parse_args()
    OUTPUT_PATH = "citydata/map_construction/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    get_net(args)
    get_map(args)