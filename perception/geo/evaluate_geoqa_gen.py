import os
import random
import pandas as pd
import sys
from pycitysim.map import Map
from pycitysim.routing import RoutingClient
from shapely import Polygon, Point

from etc.all_config import all_config
from util import gen_options, load_map, MAP_DICT
import argparse
from geopy.distance import geodesic
import copy
import signal

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_path)
sys.path.append(parent_path)

random.seed(42)

def task_files_adaption(task_file, output_path):
    task_files = copy.deepcopy(task_file)
    path_prefix = output_path
    for k in task_files:
        if path_prefix not in task_files[k]:
            task_files[k]= os.path.join(path_prefix, task_files[k])
    os.makedirs(path_prefix, exist_ok=True)
    return task_files


def get_nearby_aois(input_coor, map, radius, cfg):
    """task2: 推断POI周边的anchor """
    if cfg.use_english:
        category_supported = {
            "E3": "OtherNon-construction", "R": "Residential", "S4": "TrafficStation&Park", "A4": "Sports",
            "B31": "Entertainment", "B1": "CommercialFacilities", "U9": "OtherPublicFacilities", "A3": "Education",
            "G1": "Park&GreenLand", "B": "CommercialService&IndustryFacilities", "B32": "Resort&Fitness",
            "B13": "Restaurant&Bar", "A9": "ReligiousFacilities", "A5": "Hospital"
        }
    else:
        category_supported = {
            "E3": "其他非建设用地", "R": "居住用地", "S4": "交通场站用地", "A4": "体育用地",
            "B31": "娱乐用地", "B1": "商业设施用地", "U9": "其他公用设施用地", "A3": "教育科研用地",
            "G1": "公园绿地", "B": "商业服务业设施用地", "B32": "康体用地",
            "B13": "饭店、餐厅、酒吧等用地", "A9": "宗教设施用地", "A5": "医疗卫生用地"
        }
    lng, lat = input_coor
    # radius = 500
    limit = 10
    nearby_aois = []
    input_coor = map.lnglat2xy(lng, lat)
    for category_prefix in category_supported.keys():
        aoi_list = map.query_aois(input_coor, radius, category_prefix, limit)
        # poi_list = [LANGUAGE.poi_name_choose(poi[0],poi[0]["id"],USE_ENGLISH) for poi in poi_list]
        aoi_list = [aoi[0]["id"] for aoi in aoi_list]
        nearby_aois.extend(aoi_list)
    return nearby_aois


def AOI_POI3_task_gen(cfg, map, aoi_dict, aoi_id, type_aois, landuse_name):
    aoi_address = aoi_dict[aoi_id]['Address']
    aoi_coor = aoi_dict[aoi_id]['coord'][0][0], aoi_dict[aoi_id]['coord'][0][1]
    nearby_aois = get_nearby_aois(aoi_coor, map, 100, cfg)
    nearby_aois = set(nearby_aois)&set(aoi_name_dict_select.keys())
    multi_aoi_types = set()
    nearby_aoi_types = set()
    for aoi in nearby_aois:
        nearby_aoi_types.add(aoi_dict[aoi]['type'])
    for type_,aois in type_aois.items():
        if len(aois) > 3:
            multi_aoi_types.add(type_)
    tar_aoi_types = nearby_aoi_types&multi_aoi_types
    if len(list(tar_aoi_types)) < 1:
        return None
    tar_aoi_type = random.choice(list(tar_aoi_types))
    for aoi in nearby_aois:
        type_ = aoi_dict[aoi]['type']
        if type_ ==  tar_aoi_type:
            tar_aoi = aoi
            break
        else:
            return None
    tar_aoi_name = aoi_dict[tar_aoi]['name']
    res_aois = random.choices(type_aois[tar_aoi_type], k=3)
    res_aois_name = [aoi_dict[res_aoi]['name'] for res_aoi in res_aois]
    res_aois_name.append(tar_aoi_name)
    question3 = "Which AOI in the category {} are located within a 100m radius of {}?".format(landuse_name[tar_aoi_type], aoi_address)
    answer3 = tar_aoi_name
    res_dict3 = gen_options(res_aois_name, question3, answer3)
    return res_dict3

def AOI_POI4_task_gen(cfg, map, aoi_dict,aoi_id, landuse_name):
    aoi_name = aoi_dict[aoi_id]['name']
    aoi_coor = aoi_dict[aoi_id]['coord'][0][0], aoi_dict[aoi_id]['coord'][0][1]
    tar_type = random.choice(list(landuse_types))
    further_aois = get_nearby_aois(aoi_coor, map, 700, cfg)
    further_aois = set(further_aois) & set(aois_name_dict.keys())
    further_aois_type = further_aois&set(type_aois[tar_type])
    if len(further_aois_type) < 4:
        return
    res_aois = random.choices(list(further_aois_type), k=4)
    min_dist = 100
    nearest_aoi = None
    for aoi in res_aois:
        coor = aoi_dict[aoi]['coord'][0][0], aoi_dict[aoi]['coord'][0][1]
        dist = geodesic((coor[1], coor[0]), (aoi_coor[1], aoi_coor[0])).km
        if dist < min_dist:
            min_dist = dist
            nearest_aoi = aoi
    if not nearest_aoi:
        return
    nearest_aoi_name =  aoi_dict[nearest_aoi]['name']
    res_aois_name = [aoi_dict[res_aoi]['name'] for res_aoi in res_aois]
    answer4 = nearest_aoi_name
    question4 = "Which is the nearest {} AOI to {}?".format(landuse_name[tar_type], aoi_name)
    res_dict4 = gen_options(res_aois_name, question4, answer4)
    return res_dict4

def AOI_POI5_task_gen(cfg, map, aoi_dict, aoi_id):
    aoi_name = aoi_dict[aoi_id]['name']
    aoi_area = map.get_aoi(aoi_id)['area']
    question5 = "What is the total area of {}?".format(aoi_name)
    answer5 = int(aoi_area)
    res_areas = [int(aoi_area*0.8), int(aoi_area*2), aoi_area+200, int(aoi_area)]
    res_dict5 = gen_options(res_areas, question5, answer5)
    return res_dict5

def AOI_POI6_task_gen(cfg, map, aoi_dict, aoi_id):
    other_aois = list(set(aoi_dict.keys()).difference({aoi_id}))
    res_aois = random.choices(other_aois, k=3)
    res_aois.append(aoi_id)
    res_aois_name = [aoi_dict[aoi_id]['name'] for aoi_id in res_aois]
    max_area = 0
    biggest_aoi = None
    for aoi in res_aois:
        area = map.get_aoi(aoi)['area']
        if area > max_area:
            max_area = area
            biggest_aoi = aoi
    tar_aoi_name = aoi_dict[biggest_aoi]['name']
    question6 = "Which AOI has the largest area?"
    answer6 = tar_aoi_name
    res_dict6 = gen_options(res_aois_name, question6, answer6)
    return res_dict6


def AOI_POI2road_task_gen(cfg, map, aoi_dict, road_dict, road_with_name, aoi_id):  #选取有名字的AOI,且poi_ids数量不少于2个
    aoi_name = aoi_dict[aoi_id]['name']
    lanes = map.get_aoi(aoi_id)['driving_positions']

    roads2aoi = []
    for lane in lanes:
        lane_id = lane['lane_id']
        road_id = map.get_lane(lane_id)['parent_id']
        if road_id not in road_dict or road_id in roads2aoi:
            continue
        roads2aoi.append(road_id)

    other_roads = list(set(road_with_name.keys()).difference(set(roads2aoi)))

    roads_name = list(set([road_dict[road]['name'] for road in roads2aoi if road in set(roads2aoi)&set(road_with_name.keys())]))
    if len(roads_name) > 2:
        res_roads_name = random.choices(roads_name, k=3)
        other_road = random.choice(other_roads)
        tar_road_name = road_with_name[other_road]['name']
        res_roads_name.append(tar_road_name)
        question = "Which of the following roads is not connected to {}?".format(aoi_name)
    elif 0 < len(roads_name) <= 2:
        tar_road_name = random.choice(roads_name)
        other_roads = random.choices(other_roads, k=3)
        res_roads_name = [road_with_name[other_road]['name'] for other_road in other_roads]
        res_roads_name.append(tar_road_name)
        question = "Which of the following roads is connected to {}?".format(aoi_name)
    else:
        return
    answer = tar_road_name
    res_dict = gen_options(res_roads_name, question, answer)

    max_length = 0
    max_length_road = None
    for road_id in roads2aoi:
        length = road_dict[road_id]['length']
        if length > max_length:
            max_length_road = road_id
    if max_length_road in road_with_name:
        tar_road_name = road_with_name[max_length_road]['name']
    else:
        if cfg.use_english:
            tar_road_name = 'unknown road'
        else:
            tar_road_name = '未知路名'
    res_roads = random.choices(other_roads, k=3)
    res_roads_name = [road_dict[road_id]['name'] for road_id in res_roads]
    question2 = "What's the longest road that meets {}?".format(aoi_name)
    answer2 = tar_road_name
    res_roads_name.append(tar_road_name)
    res_dict2 = gen_options(res_roads_name, question2, answer2)

    tar_road = None
    for road_id in roads2aoi:
        if len(road_dict[road_id]['aoi_ids']) > 2 and road_id in road_with_name:
            tar_road = road_id
            continue
    if not tar_road:
        return
    other_aois = list(set(aoi_dict.keys()).difference(set(road_dict[tar_road]['aoi_ids'])))
    tar_aois = random.choices(list(set(road_dict[tar_road]['aoi_ids'])&set(aoi_dict.keys())), k=2)
    tar_aoi,res_aoi = tar_aois[0], tar_aois[1]
    res_aois = random.choices(other_aois, k=3)
    res_aois.append(tar_aoi)
    res_aois_name = [aoi_dict[aoi_id]['name'] for aoi_id in res_aois]
    tar_aoi_name = aoi_dict[tar_aoi]['name']
    question3 = "Which AOI is connected to {} by {}?".format(aoi_dict[res_aoi]['name'], road_dict[tar_road]['name'])
    answer3 = tar_aoi_name
    res_dict3 = gen_options(res_aois_name, question3, answer3)

    aoi_address = aoi_dict[aoi_id]['Address']
    aoi_coor = aoi_dict[aoi_id]['coord'][0][0], aoi_dict[aoi_id]['coord'][0][1]
    nearby_aois = get_nearby_aois(aoi_coor, map, 100, cfg)
    nearby_aois = set(nearby_aois)&set(aoi_name_dict_select.keys())
    if len(nearby_aois) == 0:
        return
    other_aois = list(set(aoi_name_dict_select.keys()).difference(set(nearby_aois)))
    res_aois = random.choices(other_aois, k=3)
    tar_aoi = random.choice(list(nearby_aois))
    res_aois.append(tar_aoi)
    tar_aoi_name = aoi_dict[tar_aoi]['name']
    res_aois_name = [aoi_dict[aoi_id]['name'] for aoi_id in res_aois]
    question4 = "Which AOI is adjacent to {} in this area?".format(aoi_dict[aoi_id]['name'])
    answer4 = tar_aoi_name
    res_dict4 = gen_options(res_aois_name, question4, answer4)
    return res_dict, res_dict2, res_dict3, res_dict4

##def AOI_POI2road_task_gen(cfg, map, aoi_dict, poi_dict, road_dict, road_with_name, aoi_id):  #选取有名字的AOI,且poi_ids数量不少于2

def generate_AOI_POI3_task(cfg, map, aoi_dict, type_aois, landuse_name):
    # AOI_POI_task_gen(cfg, map, aoi_dict, poi_dict, aoi_id):  # 选择poi_ids > 3,poi_id有至少一个具有名字，且具有名字的的aoi_id
    task_name = "AOI_POI3"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        if not isinstance(name, str):
            continue
        result = AOI_POI3_task_gen(cfg, map, aoi_dict, id, type_aois, landuse_name)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES[task_name])

def generate_AOI_POI4_task(cfg, map, aoi_dict, landuse_name):
    # AOI_POI_task_gen(cfg, map, aoi_dict, poi_dict, aoi_id):  # 选择poi_ids > 3,poi_id有至少一个具有名字，且具有名字的的aoi_id
    task_name = "AOI_POI4"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        if not isinstance(name, str):
            continue
        result = AOI_POI4_task_gen(cfg, map, aoi_dict, id, landuse_name)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES[task_name])

def generate_AOI_POI5_task(cfg, map, aoi_dict):
    # AOI_POI_task_gen(cfg, map, aoi_dict, poi_dict, aoi_id):  # 选择poi_ids > 3,poi_id有至少一个具有名字，且具有名字的的aoi_id
    task_name = "AOI_POI5"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        if not isinstance(name, str):
            continue
        result = AOI_POI5_task_gen(cfg, map, aoi_dict, id)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES[task_name])

def generate_AOI_POI6_task(cfg, map, aoi_dict):
    # AOI_POI_task_gen(cfg, map, aoi_dict, poi_dict, aoi_id):  # 选择poi_ids > 3,poi_id有至少一个具有名字，且具有名字的的aoi_id
    task_name = "AOI_POI6"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        if not isinstance(name, str):
            continue
        result = AOI_POI6_task_gen(cfg, map, aoi_dict, id)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES[task_name])
# TaskA.2(道路起终点判断) & TaskA.3(道路是否相连判断) DONE
def generate_AOI_POI2road_task_gen(cfg, map, aoi_dict, road_dict, road_with_name):
    #cfg, map, aoi_dict, poi_dict, road_dict, road_with_name, aoi_id
    AOI_connect_rd = []
    AOI_longest_rd = []
    AOI2AOI = []
    AOI2rd = []
    for id, aoi in aoi_dict.items():
        name = aoi['name']
        if not isinstance(name, str):
            continue
        result = AOI_POI2road_task_gen(cfg, map, aoi_dict, road_dict, road_with_name, id)
        if not result:
            continue
        AOI_connect_rd.append(result[0])
        AOI_longest_rd.append(result[1])
        AOI2AOI.append(result[2])
        AOI2rd.append(result[3])
    # Can you tell me the roads that {aoi_name} is connected to?
    # What's the longest road that meets {aoi_name}?
    # Which POIs are adjacent to {chosen_poi_name} in this area?
    # Which AOI is connected to {chosen_aoi_name} by {road_name}?
    datas = [AOI_connect_rd, AOI_longest_rd, AOI2AOI, AOI2rd]
    for idx,data in enumerate(datas):
        task_df = pd.DataFrame(data)
        task_df.to_csv(TASK_FILES["AOI_POI_road{}".format(idx+1)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="shanghai")
    parser.add_argument("--evaluate_version", type=str, default="v82")
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    config = all_config[args.city]

    city_map = MAP_DICT[args.city]
    cache_dir = "../../data/map_cache/"
    resource_dir = "./../data/resource/"
    routing_path = "./../config/routing_linux_amd64"
    port = 54319
    map, process, routing_client = load_map(
        city_map=city_map, # config.map_config.mongo_coll 
        cache_dir=cache_dir, 
        routing_path=routing_path, 
        port=port)
    
    if not isinstance(args.output_path, str):
        output_path = "task_Geo_knowledge/{}/{}".format(config.region_exp, args.evaluate_version)
    else:
        output_path = args.output_path

    random.seed(42)

    TASK_FILES_ = {
        "AOI_POI_road1": "AOI_POI_road1.csv",
        "AOI_POI_road2": "AOI_POI_road2.csv",
        "AOI_POI_road3": "AOI_POI_road3.csv",
        "AOI_POI_road4": "AOI_POI_road4.csv",
        "AOI_POI": "AOI_POI.csv",
        "AOI_POI2": "AOI_POI2.csv",
        "AOI_POI3": "AOI_POI3.csv",
        "AOI_POI4": "AOI_POI4.csv",
        "AOI_POI5": "AOI_POI5.csv",
        "AOI_POI6": "AOI_POI6.csv",
    }

    TASK_FILES = task_files_adaption(TASK_FILES_, output_path)
    aoi_message = pd.read_csv(os.path.join(resource_dir, "{}_aois.csv".format(config.region_exp)))
    road_message = pd.read_csv(os.path.join(resource_dir, "{}_roads.csv".format(config.region_exp)))

    all_roads = map.roads
    all_lanes = map.lanes
    all_aois = map.aois
    all_juncs = map.juncs

    aois_name_dict = aoi_message.set_index("aoi_id")["aoi_name"].to_dict()
    # aoi_id,type,land_use,coords,aoi_name,English_Address
    aoi_dict = {}  # 过滤掉名字里含有“nearby”的AOI
    for row in aoi_message.itertuples():
        aoi_id = row.aoi_id
        if config.use_english:
            aoi_name = row.aoi_name
            address = row.English_Address
            extra = "nearby"
        else:
            aoi_name = row.aoi_name
            address = row.Address
            extra = "周边|附近"
        if not isinstance(aoi_name, str) or extra in aoi_name or not isinstance(address, str):
            continue
        aoi_dict[aoi_id] = {}
        aoi_dict[aoi_id]['type'] = map.get_aoi(aoi_id)['urban_land_use']
        aoi_dict[aoi_id]['coord'] = eval(row.coords)
        aoi_dict[aoi_id]['name'] = aoi_name
        aoi_dict[aoi_id]['Address'] = address
        # aoi_dict[aoi_id]['area'] = poi_ids
    print("aois:{}".format(len(aoi_dict)))

    if config.use_english:
        landuse_name = {
            "E3": "OtherNon-construction", "R": "Residential", "S4": "TrafficStation&Park", "A4": "Sports",
            "B31": "Entertainment", "B1": "CommercialFacilities", "U9": "OtherPublicFacilities", "A3": "Education",
            "G1": "Park&GreenLand", "B": "CommercialService&IndustryFacilities", "B32": "Resort&Fitness",
            "B13": "Restaurant&Bar", "A9": "ReligiousFacilities", "A5": "Hospital", "U": "PublicFacilities"
        }
    else:
        landuse_name = {
            "E3": "其他非建设用地", "R": "居住用地", "S4": "交通场站用地", "A4": "体育用地",
            "B31": "娱乐用地", "B1": "商业设施用地", "U9": "其他公用设施用地", "A3": "教育科研用地",
            "G1": "公园绿地", "B": "商业服务业设施用地", "B32": "康体用地",
            "B13": "饭店、餐厅、酒吧等用地", "A9": "宗教设施用地", "A5": "医疗卫生用地", "U":"公用设施用地"
        }

    aoi_name_dict_select = {}
    for id, aoi in aoi_dict.items():
        aoi_name_dict_select[id] = aoi['name']

    type_aois = {}  # {用地类型：[对应的所有aoi_id]}
    for id, aoi in aoi_dict.items():
        type_ = aoi['type']
        if type_ not in landuse_name:
            continue
        if type_ not in type_aois:
            type_aois[type_] = []
        type_aois[type_].append(id)

    landuse_types = set()
    for id, aoi in aoi_dict.items():
        landuse = aoi['type']
        if landuse not in landuse_name:
            continue
        if landuse not in landuse_types:
            landuse_types.add(landuse)

    road_dict = {}
    road_with_name = {}
    for row in road_message.itertuples():
        all_aois = []
        road_id = row.road_id
        length = all_roads[road_id]['length']
        road_name = row.road_name
        lane_ids = all_roads[road_id]['lane_ids']
        for lane_id in lane_ids:
            aoi_ids = all_lanes[lane_id]['aoi_ids']
            all_aois += aoi_ids
        road_dict[road_id] = {}
        road_dict[road_id]['name'] = road_name
        road_dict[road_id]['length'] = length
        road_dict[road_id]['lane_ids'] = lane_ids
        road_dict[road_id]['aoi_ids'] = list(set(all_aois))
        if road_name == "未知路名" or road_name == "unknown road" or not isinstance(road_name, str):
            continue
        road_with_name[road_id] = {'name':road_name,'length':length,'lane_ids':lane_ids,'aois': list(set(all_aois)) }

    generate_AOI_POI3_task(config, map, aoi_dict, type_aois, landuse_name)
    generate_AOI_POI4_task(config, map, aoi_dict, landuse_name)
    generate_AOI_POI5_task(config, map, aoi_dict)
    generate_AOI_POI6_task(config, map, aoi_dict)
    generate_AOI_POI2road_task_gen(config, map, aoi_dict, road_dict, road_with_name)

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
