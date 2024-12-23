import argparse
import asyncio
import os
import random
import shapely
import signal
from itertools import chain
import numpy as np
import pandas as pd
from shapely import Polygon, Point
from geopy.distance import geodesic

from pycityproto.city.geo.v2.geo_pb2 import AoiPosition, Position, XYPosition, LongLatPosition

from .utils import task_files_adaption, gen_options, save_data, get_landuse_dict, get_category_supported
from citysim.player import Player 
from global_utils import load_map
from config import MONGODB_URI, MAP_DICT, RESOURCE_PATH, MAP_CACHE_PATH, ROUTING_PATH, CITY_BOUNDARY, GEOQA_TASK_MAPPING_v2

def generate_evaluation_task_road(map, poi_dict, all_roads, all_lanes, all_aois, TASK_FILES):
    pois_ids_set = {poi_id for poi_id in poi_dict if map.get_poi(poi_id)['name']}

    cared_roads = {}
    cared_roads_name = {}

    for road_id, road in all_roads.items():
        lane_ids = road["lane_ids"]
        road_name = road["name"]
        if road_name == '':
            continue

        road_length = 0
        road_aoi_ids = []
        near_by_lanes = []
        one_lane_in_road = []
        road_level_count = []
        for i, lane_id in enumerate(lane_ids):
            if isinstance(lane_id, list):
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            length = lane["length"]
            aoi_ids = lane["aoi_ids"]
            road_aoi_ids.extend(aoi_ids)
            
            # using the lane information under the road to estimate the road length
            if len(one_lane_in_road) == 0:
                one_lane_in_road.append(lane_id)
                road_length += length

                left_lanes = lane["left_lane_ids"]
                right_lanes = lane["right_lane_ids"]
                near_by_lanes.extend(left_lanes)
                near_by_lanes.extend(right_lanes)
                road_level_count.append(len(left_lanes) + len(right_lanes))
            else:
                if lane_id not in near_by_lanes:
                    one_lane_in_road.append(lane_id)
                    road_length += length

                    left_lanes = lane["left_lane_ids"]
                    right_lanes = lane["right_lane_ids"]
                    near_by_lanes.extend(left_lanes)
                    near_by_lanes.extend(right_lanes)
                    road_level_count.append(len(left_lanes) + len(right_lanes))


        cared_roads[road_id] = {"lane_ids": lane_ids, "name": road_name, "length": road_length, "aoi_ids": road_aoi_ids}
        if road_name in cared_roads_name:
            cared_roads_name[road_name]["lane_ids"].append(lane_ids)
            cared_roads_name[road_name]["road_ids"].append(road_id)
            cared_roads_name[road_name]["length"].append(road_length)
            cared_roads_name[road_name]["aoi_ids"].append(road_aoi_ids)
        else:
            cared_roads_name[road_name] = {"road_ids": [road_id], "lane_ids": [lane_ids], "length": [road_length],
                                            "aoi_ids": [road_aoi_ids],
                                            "level_count": road_level_count}

    for road_name in cared_roads_name:
        road_cut = max(cared_roads_name[road_name]["level_count"][0] + 1, 1)
        cared_roads_name[road_name]["road_length_estimate"] = float(
            np.sum(cared_roads_name[road_name]["length"]) / road_cut)

    res_length = []
    for road_name in cared_roads_name:
        road_length = int(cared_roads_name[road_name]["road_length_estimate"] / 100) * 100

        res = [max(road_length * 0.1, 100), max(road_length * 0.3, 100), road_length, road_length * 2, road_length * 3]
        question = "How long is {} road?".format(road_name)
        answer = road_length
        res_dict = gen_options(res, question, answer)
        res_length.append(res_dict)

    task_df = pd.DataFrame(data=res_length)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["path"]["road_length"])
    print("path road_length task OK!")

    for road_name in cared_roads_name:
        road_aoi_ids = cared_roads_name[road_name]["aoi_ids"]
        road_poi_ids = []
        for aoi_ids in road_aoi_ids:
            for aoi_id in aoi_ids:
                road_poi_ids.extend(all_aois[aoi_id]["poi_ids"])
        road_poi_ids = list(set(road_poi_ids))

        final_poi_ids = []
        for rpi in road_poi_ids:
            if rpi in pois_ids_set:
                final_poi_ids.append(rpi)

        filter_final_poi_ids = []
        for p in final_poi_ids:
            poi_name = map.get_poi(p)['name']
            if poi_name != '':
                filter_final_poi_ids.append(p)

        cared_roads_name[road_name]["arrived_pois"] = filter_final_poi_ids
    res_arrived_pois = []
    for road_name in cared_roads_name:
        res_dict = {}
        arrived_pois = cared_roads_name[road_name]["arrived_pois"]
        if len(arrived_pois) > 15:
            question_type = "FindCannot"
        else:
            question_type = "FindCan"

        if question_type == "FindCannot":
            res_temp = random.sample(arrived_pois, 15)
            res_temp_name = [map.get_poi(p)['name'] for p in res_temp]
            not_temp = random.sample(list(pois_ids_set.difference(set(arrived_pois))), 5)
            not_temp_name = [map.get_poi(p)['name'] for p in not_temp]
            res = [res_temp_name[:5], res_temp_name[5:10], res_temp_name[10:], not_temp_name]
            question = "Which POIs cannot be directly accessed via {}?".format(road_name)
            answer = not_temp_name
            res_dict = gen_options(res, question, answer)
        else:
            try:
                res_temp = random.sample(arrived_pois, 5)
            except ValueError as e:
                continue
            res_temp_name = [map.get_poi(p)['name'] for p in res_temp]
            not_temp = random.sample(list(pois_ids_set.difference(set(arrived_pois))), 15)
            not_temp_name = [map.get_poi(p)['name'] for p in not_temp]
            res = [not_temp_name[:5], not_temp_name[5:10], not_temp_name[10:], res_temp_name]
            answer = res_temp_name
            question = "Which POIs can be directly accessed via {}?".format(road_name)
            res_dict = gen_options(res, question, answer)
        res_arrived_pois.append(res_dict)

    task_df = pd.DataFrame(data=res_arrived_pois)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["path"]["road_arrived_pois"])
    print("path road_arrived_pois task OK!")


def generate_evaluation_task_road_junc(all_juncs, all_roads, all_lanes, REGION_EXP_POLYGON, TASK_FILES):

    cared_juncs = {}
    cared_roads_name = []

    for road_id, road in all_roads.items():
        lane_ids = road["lane_ids"]
        road_name = road['name']
        if road_name == '':
            continue

        road_in_region = False
        for i, lane_id in enumerate(lane_ids):
            if isinstance(lane_id, list):
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            length = lane["length"]
            last_point = Point(lane["shapely_lnglat"].coords[-1])

            if REGION_EXP_POLYGON.contains(last_point):
                road_in_region = True
                break

        if road_in_region:
            if road_name not in cared_roads_name:
                cared_roads_name.append(road_name)

    # get basic information of junctions and roads
    for junc_id in all_juncs:
        road_in_region = False
        lane_ids = all_juncs[junc_id]["lane_ids"]

        pre_road_names = []
        suc_road_names = []
        pre_road_ids = []
        suc_road_ids = []
        for lane_id in lane_ids:
            lane = all_lanes[lane_id]
            pre_lane_id = lane["predecessors"][0]["id"]
            suc_lane_id = lane["successors"][0]["id"]
            pre_lane = all_lanes[pre_lane_id]
            suc_lane = all_lanes[suc_lane_id]

            last_point = Point(lane["shapely_lnglat"].coords[-1])

            pre_road_id = pre_lane["parent_id"]
            if pre_road_id in all_roads:
                pre_road_name = all_roads[pre_road_id]["name"]
                if pre_road_name == "":
                    continue
                pre_road_names.append(pre_road_name)
                pre_road_ids.append(pre_road_id)
            else:
                continue

            suc_road_id = suc_lane["parent_id"]
            if suc_road_id in all_roads:
                suc_road_name = all_roads[suc_road_id]['name']
                if suc_road_name == "":
                    continue
                suc_road_names.append(suc_road_name)
                suc_road_ids.append(suc_road_id)
            else:
                continue

            if REGION_EXP_POLYGON.contains(last_point):
                road_in_region = True

        if road_in_region:
            pre_road_name = set(pre_road_names)
            suc_road_name = set(suc_road_names)
            pre_road_id = set(pre_road_ids)
            suc_road_id = set(suc_road_ids)
            cared_juncs[junc_id] = {"pre_road_id": pre_road_id, "suc_road_id": suc_road_id,
                                    "pre_road_name": pre_road_name, "suc_road_name": suc_road_name, "gps": last_point}

    for junc in cared_juncs:
        road_list = list(cared_juncs[junc]["pre_road_name"])
        if len(road_list) <= 1:
            junc_name = ""
        elif len(road_list) == 2:
            junc_name = "the junction of {} and {}".format(road_list[0], road_list[1])
        elif len(road_list) == 3:
            junc_name = "the junction of {}, {} and {}".format(road_list[0], road_list[1], road_list[2])
        cared_juncs[junc]["name"] = junc_name

    road_junc_gen(cared_juncs, cared_roads_name, TASK_FILES)
    road_linkage_gen(cared_juncs, TASK_FILES)


def road_junc_gen(cared_juncs, cared_roads_name, TASK_FILES):
    road_juncs = {}
    for junc in cared_juncs:
        road_list = cared_juncs[junc]["pre_road_name"]
        junc_name = cared_juncs[junc]["name"]
        junc_coord = cared_juncs[junc]["gps"]
        if len(road_list) <= 1:
            continue

        for road in list(road_list):
            if road not in road_juncs:
                road_juncs[road] = {"detail": [(junc_name, junc_coord)]}
            else:
                road_juncs[road]["detail"].append((junc_name, junc_coord))

    for rj in road_juncs:
        if len(road_juncs[rj]["detail"]) >= 2:
            lng_list = []
            lat_list = []
            for item in road_juncs[rj]["detail"]:
                lng_list.append(item[1].x)
                lat_list.append(item[1].y)
            lng_max = np.ptp(lng_list)
            lat_max = np.ptp(lat_list)
            start_junc, end_junc = [p for p in road_juncs[rj]["detail"]][:2]

            if lng_max >= lat_max:
                # the difference of longitude is larger
                lng_list_index = list(np.argsort(lng_list))
                for i, idx in enumerate(lng_list_index):
                    if i == 0:
                        start_junc = road_juncs[rj]["detail"][idx]
                    elif idx == (len(lng_list) - 1):
                        end_junc = road_juncs[rj]["detail"][idx]
            else:
                # the difference of latitude is larger
                lat_list_index = list(np.argsort(lat_list))
                for i, idx in enumerate(lat_list_index):
                    if i == 0:
                        start_junc = road_juncs[rj]["detail"][idx]
                    elif i == (len(lat_list) - 1):
                        end_junc = road_juncs[rj]["detail"][idx]
            # print(rj, "start:", start_junc, "end:", end_junc)
            road_juncs[rj]["start_junc"] = start_junc
            road_juncs[rj]["end_junc"] = end_junc

    res_road_endpoint = []
    for rj in road_juncs:
        # only one junction
        if len(road_juncs[rj]["detail"]) < 2:
            continue

        res_label = [road_juncs[rj]["start_junc"][0], road_juncs[rj]["end_junc"][0]]

        res = [res_label]
        try:
            # negative sample
            current_road_name = rj
            start_road_names = random.sample(list(set(cared_roads_name).difference({rj})), 3)
            end_road_names = random.sample(list(set(cared_roads_name).difference(set([rj] + start_road_names))), 3)
            for i, (sr, er) in enumerate(zip(start_road_names, end_road_names)):
                start_juncs = ["the junction of {} and {}".format(current_road_name, sr),
                            "the junction of {} and {}".format(sr, current_road_name)]
                end_juncs = ["the junction of {} and {}".format(current_road_name, er),
                            "the junction of {} and {}".format(er, current_road_name)]
                for junc in start_juncs + end_juncs:
                    assert junc not in res_label, "make sure negative sample is not in positive sample"
                res.append([random.choice(start_juncs), random.choice(end_juncs)])
        except:
            # print("randomly sample error")
            # print("current_label:{}", res_label)
            # print("negative_sample:{}", start_juncs + end_juncs)
            continue
        question = "Which of the following is the starting intersection and ending intersection of {}?".format(rj)
        answer = res_label
        res_dict = gen_options(res, question, answer)

        res_road_endpoint.append(res_dict)

    task_df = pd.DataFrame(data=res_road_endpoint)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["path"]["road_od"])
    print("path road_od task OK!")


def road_linkage_gen(cared_juncs, TASK_FILES):
    road_linkage = {}
    for junc in cared_juncs:
        road_list = cared_juncs[junc]["pre_road_name"]
        if len(road_list) <= 1:
            continue

        for road in list(road_list):
            if road not in road_linkage:
                road_linkage[road] = set(road_list)
            else:
                road_linkage[road] = road_linkage[road].union(set(road_list))
    for road in road_linkage:
        road_linkage[road].remove(road)

    all_roads = set(road_linkage.keys())
    road_nearby = []
    for road_name in road_linkage:
        link_roads = road_linkage[road_name]
        if len(link_roads) > 2:
            res = random.sample(list(link_roads), 3)
            select = random.sample(list(all_roads.difference(link_roads)), 1)
            res.extend(select)
            random.shuffle(res)
            res_dict = dict(zip(["A", "B", "C", "D"], res))
            for k in res_dict:
                if res_dict[k] == select[0]:
                    label = k
            res_dict["question"] = "Which road cannot directly reach {}?".format(road_name)
            res_dict["answer"] = label
        else:
            select = random.sample(list(link_roads), 1)
            res = random.sample(list(all_roads.difference(link_roads)), 3)
            res.extend(select)
            random.shuffle(res)
            question = "Which road directly reach {}?".format(road_name)
            answer = select[0]
            res_dict = gen_options(res, question, answer)
        road_nearby.append(res_dict)

    task_df = pd.DataFrame(data=road_nearby)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["path"]["road_link"])
    print("path road_link task OK!")

def generate_evalation_task_node(map, poi_message, category_supported, TASK_FILES):
    filtered_pois_data = poi_message.dropna(subset=["name"]).query('name != ""')

    if filtered_pois_data.shape[0] < 1000:
        node_num = filtered_pois_data.shape[0]
    else:
        node_num = 1000

    pois_data = filtered_pois_data.sample(node_num, random_state=42)
    pois2name = pois_data.set_index("poi_id")["name"].to_dict()
   
    seen_pois = []
    unseen_pois = []
    for p in pois2name:
        coords = map.get_poi(p)["position"]
        (x, y) = (coords["x"], coords["y"])
        p_name = pois2name[p]
        try:
            res_dict = landmark_gen((x, y), p_name, category_supported, map)
            if res_dict:
                unseen_pois.append(res_dict)
        except IndexError as e:
            pass

    save_data(unseen_pois, TASK_FILES["landmark"]["landmark_env"])
    print("landmark landmark_env task OK!")

def gps_gen(input_coor, poi_name):
    resolution = 0.01
    round_demical = 4

    lng, lat = input_coor
    lng, lat = round(lng, round_demical), round(lat, round_demical)

    res = [[lng, lat], [lng - resolution, lat - resolution], [lng - resolution, lat + resolution],
           [lng + resolution, lat - resolution], [lng + resolution, lat + resolution]]
    for i, coor in enumerate(res):
        coor[0] = round(coor[0], round_demical)
        coor[1] = round(coor[1], round_demical)
        res[i] = coor
    coor_str = ",".join([str(lng), str(lat)])
    res_str = [",".join([str(xx) for xx in x]) for x in res]
    answer = coor_str
    question = "What is the longitude and latigude coordinates of {}.".format(poi_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def landmark_gen(input_coor, poi_name, category_supported, map):
    x, y = input_coor
    radius = 100
    limit = 10
    resolution = 500

    nearby_pois = []
    for category_prefix in category_supported.keys():
        poi_list = map.query_pois(input_coor, radius, category_prefix, limit)
        poi_list = [poi[0]['name'] for poi in poi_list if poi[0]['name']]
        nearby_pois.extend(poi_list[:min(limit, len(poi_list))])
    
    if len(nearby_pois) < 5:
        return None

    candidate_poi = [poi_name]
    for center in [(x - resolution, y - resolution), (x - resolution, y + resolution), (x + resolution, y - resolution),
                   (x + resolution, y + resolution)]:
        poi_list = map.query_pois(center, radius, "", 100)
        poi_list = [poi[0]['name'] for poi in poi_list if poi[0]['name']]
        candidate_poi.append(random.choice(poi_list))
    res_str = [str(x) for x in candidate_poi]
    question = "Which point of interest (POI) is most likely to appear in the described environment among the following multiple POIs? Environment:{}".format(
        ",".join(nearby_pois))
    answer = str(poi_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def get_nearby_pois(input_coor, map, category_supported):
    lng, lat = input_coor
    radius = 500
    limit = 10
    resolution = 500
    nearby_pois = []
    input_coor = map.lnglat2xy(lng, lat)
    for category_prefix in category_supported.keys():
        poi_list = map.query_pois(input_coor, radius, category_prefix, limit)
        poi_list = [poi[0]["id"] for poi in poi_list]
        nearby_pois.extend(poi_list)
    return nearby_pois


def poi2cor_gen(input_coor, poi_id, poi_dict):
    poi_name = poi_dict[int(poi_id)]['name']
    resolution = 0.01
    round_demical = 4
    lng, lat = input_coor
    lng, lat = round(lng, round_demical), round(lat, round_demical) 
    res = [(lng, lat), (lng - resolution, lat - resolution), (lng - resolution, lat + resolution),
           (lng + resolution, lat - resolution), (lng + resolution, lat + resolution)]
    coor_str = ",".join([str(lng), str(lat)])
    res_str = [",".join([str(round(xx, round_demical)) for xx in x]) for x in res]
    answer = coor_str
    question = "What is the longitude and latigude coordinates of {}.".format(poi_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def poi2addr_gen(nearby_addrs, poi_dict, poi_id):
    tar_addr = poi_dict[int(poi_id)]['Address']
    tar_name = poi_dict[int(poi_id)]['name']
    res_str = []
    for addr in nearby_addrs:
        if len(res_str) < 4:
            if addr != tar_addr:
                res_str.append(addr)
    res_str.append(tar_addr)
    answer = tar_addr
    question = "What is the address of {}?".format(tar_name)
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def poi2type_gen(tar_type, all_pois, type_pois):
    all_poi_cate = type_pois.keys()
    available_types = [cate for cate in all_poi_cate if cate != tar_type]

    res_str = random.sample(available_types, 4)
    res_str.append(tar_type)
    if len(set(type_pois[tar_type])) < 4:
        return None
    tar_poi_ids = random.sample(list(set(type_pois[tar_type])), 4)
    poi_names = [all_pois[poi_id]['name'] for poi_id in tar_poi_ids]
    answer = tar_type
    question = "Which type do following POIs belong to? POIs:{}".format(",".join(poi_names))
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def type2poi_gen(tar_type, all_pois, type_pois):  
    type1_ids = []  # POI_ID not in tar_type
    for type1, ids in type_pois.items():
        if type1 != tar_type:
            type1_ids += ids
    if len(set(type1_ids)) < 4 or len(set(type_pois[tar_type])) < 1:
        return None
    res_poi_ids = random.sample(list(set(type1_ids)), 4)
    tar_poi_id = random.sample(list(set(type_pois[tar_type])), 1)[0]
    tar_poi_name = all_pois[tar_poi_id]['name']
    poi_names = [all_pois[poi_id]['name'] for poi_id in res_poi_ids]
    poi_names.append(tar_poi_name)
    answer = tar_poi_name
    question = "Which POI belongs to {}?".format(tar_type)
    res_dict = gen_options(poi_names, question, answer)
    return res_dict


def aoi_poi_gen(aoi_id, all_pois, all_aois, poi_dict, aoi_dict):
    # aoi_id = random.sample(list(all_aois.keys()),1)
    poi_ids = [poi_id for poi_id in all_aois[aoi_id]['poi_ids'] if all_pois[poi_id]['name']]
    if len(poi_ids) < 1:
        return None
    tar_poi_id = random.sample(poi_ids, 1)[0]
    tar_poi = all_pois[tar_poi_id]['name']
    res_poi_ids = random.sample(list(set(poi_dict.keys()) - set(poi_ids)), 3)
    res_pois = [all_pois[poi_id]['name'] for poi_id in res_poi_ids]
    pois_in_aoi = [all_pois[poi_id]['name'] for poi_id in poi_ids]
    while len(set(res_pois) - set(pois_in_aoi)) < 3: 
        res_poi_id = random.sample(list(set(poi_dict.keys()) - set(poi_ids)), 1)[0]
        res_pois.append(all_pois[res_poi_id]['name'])
    res_pois.append(tar_poi)
    question = "Which of the following POIs is likely to belong to the AOI {}?POIs:{}".format(
        aoi_dict[aoi_id]['name'], ",".join(res_pois))
    answer = tar_poi
    res_dict = gen_options(res_pois, question, answer)
    return res_dict


def poi_aoi_gen(aoi_id, all_pois, all_aois, aoi_dict):
    tar_aoi = aoi_dict[aoi_id]['name']
    valid_poi_ids = [poi_id for poi_id in all_aois[aoi_id]['poi_ids'] if all_pois[poi_id]['name']]
    if len(valid_poi_ids) < 4:
        return None
    poi_ids = random.sample(valid_poi_ids, 4)
    res_aoi_ids = random.sample(list(set(aoi_dict.keys()) - {aoi_id}), 3)
    res_aois = [aoi_dict[aoi_id]['name'] for aoi_id in res_aoi_ids]
    res_aois.append(tar_aoi)
    pois_in_aoi = [all_pois[poi_id]['name'] for poi_id in poi_ids]
    answer = tar_aoi
    question = "Which AOI does the following POIs belong to？POIs:{}".format(",".join(pois_in_aoi))
    res_dict = gen_options(res_aois, question, answer)
    return res_dict


def select_types(poi_ids, poi_dict):
    result_type = []
    poi_ids = [poi_id for poi_id in poi_ids if poi_id in poi_dict]
    poi_types = [poi_dict[poi_id]['category'] for poi_id in poi_ids]
    result_type += [random.sample([poi_id for poi_id in poi_ids if poi_dict[poi_id]['category'] == poi_type],
                                  int((poi_types.count(poi_type)) / 4) + 1) for poi_type in set(poi_types)]
    return list(chain(*result_type))


def aoi2addr_gen(nearby_addrs, aoi_id, aoi_dict):
    tar_addr = aoi_dict[aoi_id]['address']
    tar_name = aoi_dict[aoi_id]['name']
    nearby_addrs.append(tar_addr)

    return gen_options(nearby_addrs, "What is the address of {}?".format(tar_name), tar_addr)


def aoi2type_gen(map, aoi_id, landuse_dict, aoi_dict, poi_dict):
    landuse_str = ["OtherNon-construction", "Residential", "TrafficStation&Park", "Sports", "Entertainment", "OtherPublicFacilities", "Education","Park&GreenLand","CommercialService&IndustryFacilities","Resort&Fitness","Restaurant&Bar","ReligiousFacilities","Hospital"]
    try:
        tar_type = landuse_dict[aoi_dict[aoi_id]['category']]
    except KeyError as e:
        return None
    options = [landuse for landuse in landuse_str if landuse != tar_type]
    random_options = random.sample(options, 4)
    random_options.append(tar_type)
    poi_ids = map.get_aoi(aoi_id)['poi_ids']
    poi_in_type = select_types(poi_ids, poi_dict)
    poi_names = [poi_dict[poi_id]['name'] for poi_id in poi_in_type if poi_id in poi_dict]
    if len(poi_names) < 3:
        return None
    question = "Given that {} contains these POIs:{}.Which landuse type does {} belong to?".format(
        map.get_aoi(aoi_id)['name'], ",".join(poi_names), map.get_aoi(aoi_id)['name']
    )

    return gen_options(random_options, question, tar_type)


def type2aoi_gen(map, landuse, landuse_dict, aoi_dict, poi_dict):  
    tar_aoi = []
    res_aoi = []
    for id, aoi in aoi_dict.items():
        type_ = aoi['category']
        if type_ == landuse:
            tar_aoi.append(id)
        else:
            res_aoi.append(id)
    if tar_aoi == []:
        return None
    aoi_id1 = random.sample(tar_aoi, 1)[0]
    aoi_id2, aoi_id3, aoi_id4 = random.sample(res_aoi, 3)
    poi_names1 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id1)['poi_ids'], poi_dict)]
    poi_names2 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id2)['poi_ids'], poi_dict) if poi_id in poi_dict]
    poi_names3 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id3)['poi_ids'], poi_dict) if poi_id in poi_dict]
    poi_names4 = [poi_dict[poi_id]['name'] for poi_id in select_types(map.get_aoi(aoi_id4)['poi_ids'], poi_dict) if poi_id in poi_dict]
    aoi_names = [map.get_aoi(aoi_id1)['name'], map.get_aoi(aoi_id2)['name'], map.get_aoi(aoi_id3)['name'], map.get_aoi(aoi_id4)['name']]
    if len(poi_names1) < 2 or len(poi_names2) < 2 or len(poi_names3) < 2 or len(poi_names4) < 2:
        return None
    answer =map.get_aoi(aoi_id1)['name']
    question = "Given that:1.{} contains these POIs:{}\n2.{} contains these POIs:{}\n3.{} contains these POIs:{}\n4.{} contains these POIs:{}\nWhich of the above AOIs({},{},{},{}) is designated as {}?".format(
        map.get_aoi(aoi_id1)['name'], ",".join(poi_names1), map.get_aoi(aoi_id2)['name'], ",".join(poi_names2), map.get_aoi(aoi_id3)['name'], ",".join(poi_names3), map.get_aoi(aoi_id4)['name'],
        ",".join(poi_names4), map.get_aoi(aoi_id1)['name'], map.get_aoi(aoi_id2)['name'], map.get_aoi(aoi_id3)['name'], map.get_aoi(aoi_id4)['name'], landuse_dict[landuse])
    res_dict = gen_options(aoi_names, question, answer)
    return res_dict


def districts_poi_type_gen(aoi_id, type_num, map):
    aoi_info = map.get_aoi(aoi_id)
    res_str = [3, 4, 5, 6, 7]
    answer = type_num
    question = "How many types of POIs are there in {}?".format(aoi_info['name'])
    res_dict = gen_options(res_str, question, answer)
    return res_dict


def save_data(unseen_pois, save_path):
    unseen_pois_df = pd.DataFrame(data=unseen_pois)
    unseen_pois_df.to_csv(save_path)


def get_node_num(dict):
    if len(dict.keys()) > 200:
        return 200
    else:
        return len(dict.keys())


def generate_evaluation_task_aoi_loc(aoi_dict, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    aoi2addr = []
    for aoi_id in aois:
        res = [item for item in aois if item != aoi_id]
        aoi_id1, aoi_id2, aoi_id3 = random.sample(res, 3)
        if len({aoi_dict[aoi_id1]['address'], aoi_dict[aoi_id2]['address'], aoi_dict[aoi_id3]['address']}) < 3:
            continue
        nearby_addrs = [aoi_dict[aoi_id1]['address'], aoi_dict[aoi_id2]['address'], aoi_dict[aoi_id3]['address']]
        aoi2addr.append(aoi2addr_gen(nearby_addrs, aoi_id, aoi_dict))
    save_data(aoi2addr, TASK_FILES["districts"]["aoi2addr"])
    print("districts aoi2addr task OK!")


def get_land_uses(aoi_dict):
    landuses = set()
    for id, aoi in aoi_dict.items():
        landuse = aoi['category']
        landuses.add(landuse)
    return list(landuses)


def generate_evaluation_task_aoi_type(map, aoi_dict, landuse_dict, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    aoi2type = []
    for aoi_id in aois:
        res_dict_aoi2type = aoi2type_gen(map, aoi_id, landuse_dict, aoi_dict, poi_dict)
        if res_dict_aoi2type:
            aoi2type.append(res_dict_aoi2type)
    save_data(aoi2type, TASK_FILES["districts"]["aoi2type"])
    print("districts aoi2type task OK!")
    type2aoi = []
 
    landuses = [
        "E3", "R", "S4", "A4", "B31", "U9", "A3", "G1", "B", "B32", "B13", "A9", "A5"
    ]

    right = 0
    while right < 20:
        p = random.sample(landuses, 1)[0]
        res_dict_type2aoi = type2aoi_gen(map, p, landuse_dict, aoi_dict, poi_dict)
        if res_dict_type2aoi:
            type2aoi.append(res_dict_type2aoi)
            right += 1
    save_data(type2aoi, TASK_FILES["districts"]["type2aoi"])
    print("districts type2aoi task OK!")


def generate_districts_poi_type(map, aoi_dict, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = list(aoi_dict.keys())[:question_num]
    districts_poi_type = []
    for aoi_id in aois:
        poi_ids = list(set(map.get_aoi(aoi_id)['poi_ids']) & set(poi_dict.keys()))
        types = set()
        for p in poi_ids:
            types.add(map.get_poi(p)['category'])
        type_num = len(types)
        # print(f"type num: {aoi.type_num}")
        if type_num < 3 or type_num > 7:
            continue
        districts_poi_type.append(districts_poi_type_gen(aoi_id, type_num, map))
    save_data(districts_poi_type, TASK_FILES["districts"]["districts_poi_type"])
    print("districts districts_poi_type task OK!")


def generate_evaluation_task_poi_loc(map, poi_dict, poi_message, all_pois, category_supported, TASK_FILES):
    question_num = get_node_num(poi_dict)
    pois = random.sample(list(poi_dict.keys()), question_num)
    name_counts = poi_message['name'].value_counts(sort=True)
    many_names = name_counts[name_counts > 10]  
    poi2cor = []
    for p in pois:
        coords = map.get_poi(p)["position"]
        if all_pois[p]['name'] in many_names:
            continue

        if len(coords) > 0:
            lng, lat = map.xy2lnglat(x=coords["x"], y=coords["y"])  
            poi2cor.append(poi2cor_gen((lng, lat), p, poi_dict))
    save_data(poi2cor, TASK_FILES["node"]["poi2coor"])
    print("node poi2coor task OK!")
    poi2addr = []
    for p in pois:
        coords = map.get_poi(p)["position"] 
        lng, lat = map.xy2lnglat(x=coords["x"], y=coords["y"])
        nearby_id = get_nearby_pois((lng, lat), map, category_supported)
        nearby_addrs = [poi_dict[id]['Address'] for id in nearby_id if id in poi_dict]
        if all_pois[p]['name'] in many_names:
            continue
        if len(list(set(nearby_addrs))) > 4 and p in poi_dict:
            poi2addr.append(poi2addr_gen(nearby_addrs, poi_dict, p))
    save_data(poi2addr, TASK_FILES["node"]["poi2addr"])
    print("node poi2addr task OK!")


def generate_evaluation_task_poi_type(aoi_dict, all_aois, all_pois, type_pois, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = random.sample(list(aoi_dict.keys()), question_num)
    poi2type = []
    for key in type_pois.keys():
        res_poi2type = poi2type_gen(key, all_pois, type_pois)
        if res_poi2type:
            poi2type.append(res_poi2type)
    save_data(poi2type, TASK_FILES["node"]["poi2type"])
    print("node poi2type task OK!")
    type2poi = []
    for key in type_pois.keys():
        res_type2poi = type2poi_gen(key, all_pois, type_pois)
        if res_type2poi:
            type2poi.append(res_type2poi)
    save_data(type2poi, TASK_FILES["node"]["type2poi"])
    print("node type2poi task OK!")


def generate_evaluation_task_poi_aoi(aoi_dict, all_pois, all_aois, poi_dict, TASK_FILES):
    question_num = get_node_num(aoi_dict)
    aois = random.sample(list(aoi_dict.keys()), question_num)
    aoi_poi = []
    for p in aois:
        if len(all_aois[p]['poi_ids']) <= 3 or not isinstance(aoi_dict[p]['name'], str):
            continue
        ret_aoi_poi = aoi_poi_gen(p, all_pois, all_aois, poi_dict, aoi_dict)
        if ret_aoi_poi:
            aoi_poi.append(ret_aoi_poi)
    save_data(aoi_poi, TASK_FILES["districts"]["aoi_poi"])
    print("districts aoi_poi task OK!")
    poi_aoi = []
    for p in aois:
        if len(all_aois[p]['poi_ids']) <= 3 or not isinstance(aoi_dict[p]['name'], str):
            continue
        res_poi_aoi = poi_aoi_gen(p, all_pois, all_aois, aoi_dict)
        if res_poi_aoi:
            poi_aoi.append(res_poi_aoi)
        
    save_data(poi_aoi, TASK_FILES["districts"]["poi_aoi"])
    print("districts poi_aoi task OK!")

def landmark_path(map, poi_dict, pois_along_route, start_poi, end_poi):
    pois_along_route = list(set(pois_along_route))
    all_pois = random.choices(list(poi_dict.keys()), k=10)
    all_pois_name = set([map.get_poi(poi_id)['name'] for poi_id in all_pois])
    negative_samples = all_pois_name.difference(set(pois_along_route))
    negative_sample = random.choice(list(negative_samples))
    if len(pois_along_route) >= 3:
        res_pois = random.sample(pois_along_route, k=3)
    else:
        print("pois_along_route is less than 3")
        return None

    res_pois.append(negative_sample)
    question = "Which of the following POIs will not be passed when traveling from {} to {}?".format(
        start_poi, end_poi)
    answer = negative_sample
    res_dict = gen_options(res_pois, question, answer)
    return res_dict


async def generate_evaluation_task_landmark(map, routing_client, poi_dict, TASK_FILES):
    route_arrive_pois = []
    count = 0
    while count < 50:
        # 随机选出两个不同的POI ID
        poi_ids = list(poi_dict.keys())
        start_poi_id, end_poi_id = random.sample(poi_ids, 2)
        start_aoi_id = poi_dict[start_poi_id]["aoi_id"]
        end_aoi_id = poi_dict[end_poi_id]["aoi_id"]
        if start_aoi_id == end_aoi_id:
            continue
        player = Player(map, routing_client, start_aoi_id, "poi")
        route = await player.get_driving_route(end_aoi_id)
        if route is None:
            continue
        # 限制步数不超过12步，不少于3步
        if len(route["road_ids"]) > 12 or len(route["road_ids"]) < 3:
            continue
        road_list = []
        for road_id in route["road_ids"]:
            road_info = player._city_map.get_road(road_id)
            lane_info = player._city_map.get_lane(road_info["lane_ids"][0])
            road_list = player.road_info_collect(road_info, lane_info, road_list)

        start_poi_name = poi_dict[start_poi_id]['name']
        start_poi_addr = poi_dict[start_poi_id]['Address']
        start_poi = "{}{}".format(start_poi_name, start_poi_addr)
        end_poi_name = poi_dict[end_poi_id]['name']
        end_poi_addr = poi_dict[end_poi_id]['Address']
        end_poi = "{}{}".format(end_poi_name, end_poi_addr)
        if not start_poi_name or not end_poi_name:
            continue
        pois_along_route = []
        while(len(road_list) > 0):
            road_name, road_length, lane_id, direction, _ = road_list.pop(0)
            lane_info = map.get_lane(lane_id)
            endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
            endpoint_xy = lane_info["shapely_xy"].coords[-1]

            # update position
            player.position = Position(
                xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
                longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            )
            interest_list = player.get_nearby_interests()
            pois_along_route.extend(interest_list)
        # print("length of pois_along_route: ", len(set(pois_along_route)))
        res_dict = landmark_path(map, poi_dict, pois_along_route, start_poi, end_poi)
        # print("res_dict: ", res_dict) 
        if not res_dict:
            continue
        route_arrive_pois.append(res_dict)
        count += 1
    save_data(route_arrive_pois, TASK_FILES["landmark"]["landmark_path"])
    print("landmark landmark_path task OK!")

def generate_evaluation_task_boudary(all_roads, all_lanes, aoi_dict, REGION_EXP_POLYGON, TASK_FILES):
    cared_roads_name = {}
    for road_id in all_roads:
        lane_ids = all_roads[road_id]["lane_ids"]
        road_name = all_roads[road_id]['name']
        if road_name == '':
            continue

        road_in_region = False
        aoi_ids_include = []
        for i, lane_id in enumerate(lane_ids):
            if isinstance(lane_id, list):
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            last_point = Point(lane["shapely_lnglat"].coords[-1])
            aoi_ids = lane["aoi_ids"]

            if REGION_EXP_POLYGON.contains(last_point):
                road_in_region = True
                aoi_ids_include.extend(aoi_ids)

        if road_in_region:
            include_aois = []
            for aoi_id in aoi_ids_include:
                if aoi_id in aoi_dict:
                    include_aois.append(aoi_dict[aoi_id]['name'])
            if road_name not in cared_roads_name:
                cared_roads_name[road_name] = include_aois
            else:
                cared_roads_name[road_name].extend(include_aois)
    for road_name in cared_roads_name:
        cared_roads_name[road_name] = list(set(cared_roads_name[road_name]))

    aoi_boundary_dict = {}
    for aoi_id in aoi_dict:
        aoi_name = aoi_dict[aoi_id]['name']
        aoi_boundary_dict[aoi_id] = {"name": aoi_name, "boundary": []}
        for road_name in cared_roads_name:
            if aoi_name in cared_roads_name[road_name]:
                aoi_boundary_dict[aoi_id]["boundary"].append(road_name)

    res_aoi_boundary = []
    candidates_roads = list(cared_roads_name.keys())
    for aoi_id in aoi_boundary_dict:
        if len(aoi_boundary_dict[aoi_id]["boundary"]) == 0:
            continue

        aoi_name = aoi_boundary_dict[aoi_id]["name"]
        boundary = aoi_boundary_dict[aoi_id]["boundary"]

        if len(boundary) >= 3:
            res = random.sample(boundary, 3)

            random.shuffle(candidates_roads)
            for x in candidates_roads:
                if x not in boundary:
                    negative_sample = x
                    break
            res.append(negative_sample)
            question = "Which road is not the boundary of AOI {}".format(aoi_name)
            answer = negative_sample
            res_dict = gen_options(res, question, answer)
        else:
            # 边界较少，询问哪个是边界
            current_roads = list(set(candidates_roads).difference(set(boundary)))
            res = random.sample(current_roads, 3) + random.sample(boundary, 1)
            answer = res[-1]
            question = "Which road is the boundary of AOI {}".format(aoi_name)
            res_dict = gen_options(res, question, answer)
        res_aoi_boundary.append(res_dict)

    task_df = pd.DataFrame(data=res_aoi_boundary)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["boundary"]["boundary_road"])
    print("boundary boundary_road task OK!")

def generate_evalation_task_boundary_poi(map, aoi_dict, poi_dict, TASK_FILES):
    pois_ids = poi_dict.keys()
    aois_data = []
    available_aois = []
    min_pois = 5
    for aoi_id in aoi_dict.keys():
        info = map.get_aoi(id=aoi_id)
        if len(info["poi_ids"]) >= min_pois:
            aois_data.append(info)
            available_aois.append(aoi_id)
    print("available aois:{}".format(len(available_aois)))

    aoi_boudary_pois = []
    for i, aoi_id in enumerate(available_aois):
        info = aois_data[i]

        if aoi_id in aoi_dict:
            aoi_name = aoi_dict[aoi_id]['name']
        else:
            continue

        poi_ids = info["poi_ids"]
        boudary = info["shapely_xy"]
        center = boudary.centroid
        poi_distance = []
        for poi_id in poi_ids:
            poi_info = map.get_poi(poi_id)
            if poi_info['name'] == '':
                continue
            pos = poi_info["position"]
            pos_shapely = Point((pos["x"], pos["y"]))
            dis = shapely.distance(center, pos_shapely)
            poi_distance.append((poi_id, dis))

        poi_distance = sorted(poi_distance, key=lambda x: x[1])

        if len(poi_distance) < 3:
            continue
        random_pois = random.sample(list(pois_ids), 10)
        random_pois_clean = []
        for poi_id in random_pois:
            if poi_id != poi_distance[0][0] and poi_id != poi_distance[-1][0]:
                if map.get_poi(poi_id)['name'] != '':
                    random_pois_clean.append(poi_id)
        if len(random_pois_clean) < 2:
            continue

        poi_names = [map.get_poi(p[0])['name'] for p in poi_distance]
        res_pois = [poi_names[0], poi_names[1], poi_names[-1],
                    map.get_poi(random_pois_clean[0])['name'],
                    map.get_poi(random_pois_clean[1])['name']]
        label_name = poi_names[-1]
        question = "Which point of interest (POI) is most likely to appear in the boundary of AOI:{}?".format(aoi_name)
        answer = str(label_name)
        res_dict = gen_options(res_pois, question, answer)
        aoi_boudary_pois.append(res_dict)

    task_df = pd.DataFrame(data=aoi_boudary_pois)
    task_df.to_csv(TASK_FILES["boundary"]["aoi_boundary_poi"])
    print("boundary aoi_boundary_poi task OK!")

def generate_evalation_task_districts(map, aoi_message, TASK_FILES):

    def districts_gen(aois, max_group, pois_name_str):
        res = list(range(1, max_group + 1))
        answer = len(aois)
        question = "How many regions can the following POIs be divided into? POIs:{}".format(pois_name_str)
        res_dict = gen_options(res, question, answer)
        return res_dict

    aois_data = []
    available_aois = []
    min_pois = 2
    for aoi_id in aoi_message.aoi_id.to_list():
        info = map.get_aoi(id=aoi_id)
        if len(info["poi_ids"]) >= min_pois:
            aois_data.append(info)
            available_aois.append(aoi_id)
    print("available aois:{}".format(len(available_aois)))

    aois_group = []
    group_size = 50
    max_group = 5
    for i in range(max_group):
        for _ in range(2 * group_size):
            item = tuple(random.sample(available_aois, i + 1))
            if item not in aois_group:
                aois_group.append(item)
            if len(aois_group) == group_size * (i + 1):
                break
    random.shuffle(aois_group)

    tasks = []
    for aois in aois_group:
        pois = []
        for aoi in aois:
            pois_in = map.get_aoi(aoi)["poi_ids"]
            pois_in = [poi_id for poi_id in pois_in if map.get_poi(poi_id)['name']]
            max_len = min(len(pois_in), 6)
            min_len = min(len(pois_in), 1)
            if max_len <= min_len:
                continue
            pois.extend(random.sample(pois_in, random.sample(range(min_len, max_len), 1)[0]))
        pois_names = []
        for poi in pois:
            try:
                poi_name = map.get_poi(poi)['name']
                pois_names.append(poi_name)
            except IndexError as e:
                print(e)
                continue

        if len(pois_names) <= len(aois):
            continue

        random.shuffle(pois_names)
        tasks.append(districts_gen(aois, 5, ",".join(pois_names)))

    task_df = pd.DataFrame(data=tasks)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["districts"]["aoi_group"])
    print("districts aoi_group task OK!")



def get_aoi_type(aoi_id, poi_ids, aoi_dict, poi_dict):
    type_dict = {}
    aoi = aoi_dict[aoi_id]
    for poi_id in poi_ids:
        type1 = poi_dict[poi_id]['category']
        if type1 not in type_dict:
            type_dict[type1] = []
        type_dict[type1].append(poi_id)
    max_length = 0
    most_type = None
    for type1, poi_ids in type_dict.items():
        # 检查当前value的长度是否大于已知的最大长度
        if len(poi_ids) > max_length:
            max_length = len(poi_ids)
            most_type = type1
    return type_dict, most_type


def AOI_POI_task_gen(map, aoi_dict, poi_dict, aoi_id):   #选择poi_ids > 3,poi_id有至少一个具有名字，且具有名字的的aoi_id
    poi_ids = list(set(aoi_dict[aoi_id]['poi_ids']) & set(poi_dict.keys()))   #是否要这样做,先找名字再poi_ids
    aoi_address = aoi_dict[aoi_id]['address']
    type_in_aoi = get_aoi_type(aoi_id, poi_ids, aoi_dict, poi_dict)[0]
    most_type = get_aoi_type(aoi_id, poi_ids, aoi_dict, poi_dict)[1]
    tar_poi = random.choice(type_in_aoi[most_type])
    tar_poi_name = poi_dict[tar_poi]['name']

    res_pois = random.choices(list(set(poi_dict.keys()).difference({tar_poi})),k=3)
    res_pois_name = [poi_dict[res_poi]['name'] for res_poi in res_pois]
    res_pois_name.append(tar_poi_name)
    question = "Which of the following POIs is in the '{}' category within the area at {}?".format(most_type, aoi_address)
    answer = tar_poi_name
    res_dict = gen_options(res_pois_name,question,answer)
    return res_dict

def AOI_POI2_task_gen(map, aoi_dict, poi_dict, aoi_id):
    res_aois = random.choices(list(set(aoi_dict.keys()).difference({aoi_id})),k=3)
    res_aois.append(aoi_id)
    res_aois_name = [aoi_dict[res_aoi]['name'] for res_aoi in res_aois]
    max_length = 0
    most_aoi = None
    for aoi in res_aois:
        poi_ids = aoi_dict[aoi]['poi_ids']
        # 检查当前value的长度是否大于已知的最大长度
        if len(poi_ids) > max_length:
            max_length = len(poi_ids)
            most_aoi = aoi
    most_aoi_name = aoi_dict[most_aoi]['name']
    question2 = "Which AOI has the most points of interest?"
    answer2 = most_aoi_name
    res_dict2 = gen_options(res_aois_name, question2, answer2)
    return res_dict2

def AOI_POI3_task_gen(map, aoi_dict, poi_dict, aoi_id, category_supported, type_pois):
    aoi_address = aoi_dict[aoi_id]['address']
    aoi_coor = aoi_dict[aoi_id]['coord']
    nearby_pois = get_nearby_pois(aoi_coor, map, category_supported)
    nearby_pois = set(nearby_pois)&set(poi_dict.keys())
    if len(nearby_pois) == 0:
        return
    poi = random.choice(list(nearby_pois))
    tar_poi_name = poi_dict[poi]['name']
    tar_poi_type = poi_dict[poi]['category']
    possible_pois = list(set(type_pois[tar_poi_type]).difference(set(nearby_pois)))
    if len(possible_pois) < 3:
        return None  
    res_pois = random.choices(possible_pois, k=3)
    res_pois_name = [poi_dict[res_poi]['name'] for res_poi in res_pois]
    res_pois_name.append(tar_poi_name)
    question3 = "Which POIs in the category {} are located within a 100m radius of {}?".format(tar_poi_type, aoi_address)
    answer3 = tar_poi_name
    res_dict3 = gen_options(res_pois_name, question3, answer3)
    return res_dict3

def AOI_POI4_task_gen(map, aoi_dict, poi_dict, aoi_id, category_supported, type_pois):
    poi_ids = list(set(aoi_dict[aoi_id]['poi_ids']) & set(poi_dict.keys()))   #是否要这样做,先找名字再poi_ids
    aoi_name = aoi_dict[aoi_id]['name']
    aoi_coor = aoi_dict[aoi_id]['coord']

    type_in_aoi = get_aoi_type(aoi_id, poi_ids, aoi_dict, poi_dict)[0]
    most_type = get_aoi_type(aoi_id, poi_ids, aoi_dict, poi_dict)[1]
    tar_poi = random.choice(type_in_aoi[most_type])
    further_pois = get_nearby_pois(aoi_coor, map, category_supported)
    further_pois = set(further_pois) & set(poi_dict.keys())
    further_pois_type = further_pois&set(type_pois[most_type])
    if len(further_pois_type) < 4:
        # print("no enough pois <4")
        return
    res_pois = random.choices(list(further_pois_type), k=4)
    min_dist = 100
    nearest_poi = None
    for poi in res_pois:
        coor = poi_dict[poi]['coord']
        dist = geodesic((coor[1], coor[0]), (aoi_coor[1], aoi_coor[0])).km
        if dist < min_dist:
            min_dist = dist
            nearest_poi = poi
    if not nearest_poi:
        print("no nearest poi")
        return
    nearest_poi_name = poi_dict[nearest_poi]['name']
    res_pois_name = [poi_dict[res_poi]['name'] for res_poi in res_pois]
    answer4 = nearest_poi_name
    question4 = "Which is the nearest {} POI to {}?".format(most_type, aoi_name)
    res_dict4 = gen_options(res_pois_name, question4, answer4)
    return res_dict4

def AOI_POI5_task_gen(map, aoi_dict, poi_dict, aoi_id):
    aoi_name = aoi_dict[aoi_id]['name']
    aoi_area = map.get_aoi(aoi_id)['area']
    question5 = "What is the total area of {}?".format(aoi_name)
    answer5 = int(aoi_area)
    res_areas = [int(aoi_area*0.8), int(aoi_area*2), int(aoi_area+200), int(aoi_area)]
    res_dict5 = gen_options(res_areas, question5, answer5)
    return res_dict5

def AOI_POI6_task_gen(map, aoi_dict, poi_dict, aoi_id):
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


def AOI_POI2road_task_gen(map, aoi_dict, poi_dict, road_dict, aoi_id):  #选取有名字的AOI,且poi_ids数量不少于2个
    aoi = map.get_aoi(aoi_id)
    lanes = map.get_aoi(aoi_id)['driving_positions']

    roads2aoi = []
    for lane in lanes:
        lane_id = lane['lane_id']
        road_id = map.get_lane(lane_id)['parent_id']
        if road_id not in road_dict or road_id in roads2aoi:
            continue
        roads2aoi.append(road_id)

    other_roads = list(set(road_dict.keys()).difference(set(roads2aoi)))

    roads_name = list(set([road_dict[road]['name'] for road in roads2aoi if road in set(roads2aoi)&set(road_dict.keys())]))
    if len(roads_name) > 2:
        res_roads_name = random.choices(roads_name, k=3)
        other_road = random.choice(other_roads)
        tar_road_name = road_dict[other_road]['name']
        res_roads_name.append(tar_road_name)
        question = "Which of the following roads is not connected to {}?".format(aoi['name'])
    elif 0 < len(roads_name) <= 2:
        tar_road_name = random.choice(roads_name)
        other_roads = random.choices(other_roads, k=3)
        res_roads_name = [road_dict[other_road]['name'] for other_road in other_roads]
        res_roads_name.append(tar_road_name)
        question = "Which of the following roads is connected to {}?".format(aoi['name'])
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
    if max_length_road in road_dict:
        tar_road_name = road_dict[max_length_road]['name']
    else:
        tar_road_name = 'unknown road'
        
    res_roads = random.choices(other_roads, k=3)
    res_roads_name = [road_dict[road_id]['name'] for road_id in res_roads]
    question2 = "What's the longest road that meets {}?".format(aoi['name'])
    answer2 = tar_road_name
    res_roads_name.append(tar_road_name)
    res_dict2 = gen_options(res_roads_name, question2, answer2)

    tar_road = None
    for road_id in roads2aoi:
        if len(road_dict[road_id]['aoi_ids']) > 2 and road_id in road_dict:
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

    poi_ids = aoi['poi_ids']
    other_pois = list(set(poi_dict.keys()).difference(set(poi_ids)))
    res_pois = random.choices(other_pois, k=3)
    # tar_poi, res_poi = poi_ids[0], poi_ids[1]
    tar_poi, res_poi = None, None  
    valid_pois = [poi_id for poi_id in poi_ids if poi_id in poi_dict]  
    if len(valid_pois) >= 2:  
        tar_poi, res_poi = valid_pois[0], valid_pois[1]
    else:
        return
    res_pois.append(tar_poi)
    tar_poi_name = poi_dict[tar_poi]['name']
    res_pois_name = [poi_dict[poi_id]['name'] for poi_id in res_pois]
    question4 = "Which POIs are adjacent to {} in this area?".format(poi_dict[res_poi]['name'])
    answer4 = tar_poi_name
    res_dict4 = gen_options(res_pois_name, question4, answer4)
    return res_dict, res_dict2, res_dict3, res_dict4


def generate_AOI_POI_task(map, aoi_dict, poi_dict, TASK_FILES):
    task_name = "AOI_POI"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <= 3:
            continue
        if len(set(poi_ids)&set(poi_dict.keys())) == 0:
            continue
        result = AOI_POI_task_gen(map, aoi_dict, poi_dict, id)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES["others"][task_name])
    print("others AOI_POI task OK!")

def generate_AOI_POI2_task(map, aoi_dict, poi_dict, TASK_FILES):
    task_name = "AOI_POI2"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <= 3:
            continue
        if len(set(poi_ids)&set(poi_dict.keys())) == 0:
            continue
        result = AOI_POI2_task_gen(map, aoi_dict, poi_dict, id)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES["others"][task_name])
    print("others AOI_POI2 task OK!")

def generate_AOI_POI3_task(map, aoi_dict, poi_dict, category_supported, type_pois, TASK_FILES):
    task_name = "AOI_POI3"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <= 3:
            continue
        if len(set(poi_ids)&set(poi_dict.keys())) == 0:
            continue
        result = AOI_POI3_task_gen(map, aoi_dict, poi_dict, id, category_supported, type_pois)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES["others"][task_name])
    print("others AOI_POI3 task OK!")

def generate_AOI_POI4_task(map, aoi_dict, poi_dict, category_supported, type_pois, TASK_FILES):
    task_name = "AOI_POI4"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <= 3:
            continue
        if len(set(poi_ids)&set(poi_dict.keys())) == 0:
            continue
        result = AOI_POI4_task_gen(map, aoi_dict, poi_dict, id, category_supported, type_pois)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES["others"][task_name])
    print("others AOI_POI4 task OK!")

def generate_AOI_POI5_task(map, aoi_dict, poi_dict, TASK_FILES):
    task_name = "AOI_POI5"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <= 3:
            continue
        if len(set(poi_ids)&set(poi_dict.keys())) == 0:
            continue
        result = AOI_POI5_task_gen(map, aoi_dict, poi_dict, id)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES["districts"][task_name])
    print("districts AOI_POI5 task OK!")

def generate_AOI_POI6_task(map, aoi_dict, poi_dict, TASK_FILES):
    task_name = "AOI_POI6"
    AOI_POI = []
    for id,aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <= 3:
            continue
        if len(set(poi_ids)&set(poi_dict.keys())) == 0:
            continue
        result = AOI_POI6_task_gen(map, aoi_dict, poi_dict, id)
        if not result:
            continue
        AOI_POI.append(result)

    task_df = pd.DataFrame(AOI_POI)
    task_df.to_csv(TASK_FILES["districts"][task_name])
    print("districts AOI_POI6 task OK!")


def generate_AOI_POI2road_task_gen(map, aoi_dict, poi_dict, road_dict, TASK_FILES):
    AOI_connect_rd = []
    AOI_longest_rd = []
    POI2POI = []
    AOI2rd = []
    for id, aoi in aoi_dict.items():
        name = aoi['name']
        poi_ids = aoi['poi_ids']
        if not isinstance(name, str) or len(poi_ids) <2:
            continue
        result = AOI_POI2road_task_gen(map, aoi_dict, poi_dict, road_dict, id)
        if not result:
            continue
        AOI_connect_rd.append(result[0])
        AOI_longest_rd.append(result[1])
        POI2POI.append(result[2])
        AOI2rd.append(result[3])

    datas = [AOI_connect_rd, AOI_longest_rd, POI2POI, AOI2rd]
    for idx,data in enumerate(datas):
        task_df = pd.DataFrame(data)
        if idx == 3:
            task_df.to_csv(TASK_FILES["node"]["AOI_POI_road4"])
            print("node AOI_POI_road4 task OK!")
        else:
            task_df.to_csv(TASK_FILES["boundary"]["AOI_POI_road{}".format(idx+1)])
            print("boundary AOI_POI_road{} task OK!".format(idx+1))

async def main(args):
    city_map = MAP_DICT[args.city_name]
    
    map, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=args.port)


    if not isinstance(args.output_path, str):
        output_path = "citydata/task_Geo_knowledge/{}/{}".format(args.city_name, args.evaluate_version)
    else:
        output_path = args.output_path

    random.seed(42)
    REGION_EXP_POLYGON = Polygon(CITY_BOUNDARY[args.city_name])

    TASK_FILES = task_files_adaption(GEOQA_TASK_MAPPING_v2, output_path)
    category_supported = get_category_supported()

    all_roads = map.roads
    all_lanes = map.lanes
    all_aois = map.aois
    all_pois = map.pois
    all_juncs = map.juncs
    poi_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_pois.csv".format(args.city_name)))
    aoi_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_aois.csv".format(args.city_name)))
    road_message = pd.read_csv(os.path.join(RESOURCE_PATH, "{}_roads.csv".format(args.city_name)))

    poi_dict = {}
    all_poi_dict = {}
    for row in poi_message.itertuples():
        key = int(row.poi_id)
        category = map.get_poi(key)['category']
        name = row.name
        addr = row.Address
        if not isinstance(name, str):
            continue
        poi_dict[key] = {
            "aoi_id": map.get_poi(key)['aoi_id'], "category": category, "name": name, "Address": addr,
            "coord": map.get_poi(key)['shapely_lnglat'].coords[0],
        }
    print(f"define poi_dict done! len:{len(poi_dict)}")

    landuse_dict = get_landuse_dict()
    aoi_dict = {} 
    # filter "nearby" aoi
    for row in aoi_message.itertuples():
        aoi_id = row.aoi_id
        aoi_name = row.aoi_name
        address = row.Address
        extra = "nearby"
        poi_ids = map.get_aoi(aoi_id)['poi_ids']
        if not isinstance(aoi_name, str) or extra in aoi_name:
            continue
        if not isinstance(address, str):
            continue
        aoi_dict[aoi_id] = {}
        aoi_dict[aoi_id]['category'] = map.get_aoi(aoi_id)['urban_land_use']
        aoi_dict[aoi_id]['coord'] = eval(row.coords)[0]
        aoi_dict[aoi_id]['name'] = aoi_name
        aoi_dict[aoi_id]['address'] = address
        aoi_dict[aoi_id]['poi_ids'] = poi_ids
    print(f"define aoi_dict done! len:{len(aoi_dict)}")

    type_pois = {}  

    for poi_id, poi in poi_dict.items():
        poi_type = poi["category"]
        if poi_type in type_pois:
            type_pois[poi_type].append(poi_id)
        else:
            type_pois[poi_type] = [poi_id]
    
    road_dict = {}
    for row in road_message.itertuples():
        road_aois = []
        road_id = row.road_id
        length = all_roads[road_id]['length']
        road_name = row.road_name
        lane_ids = all_roads[road_id]['lane_ids']
        for lane_id in lane_ids:
            aoi_ids = all_lanes[lane_id]['aoi_ids']
            road_aois += aoi_ids
        if road_name == "未知路名" or road_name == "unknown road" or not isinstance(road_name, str):
            continue
        road_dict[road_id] = {}
        road_dict[road_id]['name'] = road_name
        road_dict[road_id]['length'] = length
        road_dict[road_id]['lane_ids'] = lane_ids
        road_dict[road_id]['aoi_ids'] = list(set(road_aois))
        
    print("Start synthesizing evaluation data")

    print("Start task: road length & road poi")
    generate_evaluation_task_road(map, poi_dict, all_roads, all_lanes, all_aois, TASK_FILES)

    print("Start task: road od & road link")
    generate_evaluation_task_road_junc(all_juncs, all_roads, all_lanes, REGION_EXP_POLYGON, TASK_FILES)

    print("Start task: landmark env")
    generate_evalation_task_node(map, poi_message, category_supported, TASK_FILES)

    print("Start task: landmark path")
    await generate_evaluation_task_landmark(map, routing_client, poi_dict, TASK_FILES)

    print("Start task: boundary road")
    generate_evaluation_task_boudary(all_roads, all_lanes, aoi_dict, REGION_EXP_POLYGON, TASK_FILES)

    print("Start task: boundary poi")
    generate_evalation_task_boundary_poi(map, aoi_dict, poi_dict, TASK_FILES)

    print("Start task: districts group")
    generate_evalation_task_districts(map, aoi_message, TASK_FILES)

    print("Start task: districts poi type & districts addr & districts type") 
    generate_districts_poi_type(map, aoi_dict, poi_dict, TASK_FILES)
    generate_evaluation_task_aoi_loc(aoi_dict, poi_dict, TASK_FILES)
    generate_evaluation_task_aoi_type(map, aoi_dict, landuse_dict, poi_dict, TASK_FILES)

    print("Start task: node coor & node addr & node type") 
    generate_evaluation_task_poi_loc(map, poi_dict, poi_message, all_pois, category_supported, TASK_FILES)
    generate_evaluation_task_poi_type(aoi_dict, all_aois, all_pois, type_pois, TASK_FILES)
    generate_evaluation_task_poi_aoi(aoi_dict, all_pois, all_aois, poi_dict, TASK_FILES)

    print("Start task: AOI_POI")
    generate_AOI_POI_task(map, aoi_dict, poi_dict, TASK_FILES)
    generate_AOI_POI2_task(map, aoi_dict, poi_dict, TASK_FILES)
    generate_AOI_POI3_task(map, aoi_dict, poi_dict, category_supported, type_pois, TASK_FILES)
    generate_AOI_POI4_task(map, aoi_dict, poi_dict, category_supported, type_pois, TASK_FILES)
    generate_AOI_POI5_task(map, aoi_dict, poi_dict, TASK_FILES)
    generate_AOI_POI6_task(map, aoi_dict, poi_dict, TASK_FILES)

    print("Start task: AOI_POI2road")
    generate_AOI_POI2road_task_gen(map, aoi_dict, poi_dict, road_dict, TASK_FILES)

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Beijing")
    parser.add_argument("--evaluate_version", type=str, default="v2")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--port", type=str, default="54319")
    args = parser.parse_args()

    asyncio.run(main(args))
    
