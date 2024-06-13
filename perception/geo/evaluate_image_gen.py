import argparse
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd
from pycitysim.map import Map
from pycitysim.routing import RoutingClient
from shapely import Point
from shapely.geometry.polygon import Polygon

from etc.all_config import all_config
from util import task_files_adaption, gen_options, save_data, get_region_exp, load_map, MAP_DICT

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_path)
sys.path.append(parent_path)


def generate_evaluation_task_road(aoi_dict, road_dict, aois_name_dict, aoi_name_dict_select):
    aoi_ids_set = set(aoi_dict.keys())
    aoi_ids = list(aoi_ids_set)

    cared_roads = {}
    cared_roads_name = {}

    count = {"all": 0, "name": 0}
    for road_id, road in road_dict.items():
        road_name = road['road_name']
        lane_ids = all_roads[road_id]["lane_ids"]

        if road_name == '':
            continue

        road_length = 0
        road_aoi_ids = []
        near_by_lanes = []
        one_lane_in_road = []
        road_level_count = []
        for i, lane_id in enumerate(lane_ids):
            if type(lane_id) == list:
                print("lane_id in lane_ids is List!!!!!")
                continue
            lane = all_lanes[lane_id]
            length = lane["length"]
            # aoi_ids = lane["aoi_ids"]
            # road_aoi_ids.extend(aoi_ids)

            # 利用道路下属车道信息进行道路长度估计
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
        else:
            cared_roads_name[road_name] = {"road_ids": [road_id], "lane_ids": [lane_ids], "length": [road_length],
                                           "level_count": road_level_count}

    for road_name in cared_roads_name:
        road_cut = max(cared_roads_name[road_name]["level_count"][0] + 1, 1)
        # 原因1：基于此类型近似判断车道，但是并不准确，导致错误；原因2：地图本身只切割了一部分，所以长度也会有差异
        # 实际计算临近车道数量
        cared_roads_name[road_name]["road_length_estimate"] = float(
            np.sum(cared_roads_name[road_name]["length"]) / road_cut)

    # 生成评估问题及对应答案：TaskA.1 道路长度
    res_length = []
    for road_name in cared_roads_name:
        road_length = int(cared_roads_name[road_name]["road_length_estimate"] / 100) * 100

        res = [max(road_length * 0.1, 100), max(road_length * 0.3, 100), road_length, road_length * 2, road_length * 3]
        random.shuffle(res)
        res_dict = dict(zip(["A", "B", "C", "D", "E"], res))
        label = None
        for k in res_dict:
            if res_dict[k] == road_length:
                label = k
        assert (label is not None)
        res_dict["question"] = "How long is {} road?".format(road_name)
        res_dict["answer"] = label

        res_length.append(res_dict)

    task_df = pd.DataFrame(data=res_length)
    task_df["is_seen"] = False
    task_df.to_csv(TASK_FILES["road"]["length"])

    # 生成评估问题及对应答案：TaskA.4 道路可达POI
    res_arrived_aois = []
    for rd_id in road_dict:
        road_name = road_dict[rd_id]['road_name']
        if road_name == '':
            continue
        arrived_aois = set(road_dict[rd_id]['aoi_ids']) & set(aoi_name_dict_select.keys())
        if len(arrived_aois) == 0:
            continue
        if len(arrived_aois) > 15:
            question_type = "FindCannot"
        else:
            question_type = "FindCan"

        if question_type == "FindCannot":
            res_temp = random.sample(list(arrived_aois), 15)
            res_temp_name = [aois_name_dict[aoi_id] for aoi_id in res_temp if aoi_id in aois_name_dict]
            not_temp = random.sample(list(aoi_ids_set.difference(set(arrived_aois))), 5)
            not_temp_name = [aois_name_dict[aoi_id] for aoi_id in not_temp if aoi_id in aois_name_dict]
            res = [res_temp_name[:5], res_temp_name[5:10], res_temp_name[10:], not_temp_name]

            res_q = ["A", "B", "C", "D"]
            random.shuffle(res_q)
            label = res_q[-1]
            res_dict = dict(zip(res_q, res))
            res_dict["question"] = "Which AOIs cannot be directly accessed via {}?".format(road_name)
            res_dict["answer"] = label
        else:
            try:
                res_temp = random.sample(list(arrived_aois), 5)
            except ValueError as e:
                continue
            res_temp_name = [aois_name_dict[aoi_id] for aoi_id in res_temp]
            not_temp = random.sample(list(aoi_ids_set.difference(set(arrived_aois))), 15)
            not_temp_name = [aois_name_dict[aoi_id] for aoi_id in not_temp]
            res = [not_temp_name[:5], not_temp_name[5:10], not_temp_name[10:], res_temp_name]

            res_q = ["A", "B", "C", "D"]
            random.shuffle(res_q)
            label = res_q[-1]
            res_dict = dict(zip(res_q, res))
            res_dict["answer"] = label
            res_dict["question"] = "Which AOIs can be directly accessed via {}?".format(road_name)

        res_arrived_aois.append(res_dict)

    task_df = pd.DataFrame(data=res_arrived_aois)
    task_df["is_seen"] = False
    print('road_aoi:ok!!!')
    task_df.to_csv(TASK_FILES["road"]["aoi"])


def generate_evaluation_task_road_junc(road_dict, all_lanes, all_juncs):
    cared_juncs = {}
    cared_roads_name = []

    # 获取所有在关心区域内的道路信息
    for road_id in road_dict:
        lane_ids = road_dict[road_id]["lane_ids"]
        road_name = road_dict[road_id]['road_name']

        if road_name == '':
            continue
        for i, lane_id in enumerate(lane_ids):
            if type(lane_id) == list:
                print("lane_id in lane_ids is List!!!!!")
                continue

            lane = all_lanes[lane_id]
            length = lane["length"]
            last_point = Point(lane["shapely_lnglat"].coords[-1])
        if road_name not in cared_roads_name:
            cared_roads_name.append(road_name)

    # 获取交叉口和道路基础信息
    for junc_id in all_juncs:
        lane_ids = all_juncs[junc_id]["lane_ids"]

        pre_road_names = []
        suc_road_names = []
        pre_road_ids = []
        suc_road_ids = []
        for lane_id in lane_ids:
            lane = all_lanes[lane_id]
            pre_lane_id = lane["predecessors"][0]['id']
            suc_lane_id = lane["successors"][0]['id']
            pre_lane = all_lanes[pre_lane_id]
            suc_lane = all_lanes[suc_lane_id]
            last_point = Point(lane["shapely_lnglat"].coords[-1])
            pre_road_id = pre_lane["parent_id"]
            suc_road_id = suc_lane["parent_id"]
            if pre_road_id in road_dict:
                pre_road_name = road_dict[pre_road_id]['road_name']
                if not isinstance(pre_road_name, str):
                    continue
                pre_road_names.append(pre_road_name)
                pre_road_ids.append(pre_road_id)
            else:
                continue
            if suc_road_id in road_dict:
                suc_road_name = road_dict[suc_road_id]['road_name']
                if not isinstance(suc_road_name, str):
                    continue
                suc_road_names.append(suc_road_name)
                suc_road_ids.append(suc_road_id)
            else:
                continue
        pre_road_name = set(pre_road_names)
        suc_road_name = set(suc_road_names)
        pre_road_id = set(pre_road_ids)
        suc_road_id = set(suc_road_ids)
        cared_juncs[junc_id] = {"pre_road_id": pre_road_id, "suc_road_id": suc_road_id, "pre_road_name": pre_road_name,
                                "suc_road_name": suc_road_name, "gps": last_point}

    # 补充道路交叉口名字，获取道路的起终点（交叉口）
    for junc in cared_juncs:
        road_list = list(cared_juncs[junc]["pre_road_name"])
        # road2 = cared_juncs[junc]["suc_road_name"]
        if len(road_list) <= 1:
            junc_name = ""
        elif len(road_list) == 2:
            junc_name = "the junction of {} and {}".format(road_list[0], road_list[1])
        elif len(road_list) == 3:
            junc_name = "the junction of {},{} and {}".format(road_list[0], road_list[1], road_list[2])
        cared_juncs[junc]["name"] = junc_name
        # if junc_name != "":
        #     print(junc, cared_juncs[junc]["name"])

    # TaskA.2
    road_junc_gen(cared_juncs, cared_roads_name)
    # TaskA.3
    road_linkage_gen(cared_juncs)


def landmark_gen(input_coor, aoi_name, aoi_name_dict_select, map, category_supported, cfg):
    """task2: 推断POI周边的anchor """
    lng, lat = input_coor
    x, y = map.lnglat2xy(lng, lat)
    radius = cfg.env_radius
    limit = cfg.get_limit
    resolution = 500
    # self,
    # center: Union[Tuple[float, float], Point],
    # radius: float,
    # urban_land_uses: Optional[List[str]] = None,
    # limit: Optional[int] = None,
    nearby_aois = []
    for category_prefix in category_supported.keys():
        aoi_list = map.query_aois(input_coor, radius, category_prefix, limit)
        aoi_list = [aoi[0]["id"] for aoi in aoi_list]
        aoi_list = list(set(aoi_list) & set(aoi_name_dict_select.keys()))
        aoi_list_name = [aoi_dict[aoi]['name'] for aoi in aoi_list]
        nearby_aois.extend(aoi_list_name)

    candidate_aoi = [aoi_name]
    for center in [(x - resolution, y - resolution), (x - resolution, y + resolution), (x + resolution, y - resolution),
                   (x + resolution, y + resolution)]:
        aoi_list = map.query_aois(center, radius, "", limit)
        aoi_list = [aoi[0]["id"] for aoi in aoi_list]
        aoi_list = list(set(aoi_list) & set(aoi_name_dict_select.keys()))
        if len(aoi_list) == 0:
            aoi_list.append(random.choice(list(set(aoi_name_dict_select.keys()).difference(set(aoi_list)))))
        aoi_list = [aoi_dict[aoi]['name'] for aoi in aoi_list]
        candidate_aoi.append(random.choice(aoi_list))
    res_str = [str(x) for x in candidate_aoi]
    random.shuffle(res_str)
    res_dict = dict(zip(["A", "B", "C", "D", "E"], res_str))

    for k in res_dict:
        if res_dict[k] == str(aoi_name):
            label = k
    res_dict[
        "question"] = "Which area of interest (AOI) is most likely to appear in the described environment among the following multiple AOIs? Environment:{}".format(
        ",".join(nearby_aois))
    res_dict["answer"] = label

    return res_dict


def road_junc_gen(cared_juncs, cared_roads_name):
    road_juncs = {}
    for junc in cared_juncs:
        road_list = cared_juncs[junc]["pre_road_name"]
        junc_name = cared_juncs[junc]["name"]
        junc_coord = cared_juncs[junc]["gps"]
        # road2 = cared_juncs[junc]["suc_road_name"]
        if len(road_list) <= 1:
            continue

        for road in list(road_list):
            if road not in road_juncs:
                road_juncs[road] = {"detail": [(junc_name, junc_coord)]}
            else:
                road_juncs[road]["detail"].append((junc_name, junc_coord))

    for rj in road_juncs:
        if len(road_juncs[rj]["detail"]) >= 2:
            # print(rj, road_juncs[rj])

            lng_list = []
            lat_list = []
            for item in road_juncs[rj]["detail"]:
                lng_list.append(item[1].x)
                lat_list.append(item[1].y)
            lng_max = np.ptp(lng_list)
            lat_max = np.ptp(lat_list)
            start_junc, end_junc = [p for p in road_juncs[rj]["detail"]][:2]
            if lng_max >= lat_max:
                # 经度差别比较大，东西向
                lng_list_index = list(np.argsort(lng_list))
                # print("lng_list_index", lng_list_index)
                for i, idx in enumerate(lng_list_index):
                    if idx == 0:
                        start_junc = road_juncs[rj]["detail"][i]
                    elif idx == (len(lng_list) - 1):
                        end_junc = road_juncs[rj]["detail"][i]
            else:
                # 纬度差别比较大，南北向
                lat_list_index = list(np.argsort(lat_list))
                # print("lat_list_index", lat_list_index)
                for i, idx in enumerate(lat_list_index):
                    if i == 0:
                        start_junc = road_juncs[rj]["detail"][idx]
                    elif i == (len(lat_list) - 1):
                        end_junc = road_juncs[rj]["detail"][idx]
            # print(rj, "start:", start_junc, "end:", end_junc)
            road_juncs[rj]["start_junc"] = start_junc
            road_juncs[rj]["end_junc"] = end_junc

    # 构造问题和答案
    res_road_endpoint = []
    for rj in road_juncs:
        # 只有一个路口的道路不构造问题，直接跳过
        if len(road_juncs[rj]["detail"]) < 2:
            continue

        # 正确答案
        res_label = [road_juncs[rj]["start_junc"][0], road_juncs[rj]["end_junc"][0]]

        res = [res_label]
        try:
            # 构造错误答案：随机组合所有路口
            current_road_name = rj
            start_road_names = random.sample(list(set(cared_roads_name).difference({rj})), 3)
            end_road_names = random.sample(list(set(cared_roads_name).difference(set([rj] + start_road_names))), 3)
            for i, (sr, er) in enumerate(zip(start_road_names, end_road_names)):
                start_juncs = ["the junction of {} and {}".format(current_road_name, sr),
                               "the junction of {} and {}".format(sr, current_road_name)]
                end_juncs = ["the junction of {} and {}".format(current_road_name, er),
                             "the junction of {} and {}".format(er, current_road_name)]
                for junc in start_juncs + end_juncs:
                    assert junc not in res_label, "确保合成的答案不与正确答案重合"
                res.append([random.choice(start_juncs), random.choice(end_juncs)])
        except:
            print("随机组合构造负样本出现问题，临时丢弃该数据")
            print("current_label:{}", res_label)
            print("negative_sample:{}", start_juncs + end_juncs)
            continue

        res_q = ["A", "B", "C", "D"]
        random.shuffle(res_q)
        res_dict = dict(zip(res_q, res))
        label = res_q[0]
        res_dict[
            "question"] = "Which of the following is the starting intersection and ending intersection of {}?".format(
            rj)
        res_dict["answer"] = label

        res_road_endpoint.append(res_dict)

    task_df = pd.DataFrame(data=res_road_endpoint)
    task_df["is_seen"] = False
    print('road_od:ok!!!')
    task_df.to_csv(TASK_FILES["road"]["od"])


def road_linkage_gen(cared_juncs):
    # 从路口格式整理为道路相邻格式
    road_linkage = {}
    for junc in cared_juncs:
        road_list = cared_juncs[junc]["pre_road_name"]
        # road2 = cared_juncs[junc]["suc_road_name"]
        if len(road_list) <= 1:
            continue

        for road in list(road_list):
            if road not in road_linkage:
                road_linkage[road] = set(road_list)
            else:
                road_linkage[road] = road_linkage[road].union(set(road_list))
    for road in road_linkage:
        road_linkage[road].remove(road)

    # 生成问题和答案
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
            res_dict = dict(zip(["A", "B", "C", "D"], res))
            for k in res_dict:
                if res_dict[k] == select[0]:
                    label = k
            res_dict["question"] = "Which road directly reach {}?".format(road_name)
            res_dict["answer"] = label
        road_nearby.append(res_dict)

    task_df = pd.DataFrame(data=road_nearby)
    task_df["is_seen"] = False
    print('road_link:ok!!!')
    task_df.to_csv(TASK_FILES["road"]["link"])


def aoi_near_gen(aoi_dict, rd_id):
    aois = road_dict[rd_id]['aoi_ids']
    junc_dict = {}  # 位于相同路口附近的AOI
    for aoi in aois:
        if aoi not in aoi_dict:
            continue
        address = aoi_dict[aoi]['Address']  # 从AOI的地址信息中提取出与其最近的相交道路名字
        if not isinstance(address, str):
            continue
        match_location = re.search(r'the junction of (.*?) and', address.split(',')[1])
        if match_location is None:
            continue
        another_rd = match_location.group(1)
        if another_rd not in junc_dict:
            junc_dict[another_rd] = []
        junc_dict[another_rd].append(aoi)
    for rd_name, aois in junc_dict.items():
        if len(aois) < 2:
            continue
        no_junc = set(list(aoi_name_dict_select.keys())).difference(set(aois))
        res = random.sample(list(no_junc), 3) + [aois[1]]
        random.shuffle(res)
        res = [aois_name_dict[aoi_id] for aoi_id in res]
        res_dict = dict(zip(["A", "B", "C", "D"], res))
        for k in res_dict:
            if res_dict[k] == aois_name_dict[aois[1]]:
                label = k
        res_dict["question"] = "Which AOI is adjacent to {}?".format(aois_name_dict[aois[0]])
        res_dict["answer"] = label
        return res_dict


def generate_evalation_task_node(aoi_dict, aoi_name_dict_select, aois_name_dict, map, category_supported, cfg):
    # TaskB.2: 推断AOI经纬度
    unseen_aois = []
    for p in aoi_name_dict_select:
        coords = aoi_dict[p]["coord"]
        lng, lat = coords[0]
        p_name = aoi_name_dict_select[p]
        unseen_aois.append(gps_gen((lng, lat), p_name, p))
    print('node_loc:ok!!!')
    save_data(unseen_aois, TASK_FILES["node"]["loc"])

    # 推断AOI的地址
    unseen_aois = []
    # def addr_gen(aoi_id=0):
    for p in aoi_name_dict_select:
        if p not in aoi_dict:
            continue
        unseen_aois.append(addr_gen(p, aoi_dict, aoi_name_dict_select))
    print('node_addr:ok!!!')
    save_data(unseen_aois, TASK_FILES["node"]["addr"])

    # 推断AOI的地址
    unseen_aois = []
    unseen_aois2 = []
    for p in aoi_name_dict_select:
        if aoi_dict[p]['type'] not in category_supported:
            continue
        unseen_aois.append(type_gen(p, aoi_dict, aoi_name_dict_select, category_supported)[0])
        unseen_aois2.append(type_gen(p, aoi_dict, aoi_name_dict_select, category_supported)[1])
    print('node_addr:ok!!!')
    save_data(unseen_aois, TASK_FILES["node"]["type"])
    save_data(unseen_aois2, TASK_FILES["node"]["type2"])

    # TaskC.1: 推断POI周边的anchor
    seen_aois = []
    unseen_aois = []
    ids = list(aoi_name_dict_select.keys())
    for i in range(cfg.question_num):
        aoi_id = random.choice(ids)
        x, y = aoi_dict[aoi_id]['coord'][0]
        p_name = aoi_dict[aoi_id]['name']
        unseen_aois.append(landmark_gen((x, y), p_name, aoi_name_dict_select, map, category_supported, cfg))
    print('node_env:ok!!!')
    save_data(unseen_aois, TASK_FILES["landmark"]["env"])

    # TaskC.1: 推断AOI的相邻关系
    unseen_aois = []
    junc_dict = {}  # 交叉口及附近的AOI
    for aoi in aoi_name_dict_select:
        address = aoi_dict[aoi]['Address']
        match = re.search(r'on the (\w+) side of ([^,]+),', address)
        match_location = re.search(r'the junction of (.*?) and', address.split(',')[1])
        if match_location is None or match is None:
            continue
        another_rd = match_location.group(1)
        rd_name = match.group(2)
        key = "{}_{}".format(rd_name, another_rd)
        if key not in junc_dict:
            junc_dict[key] = []
        junc_dict[key].append(aoi)
    for key in junc_dict:
        aois = junc_dict[key]
        if len(aois) < 2:
            continue
        no_junc = []
        for id in road_dict:
            tar_aois = set(road_dict[id]['aoi_ids']) & set(aoi_name_dict_select.keys())
            if len(set(aois) & set(road_dict[id]['aoi_ids'])) == 0 and len(tar_aois) >= 3:
                no_junc = random.sample(list(tar_aois), 3)
                break
        res = no_junc + [aois[1]]
        random.shuffle(res)
        res = [aois_name_dict[aoi_id] for aoi_id in res]
        res_dict = dict(zip(["A", "B", "C", "D"], res))
        for k in res_dict:
            if res_dict[k] == aois_name_dict[aois[1]]:
                label = k
        res_dict["question"] = "Which AOI is adjacent to {}?".format(aois_name_dict[aois[0]])
        res_dict["answer"] = label
        unseen_aois.append(res_dict)
    print('node_near:ok!!!')
    save_data(unseen_aois, TASK_FILES["node"]["near"])


def gps_gen(input_coor, aoi_name, aoi_id):
    """task1: 推断POI经纬度"""
    resolution = 0.01
    round_demical = 4

    lng, lat = input_coor
    lng, lat = round(lng, round_demical), round(lat, round_demical)

    # 保证经纬度位数在合理范围内
    res = [[lng, lat], [lng - resolution, lat - resolution], [lng - resolution, lat + resolution],
           [lng + resolution, lat - resolution], [lng + resolution, lat + resolution]]
    for i, coor in enumerate(res):
        coor[0] = round(coor[0], round_demical)
        coor[1] = round(coor[1], round_demical)
        res[i] = coor
    coor_str = ",".join([str(lng), str(lat)])
    res_str = [",".join([str(xx) for xx in x]) for x in res]
    random.shuffle(res_str)
    res_dict = dict(zip(["A", "B", "C", "D", "E"], res_str))

    for k in res_dict:
        if res_dict[k] == coor_str:
            label = k
    res_dict["answer"] = label
    res_dict["question"] = "What is the longitude and latigude coordinates of {}.".format(aoi_name)
    res_dict["poi_id"] = aoi_id

    return res_dict


def addr_gen(aoi_id, aoi_dict, aoi_name_dict_select):
    """推断AOI的地址"""
    addr = aoi_dict[aoi_id]['Address']
    aoi_name = aoi_dict[aoi_id]['name']
    res_ids = set(aoi_name_dict_select.keys()).difference({aoi_id})
    rd_belongs = []
    res_addrs = []
    for res_id in res_ids:
        addr = aoi_dict[res_id]['Address']
        match = re.search(r'on the (\w+) side of ([^,]+),', addr)  # 选取所属道路不同的AOI的地址作为其他错误选项
        if match:
            rd_belong = match.group(2)
            if rd_belong not in rd_belongs:
                res_addrs.append(aoi_dict[res_id]['Address'])
        if len(res_addrs) == 3:
            break
    res_addrs.append(addr)
    random.shuffle(res_addrs)
    res_dict = dict(zip(["A", "B", "C", "D"], res_addrs))
    for k in res_dict:
        if res_dict[k] == addr:
            label = k
    res_dict["answer"] = label
    res_dict["question"] = "What is the address of {}?".format(aoi_name)

    return res_dict


def type_gen(aoi_id, aoi_dict, aoi_name_dict_select, category_supported):
    """推断AOI的类型"""
    type_ = aoi_dict[aoi_id]['type']
    aoi_name = aoi_dict[aoi_id]['name']
    type_name = category_supported[type_]
    res_types = list(set(category_supported.keys()).difference(set([type_name])))
    res_type = random.choices(res_types, k=3)
    res_type_name = [category_supported[type] for type in res_type]
    res_type_name.append(type_name)
    random.shuffle(res_type_name)
    res_dict = dict(zip(["A", "B", "C", "D"], res_type_name))
    for k in res_dict:
        if res_dict[k] == type_name:
            label = k
    res_dict["answer"] = label
    res_dict["question"] = "What is the land use type of {}?".format(aoi_name)
    coords = aoi_dict[aoi_id]['coord'][0]
    res_aois = []
    ids = list(aoi_name_dict_select.keys())
    if len(res_aois) < 5:
        random.shuffle(ids)
        for id in ids:
            type = aoi_dict[id]['type']
        if type in res_types:
            res_aois.append(id)
    res_aois.append(aoi_id)
    res_aoi_name = [aoi_dict[id]['name'] for id in res_aois]
    random.shuffle(res_aoi_name)
    res_dict2 = dict(zip(["A", "B", "C", "D"], res_aoi_name))
    for k in res_dict2:
        if res_dict2[k] == aoi_name:
            label = k
    res_dict2["answer"] = label
    res_dict2["question"] = "Which of the following Area of Interests(AOI) has a land use type of {}?".format(type_name)

    return res_dict, res_dict2


def landmark_path(cfg, diag, aoi_dict):
    # diag[1]["content"]
    content = json.loads(diag[1]["content"])
    # TODO 临时处理，后续应当推广到没有去过的起终点来考察
    start_aoi_name = content['start_name']
    start_aoi_addr = content['start_addr']
    start_aoi = "{}({})".format(start_aoi_name, start_aoi_addr)
    # 提取"destination"和"by"之间的信息
    end_aoi_name = content['dest_name']
    end_aoi_addr = content['dest_addr']
    end_aoi = "{}({})".format(end_aoi_name, end_aoi_addr)
    aois_along_route = []
    for route in diag[2:]:
        if route["role"] == "user":
            continue
        route_content = route["content"].split('\n')
        aois_info = json.loads(route_content[-1])
        if not aois_info:
            continue
        aois = aois_info["aois"]
        aois = [aoi["name"] for aoi in aois]
        aois_along_route += aois
    aois_along_route = list(set(aois_along_route))
    all_aois = random.choices(list(aoi_dict.keys()), k=10)
    all_aois_name = set([aoi_dict[aoi_id]['name'] for aoi_id in all_aois])
    negative_samples = all_aois_name.difference(set(aois_along_route))
    negative_sample = random.choice(list(negative_samples))
    if len(aois_along_route) < 3:
        return None
    res_aois = random.choices(aois_along_route, k=3)
    res_aois.append(negative_sample)
    question = "Which of the following AOIs will not be passed when traveling from {} to {}?".format(
        start_aoi, end_aoi)
    answer = negative_sample
    res_dict = gen_options(res_aois, question, answer)
    return res_dict


def generate_evaluation_task_landmark(diags, aoi_dict, cfg):
    route_arrive_aois = []
    for diag in diags:
        content = json.loads(diag[1]["content"])
        steps = (len(content['routes']) + 1) / 2
        if steps > cfg.step:
            continue
        if not landmark_path(cfg, diag, aoi_dict):
            continue
        route_arrive_aois.append(landmark_path(cfg, diag, aoi_dict))
    save_data(route_arrive_aois, TASK_FILES["landmark"]["path"])
    print("districts_road_pois done!")


def generate_evaluation_task_boudary(road_dict, aois_name_dict, aoi2rd):
    # 构造问题和答案
    res_aoi_boundary = []
    candidates_roads = [road_dict[rd_id]['road_name'] for rd_id in list(road_dict.keys())]
    for aoi_id in aoi2rd:
        if aoi_id not in aois_name_dict:
            continue
        aoi_name = aois_name_dict[aoi_id]
        if len(aoi2rd[aoi_id]) == 0:
            continue
        boundary = aoi2rd[aoi_id]
        if len(boundary) >= 3:
            # 边界较多，询问哪个不是边界
            res = random.sample(boundary, 3)
            random.shuffle(candidates_roads)
            for x in candidates_roads:
                if x not in boundary:
                    negative_sample = x
                    res.append(negative_sample)
                    break
            res_q = ["A", "B", "C", "D"]
            random.shuffle(res_q)
            res_dict = dict(zip(res_q, res))
            label = res_q[-1]
            res_dict["question"] = "Which road is not the boundary of AOI {}".format(aoi_name)
            res_dict["answer"] = label
        else:
            # 边界较少，询问哪个是边界
            current_roads = list(set(candidates_roads).difference(set(boundary)))
            res = random.sample(current_roads, 3) + random.sample(boundary, 1)

            res_q = ["A", "B", "C", "D"]
            random.shuffle(res_q)
            res_dict = dict(zip(res_q, res))
            label = res_q[-1]
            res_dict["question"] = "Which road is the boundary of AOI {}".format(aoi_name)
            res_dict["answer"] = label
        res_aoi_boundary.append(res_dict)

    task_df = pd.DataFrame(data=res_aoi_boundary)
    task_df["is_seen"] = False
    print('boundary_rd:ok!!!')
    task_df.to_csv(TASK_FILES["boundary"]["road"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="shanghai")
    parser.add_argument("--evaluate_version", type=str, default="v83")
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    config = all_config[args.city]

    city_map = MAP_DICT[args.city]
    cache_dir = "../../data/map_cache/"
    resource_dir = "../../data/resource/"
    routing_path = "../../config/routing_linux_amd64"
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
        "road": {
            "length": "eval_road_length.csv",  # road
            "od": "eval_road_od.csv",  # road, node
            "link": "eval_road_link.csv",  # road
            "aoi": "eval_road_aoi.csv"  # road, node
        },
        "node": {
            "loc": "poi2coor.csv",  # node
            "near": "aoi_near.csv",
            "addr": "aoi2addr.csv",
            "type": "aoi2type.csv",
            "type2": "type2aoi.csv"
        },
        "landmark": {
            "env": "eval_landmark_env.csv",  # landmark, node
            "path": "eval_landmark_path.csv"  # landmark, node
        },
        "boundary": {
            "road": "eval_boundary_road.csv",  # boundary, districts, road
        },

        "navigation": {
            "navigation": ""  # navigation
        }
    }

    TASK_FILES = task_files_adaption(TASK_FILES_, output_path)
    road_message = pd.read_csv(os.path.join(resource_dir, "{}_roads.csv".format(config.region_exp)))
    aoi_message = pd.read_csv(os.path.join(resource_dir, "{}_aois.csv".format(config.region_exp)))

    if config.use_english:
        category_supported = {
            "E3": "OtherNon-construction", "R": "Residential", "S4": "TrafficStation&Park", "A4": "Sports",
            "B31": "Entertainment", "B1": "CommercialFacilities", "U9": "OtherPublicFacilities", "A3": "Education",
            "G1": "Park&GreenLand", "B": "CommercialService&IndustryFacilities", "B32": "Resort&Fitness",
            "B13": "Restaurant&Bar", "A9": "ReligiousFacilities", "A5": "Hospital"
        }
        version = "eng"
    else:
        category_supported = {
            "E3": "其他非建设用地", "R": "居住用地", "S4": "交通场站用地", "A4": "体育用地",
            "B31": "娱乐用地", "B1": "商业设施用地", "U9": "其他公用设施用地", "A3": "教育科研用地",
            "G1": "公园绿地", "B": "商业服务业设施用地", "B32": "康体用地",
            "B13": "饭店、餐厅、酒吧等用地", "A9": "宗教设施用地w", "A5": "医疗卫生用地"
        }
        version = "chi"
    all_roads = map.roads
    all_lanes = map.lanes
    all_aois = map.aois
    all_juncs = map.juncs

    aois_name_dict = aoi_message.set_index("aoi_id")["aoi_name"].to_dict()
    # aoi_id,type,land_use,coords,aoi_name,English_Address
    aoi_dict = {}
    for row in aoi_message.itertuples():
        aoi_id = row.aoi_id
        if config.use_english:
            name = row.aoi_name
            addr = row.English_Address
            extra = "nearby"
        else:
            name = row.aoi_name
            addr = row.Address
            extra = "周边|附近"
        if not isinstance(addr,str) or not isinstance(name,str):
            continue
        if aoi_id in aoi_dict or extra in row.aoi_name:
            continue
        aoi_dict[aoi_id] = {}
        aoi_dict[aoi_id]['coord'] = eval(row.coords)
        aoi_dict[aoi_id]['name'] = name
        aoi_dict[aoi_id]['Address'] = addr
        aoi_dict[aoi_id]['type'] = map.get_aoi(aoi_id)['urban_land_use']
        aoi_dict[aoi_id]['poi_ids'] = []

    aoi_name_dict_select = {}
    for id, aoi in aoi_dict.items():
        aoi_name_dict_select[id] = aoi['name']

    poi_dict = {}

    # road_id,road_name
    road_dict = {}
    for row in road_message.itertuples():
        rd_id = row.road_id
        if rd_id in road_dict:
            continue
        road_dict[rd_id] = {}
        road_dict[rd_id]['road_name'] = row.road_name
        if not row.road_name:
            continue
        road_dict[rd_id]['lane_ids'] = all_roads[rd_id]['lane_ids']
        road_dict[rd_id]['length'] = all_roads[rd_id]['length']
        road_dict[rd_id]['aoi_ids'] = []
        for lane_id in road_dict[rd_id]['lane_ids']:
            if lane_id in all_lanes:
                road_dict[rd_id]['aoi_ids'] += all_lanes[lane_id]['aoi_ids']

    road_name_dict = {}  # 使用道路名称进行匹配
    for row in road_message.itertuples():
        rd_id = row.road_id
        rd_name = row.road_name
        if not rd_name:
            continue
        if rd_name in road_name_dict:
            continue
        road_name_dict[rd_name] = {}
        road_name_dict[rd_name]['lane_ids'] = all_roads[rd_id]['lane_ids']
        road_name_dict[rd_name]['aoi_ids'] = []
        for lane_id in road_name_dict[rd_name]['lane_ids']:
            if lane_id in all_lanes:
                road_name_dict[rd_name]['aoi_ids'] += all_lanes[lane_id]['aoi_ids']

    aoi2rd = {}  # aoi_id附近的所有道路名字
    for rd_id in road_dict:
        if len(road_dict[rd_id]['aoi_ids']) == 0:
            continue
        for aoi_id in road_dict[rd_id]['aoi_ids']:
            if aoi_id not in aoi2rd:
                aoi2rd[aoi_id] = []
            aoi2rd[aoi_id].append(road_dict[rd_id]['road_name'])

    train_data = []

    print("start generate data")
    generate_evaluation_task_road(aoi_dict, road_dict, aois_name_dict, aoi_name_dict_select)
    generate_evaluation_task_road_junc(road_dict, all_lanes, all_juncs)
    generate_evalation_task_node(aoi_dict, aoi_name_dict_select, aois_name_dict, map, category_supported, config)
    generate_evaluation_task_boudary(road_dict, aois_name_dict, aoi2rd)
