import copy
import os
import random
import re
import pandas as pd
import numpy as np
import subprocess
from pycitydata.map import Map
from citysim.routing import RoutingClient
from config import MONGODB_URI

primary_directions = ['east', 'south', 'west', 'north']
secondary_directions = ['southeast', 'northeast', 'southwest', 'northwest']
EW = {'east', 'west'}
NS = {'south', 'north'}
dir_map = {"north": "south-north", "south": "south-north", "west": "east-west", "east": "east-west"}
dir_map2 = {"south-north": "east-west", "east-west": "south-north"}

secondary_dir_to_primary_dirs = {
    "southeast": ("south", "east"),
    "northeast": ("north", "east"),
    "northwest": ("north", "west"),
    "southwest": ("south", "west"),
}


def task_files_adaption(task_file, output_path):
    task_files = copy.deepcopy(task_file)
    path_prefix = output_path
    for k in task_files:
        for kk in task_files[k]:
            if path_prefix not in task_files[k][kk]:
                task_files[k][kk] = os.path.join(path_prefix, task_files[k][kk])
    os.makedirs(path_prefix, exist_ok=True)
    return task_files


def gen_options(options, question, answer):
    """
    生成字典:
    {"A": "选项1", "B": "选项2”, ..., "question": $question$, "answer": answer对应的选项}
    """
    options = copy.copy(options)
    random.shuffle(options)
    result = {}
    start_option_ascii = ord('A')
    for index, option in enumerate(options):
        selection = chr(start_option_ascii + index)
        result[selection] = option
        if option == answer:
            result["answer"] = selection

    if "answer" not in result:
        raise LookupError("未找到option=answer")

    result["question"] = question

    return result

def save_data(unseen_aois, save_path):
    unseen_aois_df = pd.DataFrame(data=unseen_aois)
    unseen_aois_df["is_seen"] = False

    task_df = pd.concat([unseen_aois_df])
    task_df.to_csv(save_path)

def dir_all_dis(routes, secondary_directions, primary_directions,secondary_dir_to_primary_dirs):
    """
    计算输入数据包含的移动方向以及每个方向移动的总距离
    """
    distances = []
    directions = []
    dir_dis_dep = []
    dir_dis = []
    for cnt, route in enumerate(routes):
        if route['type'] == "junc":
            continue
        distance = route['road_length']
        direction = route['direction'].split()[-1]
        distances.append(distance)
        directions.append(direction)

    for cnt2, direction in enumerate(directions):
        if direction in secondary_directions:
            distance = int(distances[cnt2]) * 0.7
            distance_str = str(distances[cnt2]) + "m,equals to ({},{}m) and ({},{}m)".format(direction[0],
                                                                                        "0.7*" + str(distances[
                                                                                            cnt2]) + '=' + str(
                                                                                            distance), direction[1],
                                                                                        "0.7*" + str(distances[
                                                                                            cnt2]) + '=' + str(
                                                                                            distance))
            dir_dis_dep.append((secondary_dir_to_primary_dirs[direction][0], distance))
            dir_dis_dep.append((secondary_dir_to_primary_dirs[direction][1], distance))
            dir_dis.append((direction, distance_str))
        elif direction in primary_directions:
            distance = int(distances[cnt2])
            distance_str = str(distances[cnt2]) + 'm'
            dir_dis_dep.append((direction, distance))
            dir_dis.append((direction, distance_str))
        else:
            print(direction)

    # 遍历原始列表，将相同键值的元素进行累加处理
    mid = {}
    for cnt, ddlist in enumerate(dir_dis_dep):
        dir, dis = ddlist
        if dir not in mid:
            mid[dir] = 0
        mid[dir] += dis
    dir_dis_fin = [(key, value) for key, value in mid.items()]  # 起始点到终点各个方向位移距离，[(方向，位移距离),(),...]
    dirs = set()
    for dir, dis in dir_dis_fin:
        dirs.add(dir)
    short_dir = list(set(primary_directions).difference(dirs))
    if len(short_dir) > 0:
        for dir in short_dir:
            dir_dis_fin.append((dir, 0))
    return dir_dis_fin, dir_dis

def compute_length(routine):  # 计算导航路径的总长度
    float_values = re.findall(r'for (\d+) meters', routine)
    length = np.sum([int(num) for num in float_values])
    return length


def angle2dir(angle):
    Direction = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
    s = 22.5
    for i in range(8):
        if angle < s + 45 * i:
            return Direction[i]
    return Direction[0]


def angle2dir_4(angle):
    Direction = ['north', 'east', 'south', 'west']
    if angle < 45 or angle >= 315:
        return Direction[0]
    elif 45 <= angle < 135:
        return Direction[1]
    elif 135 <= angle < 225:
        return Direction[2]
    else:
        return Direction[3]


def get_landuse_dict():
    landuse_dict = {
            "E3":"OtherNon-construction", "R":"Residential", "S4":"TrafficStation&Park", "A4":"Sports", "B31":"Entertainment",  "U9":"OtherPublicFacilities", "A3":"Education","G1":"Park&GreenLand","B":"CommercialService&IndustryFacilities","B32":"Resort&Fitness","B13":"Restaurant&Bar","A9":"ReligiousFacilities","A5":"Hospital"
            }

    return landuse_dict

def get_category_supported():
    category_supported = {"leisure":"leisure", "amenity":"amenity", "building":"building"}
    return category_supported
