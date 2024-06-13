import copy
import os
import random
import re
import pandas as pd
import numpy as np
import subprocess
from pycitysim.map import Map
from pycitysim.routing import RoutingClient


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

MAP_DICT={
    "beijing":"map_beijing5ring_withpoi_0424",
    "shanghai":"map_shanghai_20240525",
    "mumbai":"map_mumbai_20240525",
    "tokyo":"map_tokyo_20240526",
    "london":"map_london_20240526",
    "paris":"map_paris_20240512",
    "moscow":"map_moscow_20240526",
    "newyork":"map_newyork_20240512",
    "san_francisco":"map_san_francisco_20240526",
    "sao_paulo":"map_san_paulo_20240530",
    "nairobi":"map_nairobi_20240529",
    "cape_town":"map_cape_town_20240529",
    "sydney":"map_sydney_20240529"
}

MONGODB_URI = ""

def load_map(city_map, cache_dir, routing_path, port):
    m = Map(
            mongo_uri=f"{MONGODB_URI}",
            mongo_db="llmsim",
            mongo_coll=city_map,
            cache_dir=cache_dir,
        )
    route_command = f"{routing_path} -mongo_uri {MONGODB_URI} -map llmsim.{city_map} -cache {cache_dir} -listen localhost:{port}"
    cmd = route_command.split(" ")
    print("loading routing service")
    process = subprocess.Popen(args=cmd, cwd="./")
    routing_client = RoutingClient(f"localhost:{port}")

    return m, process, routing_client


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
    # TODO
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


def get_region_exp():
    # 定义实验区域region，限定可用POI，AOI和路网
    return {
        "wudaokou_small": [(116.293669, 39.99254), (116.359928, 39.99254), (116.359928,40.01967), (116.293669, 40.01967), (116.293669, 39.99254)],
        "wudaokou_large": [(116.26, 39.96), (116.40,39.96), (116.40, 40.03), (116.26, 40.03), (116.26, 39.96)],
        "wangjing": [(116.447387, 39.986287), (116.504592, 39.986287), (116.504592, 40.020057), (116.447387, 40.020057), (116.447387, 39.986287)],
        "dahongmen": [(116.37778, 39.838326), (116.450219, 39.838326), (116.450219, 39.863752), (116.37778, 39.863752), (116.37778, 39.838326)],
        "yuyuantan": [(116.287375, 39.908043), (116.368366, 39.908043), (116.368366, 39.942128), (116.287375, 39.942128), (116.287375, 39.908043)],
        "beijing":[(116.293669, 39.99254), (116.359928, 39.99254), (116.359928,40.01967), (116.293669, 40.01967), (116.293669, 39.99254)],
        "paris": [(2.2493, 48.8115), (2.4239, 48.8115), (2.4239, 48.9038), (2.2493, 48.9038), (2.2493, 48.8115)],
        "newyork": [(-74.0128, 40.7028), (-73.9445, 40.7028), (-73.9445, 40.7314), (-74.0128, 40.7314), (-74.0128, 40.7028)],
        "shanghai": [(121.4010, 31.2084), (121.5320, 31.2084), (121.5320, 31.2706), (121.4010, 31.2706), (121.4010, 31.2084)],
        "mumbai": [(72.8321, 19.0448), (72.9202, 19.0448), (72.9202, 19.0975), (72.8321, 19.0975), (72.8321, 19.0448)],
        "tokyo": [(139.7367, 35.6714), (139.7871, 35.6714), (139.7871, 35.6958), (139.7367, 35.6958), (139.7367, 35.6714)],
        "london": [(-0.1868, 51.4874), (-0.0748, 51.4874), (-0.0748, 51.5194), (-0.1868, 51.5194), (-0.1868, 51.4874)],
        "moscow": [(37.5025, 55.6834), (37.7593, 55.6834), (37.7593, 55.8136), (37.5025, 55.8136), (37.5025, 55.6834)],
        "san_francisco": [(-122.4526, 37.7482), (-122.3915, 37.7482), (-122.3915, 37.7726), (-122.4526, 37.7726), (-122.4526, 37.7482)],
        "sao_paulo": [(-46.6490, -23.5599), (-46.6133, -23.5599), (-46.6133, -23.5423), (-46.6490, -23.5423), (-46.6490, -23.5599)],
        "nairobi": [(36.7915, -1.2970), (36.8454, -1.2970), (36.8454, -1.2692), (36.7915, -1.2692), (36.7915, -1.2970)],
        "cape_town": [(18.4021, -33.9690), (18.5216, -33.9690), (18.5216, -33.9189), (18.4021, -33.9189), (18.4021, -33.9690)],
        "sydney": [(151.1489, -33.9132), (151.2443, -33.9132), (151.2443, -33.8707), (151.1489, -33.8707), (151.1489, -33.9132)]
    }
