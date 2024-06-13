import asyncio
import subprocess
from typing import Tuple, cast,Optional, Union, Dict, Any
import math
from shapely.geometry import Point, Polygon

from pycityproto.city.geo.v2.geo_pb2 import (
    AoiPosition,
    LanePosition,
    Position,
    XYPosition,
    LongLatPosition,
)
import pycityproto.city.routing.v2.routing_pb2 as routing_pb
import pycityproto.city.routing.v2.routing_service_pb2 as routing_service
from pycitysim.map import Map
from pycitysim.routing import RoutingClient

from typing import Any, Awaitable, TypeVar
from google.protobuf.message import Message
from google.protobuf.json_format import MessageToDict

from openai import OpenAI #type
import httpx
import time

PROXY = "http://127.0.0.1:10190"
# your key for LLM APIs
OPENAI_APIKEY = ""
DEEPINFRA_APIKEY = ""
DEEPSEEK_APIKEY = ""
MONGODB_URI = ""

T = TypeVar("T", bound=Message)
def parse(res: T, dict_return: bool) -> Optional[Dict[str, Any]]:
    if dict_return:
        return MessageToDict(
            res,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )
    else:
        return res


async def async_parse(res: Awaitable[T], dict_return: bool) -> Optional[Dict[str, Any]]:
    if dict_return:
        return MessageToDict(
            await res,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )
    else:
        return await res


def calculate_direction_and_distance(start_x, start_y, end_x, end_y):
    # 计算距离
    distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    # 计算方位角
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    if delta_x == 0 and delta_y == 0:
        # 如果起点和终点相同，则方位无法确定
        direction = "同一位置"
    elif delta_x == 0:
        # 如果只在y轴上移动，则方位为北或南
        direction = "北" if delta_y > 0 else "南"
    else:
        # 计算方位角的弧度
        angle = math.atan(abs(delta_y / delta_x))
        # 将弧度转换为度
        angle_deg = math.degrees(angle)
        # 根据象限确定方位
        if delta_x > 0 and delta_y > 0:  # 第一象限
            direction = "东北" if angle_deg < 45 else "北"
        elif delta_x < 0 and delta_y > 0:  # 第二象限
            direction = "西北" if angle_deg < 45 else "北"
        elif delta_x < 0 and delta_y < 0:  # 第三象限
            direction = "西北" if angle_deg > 45 else "西"
        elif delta_x > 0 and delta_y < 0:  # 第四象限
            direction = "东北" if angle_deg > 45 else "东"

    return direction, distance

def get_distance(start_x,start_y,end_x,end_y):
    return math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

def filter_road_info(roads):
    # 初始化一个新的字典来存储结果
    filtered_roads = {}

    # 遍历输入的字典
    for road_id, info in roads.items():
        road_name, direction, distance = info
        # 创建一个复合键由road_name和direction组成
        key = (road_name, direction)

        # 如果这个复合键还没有在结果字典中，或者找到了更小的distance，则更新结果字典
        if key not in filtered_roads or distance < filtered_roads[key][2]:
            filtered_roads[key] = [road_name, direction, distance, road_id]

    # 由于我们存储了额外的road_id，我们需要重新整理结果字典，只保留road_id
    result = {info[-1]: info[:-1] for info in filtered_roads.values()}

    return dict(result)

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

def load_map(city_map, cache_dir, routing_path, port):
    m = Map(
            mongo_uri=MONGODB_URI,
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


class Player:
    def __init__(
        self,
        city_map: Map,
        city_routing_client: RoutingClient,
        init_aoi_id: int,
        search_type: str
    ):
        self._city_map = city_map
        self.search_type = search_type
        self._city_routing_client = city_routing_client

        self.init_position(init_aoi_id)
        self.time_cost = 0 
        self.price_cost = 0 

        self.current_road_list = []

    def init_position(self, init_aoi_id):
        aoi = self._city_map.get_aoi(init_aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {init_aoi_id} not found")
        xy = cast(Polygon, aoi["shapely_xy"]).centroid
        x, y = xy.coords[0][0], xy.coords[0][1]
        lng, lat = self._city_map.xy2lnglat(x, y)
        if self._city_map.get_aoi(init_aoi_id)["driving_positions"]:
            lane_pos = self._city_map.get_aoi(init_aoi_id)["driving_positions"][0]
        else:
            lane_pos={"lane_id":0,"s":0}

        self.position = Position(
            aoi_position=AoiPosition(aoi_id=init_aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
            lane_position=LanePosition(lane_id=lane_pos["lane_id"], s=lane_pos["s"])
        )
    

    def get_position(self):
        """
        获取当前位置
        """
        return parse(self.position, True)
    
    def lnglat2xy(self, lng: float, lat: float) -> Tuple[float, float]:
        """
        经纬度转xy坐标
        Convert latitude and longitude to xy coordinates

        Args:
        - lng (float): 经度。longitude.
        - lat (float): 纬度。latitude.

        Returns:
        - Tuple[float, float]: xy坐标。xy coordinates.
        """
        return self.projector(lng, lat)

    def search(
        self,
        center:  Union[Tuple[float, float], Point],
        radius: float,
        category_prefix: str,
        limit: int = 10,
    ):
        """
        搜索给定范围内的POI
        """
        if self.search_type=="poi":
            return self._city_map.query_pois(center, radius, category_prefix, limit)
        elif self.search_type=="aoi":
            return self._city_map.query_aois(center, radius, category_prefix, limit)

    async def get_walking_route(self, aoi_id: int):
        """
        获取步行路线和代价
        """
        print(f"get_walking_route: {self.position.aoi_position.aoi_id} -> {aoi_id}")
        resp = await self._city_routing_client.GetRoute(
            routing_service.GetRouteRequest(
                type=routing_pb.ROUTE_TYPE_WALKING,
                start=self.position,
                end=Position(aoi_position=AoiPosition(aoi_id=aoi_id)),
            ),
            dict_return=False,
        )
        resp = cast(routing_service.GetRouteResponse, resp)
        if len(resp.journeys) == 0:
            return None
        return parse(resp.journeys[0].walking, True)

    async def get_driving_route(self, aoi_id: int):
        """
        获取开车路线和代价
        """
        # print(f"get_driving_route: {self.position.aoi_position.aoi_id} -> {aoi_id}")
        resp = await self._city_routing_client.GetRoute(
            routing_service.GetRouteRequest(
                type=routing_pb.ROUTE_TYPE_DRIVING,
                start=self.position,
                end=Position(aoi_position=AoiPosition(aoi_id=aoi_id)),
            ),
            dict_return=False,
        )
        resp = cast(routing_service.GetRouteResponse, resp)
        if len(resp.journeys) == 0:
            return None
        return parse(resp.journeys[0].driving, True)

    async def walk_to(self, aoi_id: int) -> bool:
        """
        步行到POI，实际产生移动，更新人物位置
        """
        # 起终点相同，直接返回True
        if aoi_id == self.position.aoi_position.aoi_id:
            return True
        
        route = await self.get_walking_route(aoi_id)
        if route is None:
            return False
        last_lane_id = route["route"][-1]["lane_id"]
        # 检查对应的AOI Gate
        aoi = self._city_map.get_aoi(aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {aoi_id} not found")
        gate_index = -1
        for i, p in enumerate(aoi["walking_positions"]):
            if p["lane_id"] == last_lane_id:
                gate_index = i
                break
        if gate_index == -1:
            raise ValueError(
                f"aoi {aoi_id} has no walking gate for lane {last_lane_id}"
            )
        # 更新人物位置
        gate_xy = aoi["walking_gates"][gate_index]
        x, y = gate_xy["x"], gate_xy["y"]
        lng, lat = self._city_map.xy2lnglat(x, y)
        self.position = Position(
            aoi_position=AoiPosition(aoi_id=aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )
        self.time_cost += route["eta"]
        self.price_cost += 0
        return True

    async def drive_to(self, aoi_id: int):
        """
        开车到POI，实际产生移动，更新人物位置
        """
         # 起终点相同，直接返回True
        if aoi_id == self.position.aoi_position.aoi_id:
            return True
        
        route = await self.get_driving_route(aoi_id)
        if route is None:
            return False
        last_road_id = route["road_ids"][-1]
        # 检查对应的AOI Gate
        aoi = self._city_map.get_aoi(aoi_id)
        if aoi is None:
            raise ValueError(f"aoi {aoi_id} not found")
        gate_index = -1
        for i, p in enumerate(aoi["driving_positions"]):
            lane_id = p["lane_id"]
            lane = self._city_map.get_lane(lane_id)
            if lane is None:
                raise ValueError(f"lane {lane_id} not found")
            road_id = lane["parent_id"]
            if road_id == last_road_id:
                gate_index = i
                break
        if gate_index == -1:
            raise ValueError(
                f"aoi {aoi_id} has no driving gate for road {last_road_id}"
            )
        # 更新人物位置
        gate_xy = aoi["driving_gates"][gate_index]
        x, y = gate_xy["x"], gate_xy["y"]
        lng, lat = self._city_map.xy2lnglat(x, y)
        self.position = Position(
            aoi_position=AoiPosition(aoi_id=aoi_id),
            xy_position=XYPosition(x=x, y=y),
            longlat_position=LongLatPosition(longitude=lng, latitude=lat),
        )
        self.time_cost += route["eta"]
        self.price_cost += route["eta"]
        return True

    def get_aoi_of_poi(self, poi_id):
        if poi_id in self._city_map.pois:
            aoi_id = self._city_map.get_poi(poi_id)["aoi_id"]
            return aoi_id
        else:
            print("POI:{}的没有AOI归属信息信息".format(poi_id))
            return None
    
    async def set_routing_list(self, poi_id):
        aoi_id = self.get_aoi_of_poi(poi_id)
        self.current_road_list = await self.get_driving_route(aoi_id)
    
    async def move_step_by_step(self):
        """执行单步移动操作"""
        
        # 获取当前道路
        current_lane_id = self.position.lane_position.lane_id
        parent_id = self._city_map.get_lane(current_lane_id)["parent_id"]
        try:
            if "external" in self._city_map.get_road(parent_id):
                name = self._city_map.get_road(parent_id)["external"]["name"]
            else:
                name = self._city_map.get_road(parent_id)["name"]
            print("current_road:",name)
        except:
            print("notavailable")

        try:
            pre_lane_id = self._city_map.get_lane(current_lane_id)["predecessors"][0]["id"]
        except IndexError as e:
            return (False, "No avaiable lanes")
        
        pre_lane_info = self._city_map.get_lane(pre_lane_id)

        # 获取路口
        junc_id = pre_lane_info["parent_id"]
        junc_info = self._city_map.get_junction(junc_id)

        # 获取可行路口
        avaiable_lanes = []
        for junc_lane_id in junc_info["lane_ids"]:
            junc_lane_info = self._city_map.get_lane(junc_lane_id)

            for predecessor in junc_lane_info["predecessors"]:
                junc_pre_lane_id = predecessor["id"]
                avaiable_lanes.append(junc_pre_lane_id)
        avaiable_road_names = {}
        for lane_id in avaiable_lanes:
            parent_id = self._city_map.get_lane(lane_id)["parent_id"]
            try:
                #name = self._city_map.get_road(parent_id)["external"]["name"]
                name = self._city_map.get_road(parent_id)["name"]
            except:
                name = "unknown"
            avaiable_road_names[lane_id] = name
        
        # 选择一条路前行
        next_lane_id = avaiable_lanes[0]

        lane_info = self._city_map.get_lane(next_lane_id)
        endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
        endpoint_xy = lane_info["shapely_xy"].coords[-1]

        # 发生移动，更新位置
        self.position = Position(
            xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
            longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            lane_position=LanePosition(lane_id=next_lane_id, s=0)
        )

        return (True, "Move One Step")
    
    def get_junction_list(self):
        # 获取当前道路与当前位置
        current_lane_id = self.position.lane_position.lane_id
        current_xy=dict(self.get_position()["xy_position"])
        current_lnglat=dict(self.get_position()["longlat_position"])
        #print("cp",current_lnglat)
        #print("current_lane_id",current_lane_id)
        try:
            pre_lane_id = self._city_map.get_lane(current_lane_id)["predecessors"][0]["id"]
        except IndexError as e:
            return (False, "No avaiable lanes")
        
        pre_lane_info = self._city_map.get_lane(pre_lane_id)

        # 获取路口
        junc_id = pre_lane_info["parent_id"]
        junc_info = self._city_map.get_junction(junc_id)

        # 获取可行路口
        avaiable_lanes = []
        for junc_lane_id in junc_info["lane_ids"]:
            junc_lane_info = self._city_map.get_lane(junc_lane_id)
            for predecessor in junc_lane_info["predecessors"]:
                junc_pre_lane_id = predecessor["id"]
                #此处添加对lane方向与距离的计算
                lane_info = self._city_map.get_lane(junc_pre_lane_id)
                #1.得到行走一步以后的位置
                endpoint_lnglat_temp = lane_info["shapely_lnglat"].coords[-1]
                endpoint_lnglat={"longitude":endpoint_lnglat_temp[0], "latitude":endpoint_lnglat_temp[1]}
                endpoint_xy_temp = lane_info["shapely_xy"].coords[-1]
                endpoint_xy={'x':endpoint_xy_temp[0],'y':endpoint_xy_temp[1]}
                #计算lane对应的方向与距离
                dir,dis=calculate_direction_and_distance(current_xy['x'], current_xy['y'], endpoint_xy['x'],endpoint_xy['y'])

                avaiable_lanes.append([junc_pre_lane_id,dir,dis])
        available_road_names = {}
        for lane in avaiable_lanes:
            parent_id = self._city_map.get_lane(lane[0])["parent_id"]
            try:
                #name = self._city_map.get_road(parent_id)["external"]["name"]
                name = self._city_map.get_road(parent_id)["name"]
            except:
                name = "unknown"
            available_road_names[lane[0]] = [name,lane[1],lane[2]]
        #只保留距离最短的lane
        filtered_info=filter_road_info(available_road_names)
        return filtered_info

    #player根据选择前进
    def move_after_decision(self,next_lane_id):
        lane_info = self._city_map.get_lane(next_lane_id)
        endpoint_lnglat = lane_info["shapely_lnglat"].coords[-1]
        endpoint_xy = lane_info["shapely_xy"].coords[-1]

        # 发生移动，更新位置
        self.position = Position(
            xy_position=XYPosition(x=endpoint_xy[0], y=endpoint_xy[1]),
            longlat_position=LongLatPosition(longitude=endpoint_lnglat[0], latitude=endpoint_lnglat[1]),
            lane_position=LanePosition(lane_id=next_lane_id, s=0)
        )

    def check_position(self,end_xy,thres):
        current_xy=dict(self.get_position()["xy_position"])
        start_x=current_xy['x']
        start_y=current_xy['y']
        end_x=end_xy['x']
        end_y=end_xy['y']
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        #print("updated distance",distance)
        """确认是否已到达指定位置"""
        if distance<thres:
            return 0
        else:
            return distance


    def get_cur_position(self):
        current_xy=dict(self.get_position()["xy_position"])
        start_x=current_xy['x']
        start_y=current_xy['y']
        if self.search_type=="aoi":
            list_temp=self.search([start_x,start_y],1000,["E3"],2)
        elif self.search_type=="poi":
            list_temp=self.search([start_x,start_y],1000,"",2)
        poi_list=[]
        for items in list_temp:
            poi_list.append(items[0]['name'])
        return poi_list



def get_agent_action_simple(session, model_name="gpt-3.5", temperature=1.0, max_tokens=200, infer_server=None):

    # 模型API设置
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        model_name_map = {
            "gpt-3.5": "gpt-3.5-turbo-0125",
            "gpt-4": "gpt-4-0125-preview"
        }
        model_name = model_name_map[model_name]
        client = OpenAI(
            http_client=httpx.Client(proxy=PROXY),
            api_key=OPENAI_APIKEY
            )
    elif "Llama-3" in model_name or "mistralai" in model_name:
        client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=DEEPINFRA_APIKEY,
        http_client=httpx.Client(proxies=PROXY),
            )
    elif "deepseek-chat" in model_name:
        client = OpenAI(
        api_key=DEEPSEEK_APIKEY,
        base_url="https://api.deepseek.com/v1"
        )
    else:
        raise NotImplementedError

    MAX_RETRIES = 1
    WAIT_TIME = 1
    for i in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                        model=model_name,
                        messages=session,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
            return response.choices[0].message.content
        except Exception as e:
            if i < MAX_RETRIES - 1:
                time.sleep(WAIT_TIME)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "OpenAI API Error."


def get_system_prompts():
    system_prompt = """
    Your navigation destination is {}. 
    You are now on {}, two nearby POIs are:{} and{}
    Given the available options of road and its corresponding direction to follow and correspoding direction,
    directly choose the option that will help lead you to the destination.
    """
    return system_prompt

def get_system_prompts2():
    system_prompt = """
    Your navigation destination is {}. 
    You are now near two POIs :{} and{}
    Given the available options of road and its corresponding direction to follow and correspoding direction,
    directly choose the option that will help lead you to the destination.
    """
    return system_prompt


def get_user_prompt():
    user_prompt="the options are:{}.Directly make a choice."
    return user_prompt


def transform_road_data(roads):
    # 创建一个新的字典用于存储转换后的数据
    transformed_roads = {}
    # 遍历原始字典
    for road_id, details in roads.items():
        road_name, direction, distance = details

        # 如果road_name为空或者为unknown，则跳过
        if not road_name or road_name.lower() == "unknown":
            continue

        # 使用[road_name, direction]作为新字典的键，road_id作为值
        transformed_roads[(road_name, direction)] = road_id

    return transformed_roads

def print_dict_with_alphabet(d):
    # 获取大写英文字母列表
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # 确保字典长度不超过字母表长度
    if len(d) > len(alphabet):
        raise ValueError("字典长度超过字母表长度")
    
    # 依次输出
    output = []
    for i, key in enumerate(d):
        output.append(f"{alphabet[i]} {key}")
    
    return "\n".join(output)

def transform_dict_keys_to_alphabet(d):
    # 获取大写英文字母列表
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # 确保字典长度不超过字母表长度
    if len(d) > len(alphabet):
        raise ValueError("字典长度超过字母表长度")
    
    # 创建一个新的字典，将大写字母作为键，输入字典的键作为值
    transformed_dict = {alphabet[i]: key for i, key in enumerate(d)}
    
    return transformed_dict


import re
from thefuzz import process
def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^A-H]{0,20}?(?:n't|not))[^A-H]{0,10}?\b(?:|is|:|be))\b)[^A-H]{0,20}?\b([A-H])\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b([A-H])\b(?![^A-H]{0,8}?(?:n't|not)[^A-H]{0,5}?(?:correct|right))[^A-H]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^([A-H])(?:\.|,|:|$)", gen)

    # simply extract the first appeared letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])([A-H])(?![a-zA-Z=])", gen)

    if res is None:
        return choice_list[choice_list.index(process.extractOne(gen, choice_list)[0])]
    
    return res.group(1)



########################################3
def get_performance(m,routing_client,init_id,init_name,destination_id,destination_name,m_name,server,temperature,thres,round,step=15, search_type="aoi"):
    total_step=0
    success_time=0
    temp=0
    for j in range(round):
        # initial Player
        player = Player(city_map=m, city_routing_client=routing_client, init_aoi_id=init_id, search_type=search_type)
        player2=Player(city_map=m, city_routing_client=routing_client, init_aoi_id=destination_id, search_type=search_type)
        end_xy=dict(player2.get_position()["xy_position"])  
        start_xy=dict(player.get_position()["xy_position"])
        initial_dis=get_distance(start_xy['x'], start_xy['y'], end_xy['x'],end_xy['y'])
        if(initial_dis>2000):
            break
        shortest_dis=initial_dis
        #max step
        for i in range(step):
            cur_roadid=player._city_map.get_lane(player.position.lane_position.lane_id)["parent_id"]
            road_info = player._city_map.get_road(cur_roadid)
            if road_info and "external" in road_info:
                current_roadname=player._city_map.get_road(cur_roadid)["external"]["name"]
            elif road_info and "name" in road_info:
                current_roadname=player._city_map.get_road(cur_roadid)["name"]
            else:
                print("not a road,out of border,task failed!")
                break
            current_poi_list=player.get_cur_position()
            if len(current_poi_list)<2:
                print("no more pois,out of border,task failed!")
                break
            diag= [dict(role="system", content=get_system_prompts().format(destination_name,current_roadname,current_poi_list[0],current_poi_list[1]))]

            road_list=player.get_junction_list()

            if isinstance(road_list, tuple):
                print("no more road,out of border,task failed!")
                break
            road=transform_road_data(road_list)
            candidate=print_dict_with_alphabet(road)
            result_index=transform_dict_keys_to_alphabet(road_list)
            diag.append(dict(role="user", content=get_user_prompt().format(candidate)))

            res=get_agent_action_simple(
            session=diag,
            model_name=m_name,
            infer_server=server,
            temperature=temperature
            )

            lane_choice=extract_choice(res,choice_list = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N'])
            if lane_choice in result_index.keys():
                next_lane_id=result_index[lane_choice]
            else:
                print("invalid answer!")
                break

            player.move_after_decision(next_lane_id)
            current_dis=player.check_position(end_xy,thres)
            if not current_dis:
                shortest_dis=0
                total_step=total_step+i+1
                success_time=success_time+1
                print("successfully found!,totalstep={}".format(i+1))
                break
            else:
                shortest_dis=min(shortest_dis,current_dis)
        temp=temp+shortest_dis
    if success_time:
        average_step=(total_step+step*(round-success_time))/round
    else:
        average_step=step
    completion=1-temp/(round*initial_dis)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("case:{}to{}".format(init_name,destination_name))
    print("model{}:success_time:{},average step:{},level of completion{}".format(m_name,success_time,average_step,completion))
    return m_name,success_time,average_step,completion
