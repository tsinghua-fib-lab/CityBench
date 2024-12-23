import subprocess
import math
import os
import json
from pycitydata.map import Map
from citysim.routing import RoutingClient

from citysim.player import Player
from serving.llm_api import get_chat_completion, extract_choice, get_model_response_hf
from config import MONGODB_URI, MAP_DICT, RESULTS_PATH


def get_distance(start_x,start_y,end_x,end_y):
    return math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)


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
    transformed_roads = {}
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


########################################3
def get_performance(m,routing_client,init_id,init_name,destination_id,destination_name,m_name,log_file, temperature,thres,round,step=15, search_type="poi", model=None):
    total_step=0
    success_time=0
    temp=0
    
    all_session_logs = []
    for j in range(round):
        # initial Player
        session_logs = []
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
            
            session_logs.append({"role": "system", "content": get_system_prompts().format(destination_name,current_roadname,current_poi_list[0],current_poi_list[1])})

            road_list=player.get_junction_list()

            if isinstance(road_list, tuple):
                print("no more road,out of border,task failed!")
                break
            road=transform_road_data(road_list)
            candidate=print_dict_with_alphabet(road)
            result_index=transform_dict_keys_to_alphabet(road_list)
            diag.append(dict(role="user", content=get_user_prompt().format(candidate)))

            session_logs.append({"role": "user", "content": get_user_prompt().format(candidate)})

            if model is not None:
                text = get_system_prompts().format(destination_name,current_roadname,current_poi_list[0],current_poi_list[1]) + "\n" + get_user_prompt().format(candidate)
                res = get_model_response_hf(text, model)
            else:
                res, _=get_chat_completion(
                    session=diag,
                    model_name=m_name,
                    temperature=temperature
                    )
                
            session_logs.append({"role": "response", "content": res})
            
            lane_choice=extract_choice(res,choice_list = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N'])

            session_logs.append({"role": "extract", "content": lane_choice})
            
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
                print("successfully found! totalstep={}".format(i+1))
                session_logs.append({"role": "system", "content": "successfully found! totalstep={}".format(i+1)})
                break
            else:
                shortest_dis=min(shortest_dis,current_dis)
        temp=temp+shortest_dis
        all_session_logs.extend(session_logs)
        
    if success_time:
        average_step=(total_step+step*(round-success_time))/round
    else:
        average_step=step
    completion=1-temp/(round*initial_dis)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("case:{}to{}".format(init_name,destination_name))
    print("model: {}: success_time:{}, average step:{}, level of completion{}".format(m_name,success_time,average_step,completion))
    if all_session_logs != []:
        with open(log_file, 'a') as f:
            f.write(json.dumps(all_session_logs) + '\n')
    return m_name,success_time,average_step,completion
