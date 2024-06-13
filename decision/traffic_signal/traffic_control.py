import numpy as np
from moss import Engine, LaneChange, TlPolicy
from typing import List, Tuple
from utils.utils import JunctionState, validate_response, get_response, process_phase, get_chosen_number_from_phase_index, whether_junc_in_region, get_coords
import tqdm
import time
from multiprocessing import Pool
import json
import jsonlines
import uuid
import argparse

# 基于LLM输出进行交通灯控制
def post_process(response, phase_map, qualified_phase_num, j, state, eng):
    answer = validate_response(response, phase_map)
    
    if answer[0] == True:
        chosen_number = answer[1]
        state.phase_index = phase_map[chosen_number]
        # print(f"Chosen number: {chosen_number}")
        # print(f"Chosen phase index: {state.phase_index}")
    else:
        print(answer[1])
        # print(state.phase_index)
        print(j.index)
        phase_index = state.phase_index if state.phase_index>0 else list(phase_map.values())[0]
        chosen_number = get_chosen_number_from_phase_index(phase_index, phase_map)
        
        new_chosen_number = (chosen_number + 1) % qualified_phase_num
        state.phase_index = phase_map[new_chosen_number]
        # print(f"Chosen number: {chosen_number}")
        # print(f"New Chosen number: {new_chosen_number}")
        # print(f"Chosen phase index: {state.phase_index}")
    
    # 设置路口相位
    # print(f"j.index: {j.index}, junction_index: {junction_index}")
    eng.set_tl_phase(j.index, state.phase_index)
    # 返回处理的交叉口索引
    return answer[0]


# 基于模拟器状态构建LLM的prompt
def get_prompt(j, cnt, waiting_cnt):
    # 保存相位信息
    phase_options = []
    # 相位选项序号映射
    phase_map = {} 
    # 过滤掉黄灯后的相位选项序号
    filtered_index = 0
    
    for original_index, phase in enumerate(j.tl.phases):
        phase_valid = True
        green_indices = []

        for i, phase_state in enumerate(phase.states):
            if phase_state.name == "YELLOW":
                # 过滤掉黄灯选项
                phase_valid = False
                break
            if phase_state.name == "GREEN":
                # 记录绿灯序号
                green_indices.append(i)

        if phase_valid:
            # 相位序号做映射
            phase_map[filtered_index] = original_index
            description = process_phase(j, green_indices, filtered_index, cnt, waiting_cnt)
            filtered_index += 1
            phase_options.append(description)

    # 过滤掉黄灯的总相位数
    qualified_phase_num = filtered_index
    # 合并车道信息和相位选项信息
    basic_descriptions = f"""A traffic light controls a complex intersection with {qualified_phase_num} signal phases. Each signal relieves vehicles' flow in the allowed lanes.
    The state of the intersection is listed below. It describes:
    - The number of lanes relieving vehicles' flow under each traffic light phase.
    - The total count of vehicles present on lanes permitted to move of each signal.
    - The count of vehicles queuing on lanes allowed to move, which are close to the end of the road or intersection and moving at low speeds of each signal."""
    prompt = "\n".join([basic_descriptions] + ["Available Phase Options"] + phase_options)
    
    prompt += """
    Please answer:
    Which is the most effective traffic signal that will most significantly improve the traffic condition during the
    next phase?

    Note:
    The traffic congestion is primarily dictated by the waiting vehicles, with the MOST significant impact. You MUST pay the MOST attention to lanes with long queue lengths.

    Requirements:
    - Let's think step by step.
    - You can only choose one of the phase option NUMBER listed above.
    - You must follow the following steps to provide your analysis:
        Step 1: Provide your analysis for identifying the optimal traffic signal.
        Step 2: Answer your chosen phase option number.
    - Your choice can only be given after finishing the analysis.
    - Your choice must be identified by the tag: <signal>YOUR_NUMBER</signal>.

    """
    
    return prompt, phase_map, qualified_phase_num


if __name__ == '__main__':

    ###### 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="newyork", choices=["beijing", "paris", "newyork"])
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--total_time", type=int, default=3600)

    args = parser.parse_args()

    # 设置切换时长为30秒
    CITY = args.city
    TIME_THRESHOLD = 30
    TOTAL_TIME = args.total_time
    START_TIME = 30000
    MAP_FILE = './EXP_ORIG_DATA/{}/{}.map.pb'.format(CITY, CITY)
    AGENT_FILE = './trips/{}_trip_filtered_start_29000_end_32000_extend_5.pb'.format(CITY, CITY)
    SAVE_FILE_NAME = "./results/{}_results_20240609.json".format(CITY)
    LOG_FILE_NAME = "./results/logs/{}.jsonl".format(uuid.uuid4())
    RUNNING_LABEL = "exp"
    GPU_DEVICE_NUM = 0
    
    # "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"
    # "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"
    # "gpt-3.5-turbo-0125", "gpt-4o", "gpt-4-turbo-2024-04-09"
    # "fixed-time", "deepseek-chat"
    MODEL_NAME = args.model
    #################


    print("Start Engine")
    eng = Engine(
        map_file=MAP_FILE,
        agent_file= AGENT_FILE,
        start_step=START_TIME,
        total_step=10000000,
        lane_change=LaneChange.MOBIL,
        device=GPU_DEVICE_NUM
    )
    # 设置所有路口的信控为FIXED_TIME
    eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.FIXED_TIME)
    eng.set_tl_duration_batch(range(eng.junction_count), TIME_THRESHOLD)

    # 获取地图
    M = eng.get_map()
    # 划分的交通灯控制区域
    coords = get_coords(CITY)
    # 保存需要的交叉口索引
    valid_junctions = []
    junc_states = {}
    for j in M.junctions:
        # j = M.junctions[junction_index]
            if j.tl is None:
                continue
            if whether_junc_in_region(M, j, coords) == False:
                continue
            # 设置指定路口的信控为手动切换
            if MODEL_NAME=="fixed-time":
                pass
            else:
                eng.set_tl_policy(j.index, TlPolicy.MANUAL)
            junc_states[j.index] = JunctionState()
            valid_junctions.append(j.index)
    
    print(f"valid_junctions: {len(valid_junctions)}")

    aql = []

    print("Start Simulate")
    sim_time = time.time()
    llm_cost_time_sum = 0
    invalid_actions = 0
    llm_inout = []
    for epoch in tqdm.tqdm(range(TOTAL_TIME//TIME_THRESHOLD)):
        # 获取lane上所有的车辆数
        cnt = eng.get_lane_vehicle_counts()  
        waiting_cnt = eng.get_lane_waiting_at_end_vehicle_counts() 
        # Average queue length
        print(f"current vehicle sum:{cnt.sum()} waiting vehicles sum:{waiting_cnt.sum()}")
        aql.append(waiting_cnt.sum()/len(valid_junctions))
        
        junction_info = []
        junction_prompt = []
        # 获取所有junction参数
        for junction_index in valid_junctions:
            j = M.junctions[junction_index]
            state = junc_states[j.index]
            prompt, phase_map, qualified_phase_num = get_prompt(j, cnt, waiting_cnt)
            junction_info.append([prompt, phase_map, qualified_phase_num, j, state])
            junction_prompt.append([prompt, MODEL_NAME])
        
        # 并行处理交叉路口LLM访问请求
        llm_start_time = time.time()
        with Pool(processes=len(valid_junctions)) as pool:
            results = pool.starmap(get_response, junction_prompt)
        llm_cost_time = time.time()-llm_start_time
        llm_cost_time_sum += llm_cost_time
        
        # 基于LLM回答进行后处理
        for i, res in enumerate(results):
            if MODEL_NAME == "fixed-time":
                pass
            else:
                llm_inout.append([{"role":"user", "content": junction_prompt[i][0]}, {"role":"assistant", "content": res}])
                ans_state = post_process(res, junction_info[i][1], junction_info[i][2], junction_info[i][3], junction_info[i][4], eng)
                if not ans_state:
                    invalid_actions += 1

        print("Steps: {} LLM:{} Cost:{}".format(epoch, MODEL_NAME, llm_cost_time))
        eng.next_step(TIME_THRESHOLD)

    ####### 计算指标
    sim_time = time.time()-sim_time
    # Average queue length
    aql = np.mean(aql)
    # Average traveling time
    att = eng.get_departed_vehicle_average_traveling_time()
    # Throughput
    tp = eng.get_finished_vehicle_count()
    # 动作失败率
    invalid_ratio = invalid_actions/(TOTAL_TIME//TIME_THRESHOLD)
    print(f"Model:{MODEL_NAME} Average queue length: {aql}, Average traveling time: {att}, Throughput: {tp} Invalid Actions: {invalid_ratio}")

    ####### 保存结果
    res = {
        "exp_time": time.asctime( time.localtime(time.time())),
        "type": RUNNING_LABEL,
        "simulator": {
            "time_threshold": TIME_THRESHOLD,
            "start_time": START_TIME,
            "total_time": TOTAL_TIME,
            "map_file": MAP_FILE,
            "agent_file": AGENT_FILE,
            "simulate_time_cost": round(sim_time, 3)
            },
        "model": {
            "model_name": MODEL_NAME.replace("/", "-"),
            "request_time_cost": round(llm_cost_time_sum, 3),
        },
        "performance": {
            "average_queue_length": round(aql, 3),
            "average_traveling_time": round(att, 3),
            "throughput": round(tp, 3),
            "invalid": round(invalid_ratio, 3)
        },
        "logs": {
            "file_name": LOG_FILE_NAME
        }
    }

    with jsonlines.open(LOG_FILE_NAME, "w") as wid:
        wid.write_all(llm_inout)

    try:
        with open(SAVE_FILE_NAME, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    data.append(res)
    with open(SAVE_FILE_NAME, "w") as wid:
        json.dump(data, wid, indent=2)
