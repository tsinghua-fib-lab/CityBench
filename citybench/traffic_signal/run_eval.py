import os
import tqdm
import time
import json
import jsonlines
import argparse

from citysim.signal import simulate
from config import MAP_DATA_PATH, TRIP_DATA_PATH, RESULTS_PATH


if __name__ == '__main__':

    ###### 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="NewYork")
    parser.add_argument("--model_name", type=str, default="LLama3-8B")
    parser.add_argument("--data_name", type=str, default="mini", choices=["all", "mini"])
    parser.add_argument("--total_time", type=int)

    args = parser.parse_args()

    signal_results_path = os.path.join(RESULTS_PATH, "signal_results")
    signal_results_logs_path = os.path.join(RESULTS_PATH, "signal_results", "logs")
    os.makedirs(signal_results_path, exist_ok=True)
    os.makedirs(signal_results_logs_path, exist_ok=True)

    # 设置切换时长为30秒
    TIME_THRESHOLD = 30
    START_TIME, END_TIME = 30000-1000, 30000+2000
    # 根据data_name设置总时间
    if args.data_name == "all":
        default_total_time = 3000
    else:
        default_total_time = 300
    # 若设置了total_time则使用设置的total_time
    TOTAL_TIME = args.total_time if args.total_time is not None else default_total_time

    MAP_FILE = os.path.join(MAP_DATA_PATH, '{}/{}.map.pb'.format(args.city_name, args.city_name))
    AGENT_FILE = os.path.join(TRIP_DATA_PATH, '{}_trip_filtered_start_{}_end_{}_extend_5.pb'.format(args.city_name, START_TIME, END_TIME))  
    SAVE_FILE_NAME = os.path.join(signal_results_path, "{}_results.json".format(args.city_name))
    LOG_FILE_NAME = os.path.join(signal_results_logs_path, "{}_{}.jsonl".format(args.city_name, args.model_name))
    RUNNING_LABEL = "exp"
    MODEL_NAME = args.model_name
    #################

    
    #### Start to Simulate
    aql, att, tp, sim_time, llm_cost_time_sum, invalid_actions, llm_inout = simulate(CITY=args.city_name, MAP_FILE=MAP_FILE, AGENT_FILE=AGENT_FILE, TOTAL_TIME=TOTAL_TIME, START_TIME=START_TIME, MODEL_NAME=MODEL_NAME)
    invalid_ratio = invalid_actions/(TOTAL_TIME//TIME_THRESHOLD)
    print(f"Model:{MODEL_NAME} Average queue length: {aql}, Average traveling time: {att}, Throughput: {tp} Invalid Actions: {invalid_ratio}")
    #### Stop Simulation

    ####### 保存结果
    res = {
        "exp_time": time.asctime( time.localtime(time.time())),
        "type": RUNNING_LABEL,
        "city_name": args.city_name,
        "simulator": {
            "time_threshold": TIME_THRESHOLD,
            "start_time": START_TIME,
            "total_time": TOTAL_TIME,
            "map_file": MAP_FILE,
            "agent_file": AGENT_FILE,
            "simulate_time_cost": round(sim_time, 3)
            },
        "model": {
            "model_name": MODEL_NAME.replace("/", "_"),
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
