import sys
import os
import pandas as pd
import numpy as np
import argparse
import signal
import random
import time
from .utils import *
from config import ROUTING_PATH, MAP_DATA_PATH, MAP_CACHE_PATH, RESOURCE_PATH, RESULTS_PATH, LLM_MODEL_MAPPING, VLM_API
from global_utils import load_map
from serving.vlm_serving import VLMWrapper

def data_gen(args, SAMPLES, city_map, routing_client, search_type, task_file, exploration_results_path):
    aoi_df = pd.read_csv(os.path.join(resource_dir, '{}_aois.csv'.format(args.city_name)))
    aoi_list = []
    for index, row in aoi_df.iterrows():
        if pd.notna(row['aoi_name']) and "nearby" not in row['aoi_name']:
            aoi_list.append([row['aoi_name'], row['aoi_id']])
    selected_lists = random.sample(aoi_list, 2)

    log_file = os.path.join(exploration_results_path, "logs/{}/{}_{}.jsonl".format(args.city_name, args.model_name, args.mode))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    count=0
    while count<SAMPLES:
        selected_lists = random.sample(aoi_list, 2)
        init_id=selected_lists[0][1]
        init_name=selected_lists[0][0]
        destination_id=selected_lists[1][1]
        destination_name=selected_lists[1][0]
        _,success_time,average_step,completion=get_performance(city_map,routing_client,init_id,init_name,destination_id,destination_name,args.model_name,log_file,args.temperature,args.threshold,round=args.exp_round,step=args.max_step,search_type=search_type)
        if success_time:
            count=count+1
            data ={'start_id':[init_id],'start_name':[init_name],
                'des_id':[destination_id],'des_name':[destination_name],
                'success_time':[success_time],'average_step':[average_step],'completion':[completion]}
            df=pd.DataFrame(data)
            try:
                existing_df = pd.read_csv(task_file)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(task_file, index=False)
            except FileNotFoundError:
                df.to_csv(task_file, index=False)


def eval_gen(args, city_map, routing_client, search_type, task_file, exploration_results_path):
    data = pd.read_csv(task_file)
    result_directory=os.path.join(exploration_results_path, "{}_result.csv".format(args.city_name))

    for m_name in [args.model_name]:   
        success_time=[]
        average_step=[]
        completion=[]
        log_file = os.path.join(exploration_results_path, "logs/{}/{}_{}.jsonl".format(args.city_name, m_name, args.mode))
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        model = None
        if m_name in VLM_API:
            model_full = m_name
        else:
            try:
                model_full = LLM_MODEL_MAPPING[m_name]
            except KeyError:
                model_full = m_name
                model_wrapper = VLMWrapper(model_full)
                model = model_wrapper.get_vlm_model()

        for index, row in data.iterrows():
            start_id = row['start_id']
            start_name = row['start_name']
            des_id = row['des_id']
            des_name = row['des_name']
            
            print(start_id, start_name, des_id, des_name)
            _,suc_time,ave_step,comp=get_performance(city_map,routing_client,start_id,start_name,des_id,des_name,m_name, log_file,args.temperature,args.threshold,round=args.exp_round,step=args.max_step, search_type=search_type, model=model)
            success_time.append(suc_time)
            average_step.append(ave_step)
            completion.append(comp)
        if "/" in m_name:
            model_back = m_name.replace("/", "_")
        
        df = pd.DataFrame(
            {
                "City_Name":[args.city_name],
                "Model_Name":[model_back],
                "Exploration_Success_Ratio":[np.mean(success_time)/args.exp_round],
                "Exploration_Average_Steps":[np.mean(average_step)],
                "Exploration_Completion":[np.mean(completion)]
            }
        )
        try:
            existing_df = pd.read_csv(result_directory)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(result_directory, index=False)
        except FileNotFoundError:
            df.to_csv(result_directory, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Shanghai")
    parser.add_argument("--model_name", type=str, default="LLama3-8B")
    parser.add_argument("--data_name", type=str, default="mini", choices=["all", "mini"])
    parser.add_argument("--samples", type=int)
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--threshold", type=int, default=500)
    parser.add_argument("--exp_round", type=int, default=5)
    parser.add_argument("--max_step", type=int, default=15)
    parser.add_argument("--port", type=int, default=54352)
    args = parser.parse_args()

    cache_dir = MAP_CACHE_PATH
    resource_dir = RESOURCE_PATH
    routing_path = ROUTING_PATH
    
    city_map = MAP_DICT[args.city_name]


    # 根据data_name设置采样数
    if args.data_name == "all":
        default_samples = 50
    else:
        default_samples = 5
    # 若设置了命令行参数，则使用设置的samples
    SAMPLES = args.samples if args.samples is not None else default_samples

    search_type="poi"
    exploration_tasks_path = "citydata/exploration_tasks"
    os.makedirs(exploration_tasks_path, exist_ok=True)
    task_file = os.path.join(exploration_tasks_path, 'case_{}.csv'.format(args.city_name))
    exploration_results_path = os.path.join(RESULTS_PATH, "exploration_results")

    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=cache_dir, 
        routing_path=routing_path, 
        port=args.port)
    # 等待地图加载完成
    time.sleep(10)
    
    if args.mode=="gen":
        data_gen(args, SAMPLES, city_map=m, routing_client=routing_client, search_type=search_type, task_file=task_file, exploration_results_path=exploration_results_path)
    elif args.mode=="eval":
        eval_gen(args, city_map=m, routing_client=routing_client, search_type=search_type, task_file=task_file, exploration_results_path=exploration_results_path)

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
