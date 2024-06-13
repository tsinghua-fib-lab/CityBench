import sys
import os
import pandas as pd
import numpy as np
import argparse
import signal
import random

from utils import *

def data_gen(args, city_map, routing_client, search_type, file_name):
    aoi_df = pd.read_csv(os.path.join(resource_dir, '{}_aois.csv'.format(args.city)))
    aoi_list = []
    for index, row in aoi_df.iterrows():
        if pd.notna(row['aoi_name']):
            aoi_list.append([row['aoi_name'], row['aoi_id']])
    selected_lists = random.sample(aoi_list, 2)

    count=0
    while count<args.samples:
        selected_lists = random.sample(aoi_list, 2)
        init_id=selected_lists[0][1]
        init_name=selected_lists[0][0]
        destination_id=selected_lists[1][1]
        destination_name=selected_lists[1][0]
        _,success_time,average_step,completion=get_performance(city_map,routing_client,init_id,init_name,destination_id,destination_name,args.model,None,args.temperature,args.threshold,round=args.exp_round,step=args.max_step,search_type=search_type)
        if success_time:
            count=count+1
            data ={'start_id':[init_id],'start_name':[init_name],
                'des_id':[destination_id],'des_name':[destination_name],
                'success_time':[success_time],'average_step':[average_step],'completion':[completion]}
            df=pd.DataFrame(data)
            try:
                existing_df = pd.read_csv(file_name)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(file_name, index=False)
            except FileNotFoundError:
                df.to_csv(file_name, index=False)


def eval_gen(args, city_map, routing_client, search_type, file_name):
    data = pd.read_csv(file_name)
    result_directory=os.path.join("./results/", "{}_result.csv".format(args.city))

    for m_name in [args.model]:   
        success_time=[]
        average_step=[]
        completion=[]
        for index, row in data.iterrows():
            start_id = row['start_id']
            start_name = row['start_name']
            des_id = row['des_id']
            des_name = row['des_name']
            
            print(start_id, start_name, des_id, des_name)
            _,suc_time,ave_step,comp=get_performance(city_map,routing_client,start_id,start_name,des_id,des_name,m_name," ",args.temperature,args.threshold,round=args.exp_round,step=args.max_step, search_type=search_type)
            success_time.append(suc_time)
            average_step.append(ave_step)
            completion.append(comp)

        df = pd.DataFrame(
            {
                "model_name":[m_name],
                "success_time":[np.mean(success_time)],
                "average_step":[np.mean(average_step)],
                "completion":[np.mean(completion)]
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
    parser.add_argument("--city", type=str, default="shanghai")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="DeepInfra")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--thresold", type=int, default=500)
    parser.add_argument("--exp_round", type=int, default=5)
    parser.add_argument("--max_step", type=int, default=15)
    args = parser.parse_args()

    cache_dir = "../../data/map_cache/"
    resource_dir = "../../data/resource/"
    routing_path = "../../config/routing_linux_amd64"
    
    city_map = MAP_DICT[args.city]
    port = 54310

    search_type="poi" if args.city=="beijing" else "aoi"
    file_name = './results/case_{}.csv'.format(args.city)

    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=cache_dir, 
        routing_path=routing_path, 
        port=port)
    
    if args.mode=="gen":
        data_gen(args, city_map=m, routing_client=routing_client, search_type=search_type, file_name=file_name)
    elif args.mode=="eval":
        eval_gen(args, city_map=m, routing_client=routing_client, search_type=search_type, file_name=file_name)

    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
