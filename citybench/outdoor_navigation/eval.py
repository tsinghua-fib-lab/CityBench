import os
import argparse
import numpy as np
import pandas as pd
import random
import jsonlines
import json
import signal
from datetime import datetime
from multiprocessing import Pool
from openai import OpenAI
from tqdm import tqdm


from serving.vlm_serving import VLMWrapper
from config import MAP_CACHE_PATH, RESOURCE_PATH, RESULTS_PATH, MONGODB_URI, MAP_DICT, IMAGE_FOLDER, ROUTING_PATH, VLM_API, NAVIGATION_UPGRADE
from .utils import calculate_direction, calculate_distance, get_prompt_eval, get_basic_prompt, get_prompt_eval_reason
from global_utils import load_map
from serving.llm_api import  get_chat_completion, get_model_response_hf, match_response, get_model_response_hf_image, match_response_reason

def route_process_nav(city_map, city, road_ids, match_df, url_df, meta_info_df):
    # one route landmark navigation
    url_map = dict(zip(url_df['image_name'], url_df['image_url']))
    instructions = []
    basic_prompt = get_basic_prompt()
    instructions.append({
        "type": "text",
        "text": basic_prompt
    })
    steps = []

    for i, road_id in enumerate(road_ids):
        road_matches = match_df[match_df['road_id'] == road_id]
        if road_matches.empty:
            print(f"No matches found for road_id: {road_id}")
            continue
        road_info = city_map.get_road(road_id)
        road_name = road_info['name']
        if not road_name: 
            road_name = "unknown road"
        road_len = road_info['length']
        lane_id = road_info['lane_ids'][0]

        # 按照 distance 对匹配到的文件进行排序
        sorted_matches = road_matches.sort_values(by='distance')
        for count, (idx, row) in enumerate(sorted_matches.iterrows()):
            walk_len = int(road_len - row['distance'])
            image_path_suffix = row['file_name']
            url_image = url_map.get(image_path_suffix)
            
            if count == 0:
                action = "forward"
                step_instruction = f"When you see this image, your current action is 'forward':"
                instructions.append({
                        "type": "text",
                        "text": step_instruction
                    })
                instructions.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url_image
                    }
                })
                
            # if last element
            elif count == len(sorted_matches) - 1:
                # print("enter last image")
                # check if it is the last road_id
                if i == len(road_ids) - 1:
                    action = "stop"
                    step_instruction = f"When you see this image, your current action is '{action}':"
                    instructions.append({
                        "type": "text",
                        "text": step_instruction
                    })
                    instructions.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url_image
                        }
                    })
                else:
                    next_road_id = road_ids[i+1]
                    next_road_info = city_map.get_road(next_road_id)
                    next_road_start = next_road_info['shapely_lnglat'].coords[0]
                    current_road_end = road_info['shapely_lnglat'].coords[-1]
                    action = calculate_direction(current_road_end, next_road_start)
                    step_instruction = f"When you see this image, your current action is '{action}':"
                    instructions.append({
                        "type": "text",
                        "text": step_instruction
                    })
                    instructions.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url_image
                        }
                    })
            steps.append({
                        "action": action,
                        "image_name": image_path_suffix
                    })
            
    last_instruction = f"ATTENTION: Your should describe the image and integrate the action decision into your description for EACH image."
    instructions.append({
        "type": "text",
        "text": last_instruction
    })
    return instructions, steps


def nav_gen(city, match_file, route_file, meta_file, url_file, output_file, city_map, model_name, data_sample=600):
    
    match_data_df = pd.read_csv(match_file)
    meta_info_df = pd.read_csv(meta_file)
    url_df = pd.read_csv(url_file)  

    # 控制路径数量
    sample = 0
    with jsonlines.open(output_file, mode='a') as writer:
        with jsonlines.open(route_file) as reader:
            for obj in reader:
                start_aoi_id = obj.get('start_aoi_id')
                dest_aoi_id = obj.get('dest_aoi_id')
                road_ids = obj.get('road_ids')
                instructions, steps = route_process_nav(city_map, city, road_ids, match_data_df, url_df, meta_info_df)
                response = get_chat_completion([{"role": "user", "content": json.dumps(instructions)}], model_name)

                record = {
                        "route": road_ids,
                        "response": response,
                        "steps": steps
                    }
                writer.write(record)
                sample += 1
                
                if sample == data_sample:
                    break

def single_route_process(city, url_file, route, navigation, steps, model_name):
    # print("enter single_route_process")
    name_to_url_mapping = pd.read_csv(url_file).set_index('image_name')['image_url'].to_dict()
    success_flag = True
    basic_prompt = get_prompt_eval()
    basic_prompt = f"{basic_prompt}\n{navigation}"
    prompts = []
    prompts.append({
    "type": "text",
    "text": basic_prompt
    })
    for step in steps:
        text = f"Here is the street view image of your current location."
        image_name = step['image_name']
        image_url = name_to_url_mapping[image_name]
        prompts.append({
            "type": "text",
            "text": text
        })
        prompts.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
        last_text = f"Please provide the next action('forward', 'left', 'right', or 'stop') based on the image and the navigation instruction:"
        prompts.append({
            "type": "text",
            "text": last_text
        })
        action_true = step['action']
        action,_ = get_chat_completion([{"role": "user", "content": prompts}], model_name)
        print(f"Action: {action}, True Action: {action_true}")
        if action != action_true:
            success_flag = False
            print("false")
            break
        prompts.append({
            "type": "text",
            "text": f"Action: {action}"
        })
    
    if success_flag:
        print("right")
    return success_flag
    

def process_single_route(args):
    # 在筛选路径时，gpt-4o-mini尝试5次，保留成功的路径
    city, url_file, route, response, steps, model_name = args

    for i in range(5):
        success_time = single_route_process(city, url_file, route, response, steps, model_name)
        if success_time == 1:
            print(f"Success found for route {route}")
            return {"route": route, "response": response, "steps": steps} 

    return None 


def validate_eval(city, url_file, instruction_file, instruction_validate_file, model_name, num_processes):
    results_to_save = []
    print(f"Validating routes in {instruction_file} using model {model_name}...")
    with jsonlines.open(instruction_file) as reader:
        records = list(reader)  

    print(f"Validating {len(records)} routes...")
    args_list = [(city, url_file, record['route'], record['response'], record['steps'], model_name) for record in records]
    with Pool(processes=num_processes) as pool:  
        print("Processing routes...enter parallel processing")
        for result in pool.imap(process_single_route, args_list):
            if result:
                results_to_save.append(json.dumps(result))
    with open(instruction_validate_file, 'a') as file:
        for item in results_to_save:
            file.write(item + '\n')

    print(f"Validated routes saved to {instruction_validate_file}")


def single_route_process_eval(city, url_file, logs_file, route, navigation, steps, model_name, model=None):
    # print("enter single_route_process_eval")

    name_to_url_mapping = pd.read_csv(url_file).set_index('image_name')['image_url'].to_dict()
    success_flag = True
    step_count = 0  
    if NAVIGATION_UPGRADE == True:
        basic_prompt = get_prompt_eval_reason()
    else:
        basic_prompt = get_prompt_eval()
    basic_prompt = f"{basic_prompt}\n{navigation}"
    begin_prompts = []

    if model_name in VLM_API:
        begin_prompts.append({
        "type": "text",
        "text": basic_prompt
        })
    else:
        begin_prompts.append({
        "type": "text",
        "value": basic_prompt
        })
    last_text = f"""
            Please provide the Reason and next Action('forward', 'left', 'right', or 'stop') based on the image and the navigation instruction.\n"""
    last_image_name = steps[-1]['image_name']
    last_image_url = name_to_url_mapping[last_image_name]
    actions_history = []  
    
    for step in steps:
        # 用于记录
        step_actions = []
        text = f"Here is the street view image of your current location."
        image_name = step['image_name']
        image_url = name_to_url_mapping[image_name]
        image_path = os.path.join(IMAGE_FOLDER, f"{city}_StreetView_Images/{image_name}")
        current_prompts = begin_prompts.copy()
        if NAVIGATION_UPGRADE == True:
            if actions_history:
                previous_actions = "Previous actions: " + ", ".join(actions_history)
            else:
                previous_actions = ""
            

            current_prompts = f"{basic_prompt}\n{previous_actions}\n{text}\n{last_text}"
            if model_name in VLM_API:
                api_prompts = []
                api_prompts.append({
                    "type": "text",
                    "text": current_prompts
                })
                api_prompts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
                response_action = get_chat_completion([{"role": "user", "content": api_prompts}], model_name)
                reason, action = match_response_reason(response_action)
            else:
                model_response = get_model_response_hf_image(image_path, current_prompts, model)
            # print(f"model response: {model_response}")
                reason, action = match_response_reason(model_response)
        else:
            for action in actions_history:
                if model_name in VLM_API:
                    current_prompts.append({
                        "type": "text",
                        "text": f"Previous action: {action}"
                    })
                else:
                    current_prompts.append({
                        "type": "text",
                        "value": f"Previous action: {action}"
                    })

            if model_name in VLM_API:
                current_prompts.append({
                    "type": "text",
                    "text": text
                })
                current_prompts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
            else:
                current_prompts.append({
                    "type": "text",
                    "value": text
                })
                
                current_prompts.append({
                    "type": "image",
                    "value": image_path
                    
                })
            if model_name in VLM_API:
                current_prompts.append({
                    "type": "text",
                    "text": last_text
                })
            else:
                current_prompts.append({
                    "type": "text",
                    "value": last_text
                })
            action_true = step['action']
            if model_name in VLM_API:
                response_action = get_chat_completion([{"role": "user", "content": current_prompts}], model_name)
                action = match_response(response_action)
            else:
                response_action = get_model_response_hf(current_prompts, model)
            action = match_response(response_action)
        print(f"Action: {action}, True Action: {action_true}")
        if action != action_true:
            success_flag = False
            print("false")
            distance = calculate_distance(city, last_image_name, image_name)
            break

        actions_history.append(action)
        step_count += 1
        current_prompts.append({
            "response": {response_action},
            "extract": {action},
            "ref": {action_true}
        })
        with jsonlines.open(logs_file, mode='a') as writer:
            writer.write({current_prompts})
    
    if success_flag:
        distance = 0
        print("right")

    return success_flag, step_count, distance

def process_single_route_eval(args):
    city, url_file, logs_file, route, response, steps, model_name = args
    success_found, step_count, distance = single_route_process_eval(city, url_file, logs_file, route, response, steps, model_name)
    return success_found, step_count, distance

def eval_gen(city, url_file, logs_file, instruction_validate_file, results_file, model_name, num_processes, samples):
    success_time = 0
    step_sum = 0
    distance_sum = 0

    if model_name in VLM_API:
        # 可以通过API调用的模型
        with jsonlines.open(instruction_validate_file) as reader:
            records = list(reader)  
            record_count = len(records)
            if samples < record_count:
                records = random.sample(records, samples)
            else:
                print(f"Requested sample size {samples} exceeds available records {record_count}. Evaluating all records.")
        args_list = [(city, url_file, logs_file, record['route'], record['response'], record['steps'], model_name) for record in records]
        with Pool(processes=num_processes) as pool:  
            for result in pool.imap(process_single_route_eval, args_list):
                success_found, step_count, distance = result
                if success_found:
                    success_time += 1
                step_sum += step_count
                distance_sum += distance
    else:
        # 本地部署模型
        print("other model")
        model_wrapper = VLMWrapper(model_name)
        print(model_name)
        model = model_wrapper.get_vlm_model()
        with jsonlines.open(instruction_validate_file) as reader:
            records = list(reader)  
            record_count = len(records)  
            if samples < record_count:
                records = random.sample(records, samples)
            else:
                print(f"Requested sample size {samples} exceeds available records {record_count}. Evaluating all records.")

        for record in tqdm(records):
            route = record["route"]
            response = record["response"]
            steps = record["steps"]
            success_found, step_count, distance = single_route_process_eval(city, url_file, logs_file, route, response, steps, model_name, model)
            if success_found:
                success_time += 1
            step_sum += step_count
            distance_sum += distance

    success_ratio = success_time / record_count
    avg_step = step_sum / record_count
    avg_distance = distance_sum / record_count

    if "/" in model_name:
        model_back = model_name.replace("/", "_")
    result_data = {
        "City_Name": city,
        "Model_Name": model_back,
        "Navigation_Success_Ratio": success_ratio,
        "Navigation_Average_Steps": avg_step,
        "Navigation_Average_Distance": avg_distance
    }
    print(f"Success ratio: {success_ratio}, Average steps: {avg_step}, Average distance: {avg_distance}")
     

    result_df = pd.DataFrame([result_data])
    result_df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)

    print(f"Results saved to {results_file}")


def main(args):
    city_map = MAP_DICT[args.city_name]
    m, process, routing_client = load_map(
        city_map=city_map, 
        cache_dir=MAP_CACHE_PATH, 
        routing_path=ROUTING_PATH, 
        port=args.port)

    match_file = "citydata/outdoor_navigation_tasks/{}_matched_images.csv".format(args.city_name)
    route_file = "citydata/outdoor_navigation_tasks/{}_navigation_tasks.jsonl".format(args.city_name)
    meta_file = os.path.join(IMAGE_FOLDER, "{}_StreetView_Images/combined_stitch_meta_info.csv".format(args.city_name))
    url_file = os.path.join(IMAGE_FOLDER, "url_mapping.csv")
    instruction_file = "citydata/outdoor_navigation_tasks/{}_navigation_instructions.jsonl".format(args.city_name)
    instruction_validate_file = "citydata/outdoor_navigation_tasks/{}_navigation_instructions_validate.jsonl".format(args.city_name)
    logs_file = os.path.join(RESULTS_PATH, f'outdoor_navigation_results/logs/{args.city_name}/{args.city_name}_{args.model_name}_logs.jsonl')
    os.makedirs(os.path.dirname(logs_file), exist_ok=True)  
    results_file = os.path.join(RESULTS_PATH, f'outdoor_navigation_results/{args.city_name}_results.csv')
    os.makedirs(os.path.dirname(results_file), exist_ok=True) 

    if args.data_name == "all":
        default_samples = 50
    else:
        default_samples = 5
    SAMPLES = args.samples if args.samples is not None else default_samples

    num_processes = 10
    if args.mode=="gen":
        # 获取路径的landmark导航指令
        # data_sample控制生成的导航任务数量
        data_sample = 10000
        # generate
        nav_gen(args.city_name, match_file, route_file, meta_file, url_file, instruction_file, m, args.model_name, data_sample)
        # validate
        validate_eval(args.city_name, url_file, instruction_file, instruction_validate_file, args.model_name, num_processes)

    elif args.mode=="eval":
        # evaluate
        eval_gen(args.city_name, url_file, logs_file, instruction_validate_file, results_file, args.model_name, num_processes, SAMPLES)
    
    print("send signal")
    process.send_signal(sig=signal.SIGTERM)
    process.wait()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Shanghai")
    parser.add_argument("--model_name", type=str, default="GPT4omini")
    parser.add_argument("--data_name", type=str, default="mini", choices=["all", "mini"])
    parser.add_argument("--samples", type=int)
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--port", type=int, default=54100)
    args = parser.parse_args()

    main(args)
