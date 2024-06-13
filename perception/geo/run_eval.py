import os
import sys
import tqdm
import jsonlines
import argparse
import pandas as pd
from openai import OpenAI
import time
import re
import httpx
from thefuzz import process
from multiprocessing import Pool
import copy

parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(parent_path)
sys.path.append(parent_path)

PROXY = "http://127.0.0.1:10190"
# your key for LLM APIs
OPENAI_APIKEY = ""
DEEPINFRA_APIKEY = ""
DEEPSEEK_APIKEY = ""


TASK_FILES_EXTEND = {
    "aoi2adr":"aoi2adr.csv",
    "aoi2type":"aoi2type.csv",
    "aoi_boundary_poi":"aoi_boundary_poi.csv",
    "aoi_group":"aoi_group.csv",
    "AOI_POI":"AOI_POI.csv",
    "aoi_poi":"aoi_poi.csv",
    "AOI_POI2":"AOI_POI2.csv",
    "AOI_POI3":"AOI_POI3.csv",
    "AOI_POI4":"AOI_POI4.csv",
    "AOI_POI5":"AOI_POI5.csv",
    "AOI_POI6":"AOI_POI6.csv",
    "AOI_POI_road1":"AOI_POI_road1.csv",
    "AOI_POI_road2":"AOI_POI_road2.csv",
    "AOI_POI_road3":"AOI_POI_road3.csv",
    "AOI_POI_road4":"AOI_POI_road4.csv",
    "boundary_road":"boundary_road.csv",
    "districts_poi_type":"districts_poi_type.csv",
    "landmark_env":"landmark_env.csv",
    "landmark_path":"landmark_path.csv",
    "poi2adr":"poi2adr.csv",
    "poi2cor":"poi2cor.csv",
    "poi2type":"poi2type.csv",
    "poi_aoi":"poi_aoi.csv",
    "road_arrived_pois":"road_arrived_pois.csv",
    "road_length":"road_length.csv",
    "road_link":"road_link.csv",
    "road_od":"road_od.csv",
    "type2aoi":"type2aoi.csv",
    "type2poi":"type2poi.csv",
}
TASK_FILES_GEOQA = {
    "aoi2addr":"aoi2addr.csv",
    "aoi2type":"aoi2type.csv",
    "aoi_near":"aoi_near.csv",
    "AOI_POI3":"AOI_POI3.csv",
    "AOI_POI4":"AOI_POI4.csv",
    "AOI_POI5":"AOI_POI5.csv",
    "AOI_POI6":"AOI_POI6.csv",
    "AOI_POI_road1":"AOI_POI_road1.csv",
    "AOI_POI_road2":"AOI_POI_road2.csv",
    "AOI_POI_road3":"AOI_POI_road3.csv",
    "AOI_POI_road4":"AOI_POI_road4.csv",
    "eval_boundary_road":"eval_boundary_road.csv",
    "eval_landmark_env":"eval_landmark_env.csv",
    "eval_landmark_path":"eval_landmark_path.csv",
    "eval_road_aoi":"eval_road_aoi.csv",
    "eval_road_length":"eval_road_length.csv",
    "eval_road_link":"eval_road_link.csv",
    "eval_road_od":"eval_road_od.csv",
    "poi2coor":"poi2coor.csv",
}
def task_files_adaption(task_file, region_exp, evaluate_version):
    task_files = copy.deepcopy(task_file)
    path_prefix = "./task_Geo_knowledge/{}/{}".format(region_exp, evaluate_version)
    for k in task_files:
        if path_prefix not in task_files[k]:
            task_files[k] = os.path.join(path_prefix, task_files[k])
    os.makedirs(path_prefix, exist_ok=True)
    return task_files

# evaluation codes from https://github.com/THUDM/AgentTuning/blob/main/eval_general/eval_mmlu_hf.py
INIT_PROMPT = "The following is a multiple-choice question about the geospatial knowledge of city. Please choose the most suitable one among A, B, C and D as the answer to this question. Please output the option directly. No need for explaination.\n"

def format_example(line, include_answer=True, max_choices=4):
    choices = ["A", "B", "C", "D"]
    prompt=INIT_PROMPT
    if max_choices>=5:
        choices.append("E")
        prompt = prompt.replace("A, B, C and D", "A, B, C, D and E")
    if max_choices>=6:
        choices.append("F")
        prompt = prompt.replace("A, B, C, D and E", "A, B, C, D, E and F")
    
    example = prompt + 'Question: ' + line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += '\nAnswer: ' + line["answer"] + '\n\n'
    else:
        example += '\nAnswer:'
    return example

def extract_choice(gen, choice_list):
    # Add the choices to the regex pattern dynamically based on the given choice_list
    choice_pattern = "|".join(choice_list)
    res = re.search(
        rf"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^{choice_pattern}]{{0,20}}?(?:n't|not))[^{choice_pattern}]{{0,10}}?\b(?:|is|:|be))\b)[^{choice_pattern}]{{0,20}}?\b({choice_pattern})\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            rf"\b({choice_pattern})\b(?![^{choice_pattern}]{{0,8}}?(?:n't|not)[^{choice_pattern}]{{0,5}}?(?:correct|right))[^{choice_pattern}]{{0,10}}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(rf"^({choice_pattern})(?:\.|,|:|$)", gen)

    # simply extract the first appeared letter
    if res is None:
        res = re.search(rf"(?<![a-zA-Z])({choice_pattern})(?![a-zA-Z=])", gen)

    if res:
        return res.group(1)
    else:
        return None


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

###################### 评估接口
def run_evaluate_api(task_file_path, model_name, task_name, max_validation, temperature, max_tokens, infer_server, region_exp):
    try:
        test_df = pd.read_csv(task_file_path, header=0)
    except:
        return []
    
    columns = test_df.columns.to_list()
    if "F" in columns:
        max_choices = 6
    elif "E" in columns:
        max_choices = 5
    elif "D" in columns:
        max_choices = 4
    else:
        max_choices = 4
    
    if test_df.shape[0]>max_validation*2:
        test_df = test_df.sample(max_validation, random_state=42)
    
    correct_count, count = 0, 0
    res = []
    for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, include_answer=False, max_choices=max_choices)
        output = get_agent_action_simple(
            session=[{"role":"user", "content": question}], 
            model_name=model_name, temperature=temperature, max_tokens=max_tokens, infer_server=infer_server
            )
        res.append([{"role":"user", "content": question}, {"role":"assistant", "content": output}, {"role":"ref", "content": row["answer"]}])

        # TODO 如何从输入中提取答案
        if len(output) == 0:
            pass
        else:
            ans = extract_choice(output, ["A", "B", "C", "D", "E", "F"]) 
            if ans==row["answer"]:
                correct_count += 1
        count += 1
    
    print("Success rate:{}({}/{})".format(correct_count/count, correct_count, count))

    os.makedirs("./results/logs_geo_knowledge/", exist_ok=True)
    if "/" in model_name:
        model_name = model_name.replace("/", "_")
    with jsonlines.open("./results/logs_geo_knowledge/{}_{}_{}.jsonl".format(region_exp, model_name, task_name), "w") as wid:
        for r in res:
            wid.write(r)
    return [model_name, task_name, correct_count, count, correct_count/count]

if __name__ == "__main__":
    print("start model evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--city_eval_version", type=str, default="v82")
    parser.add_argument("--city", type=str)
    parser.add_argument("--max_tokens", default=200, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_valid", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--infer_server", type=str, default="DL4-vllm")
    args = parser.parse_args()

    if args.city == "beijing":
        KNOWLEDGE_TASK_FILES_ = task_files_adaption(TASK_FILES_EXTEND, args.city, args.city_eval_version)
    else:
        KNOWLEDGE_TASK_FILES_ = task_files_adaption(TASK_FILES_GEOQA, args.city, args.city_eval_version)
    para_group = []
    for model in [args.model_name]: # "chatglm3-v8-e3:8000", 
        for task_name in KNOWLEDGE_TASK_FILES_.keys():
            print("evaluate model:{} task:{}".format(model, task_name))
            if "csv" not in KNOWLEDGE_TASK_FILES_[task_name]:
                print("task:{} is not ready, ignore it!".format(task_name))
                continue
            para_group.append((
                KNOWLEDGE_TASK_FILES_[task_name], 
                model,  
                task_name,
                args.max_valid, 
                args.temperature, 
                args.max_tokens,
                args.infer_server,
                args.city
            ))

    res_df = []
    if args.workers == 1:
        for para in para_group:
            print(para)
            res = run_evaluate_api(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7])
            if len(res)<1:
                continue
            res_df.append(res)
    else:
        with Pool(args.workers) as pool:
            results = pool.starmap(run_evaluate_api, para_group)
        for res in results:
            if len(res)<1:
                continue
            res_df.append(res)
    
    res_df = pd.DataFrame(res_df, columns=["model_name", "task_name", "corrct", "count", "accuracy"])
    print(res_df.head())
    res_df.to_csv("./results/geo_knowledge_result/geo_knowledge_{}_{}_summary_{}.csv".format(
        args.city, 
        args.city_eval_version, 
        args.model_name.replace("/", "_")))
