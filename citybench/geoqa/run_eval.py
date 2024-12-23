import os
import tqdm
import copy
import jsonlines
import argparse
import pandas as pd
from multiprocessing import Pool

from config import RESULTS_PATH, GEOQA_SAMPLE_RATIO, LLM_MODEL_MAPPING, GEOQA_TASK_MAPPING_v1, GEOQA_TASK_MAPPING_v2
from serving.llm_api import get_chat_completion, extract_choice, get_model_response_hf
from serving.vlm_serving import VLMWrapper


def task_files_adaption(task_file, path_prefix):
    for category, tasks in task_file.items():
        for task_name, file_name in tasks.items():
            task_file[category][task_name] = os.path.join(path_prefix, file_name)
    os.makedirs(path_prefix, exist_ok=True)
    return task_file


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


###################### 评估接口
def run_evaluate_api(task_file_path, model_name, task_name, max_validation, temperature, max_tokens, region_exp, data_name, model=None):
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
    
    if data_name == "mini":
        test_df = test_df.sample(min(max_validation,int(test_df.shape[0]*GEOQA_SAMPLE_RATIO)), random_state=42)
    else:
        if test_df.shape[0]>max_validation*2:
            test_df = test_df.sample(max_validation, random_state=42)
    correct_count, count = 0, 0
    res = []
    for _, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, include_answer=False, max_choices=max_choices)
 
        if model is not None:
            prompt = {
                "type": "text",
                "value": question
            }

            output = get_model_response_hf(prompt, model)
        else:
            output, token_usage = get_chat_completion(
                session=[{"role":"user", "content": question}], 
                model_name=model_name, temperature=temperature, max_tokens=max_tokens
                )
        res.append([{"role":"user", "content": question}, {"role":"response", "content": output}, {"role":"ref", "content": row["answer"]}])

        if len(output) == 0:
            pass
        else:
            ans = extract_choice(output, ["A", "B", "C", "D", "E", "F"]) 
            if ans==row["answer"]:
                correct_count += 1
        count += 1

    if count == 0:
        count = 1
    print("Success rate:{}({}/{})".format(correct_count/count, correct_count, count))

    os.makedirs("results/logs_geo_knowledge/", exist_ok=True)
    if "/" in model_name:
        model_name = model_name.replace("/", "_")
    with jsonlines.open("results/logs_geo_knowledge/{}_{}_{}.jsonl".format(region_exp, model_name, task_name), "w") as wid:
        for r in res:
            wid.write(r)
    return [model_name, task_name, correct_count, count, correct_count/count]

if __name__ == "__main__":
    print("start model evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--evaluate_version", type=str, default="v1")
    parser.add_argument("--max_tokens", default=200, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_valid", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task_path", type=str, help="same with the output path in data_gen.py")
    parser.add_argument("--data_name", default="mini", choices=["all","mini"])
    args = parser.parse_args()

    if not isinstance(args.task_path, str):
        task_path = "citydata/task_Geo_knowledge/{}/{}".format(args.city_name, args.evaluate_version)
    else:
        task_path = args.task_path

    if args.city_name == "Beijing":
        GEOQA_TASK_MAPPING = GEOQA_TASK_MAPPING_v2
    else:
        if args.evaluate_version == "v1":
            GEOQA_TASK_MAPPING = GEOQA_TASK_MAPPING_v1
        elif args.evaluate_version == "v2":
            GEOQA_TASK_MAPPING = GEOQA_TASK_MAPPING_v2
    KNOWLEDGE_TASK_FILES_ = task_files_adaption(GEOQA_TASK_MAPPING, task_path)
    
    para_group = []
    for model_name in [args.model_name]:  
        model = None
        try:
            model_full = LLM_MODEL_MAPPING[model_name]
        except KeyError:
            model_full = model_name
            model_wrapper = VLMWrapper(model_full)
            model = model_wrapper.get_vlm_model()
        for category, tasks in KNOWLEDGE_TASK_FILES_.items():
            for task_name, file_path in tasks.items():
                print(f"Evaluating model: {model_name} task: {task_name}")
                if not os.path.isfile(file_path):
                    print(f"Task: {task_name} file not found at {file_path}, ignore it!")
                    continue
            
                para_group.append((
                    file_path,
                    model_name,  
                    task_name,
                    args.max_valid, 
                    args.temperature, 
                    args.max_tokens,
                    args.city_name,
                    args.data_name,
                    model
                ))

    res_df = []
    if args.workers == 1:
        for para in para_group:
            print(para)
            res = run_evaluate_api(para[0], para[1], para[2], para[3], para[4], para[5], para[6], para[7], para[8], para[9])
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
    res_df.to_csv("results/geo_knowledge_result/geo_knowledge_{}_{}_summary_{}.csv".format(
        args.city_name, 
        args.evaluate_version, 
        args.model_name.replace("/", "_")))
