import os
import re
import pandas as pd
import numpy as np
import argparse

from config import RESULTS_PATH, GEOQA_TASK_MAPPING_v1, GEOQA_TASK_MAPPING_v2
def get_result(file_path, result_files, map_task):
    final_result = {}
    for result_file in result_files:
        match = re.match(r"geo_knowledge_(.+?)_v82_summary_(.+?)\.csv", result_file)
        if not match:
            continue
        city_name = match.group(1) 
        model_name = match.group(2) 
        result = pd.read_csv(os.path.join(file_path,result_file))
        # Check if the number of results is correct
        file_row_count = len(result)  
        if file_row_count != 29:
            print(f"Error: The number of results for model {model_name} under city {city_name} is {file_row_count}, not 29.")
        
        for _, row in result.iterrows():
            task = row['task_name']
            acc = row['accuracy']
            if (model_name, city_name) not in final_result:
                final_result[(model_name, city_name)] = {
                    "node": None,
                    "landmark": None,
                    "path": None,
                    "districts": None,
                    "boundary": None,
                    "others": None
                }
            for cat,tasks in map_task.items():
                if task in tasks:
                    final_result[(model_name, city_name)][cat] = acc
                    break
    
    data = []
    for (model_name, city_name), categories in final_result.items():
        row = [model_name, city_name, categories['node'], categories['landmark'], categories['path'], categories['districts'], categories['boundary'], categories['others']]
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Model_Name', 'City_Name', 'Node', 'Landmark', 'Path', 'Districts', 'Boundary', 'Others'])
    df['GeoQA_Average_Accuracy'] = df[['Node', 'Landmark', 'Path', 'Districts', 'Boundary', 'Others']].mean(axis=1).round(4)
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", type=str, default="Beijing")
    parser.add_argument("--evaluate_version", type=str, default="v1")
    args = parser.parse_args()
    file_path = os.path.join(RESULTS_PATH, "geo_knowledge_result")
    result_files = os.listdir(file_path)
    
    if args.city_name == "Beijing":
        GEOQA_TASK_MAPPING = GEOQA_TASK_MAPPING_v2
    else:
        if args.evaluate_version == "v1":
            GEOQA_TASK_MAPPING = GEOQA_TASK_MAPPING_v1
        elif args.evaluate_version == "v2":
            GEOQA_TASK_MAPPING = GEOQA_TASK_MAPPING_v2
    result = get_result(file_path, result_files, GEOQA_TASK_MAPPING)
    result.to_csv(os.path.join(file_path,"geoqa_benchmark_result.csv"), index=False)
    print("GeoQA results have been saved!")

