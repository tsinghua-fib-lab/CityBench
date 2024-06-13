import os
import pandas as pd
import numpy as np

def get_result(result_files):
    final_result = {}
    for result_file in result_files:
        result = pd.read_csv(os.path.join(file_path,result_file))
        for _, row in result.iterrows():
            task = row['task_name']
            model = row['model_name']
            acc = row['accuracy']
            if model not in final_result:
                final_result[model] = {}
            for cat,tasks in map_task.items():
                if cat not in final_result[model]:
                    final_result[model][cat] = []
                if task in tasks:
                    final_result[model][cat].append(acc)
                    break

    final_result2 = {}   
    for model, sub_dict in final_result.items():
        if model not in final_result2:
            final_result2[model]={}
        for cat, acc_list in sub_dict.items():
            if cat not in final_result2[model]:
                final_result2[model][cat] = {'mean':0,'var':0}
            final_result2[model][cat]['result'] = f"{format(np.mean(acc_list),'.4f')}±{format(np.var(acc_list),'.4f')}"

    columns = ['node', 'landmark', 'path', 'districts', 'boundary', 'others']
    index = list(final_result2.keys())
    df = pd.DataFrame(index=index, columns=columns)
    for model, features in final_result2.items():
        for feature, results in features.items():
            df.at[model, feature] = results['result']
    return df


if __name__ == '__main__':
    file_path = "./results/geo_knowledge_result"
    output_path = "./results"
    result_files = os.listdir(file_path)
    map_task = {
        "node": ["poi2coor", "aoi_near", "AOI_POI_road4", "poi2cor", ],
        "landmark": ["eval_landmark_env", "eval_landmark_path", "landmark_env", "landmark_path"],
        "path": ["eval_road_link", "eval_road_od", "eval_road_length", "road_link", "road_od", "road_length"],
        "districts": ["aoi2addr", "eval_road_aoi", "AOI_POI5", "AOI_POI6", "aoi2adr", "road_arrived_pois"],
        "boundary": ["eval_boundary_road", "AOI_POI_road1", "AOI_POI_road2", "AOI_POI_road3", "boundary_road"],
        "others": ["AOI_POI3", "AOI_POI4", "aoi2type"]
    }
    result = get_result(result_files)
    result.to_csv(os.path.join(output_path,"benchmark_result.csv"))

