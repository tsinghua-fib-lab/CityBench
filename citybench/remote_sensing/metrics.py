import os
import re
import json
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from config import REMOTE_SENSING_PATH, REMOTE_SENSING_RESULTS_PATH, VLM_MODELS, CITY_BOUNDARY


def compute_accuracy_regression(pred_list, true_list):
    mse = metrics.mean_squared_error(true_list, pred_list)
    mae = metrics.mean_absolute_error(true_list, pred_list)
    r2 = metrics.r2_score(true_list, pred_list)
    
    rmse = metrics.mean_squared_error(true_list, pred_list, squared=False)
    return mse, mae, r2, rmse


def normalized_fractional_ranking(numbers):
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1])

    ranks = {}
    for rank, (original_index, number) in enumerate(sorted_numbers):
        if number in ranks:
            ranks[number][0] += rank + 1
            ranks[number][1] += 1
        else:
            ranks[number] = [rank + 1, 1]

    average_ranks = {number: total_rank / count for number, (total_rank, count) in ranks.items()}

    return [(average_ranks[number] - 1) / len(numbers) for number in numbers]


def load_json_file(json_file_path):
    data = []
    with open(json_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def get_city_and_model_from_filename(folder_path):
    # 从文件名提取城市名和模型名
    pattern = re.compile(r"([A-Za-z]+(?:[A-Za-z ]+)*?)_([A-Za-z0-9\-_]+)_.*\.jsonl")
    city_model_pairs = set()
    for file in glob.glob(os.path.join(folder_path, "*.jsonl")):
        match = pattern.search(file)
        if match:
            city_name = match.group(1).replace(" ", "")  
            model_name = match.group(2)
            city_model_pairs.add((city_name, model_name))
    
    return list(city_model_pairs)


def eval_pop(city_name_list, model_name_list, save_name):
    all_csv_file_name = glob.glob(os.path.join(REMOTE_SENSING_PATH, '*_img_indicators.csv'))

    all_csv_df = pd.DataFrame()
    for i in all_csv_file_name:
        df = pd.read_csv(i)
        all_csv_df = pd.concat([all_csv_df, df], axis=0)
        
    all_csv_df.reset_index(drop=True, inplace=True)

    gt_ranking = normalized_fractional_ranking(all_csv_df['worldpop'].values)
    gt_ranking = [int(r * 100.0) / 10.0 if r < 1.0 else 9.9 for r in gt_ranking]
    # give the ranking to the dataframe
    all_csv_df["rank"] = gt_ranking
    img_name_list = all_csv_df["img_name"].to_list()

    all_r2_list = []
    all_mae_list = []
    all_rmse_list = []
    all_mse_list = []
    all_city_pred_list = []
    all_model_name_list = []

    for model_name in model_name_list:
        all_city_pred = []
        all_city_true = []

        for city_name in city_name_list:
            print(model_name, city_name)
            try:
                json_file_path = os.path.join(REMOTE_SENSING_RESULTS_PATH, city_name+'_'+model_name+'_pop.jsonl')
                data = load_json_file(json_file_path)
            except FileNotFoundError as e:
                try:
                    json_file_path = os.path.join(REMOTE_SENSING_RESULTS_PATH, city_name+'_'+model_name+'_population.jsonl')
                    data = load_json_file(json_file_path)
                except FileNotFoundError as e:
                    print("File not found! City:{} Model:{}".format(city_name, model_name))
                    continue

            pred_list = [] 
            true_list = [] 
            for item in data:
                text = str(item['text'])
                img_name = item['img_name'].split('/')[-1].split('.')[0]
                match = re.search(r"(\d+\.\d+)", text)
                if img_name not in img_name_list:
                    continue

                if match:
                    
                    rating = float(match.group(0))
                    pred_list.append(rating)
                    true_list.append(all_csv_df[all_csv_df['img_name']==img_name]['rank'].values[0])

            mse, mae, r2, rmse = compute_accuracy_regression(pred_list, true_list)
            city_name = city_name.replace(' ', '')

            
            all_r2_list.append(r2)
            all_mae_list.append(mae)
            all_rmse_list.append(rmse)
            all_mse_list.append(mse)
            all_model_name_list.append(model_name)
            all_city_pred_list.append(city_name)
            
            all_city_pred.extend(pred_list)
            all_city_true.extend(true_list)
                
        all_city_mse, all_city_mae, all_city_r2, all_city_rmse = compute_accuracy_regression(all_city_pred, all_city_true)
        
        all_mse_list.append(all_city_mse)
        all_r2_list.append(all_city_r2)
        all_mae_list.append(all_city_mae)
        all_rmse_list.append(all_city_rmse)
        all_model_name_list.append(model_name)
        all_city_pred_list.append('all_city')
        print(model_name, all_city_r2)

    df = pd.DataFrame({'City_Name': all_city_pred_list, 'Model_Name': all_model_name_list, 'r2': all_r2_list, \
            'MAE': all_mae_list, 'RMSE': all_rmse_list, 'MSE': all_mse_list})
    df.to_csv(os.path.join(REMOTE_SENSING_RESULTS_PATH, save_name), index=False)


def eval_object(city_name_list, model_name_list, save_name):
    with open(os.path.join(REMOTE_SENSING_PATH, 'all_city_img_object_set.json'),'r') as f:
        all_city_img_object_set = json.load(f)
    
    select_category_list = [
        "Bridge", "Stadium", "Ground Track Field", "Baseball Field", \
        "Overpass", "Airport", "Golf Field", "Storage Tank", \
        "Roundabout", "Swimming Pool", "Soccer Ball Field", "Harbor", "Tennis Court", \
        "Windmill", "Basketball Court", "Dam", "Train Station"]

    select_category_list = sorted(select_category_list)

    all_city_accurecy_list = []
    all_city_precision_list = []
    all_city_recall_list = []
    all_city_f1_list = []
    all_city_pred_list = []
    all_city_sub_category_list = []
    all_model_name_list = []
    
    # TODO need to prepare data, due to the absence of safegraph
    try:
        city_name_list.remove("Shanghai")
        city_name_list.remove("Beijing")
    except ValueError as e:
        pass

    for one_city_name in city_name_list:
        sample_df = pd.read_csv(os.path.join(REMOTE_SENSING_PATH, one_city_name+'_img_indicators.csv'))
        img_name_list = sample_df["img_name"].to_list()

        for model_name in model_name_list:
            print(model_name, one_city_name)
            pred_city_set = dict()
            try:
                json_file_path = os.path.join(REMOTE_SENSING_RESULTS_PATH, one_city_name+'_'+model_name+'_object.jsonl')
                data = load_json_file(json_file_path)
            except FileNotFoundError as e:
                try:
                    json_file_path = os.path.join(REMOTE_SENSING_RESULTS_PATH, one_city_name+'_'+model_name+'_objects.jsonl')
                    data = load_json_file(json_file_path)
                except FileNotFoundError as e:
                    print("File not found! City:{} Model:{}".format(one_city_name, model_name))
                    continue

            if one_city_name not in pred_city_set:
                pred_city_set[one_city_name] = dict()

            for item in data:
                img_name = item['img_name'].split('.')[0]
                if img_name not in img_name_list:
                    continue

                if img_name not in pred_city_set[one_city_name]:
                    pred_city_set[one_city_name][img_name] = dict()
                text = item['text']
                for obj in select_category_list:
                    if obj in text:
                        pred_city_set[one_city_name][img_name][obj] = 1
                    else:
                        pred_city_set[one_city_name][img_name][obj] = 0      

            for one_sub_category in select_category_list:
                one_city_pred = []
                one_city_true = []
                for img_name in pred_city_set[one_city_name]:
                    if img_name in all_city_img_object_set.keys():
                        one_city_pred.append(pred_city_set[one_city_name][img_name][one_sub_category])
                        one_city_true.append(all_city_img_object_set[img_name][one_sub_category])

                # cal the accuracy, precision, recall, f1 get report
                all_city_precision_list.append(metrics.precision_score(one_city_true, one_city_pred))
                all_city_recall_list.append(metrics.recall_score(one_city_true, one_city_pred))
                all_city_f1_list.append(metrics.f1_score(one_city_true, one_city_pred))
                all_city_accurecy_list.append(metrics.accuracy_score(one_city_true, one_city_pred))
            
                all_city_pred_list.append(one_city_name)
                all_city_sub_category_list.append(one_sub_category)
                all_model_name_list.append(model_name)

    df = pd.DataFrame()
    df["sub_category"] = all_city_sub_category_list
    df["Precision"] = all_city_precision_list
    df["Recall"] = all_city_recall_list
    df["f1"] = all_city_f1_list
    df["Infrastructure_Accuracy"] = all_city_accurecy_list
    df["City_Name"] = all_city_pred_list
    df["Model_Name"] = all_model_name_list
    df.to_csv(os.path.join(REMOTE_SENSING_RESULTS_PATH, save_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--task_name", type=str, default="all", choices=["population", "objects", "all"])
    args = parser.parse_args()

    # 获取 city_name 和 model_name
    city_model_pairs = get_city_and_model_from_filename(REMOTE_SENSING_RESULTS_PATH)
    city_name_list = list(set([pair[0] for pair in city_model_pairs]))
    model_name_list = list(set([pair[1] for pair in city_model_pairs]))

    save_name_pop = os.path.join(REMOTE_SENSING_RESULTS_PATH, "population_benchmark_results.csv")
    save_name_object = os.path.join(REMOTE_SENSING_RESULTS_PATH, "object_benchmark_results.csv")

    for city in city_name_list:
        assert city in CITY_BOUNDARY.keys()

    for model_name in model_name_list:
        assert model_name in VLM_MODELS
    
    if args.task_name in ["objects", "all"]:
        eval_object(city_name_list=city_name_list, model_name_list=model_name_list, save_name=save_name_object)
    if args.task_name in ["population", "all"]:
        eval_pop(city_name_list=city_name_list, model_name_list=model_name_list, save_name=save_name_pop)
