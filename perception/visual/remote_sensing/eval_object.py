
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle as pkl
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import os
import json
import pandas as pd
import os
import argparse
select_category_list = ["Bridge", "Stadium", "Ground Track Field", "Baseball Field", "\
    Overpass", "Airport", "Golf Field", "Storage Tank", \
        "Roundabout", "Swimming Pool", "Soccer Ball Field", "Harbor", "Tennis Court", \
            "Windmill", "Basketball Court", "Dam", "Train Station"]

select_category_list = sorted(select_category_list)

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--city_name', type=str, default='SanFrancisco', help='city name')
    parser.add_argument('--model_name', type=str, choices=['cogvlm2','llava34b','MiniCPM-V-2.5','llama3_llava_next_8b'], help='model name')
    parser.add_argument('--img_indicators_csv_path', type=str, default='', help='img_indicators_csv_path')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--jsonl_path', type=str, default='', help='jsonl_path')
    args = parser.parse_args()


    with open('all_city_img_object_set.json','r') as f:
        all_city_img_object_set = json.load(f)
    city_name_list = ['SanFrancisco', 'NewYork', 'Beijing', 'Shanghai', 'Mumbai', 'Tokyo', 'London', 'Paris', 'Moscow', 'SaoPaulo', 'Nairobi', 'CapeTown', 'Sydney']
    model_name_list = ["cogvlm2","llava34b","MiniCPM-V-2.5","llama3_llava_next_8b","GPT4o","Qwen"]
    assert args.city_name in city_name_list
    assert args.model_name in model_name_list
    
    city_name_list = [args.city_name]
    model_name_list = [args.model_name]
    
    all_city_accurecy_list = []
    all_city_precision_list = []
    all_city_recall_list = []
    all_city_f1_list = []
    all_city_pred_list = []
    all_city_sub_category_list = []
    all_model_name_list = []
    
    
    city_name_list.remove("Shanghai")
    city_name_list.remove("Beijing")


    indicator_name = 'object'

    for one_city_name in  tqdm(city_name_list):

        sample_df = pd.read_csv(args.img_indicators_csv_path+'/'+one_city_name+'_img_indicators.csv')




        for model_name in (model_name_list):
            pred_city_set = dict()


            json_file_path = args.jsonl_path+'/'+one_city_name+'/'+model_name+'/img2'+indicator_name+'.jsonl'

            data = []
            with open(json_file_path, 'r') as file:
                for line in file:
                    
                    data.append(json.loads(line))
            if one_city_name not in pred_city_set:
                pred_city_set[one_city_name] = dict()

            for item in data:
                img_name = item['img_name'].split('.')[0]
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
                one_city_true = [  ]
                for img_name in pred_city_set:
                    if img_name in all_city_img_object_set.keys():
                        one_city_pred.append(pred_city_set[one_city_name][img_name][one_sub_category])
                        one_city_true.append(all_city_img_object_set[img_name][one_sub_category])

                # cal the accuracy, precision, recall, f1 get report

                all_city_precision_list.append(precision_score(one_city_true, one_city_pred))
                all_city_recall_list.append(recall_score(one_city_true, one_city_pred))
                all_city_f1_list.append(f1_score(one_city_true, one_city_pred))
                all_city_accurecy_list.append(accuracy_score(one_city_true, one_city_pred))
            
                all_city_pred_list.append(one_city_name)
                all_city_sub_category_list.append(one_sub_category)
                all_model_name_list.append(model_name)

    df = pd.DataFrame()
    df["sub_category"] = all_city_sub_category_list
    df["precision"] = all_city_precision_list
    df["recall"] = all_city_recall_list
    df["f1"] = all_city_f1_list
    df["accuracy"] = all_city_accurecy_list
    df["city"] = all_city_pred_list
    df["model_name"] = all_model_name_list
    df.to_csv(args.output_dir+'/'+args.city_name+'_'+args.model_name+'_object.csv',index=False)


            