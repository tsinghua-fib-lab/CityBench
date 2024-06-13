import json
import pandas as pd
import re
import os
import sklearn.metrics as metrics
import pandas as pd
import argparse

import glob
def compute_accuracy_regression(pred_list, true_list):
    mse = metrics.mean_squared_error(true_list, pred_list)
    mae = metrics.mean_absolute_error(true_list, pred_list)
    r2 = metrics.r2_score(true_list, pred_list)
    
    rmse = metrics.mean_squared_error(true_list, pred_list, squared=False)
    return mse, mae, r2,rmse
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


  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--city_name', type=str, default='SanFrancisco', help='city name')
    parser.add_argument('--model_name', type=str, choices=['cogvlm2','llava34b','MiniCPM-V-2.5','llama3_llava_next_8b'], help='model name')
    parser.add_argument('--img_indicators_csv_path', type=str, default='', help='img_indicators_csv_path')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--jsonl_path', type=str, default='', help='jsonl_path')
    args = parser.parse_args()
    
    
    city_name_list = ['SanFrancisco', 'NewYork', 'Beijing', 'Shanghai', 'Mumbai', 'Tokyo', 'London', 'Paris', 'Moscow', 'SaoPaulo', 'Nairobi', 'CapeTown', 'Sydney']
    model_name_list = ["cogvlm2","llava34b","MiniCPM-V-2.5","llama3_llava_next_8b","GPT4o","Qwen"]

    city_name = args.city_name
    model_name = args.model_name


    assert city_name in city_name_list
    assert model_name in model_name_list
    

    city_name_list = [city_name]
    model_name_list = [model_name]
    indicator_name = 'pop'

    all_csv_file_name = glob.glob(args.img_indicators_csv_path+'/*.csv')
    all_csv_df = pd.DataFrame()
    for i in all_csv_file_name:
        df = pd.read_csv(i)
        all_csv_df = pd.concat([all_csv_df, df], axis=0)
        
    all_csv_df.reset_index(drop=True, inplace=True)

    gt_ranking = normalized_fractional_ranking(all_csv_df['worldpop'].values)
    gt_ranking = [int(r * 100.0) / 10.0 if r < 1.0 else 9.9 for r in gt_ranking]
    # give the ranking to the dataframe
    all_csv_df["rank"] = gt_ranking

    all_city_name_list = [x.split('_')[0] for x in all_csv_file_name]



    all_r2_list = []
    all_mae_list = []
    all_rmse_list = []
    all_mse_list = []
    all_city_pred_list = []
    all_model_name_list = []

    for model_name in model_name_list:
        all_city_pred = []
        all_city_true = []

        for city_name in all_city_name_list:

            json_file_path = args.jsonl_path+'/'+city_name+'/'+model_name+'/img2'+indicator_name+'.jsonl'

            data = []
            with open(json_file_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))

            pred_list = [] 
            true_list = [] 
            for item in data:
                text = item['text']
                
                img_name = item['img_name'].split('/')[-1].split('.')[0]


                match = re.search(r"(\d+\.\d+)", text)
                if match:
                    
                    rating = float(match.group(0))
                    pred_list.append(rating)
                    true_list.append(all_csv_df[all_csv_df['img_name']==img_name]['rank'].values[0])

            mse, mae, r2,rmse = compute_accuracy_regression(pred_list, true_list)
            city_name = city_name.replace(' ', '')

            
            all_r2_list.append(r2)
            all_mae_list.append(mae)
            all_rmse_list.append(rmse)
            all_mse_list.append(mse)
            all_model_name_list.append(model_name)
            all_city_pred_list.append(city_name)
            
            all_city_pred.extend(pred_list)
            all_city_true.extend(true_list)
                
        all_city_mse, all_city_mae, all_city_r2,all_city_rmse = compute_accuracy_regression(all_city_pred, all_city_true)
        
        all_mse_list.append(all_city_mse)
        all_r2_list.append(all_city_r2)
        all_mae_list.append(all_city_mae)
        all_rmse_list.append(all_city_rmse)
        all_model_name_list.append(model_name)
        all_city_pred_list.append('all_city')
        print(model_name, all_city_r2)
            
    

    df = pd.DataFrame({'city': all_city_pred_list, 'model': all_model_name_list, 'r2': all_r2_list, \
            'mae': all_mae_list, 'rmse': all_rmse_list,'mse': all_mse_list})
    df.to_csv(args.output_dir+'/'+args.city_name+'_'+args.model_name+'_pop.csv',index=False)


