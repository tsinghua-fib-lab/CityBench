import os
import pandas as pd
from sklearn.metrics import f1_score
import ast
import csv
import numpy as np
from config import RESULTS_PATH

def get_acc1_f1(df):
    acc1 = (df['prediction'] == df['ground_truth']).sum() / len(df)
    acc1 = round(acc1, 9)
    # print(f"acc1:{acc1}")
    # print(f"sum:{(df['prediction'] == df['ground_truth']).sum()}")
    # print(f"len:{len(df)}")
    f1 = f1_score(df['ground_truth'], df['prediction'], average='weighted')
    return acc1, f1

def get_is_correct(row):
    pred_list = row['prediction']
    if row['ground_truth'] in pred_list:
        row['is_correct'] = True
    else:
        row['is_correct'] = False
    
    return row


def get_is_correct10(row):
    pred_list = row['top10']
    if row['ground_truth'] in pred_list:
        row['is_correct10'] = True
    else:
        row['is_correct10'] = False
        
    pred_list = row['top5']
    if row['ground_truth'] in pred_list:
        row['is_correct5'] = True
    else:
        row['is_correct5'] = False

    pred = row['top1']
    if pred == row['ground_truth']:
        row['is_correct1'] = True
    else:
        row['is_correct1'] = False
    
    return row


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def get_ndcg(prediction, targets, k=10):
    """
    Calculates the NDCG score for the given predictions and targets.

    Args:
        prediction (Nxk): list of lists. the softmax output of the model.
        targets (N): torch.LongTensor. actual target place id.

    Returns:
        the sum ndcg score
    """
    for _, xi in enumerate(prediction):
        if len(xi) < k:
            #print(f"the {i}th length: {len(xi)}")
            xi += [-5 for _ in range(k-len(xi))]
        elif len(xi) > k:
            xi = xi[:k]
        else:
            pass
    
    n_sample = len(prediction)
    prediction = np.array(prediction)
    targets = np.broadcast_to(targets.reshape(-1, 1), prediction.shape)
    hits = first_nonzero(prediction == targets, axis=1, invalid_val=-1)
    hits = hits[hits>=0]
    ranks = hits + 1
    ndcg = 1 / np.log2(ranks + 1)
    return np.sum(ndcg) / n_sample

def safe_convert_to_int(x):
    try:
        # 这里尝试先将字符串转换为浮点数，然后再转换为整数
        return int(float(x))
    except (ValueError, TypeError):
        # 如果转换失败（可能是非数值型字符串），返回 -100 作为占位符
        return -100
    

def cal_metrics(output_dir):
    folder_name = os.path.basename(output_dir)  
    parts = folder_name.split('_')

    if len(parts) < 4:
        print(f"Skipping invalid folder: {folder_name}")
        return None, None, None, None
    
    model = "_".join(parts[:-3])  
    city = parts[-3]
    
    file_list = [file for file in os.listdir(output_dir) if file.endswith('.csv')]
    file_path_list = [os.path.join(output_dir, file) for file in file_list]

    df = pd.DataFrame({
        'user_id': None,
        'ground_truth': None,
        'prediction': None,
        'reason': None
    }, index=[])

    for file_path in file_path_list:
        iter_df = pd.read_csv(file_path)
        df = pd.concat([df, iter_df], ignore_index=True)
        
    df_cleaned = df.dropna(subset=['prediction', 'ground_truth'])
    df_cleaned['prediction'] = df_cleaned['prediction'].apply(safe_convert_to_int)
    df_cleaned['ground_truth'] = df_cleaned['ground_truth'].apply(safe_convert_to_int)
    
    acc1, f1 = get_acc1_f1(df_cleaned)
    return model, city, acc1, f1

def process_results(main_dir, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model_Name', 'City_Name', 'Acc@1', 'F1'])
        # 遍历主文件夹中的所有子文件夹
        for folder in os.listdir(main_dir):
            folder_path = os.path.join(main_dir, folder)
            if os.path.isdir(folder_path):
                model, city, acc1, f1 = cal_metrics(folder_path)
                if model and city:
                    writer.writerow([model, city, acc1, f1])

    print(f"Mobility results have been saved!")



if __name__ == "__main__":
    main_dir = os.path.join(RESULTS_PATH, "prediction_results")
    csv_file = os.path.join(main_dir, "mobility_benchmark_result.csv")

    process_results(main_dir, csv_file)
