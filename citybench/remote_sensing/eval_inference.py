import os
import argparse
import pandas as pd
from setproctitle import setproctitle

from tqdm import tqdm
import json

from config import REMOTE_SENSING_PATH, REMOTE_SENSING_RESULTS_PATH
from serving.vlm_serving import VLMWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--data_name', type=str, default="mini", help='dataset size')
    parser.add_argument('--task_name', type=str, default='population', help='task name', choices=["population", "objects"])
    args = parser.parse_args()

    print("Load the model")
    model_wrapper = VLMWrapper(args.model_name)
    model = model_wrapper.get_vlm_model()

    print("Load the image indicator csv")
    df = pd.read_csv(os.path.join(REMOTE_SENSING_PATH, f"{args.city_name}_img_indicators.csv"))
    print(df.shape)
    img_name_list = [os.path.join(REMOTE_SENSING_PATH, args.city_name, x+".png") for x in df['img_name'].astype(str).to_list()]
    
    #according to the task name, set the prompt
    if args.task_name == 'population':
        prompt =  "Please analyze the population density of the image shown comparing to all cities around the world. \
            Rate the population density in a degree from 0.0 to 9.9, where higher rating represents higher population density. \
                You should provide ONLY your answer in the exact format: 'My answer is X.X.', where 'X.X' represents your rating of the population density."
    elif args.task_name == 'objects':
        prompt = '''Suppose you are an expert in identifying urban infrastructure.
        Please analyze this image and choose the infrastructure type from the following list:
        ['Bridge', 'Stadium', 'Ground Track Field', 'Baseball Field',
        'Overpass', 'Airport', 'Golf Field', 'Storage Tank', 
        'Roundabout', 'Swimming Pool', 'Soccer Ball Field', 'Harbor', 'Tennis Court', 
        'Windmill', 'Basketball Court', 'Dam', 'Train Station']
        Please meet the following requirements:
        1. If you can identify multiple infrastructure types, please provide all of them.
        2. You must provide the answer in the exact format: 'My answer is X, Y, ... and Z.' where 'X, Y, ... and Z' represent the infrastructure types you choose.
        3. Don't output any other sentences.
        4. If you cannot choose any of the infrastructure types from the list, please choose 'Other'.
        '''

    #adjust the dataset size
    if args.data_name == "all":
        img_name_list = img_name_list
    elif args.data_nane == "mini":
        img_name_list = img_name_list[:int(len(img_name_list)*0.01)]
    
    # read existing data
    try:
        his_data = pd.read_json(os.path.join(REMOTE_SENSING_RESULTS_PATH, f"{args.city_name}_{args.model_name}_{args.task_name}.jsonl"), lines=True)
        his_imgs = his_data["img_name"].to_list()
        his_ress = his_data["text"].to_list()
        his_data_list = list(zip(his_imgs, his_ress))
    except:
        his_imgs, his_ress, his_data_list = [], [], []

    print("Generate the response")
    response = []
    for img_name in tqdm(img_name_list):
        img_name_slim = img_name.split('/')[-1]
        if img_name_slim in his_imgs:
            continue
        ret = model.generate([img_name, prompt])
        response.append([img_name_slim, ret])

    # Save the response
    with open(os.path.join(REMOTE_SENSING_RESULTS_PATH, f"{args.city_name}_{args.model_name}_{args.task_name}.jsonl"), "w") as fout:
        # saving old data
        for i in range(len(his_data_list)):
            value = {
                "img_name": his_data_list[i][0],
                "text": his_data_list[i][1],
            }
            fout.write(json.dumps(value) + "\n")
        
        # saving new data
        for i in range(len(response)):
            value = {
                "img_name": response[i][0],
                "text": response[i][1],
            }
            fout.write(json.dumps(value) + "\n")
    
    model_wrapper.clean_proxy()
