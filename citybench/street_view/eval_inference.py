import os
import argparse
import pandas as pd

from tqdm import tqdm
import json

from config import RESULTS_PATH, STREET_VIEW_PATH
from serving.vlm_serving import VLMWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='InternVL2-40B', help='model name')
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    parser.add_argument('--data_name', type=str, default='mini', help='dataset size')
    parser.add_argument('--task_name', type=str, default='geoloc', help='task name', choices=["geoloc"])
    args = parser.parse_args()

    # Load the model
    model_wrapper = VLMWrapper(args.model_name)
    model = model_wrapper.get_vlm_model()

    #according to the task name, set the prompt
    if args.task_name == "geoloc":
        prompt = """Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \nYou can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'. \nOnly answer with the city name and location. Do not output any explanational sentences. Example Answer: Los Angeles. (34.148331, -118.324755)."""
    else:
        print("{} task is not supported".format(args.task_name))
        exit(0)
    
    if args.data_name == "mini":
        num_test = 5
    else:
        num_test = 500

    input_folder_path = os.path.join(STREET_VIEW_PATH, args.city_name+"_CUT")
    selected_data_list = pd.read_csv(os.path.join(STREET_VIEW_PATH, args.city_name+"_data.csv"))["img_name"].to_list()
    result_folder_path = os.path.join(RESULTS_PATH, "street_view")
    os.makedirs(result_folder_path, exist_ok=True)

    # read existing data
    try:
        his_data = pd.read_json(os.path.join(result_folder_path, f"{args.city_name}_{args.model_name}_geoloc.jsonl"), lines=True)
        his_imgs = his_data["image_path"].to_list()
        his_ress = his_data["response"].to_list()
        his_data_list = list(zip(his_imgs, his_ress))
    except:
        his_imgs, his_ress, his_data_list = [], [], []

    print("City:{} Model:{} Existing Data:{}".format(args.city_name, args.model_name, len(his_data_list)))

    # generate new
    res_list = []
    for image in tqdm(os.listdir(input_folder_path)):
        if image not in selected_data_list:
            continue
        
        image_path = os.path.join(input_folder_path, image)

        if image_path in his_imgs:
            continue

        response = model.generate([image_path, prompt])
        res_list.append([image_path, response])
    
    # save data
    with open(os.path.join(result_folder_path, f"{args.city_name}_{args.model_name}_geoloc.jsonl"), "w") as f:
        # saving old data
        for i in range(len(his_data_list)):
            image_path, response = his_data_list[i]
            f.write(json.dumps({
                "image_path": image_path,
                "response": response
            }) + "\n")

        
        for r in res_list:
            image_path, response = r
            f.write(json.dumps({
                "image_path": image_path,
                "response": response
            }) + "\n")
