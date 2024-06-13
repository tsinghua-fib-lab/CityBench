import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from chat import MiniCPMVChat, img2base64

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, default="/data1/ouyangtianjian/Gemini_GPT")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon","img2cityloc"], default="img2cityloc")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/Gemini_GPT/results_final")
    parser.add_argument("--city_name", type=str, default="SanFrancisco")
    parser.add_argument("--model", type=str, default="MiniCPM-V-2.5")
    args = parser.parse_args()
    
    chat_model = MiniCPMVChat('/data1/ouyangtianjian/llm_benchmark/code/pretrain/MiniCPM-Llama3-V-2_5')

    folder_path = f"/data1/ouyangtianjian/Gemini_GPT/{args.city_name}_CUT"
    df = pd.read_csv(args.data_csv_path)
    df['img_name'] = folder_path + '/' + df['img_name'].astype(str)
    img_data = df['img_name'].tolist()
    
    prompt = "Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \
    you can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'.\
        Only answer with the city name and location. Do not output any explanational sentences. Example Answer: 'Los Angeles'. (34.148331, -118.324755)."        

    arguments = [
        {
            "image_file": l,
            "question": prompt
        }
        for l in img_data
    ]    
    states = [None] * len(img_data)

    for i in tqdm(range(len(arguments))):
        image_file = arguments[i]["image_file"]
        question = arguments[i]["question"]
        im_64 = img2base64(image_file)
        msgs = [{"role": "user", "content": question}]
        inputs = {"image": im_64, "question": json.dumps(msgs)}
        answer = chat_model.chat(inputs)
        states[i] = answer
    
    print(f"Write output to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + "/" + args.city_name):
        os.makedirs(args.output_dir + "/" + args.city_name)
    if not os.path.exists(args.output_dir + "/" + args.city_name + "/" + args.model+"/"):
        os.makedirs(args.output_dir + "/" + args.city_name + "/" + args.model+"/")
    with open(os.path.join(args.output_dir, args.city_name, args.model, args.task + ".jsonl"), "w") as fout:
        for i in range(len(img_data)):
            value = {
                "img_name": img_data[i],
                "text": states[i].strip(),
            }
            fout.write(json.dumps(value) + "\n")
