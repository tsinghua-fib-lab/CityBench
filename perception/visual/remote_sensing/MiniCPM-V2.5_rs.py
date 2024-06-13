import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from chat import MiniCPMVChat, img2base64

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, default="/data1/zhangxin/llm_benchmark/data/sample_500")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon"], default="img2pop")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/Gemini_GPT/results_rs")
    parser.add_argument("--city_name", type=str, default="SanFrancisco")
    parser.add_argument("--model", type=str, default="MiniCPM-V-2.5")
    args = parser.parse_args()
    
    chat_model = MiniCPMVChat('/data1/ouyangtianjian/llm_benchmark/code/pretrain/MiniCPM-Llama3-V-2_5')

    img_data_df = pd.read_csv(args.data_csv_path+"/"+args.city_name+"_img_indicators.csv")
    img_data = img_data_df['img_name'].tolist()

    prompt =  "Please analyze the population density of the image shown comparing to all cities around the world. \
        Rate the population density in a degree from 0.0 to 9.9, where higher rating represents higher population density. \
            You should provide ONLY your answer in the exact format: 'My answer is X.X.', where 'X.X' represents your rating of the population density."
  

    arguments = [
        {
            "image_file": os.path.abspath("/data1/zhangxin/llm_benchmark/data/" + args.city_name + "/" + l + ".png"),
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
