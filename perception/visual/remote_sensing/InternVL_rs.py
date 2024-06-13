import os
import json
import argparse
import pandas as pd
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon","img2object"], default="img2loc")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/llm_benchmark/code/InternVL/results_rs/")
    parser.add_argument("--city_name", type=str, default="NewYork")
    parser.add_argument("--dataset_type", type=str, default="si")
    parser.add_argument("--model", type=str, default="InternVL-Chat-v1-5")
    
    args = parser.parse_args()

    pipe = pipeline('/data1/ouyangtianjian/llm_benchmark/code/pretrain/InternVL-Chat-V1-5',
                    backend_config=TurbomindEngineConfig(session_len=8192))
    gen_config = GenerationConfig(temperature=0.2)

    # Load Satellite Image data
    folder_path = f"/data1/zhangxin/llm_benchmark/data/{args.city_name}"
    df = pd.read_csv(f"/data1/zhangxin/llm_benchmark/data/sample_500/{args.city_name}_img_indicators.csv")
    img_name_list = folder_path + '/' + df['img_name'].astype(str) + '.png'

    prompt =  "Please analyze the population density of the image shown comparing to all cities around the world. \
        Rate the population density in a degree from 0.0 to 9.9, where higher rating represents higher population density. \
            You should provide ONLY your answer in the exact format: 'My answer is X.X.', where 'X.X' represents your rating of the population density."
    
    # prompt = '''Suppose you are an expert in identifying urban infrastructure.
    # Please analyze this image and  choose the infrastructure type from the following list:
    # ['Bridge', 'Stadium', 'Ground Track Field', 'Baseball Field',
    # 'Overpass', 'Airport', 'Golf Field', 'Storage Tank', 
    # 'Roundabout', 'Swimming Pool', 'Soccer Ball Field', 'Harbor', 'Tennis Court', 
    # 'Windmill', 'Basketball Court', 'Dam', 'Train Station']
    # Please meet the following requirements:
    # 1. If you can identify multiple infrastructure types, please provide all of them.
    # 2. You must provide the answer in the exact format: 'My answer is X, Y, ... and Z.' where 'X, Y, ... and Z' represent the infrastructure types you choose.
    # 3. Don't output any other sentences.
    # 4. If you cannot choose any of the infrastructure types from the list, please choose 'Other'.
    # '''

    image_urls = img_name_list
    prompts = [(prompt, load_image(img_url)) for img_url in image_urls]
    response = pipe(prompts, gen_config=gen_config,use_tqdm=True)

    # Save results
    print(f"Write output to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + "/" + args.city_name):
        os.makedirs(args.output_dir + "/" + args.city_name)
    if not os.path.exists(args.output_dir + "/" + args.city_name + "/" + args.model+"/"):
        os.makedirs(args.output_dir + "/" + args.city_name + "/" + args.model+"/")
    with open(os.path.join(args.output_dir, args.city_name, args.model, args.task + "_"+args.dataset_type+".jsonl"), "w") as fout:
        for i in range(len(image_urls)):
            value = {
                "img_name": image_urls[i].split('/')[-1],
                "text": response[i].text.strip(),
            }
            fout.write(json.dumps(value) + "\n")



