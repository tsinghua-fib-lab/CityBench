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
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon","img2object","img2cityloc"], default="img2cityloc")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/llm_benchmark/code/InternVL/results/")
    parser.add_argument("--city_name", type=str, default="NewYork")
    parser.add_argument("--dataset_type", type=str, default="sv")
    parser.add_argument("--model", type=str, default="InternVL-Chat-v1-5")
    args = parser.parse_args()

    pipe = pipeline('/data1/ouyangtianjian/llm_benchmark/code/pretrain/InternVL-Chat-V1-5',
                    backend_config=TurbomindEngineConfig(session_len=8192))
    gen_config = GenerationConfig(temperature=0.2)

    # Load Street View data
    folder_path = f"/data1/ouyangtianjian/Gemini_GPT/{args.city_name}_CUT"
    df = pd.read_csv(f"/data1/ouyangtianjian/Gemini_GPT/{args.city_name}_data.csv")
    img_name_list = folder_path + '/' + df['img_name'].astype(str)

    prompt = "Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \
    you can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'.\
        Only answer with the city name and location. Do not output any explanational sentences. Example Answer: Los Angeles. (34.148331, -118.324755)."   
 
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



