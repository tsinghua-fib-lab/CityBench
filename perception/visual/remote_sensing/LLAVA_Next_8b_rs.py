import os
import json
import pandas as pd
import argparse
import sglang as sgl
from sglang.lang.chat_template import get_chat_template
from PIL import Image, ImageFile

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of truncated images

@sgl.function
def image_qa(s, image, question):
    s += sgl.user(sgl.image(image) + question)
    s += sgl.assistant(sgl.gen("answer"))


def batch():
    
    img_data_df = pd.read_csv(args.data_csv_path+"/"+args.city_name+"_img_indicators.csv")

    prompt =  "Please analyze the population density of the image shown comparing to all cities around the world. \
        Rate the population density in a degree from 0.0 to 9.9, where higher rating represents higher population density. \
            You should provide ONLY your answer in the exact format: 'My answer is X.X.', where 'X.X' represents your rating of the population density."

    arguments = [
        {
            "image": Image.open("/data1/zhangxin/llm_benchmark/data/" + args.city_name + "/" + img_data_df['img_name'][l] + ".png"),
            "question": prompt
        }
        for l in range(len(img_data_df))
    ]   
    
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=512,
        temperature=0.2, 
        num_threads=16,
        progress_bar=True,
        #top_p=0.7
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + "/" + args.city_name):
        os.makedirs(args.output_dir + "/" + args.city_name)
    if not os.path.exists(args.output_dir + "/" + args.city_name + "/" + args.model+"/"):
        os.makedirs(args.output_dir + "/" + args.city_name + "/" + args.model+"/")
    with open(os.path.join(args.output_dir, args.city_name, args.model, args.task + ".jsonl"), "w") as fout:
        for i in range(len(img_data_df)):
            print(img_data_df.iloc[i]['img_name'], states[i])
            value = {
                "img_name": img_data_df.iloc[i]['img_name'],
                "text": states[i]["answer"].strip(),
            }
            fout.write(json.dumps(value) + "\n")


import argparse
if __name__ == "__main__":
    import multiprocessing as mp
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, default="/data1/zhangxin/llm_benchmark/data/sample_500")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon","img2cityloc"], default="img2pop")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/Gemini_GPT/results_rs")
    parser.add_argument("--city_name", type=str, default="NewYork")
    parser.add_argument("--dataset_type", type=str, default="si")
    parser.add_argument("--model", type=str, default="llama3_llava_next_8b")
    args = parser.parse_args()
    
    mp.set_start_method("spawn", force=True)
    runtime = sgl.Runtime(
        model_path="/data1/ouyangtianjian/llm_benchmark/code/pretrain/llama3-llava-next-8b",
        tokenizer_path="/data1/ouyangtianjian/llm_benchmark/code/pretrain/llama3-llava-next-8b-tokenizer", 
    )
    runtime.endpoint.chat_template = get_chat_template("llama-3-instruct")
    sgl.set_default_backend(runtime)
    print(f"chat template: {runtime.endpoint.chat_template.name}")

    # Run a batch of requests
    print("\n========== batch ==========\n")
    batch()

    runtime.shutdown()
