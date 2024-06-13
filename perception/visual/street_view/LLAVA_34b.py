import argparse
import json
import os
import time
import tqdm
import pandas as pd
import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text, read_jsonl


@sgl.function
def image_qa(s, image_file, question):
    s += sgl.user(sgl.image(image_file) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=args.max_tokens))


def main(args):
    
    # Load Street View data
    folder_path = f"/data1/ouyangtianjian/Gemini_GPT/{args.city_name}_CUT"
    df = pd.read_csv(args.data_csv_path)
    df['img_name'] = folder_path + '/' + df['img_name'].astype(str)
    img_data = df['img_name'].tolist()

    prompt = "Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \
    you can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'.\
        Only answer with the city name and location. Do not output any explanational sentences. Example Answer: Los Angeles. (34.148331, -118.324755)."   
            
    arguments = [
        {
            "image_file":  l,
            "question": prompt, 
        }
        for l in img_data
    ]    
    states = [None] * len(img_data)

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    if args.parallel == 1:
        for i in tqdm.tqdm(range(len(img_data))):
            image_file = arguments[i]["image_file"]
            question = arguments[i]["question"]
            ret = image_qa.run(image_file=image_file, question=question, temperature=0)
            states[i] = ret
    else:
        states = image_qa.run_batch(
            arguments, 
            temperature=args.temperature, 
            num_threads=args.parallel, 
            progress_bar=True,
            
        )
    latency = time.time() - tic
    print(f"Latency: {latency:.3f}")

    # Write results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    print(f"Write output to {args.output_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) 
    if not os.path.exists(args.output_dir + "/" + args.city_name):
        os.makedirs(args.output_dir + "/" + args.city_name)
    if not os.path.exists(args.output_dir + "/" + args.city_name + "/" + args.model+"/"):
        os.makedirs(args.output_dir + "/" + args.city_name + "/" + args.model+"/")
    with open(os.path.join(args.output_dir, args.city_name, args.model, args.task + "_"+args.dataset_type+".jsonl"), "w") as fout:
        for i in range(len(img_data)):
            value = {
                "img_name": img_data[i],
                "text": states[i]["answer"].strip(),
            }
            fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, default="/data1/ouyangtianjian/Gemini_GPT/NewYork_CUT")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon","img2city","img2cityloc"], default="img2cityloc")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/Gemini_GPT/results_test")
    parser.add_argument("--city_name", type=str, default="NewYork")
    
    parser.add_argument("--dataset_type", type=str, default="sv")
    parser.add_argument("--model", type=str, default="llava34b")
    args = add_common_sglang_args_and_parse(parser)
    main(args)
