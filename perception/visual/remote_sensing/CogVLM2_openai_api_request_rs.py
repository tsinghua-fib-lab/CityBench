"""
This script is designed to mimic the OpenAI API interface with CogVLM2 Chat
It demonstrates how to integrate image and text-based input to generate a response.
Currently, the model can only handle a single image.
Therefore, do not use this script to process multiple images in one conversation. (includes images from history)
And it only works on the chat model, not the base model.
"""
import requests
import json
import base64

base_url = "http://0.0.0.0:8000"


def create_chat_completion(model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):
    """
    This function sends a request to the chat API to generate a response based on the given messages.

    Args:
        model (str): The name of the model to use for generating the response.
        messages (list): A list of message dictionaries representing the conversation history.
        temperature (float): Controls randomness in response generation. Higher values lead to more random responses.
        max_tokens (int): The maximum length of the generated response.
        top_p (float): Controls diversity of response by filtering less likely options.
        use_stream (bool): Determines whether to use a streaming response or a single response.

    The function constructs a JSON payload with the specified parameters and sends a POST request to the API.
    It then handles the response, either as a stream (for ongoing responses) or a single message.
    """

    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    
    
    if response.status_code == 200:
        if use_stream:
            # stream response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # single response
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
            
            return content
    else:
        print("Error:", response.status_code)
        return None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

import os
import pandas as pd

def simple_image_chat(use_stream=True, args=None):

    # Prepare data
    img_data_df = pd.read_csv(args.data_csv_path+"/"+args.city_name+"_img_indicators.csv")
    folder_path = f"/data1/zhangxin/llm_benchmark/data/{args.city_name}"
    img_data = img_data_df['img_name'].tolist()

    prompt =  "Please analyze the population density of the image shown comparing to all cities around the world. \
        Rate the population density in a degree from 0.0 to 9.9, where higher rating represents higher population density. \
            You should provide ONLY your answer in the exact format: 'My answer is X.X.', where 'X.X' represents your rating of the population density."

    arguments = [
        {
            "image_file": os.path.abspath(folder_path + "/" + l + ".png"),
            "question": prompt, 
        }
        for l in img_data
    ]    
    
    from tqdm import tqdm
    
    states = [None] * len(img_data)
    for i in tqdm(range(len(arguments))):
        image_file = arguments[i]["image_file"]
        question = arguments[i]["question"]
        img_url = f"data:image/jpeg;base64,{encode_image(image_file)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                        },
                    },
                ],
            },
        ]
        ret = create_chat_completion(args.model, messages=messages, use_stream=use_stream)
        if ret is not None:
            states[i] = ret

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
  

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, default="/data1/zhangxin/llm_benchmark/data/sample_500")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=["img2loc","img2pop","img2carbon","img2city","img2cityloc"], default="img2pop")
    parser.add_argument("--output_dir", type=str, default="/data1/ouyangtianjian/Gemini_GPT/results_rs")
    parser.add_argument("--city_name", type=str, default="NewYork")
    parser.add_argument("--model", type=str, default="cogvlm2")
    
    args = parser.parse_args()
    
    simple_image_chat(False, args=args)
