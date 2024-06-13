import os
import pickle 
import json
from PIL import Image
import base64
import numpy as np
from matplotlib import pyplot as plt
import requests
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import httpx
from multiprocessing import Pool, cpu_count

api_key = 'Your API Key'

client = OpenAI(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=api_key,
    http_client=httpx.Client(proxies='http://127.0.0.1:10190'),
)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

prompt = "Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \
you can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'.\
    Only answer with the city name and location. Do not output any explanational sentences. Example Answer: Los Angeles. (34.148331, -118.324755)."   

def call_with_local_file(file_path):
    with open(file_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }

        proxies = {
            "http": "http://127.0.0.1:10190",
            "https": "http://127.0.0.1:10190"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, proxies=proxies)
        response_json = response.json()           
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred with {file_path}: {e}")
        return "ERROR"

def process_image(args):
    file_path, City = args
    response = call_with_local_file(file_path)
    value = {
        "img_name": file_path.split('/')[-1],
        "text": response,
    }
    print(value)
    with open(f"/data1/ouyangtianjian/Gemini_GPT/results_GPT4o/{City}.jsonl", "a") as f:
        f.write(json.dumps(value) + '\n')

def process_city(City):
    folder_path = f"/data1/ouyangtianjian/Gemini_GPT/{City}_CUT"
    df = pd.read_csv(f"/data1/ouyangtianjian/Gemini_GPT/{City}_data.csv")
    img_name_list = folder_path + '/' + df['img_name'].astype(str)
    
    args = [(file_path, City) for file_path in img_name_list]

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, args), total=len(args)))

if __name__ == "__main__":
    cities = ['CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'SaoPaulo', 'Sydney', 'Tokyo', 'NewYork', 'SanFrancisco', 'Nairobi', 'Beijing', 'Shanghai']
    for City in cities:
        process_city(City)
