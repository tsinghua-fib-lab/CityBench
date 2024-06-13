import pandas as pd
from dashscope import MultiModalConversation
import dashscope
from tqdm import tqdm
import time

dashscope.api_key = 'Your API Key'

def call_with_local_file(file_path):
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': file_path
            },
            {
                'text': prompt
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    return response

prompt = "Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \
you can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'.\
    Only answer with the city name and location. Do not output any explanational sentences. Example Answer: Los Angeles. (34.148331, -118.324755)."   


for City in ['CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'SaoPaulo', 'Sydney', 'Tokyo', 'NewYork', 'SanFrancisco', 'Nairobi', 'Beijing', 'Shanghai']:

    folder_path = f"/data1/ouyangtianjian/Gemini_GPT/{City}_CUT"
    df = pd.read_csv(f"/data1/ouyangtianjian/Gemini_GPT/{City}_data.csv")
    lat = [img.split("&")[1] for img in df['img_name']]
    lng = [img.split("&")[2] for img in df['img_name']]
    img_name_list = folder_path + '/' + df['img_name'].astype(str)

    with open(f"/data1/ouyangtianjian/Gemini_GPT/results_Qwen/{City}.txt", "a") as f:
        for file_path in tqdm(img_name_list):         
            response = call_with_local_file(file_path)
            try:
                response_text = response["output"]["choices"][0]["message"]["content"][0]['text']
                f.write(file_path.split("/")[-2] + '/' + file_path.split("/")[-1] + ': ' + response_text + '\n')
            except:
                f.write(file_path.split("/")[-2] + '/' + file_path.split("/")[-1] + ': ERROR. ' + response + '\n')
