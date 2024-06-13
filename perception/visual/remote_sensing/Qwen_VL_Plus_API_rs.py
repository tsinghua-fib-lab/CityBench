import pandas as pd
from dashscope import MultiModalConversation
import dashscope
from tqdm import tqdm
import json

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

# prompt =  "Please analyze the population density of the image shown comparing to all cities around the world. \
#     Rate the population density in a degree from 0.0 to 9.9, where higher rating represents higher population density. \
#         You should provide ONLY your answer in the exact format: 'My answer is X.X.', where 'X.X' represents your rating of the population density."


prompt = '''Suppose you are an expert in identifying urban infrastructure.
Please analyze this image and  choose the infrastructure type from the following list:
['Bridge', 'Stadium', 'Ground Track Field', 'Baseball Field',
'Overpass', 'Airport', 'Golf Field', 'Storage Tank', 
'Roundabout', 'Swimming Pool', 'Soccer Ball Field', 'Harbor', 'Tennis Court', 
'Windmill', 'Basketball Court', 'Dam', 'Train Station']
Please meet the following requirements:
1. If you can identify multiple infrastructure types, please provide all of them.
2. You must provide the answer in the exact format: 'My answer is X, Y, ... and Z.' where 'X, Y, ... and Z' represent the infrastructure types you choose.
3. Don't output any other sentences.
4. If you cannot choose any of the infrastructure types from the list, please choose 'Other'.
'''

for City in ['CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'SaoPaulo', 'Sydney', 'Tokyo', 'NewYork', 'SanFrancisco', 'Nairobi', 'Beijing', 'Shanghai']:

    folder_path = f"/data1/zhangxin/llm_benchmark/data/{City}"
    df = pd.read_csv(f"/data1/zhangxin/llm_benchmark/data/sample_500/{City}_img_indicators.csv")

    with open(f"/data1/ouyangtianjian/Gemini_GPT/results_Qwen_rs_funda_facility/{City}.jsonl", "a") as f:
        for img_name in tqdm(df['img_name']):
            img_path = folder_path + '/' + img_name + '.png'
            
            response = call_with_local_file(img_path)
            try:
                response_text = response["output"]["choices"][0]["message"]["content"][0]['text']
                value = {
                    "img_name": img_name,
                    "text": response_text,
                }
                f.write(json.dumps(value) + '\n')
            except:
                print(response)
                value = {
                    "img_name": img_name,
                    "text": "ERROR",
                }
                f.write(json.dumps(value) + '\n')
