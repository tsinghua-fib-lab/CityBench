import os
import json
import random
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
models_to_test = [
    'llava_next_llama3',
    'InternVL2-40B',
    'cogvlm2-llama3-chat-19B',
    'MiniCPM-Llama3-V-2_5',
    'llava_next_yi_34b'
]
city_names = ['CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'SaoPaulo', 'Sydney', 'Tokyo', 'NewYork', 'SanFrancisco', 'Beijing', 'Shanghai', 'Nairobi']
'''

model_name = "llava_next_llama3"
city_name = "Paris"
num_test = 100

prompt = "Suppose you are an expert in geo-localization. Please first analyze which city is this image taken from, and then make a prediction of the longitude and latitude value of its location. \
you can choose among: 'Los Angeles', 'Nakuru', 'Johannesburg', 'Rio de Janeiro', 'CapeTown', 'London', 'Moscow', 'Mumbai', 'Paris', 'Sao Paulo', 'Sydney', 'Tokyo', 'New York', 'Guangzhou', 'Kyoto', 'Melbourne', 'San Francisco', 'Nairobi', 'Beijing', 'Shanghai', 'Bangalore', 'Marseille', 'Manchester', 'Saint Petersburg'.\
    Only answer with the city name and location. Do not output any explanational sentences. Example Answer: Los Angeles. (34.148331, -118.324755)."   

folder_path = f"./data/{city_name}"
result_folder_path = "./results"

model_result_path = os.path.join(result_folder_path, model_name)
os.makedirs(model_result_path, exist_ok=True)
print(f"Testing model: {model_name}")

if model_name in ['InternVL2-40B', 'cogvlm2-llama3-chat-19B']:
    os.system('python -m pip install transformers==4.37.0')
elif model_name in ['llava_next_llama3', 'MiniCPM-Llama3-V-2_5', 'llava_next_yi_34b']:
    os.system('python -m pip install transformers==4.44.0')
from vlmeval.config import supported_VLM
model = supported_VLM[model_name]()

for image in os.listdir(folder_path)[:num_test]:
    image_path = os.path.join(folder_path, image)
    question = prompt

    response = model.generate([image_path, question])

    with open(os.path.join(model_result_path, f"{city_name}_test_result.jsonl"), "a") as f:
        f.write(json.dumps({
            "image_path": image_path,
            "response": response
        }) + "\n")

print(f"{city_name} done!")