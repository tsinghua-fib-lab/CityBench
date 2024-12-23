import os
import json
import csv
from config import RESULTS_PATH

def extract_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        results = [] 
        for entry in data:
            if isinstance(entry, dict): 
                city_name = entry.get('city_name', 'Unknown') 
                model_name = entry['model']['model_name'] 
                avg_queue_length = entry['performance']['average_queue_length'] 
                avg_traveling_time = entry['performance']['average_traveling_time']
                throughput = entry['performance']['throughput']  
                results.append([city_name, model_name, avg_queue_length, avg_traveling_time, throughput])

        return results
    else:
        raise ValueError(f"The format of {json_file} is not correct!")

def process_json_files_in_folder(folder_path, output_csv_file):
   with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['City_Name', 'Model_Name', 'Average_Queue_Length', 'Average_Traveling_Time', 'Throughput'])
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                json_file_path = os.path.join(folder_path, filename)
                data = extract_data_from_json(json_file_path)
                for row in data:
                    writer.writerow(row)
                    
folder_path = os.path.join(RESULTS_PATH, "signal_results")
output_csv_file = os.path.join(folder_path, "signal_benchmark_results.csv")

process_json_files_in_folder(folder_path, output_csv_file)
print(f"Signal results have been saved!")

