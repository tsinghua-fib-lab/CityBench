import requests
import json
import time
from pathlib import Path
import pandas as pd
# from haversine import haversine
import pickle
import paramiko
import os

API_KEY = os.environ.get('GOOGLE_API_KEY')

def get_metadata_from_lati_lonti(lati,longti):
    base_url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
    params = {'location':str(lati)+','+str(longti),'key':API_KEY}
    while True:
        try:
            response = requests.get(base_url, params = params)
        except (requests.exceptions.RequestException, requests.exceptions.SSLError):
            print('查询metadata API无响应')
            time.sleep(1)
        else:
            break

    results = json.loads(response.text)
    response.close()
    if results['status'] == 'OK':
        return results['pano_id'], results['location']['lat'], results['location']['lng'], results['date']
    elif results['status'] == 'ZERO_RESULTS':
        return -1, -1, -1, -1
    elif results['status'] == 'OVER_QUERY_LIMIT':
        return -2, -2, -2, -2
    else:
        print(params)
        print(results['status'])
        return -3,-3,-3,-3

def get_image_tiles(panoid):
    base_url = 'https://streetviewpixels-pa.googleapis.com/v1/tile'
    output_img_dict = {}
    for x in range(16):
        for y in [2,3,4]:
            params = {'cb_client':'maps_sv.tactile','panoid':str(panoid),'x':str(x),'y':str(y),'zoom':'4','nbt':'1','fover':'2'}
            t = 0
            while True:
                try:
                    t += 1
                    response = requests.get(base_url, params = params, timeout=(3.05,6.05))
                except (requests.exceptions.RequestException, requests.exceptions.SSLError, requests.exceptions.Timeout):
                    print('下载图片无响应，重试中')
                    print(base_url)
                    print(params)
                    if t > 3:
                        print("下载图片多次重试无效，10秒等待！")
                        time.sleep(10)
                else:
                    if response.status_code == 400:
                        print('此地参数无此设置')
                        print(params)
                        response.close()
                        break
                    else:
                        img = response.content
                        output_img_dict[str(x)+'&'+str(y)] = img
                        response.close()
                        break

    return output_img_dict

def save_img(img_dict, save_path, panoid, query_lati, query_longti, real_lati, real_longti, date):
    panoid_list = [str(panoid) for _ in range(len(img_dict))]
    query_lati_list = [str(query_lati) for _ in range(len(img_dict))]
    query_longti_list = [str(query_longti) for _ in range(len(img_dict))]
    real_lati_list = [str(real_lati) for _ in range(len(img_dict))]
    real_longti_list = [str(real_longti) for _ in range(len(img_dict))]
    date_list = [str(date) for _ in range(len(img_dict))]
    file_name_list = []
    for key, val in img_dict.items():
        x,y = key.split('&')
        file_name = str(panoid) + '&' + str(real_lati) + '&' + str(real_longti) + '&' + str(x) + '&' + str(y) + '.jpg'
        file_name_list.append(file_name)
        total_file_name = save_path.joinpath(file_name)
        with open(total_file_name,mode='wb') as f:
            f.write(val)

    new_metainfo_dataframe = pd.DataFrame({'panoid':panoid_list,'query_lati':query_lati_list, 'query_longti':query_longti_list, 'real_lati':real_lati_list,'real_longti':real_longti_list,'date':date_list,'file_name':file_name_list})
    return new_metainfo_dataframe

def update_and_save_metainfo(old_metainfo_dataframe, new_metainfo_dataframe, save_path):
    updated_metainfo = pd.concat([old_metainfo_dataframe, new_metainfo_dataframe])
    updated_metainfo.to_csv(save_path.joinpath('meta_info.csv'), sep = ',',index = False)
    return updated_metainfo

def save_query_set(query_lati_longti_set,save_path):
    with open(save_path.joinpath('query_lati_longti_set.pkl'),'wb') as f:
        pickle.dump(query_lati_longti_set,f)


def resume_from_history_info(save_path):
    try:
        meta_info = pd.read_csv(save_path.joinpath('meta_info.csv'),dtype=str)
        
        with open(save_path.joinpath('query_lati_longti_set.pkl'),'rb') as f:
            query_lati_longti_set = pickle.load(f)
        return meta_info, query_lati_longti_set
    except:
        print('未查找到历史信息')
        return pd.DataFrame(), set()

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def transfer_file(source_path, destination_host, destination_path, username, password=None, private_key=None, port=35167):
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to source host
        if password:
            ssh.connect(destination_host, port=port, username=username, password=password)
        else:
            ssh.connect(destination_host, port=port, username=username, key_filename=private_key)

        # SCP file from source host to destination host
        scp = paramiko.SFTPClient.from_transport(ssh.get_transport())
        scp.put(source_path, destination_path)

        # Close connection
        scp.close()
        ssh.close()
        return "File transferred successfully."
    except Exception as e:
        return (f"Error: {e}")