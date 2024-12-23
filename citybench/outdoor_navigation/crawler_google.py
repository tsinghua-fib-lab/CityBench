from pathlib import Path
import pandas as pd
from utils_scp import *
import numpy as np
from datetime import datetime
from multiprocessing import Process, Pool
import copy
import pandas as pd
import os
import argparse
import ast
from config import IMAGE_FOLDER, SAMPLE_POINT_PATH

def crawl_multiple_region(city, multi_process_num, total_points, index=[]):
    csv_file_path = os.path.join(SAMPLE_POINT_PATH, f'{args.city_name}_sampled_points_expand.csv')
    df = pd.read_csv(csv_file_path)
    df['coord_y'] = df['coord_y'].apply(lambda x: ast.literal_eval(x))
    df['coord_x'] = df['coord_x'].apply(lambda x: ast.literal_eval(x))

    p = Pool(multi_process_num)
    result = [p.apply_async(crawl_single_region, args=(i, total_points, df['coord_x'][i], df['coord_y'][i], df['y_x'][i])) for i in index]

    for i in result:
        i.get()

    p.close()
    p.join()


def crawl_single_region(index, total_points, longti, lati, region_code):
    region_code = copy.deepcopy(str(region_code))
    lati = copy.deepcopy(lati)
    longti = copy.deepcopy(longti)
    lati = list(lati)
    longti = list(longti)

    # save images into local folder
    path_with_region_code = SAVE_PATH.joinpath(region_code)
    path_with_region_code.mkdir(exist_ok=True,parents=True)
    meta_info, query_lati_longti_set = resume_from_history_info(path_with_region_code)
    current_point_index = 0
    success_download = 0
    total_start_time = datetime.now()

    for j in range(total_points):
        start_time = datetime.now()
        lati[j] = np.round(float(lati[j]), 4)
        longti[j] = np.round(float(longti[j]), 4)
        if (str(lati[j]),str(longti[j])) in query_lati_longti_set:
            print('进程'+ str(index) + ': 已爬取，跳过')
        else:  
            try:
                print("lati-longti: ", str(lati[j]),str(longti[j]))
                panoid, real_lati, real_longti, date = get_metadata_from_lati_lonti(lati[j],longti[j]) # 查询经纬度对应panoid
            except:
                continue
            if panoid == -1:
                print('进程'+ str(index) + ': 无街景点，跳过')
                query_lati_longti_set.add((str(lati[j]),str(longti[j])))
                save_query_set(query_lati_longti_set,path_with_region_code)
            elif panoid == -2:
                print('进程'+ str(index) + ': 超过日限额或并发量')
                time.sleep(3600*24)
            elif panoid == -3:
                print('进程'+ str(index) + ': 不知道查询metadata发生了什么，跳过')
            else:
                img_dict = get_image_tiles(panoid)
                new_metainfo = save_img(img_dict, path_with_region_code, panoid, lati[j], longti[j], real_lati, real_longti, date)
                meta_info = update_and_save_metainfo(meta_info, new_metainfo, path_with_region_code)
                query_lati_longti_set.add((str(lati[j]),str(longti[j])))
                save_query_set(query_lati_longti_set,path_with_region_code)
                print('进程'+ str(index) + ': Downloaded Place: ' + str(panoid) + '; With ' + str(len(img_dict)) + ' Pictures!')
                if len(img_dict) > 0:
                    success_download += 1

        current_point_index += 1
        end_time = datetime.now()
        print("进程"+ str(index) + ": This point costs " + str((end_time - start_time).total_seconds()) + ' seconds.')
        print("进程"+ str(index) + ": Suceessfully downloaded " + str(success_download) + " points from " + str(current_point_index) + " points.")
        print('==================================')

        # if (success_download == SAMPLE_SIZE):
        #     break
        
        # Can not find images over time limit, then remove the dir and save the region code
        if ((end_time-total_start_time).total_seconds() > time_limit) & (len(os.listdir(path_with_region_code)) == 0):
            break

    print(f"成功下载{success_download}个地点，共需要下载{total_points}个地点")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default="Sydney")
    parser.add_argument('--multi_process_num', type=int, help='多线程数量，注意不要超过cpu核心数量')
    parser.add_argument('--index', dest="index", type=int, default=0, help='从第index组开始下载')
    parser.add_argument('--total_points', type=int, help='每一组内下载多少个点')
    args = parser.parse_args()
    print(args)

    SAVE_DIR = os.path.join(IMAGE_FOLDER, f'{args.city_name}_StreetView_Images_origin')
    SAVE_PATH = Path(SAVE_DIR)
    SAVE_PATH.mkdir(exist_ok=True,parents=True)

    time_limit = 300 #By seconds
    empty_region = []

    begin_time = datetime.now()
    crawl_multiple_region(args.city_name, args.multi_process_num, args.total_points, index=[i for i in range(args.index, args.total_points)]) 
    end_time = datetime.now()
    print('总共耗时：' + str((end_time-begin_time).total_seconds() / 3600 ) + '小时 or ' + str((end_time-begin_time).total_seconds()) + '秒')
