import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import copy
import random

from config import IMAGE_FOLDER
def stitch_multiple_region(index=None):
    # 根据目录一个一个自动进行
    p = Pool(MULTI_PROCESS_NUM)

    result = [p.apply_async(stitch_single_region, args=(0, str(region.stem))) for region in INPUT_PATH.iterdir()]
    for i in result:
        i.get()

    p.close()
    p.join()


def stitch_single_region(index, region_code):
    try:
        save_path_for_region = SAVE_PATH.joinpath(region_code)
        save_path_for_region.mkdir(exist_ok=True,parents=True)
        meta_info = pd.read_csv(INPUT_PATH.joinpath(region_code).joinpath('meta_info.csv'))
        meta_info['x'] = meta_info['file_name'].map(lambda x:int(x.split('&')[-2]))
        meta_info['y'] = meta_info['file_name'].map(lambda x:int(x.split('&')[-1][:-4]))
        pano_id_list = meta_info['panoid'].drop_duplicates()

        stitch_info = pd.DataFrame()

        # i = 0
        for pano_id in tqdm(pano_id_list):
            pano_img_infos = meta_info.loc[meta_info['panoid'] == pano_id]
            tmp_stitch_info = {'panoid':[pano_img_infos.iloc[0,0]], 'query_lati':[pano_img_infos.iloc[0,1]],'query_longti':[pano_img_infos.iloc[0,2]],'real_lati':[pano_img_infos.iloc[0,3]],'real_longti':[pano_img_infos.iloc[0,4]],'date':[pano_img_infos.iloc[0,5]]}
            x_max = pano_img_infos['x'].max()
            y_max = pano_img_infos['y'].max()
            y_min = pano_img_infos['y'].min()
            # 目前发现两种类型：一圈儿16个x和一圈12个x。这两个都对应360度，只不过采样间隙不同
            if x_max == 15:
                x_type = '15'
                to_image_A = Image.new('RGB', (IMG_SIZE * 4, IMG_SIZE * (y_max - y_min + 1)))  # x=0,1,2,3
                cut_position = random.choice([0,4,8,12])
                for _, row in pano_img_infos.iterrows():
                    try:
                        from_image = Image.open(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))
                    except:
                        print("Error File")
                        print(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))
                    try:
                        if row['x'] in range(cut_position,cut_position+4):
                            to_image_A.paste(from_image,(((row['x']-cut_position) * IMG_SIZE, (row['y']-2) * IMG_SIZE)))

                    except:
                        print('拼接图像出错')
                        print('panoid: ' + str(pano_id))
                        print('x: '+str(row['x']))
                        print('y: '+str(row['y']))
                        print(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))

            elif x_max == 12:
                x_type = '12'
                # 十分奇怪地多出来一列，直接丢弃了
                to_image_A = Image.new('RGB', (IMG_SIZE * 3, IMG_SIZE * (y_max - y_min + 1)))
                cut_position = random.choice([0,3,6,9])
                for _, row in pano_img_infos.iterrows():
                    try:
                        from_image = Image.open(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))
                    except:
                        print("Error File")
                        print(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))
                    try:
                        if row['x'] in range(cut_position,cut_position+3):
                            to_image_A.paste(from_image,(((row['x']-cut_position) * IMG_SIZE, (row['y']-2) * IMG_SIZE)))

                    except:
                        print('拼接图像出错')
                        print('panoid: ' + str(pano_id))
                        print('x: '+str(row['x']))
                        print('y: '+str(row['y']))
                        print(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))
            else:
                print("================================")
                print('此处并不符合要求')
                print(INPUT_PATH.joinpath(region_code).joinpath(row['file_name']))
                print(pano_id)
                print(x_max)
                print("================================")


            file_name_A = '&'.join(pano_img_infos.iloc[0,-3].split('&')[:-2]) + '&' + x_type + '&' + str(cut_position) + '.jpg'
            to_image_A = to_image_A.resize((OUT_IMG_SIZE_W,OUT_IMG_SIZE_H))

            to_image_A.save(save_path_for_region.joinpath(file_name_A),quality=95)

            tmp_stitch_info['type'] = [x_type]

            tmp_stitch_infoA = copy.copy(tmp_stitch_info)
            tmp_stitch_infoA['file_name'] = [file_name_A]
            stitch_info = pd.concat([stitch_info,pd.DataFrame(tmp_stitch_infoA)])

            stitch_info.to_csv(save_path_for_region.joinpath('stitch_meta_info.csv'),index=False)
            # i += 1
            # if i == 5:
            #     break
    except:
        print('Error')
        print(INPUT_PATH.joinpath(region_code))
        print(save_path_for_region)
        print('================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default="Sydney")
    parser.add_argument('--multi_process_num', type=int, help='多线程数量，注意不要超过cpu核心数量')
    parser.add_argument('--image_size', default=512, help='单个图像的尺寸')
    parser.add_argument('--out_image_size_width', default=512, help='全景图像的宽度')
    parser.add_argument('--out_image_size_height', default=512, help='全景图像的高度')

    args = parser.parse_args()
    MULTI_PROCESS_NUM = args.multi_process_num
    IMG_SIZE = args.image_size
    OUT_IMG_SIZE_W = args.out_image_size_width
    OUT_IMG_SIZE_H = args.out_image_size_height
    INPUT_PATH = os.path.join(IMAGE_FOLDER, f'{args.city_name}_StreetView_Images_origin')
    SAVE_PATH = IMAGE_FOLDER
    SAVE_PATH.mkdir(exist_ok=True,parents=True)

    begin_time = datetime.now()
    stitch_multiple_region()
    end_time = datetime.now()
    print('总共耗时：' + str((end_time-begin_time).total_seconds() / 3600 ) + '小时')
