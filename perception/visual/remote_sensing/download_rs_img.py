from pycitysim.sateimg import download_all_tiles


import argparse
from tqdm import tqdm

import pandas as pd
import os
if __name__ =="main":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--city_name', type=str, default='SanFrancisco', help='city name')
    parser.add_argument('--output_dir', type=str, default='cache', help='output directory')
    args = parser.parse_args()
    
    img_info_df = pd.read_csv(args.city_name+'_img_indicators.csv')
    need_to_download = img_info_df["img_name"].tolist()

    
    zoom_level = 15
    base_url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/27659"
    y_x = [
        "12394_26956","12394_26957","12398_26996"]

    if args.city=="Beijing":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/27659/'#2021-04-28
    if args.city=="Shanghai":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/16749/'#2021-10-13
    if args.city=="Mumbai":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/18289/'#2020-07-01
    if args.city=="Tokyo":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/27659/'#2021-04-28
    if args.city=="London":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/17825/'#2022-08-10
    if args.city=="Paris":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/119/'#2020-10-14
    if args.city=="Moscow":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/9812/'#2021-02-24
    if args.city=="SaoPaulo":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/29260/'#2020-12-16
    if args.city=="Nairobi":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/19187/'#2020-09-23
    if args.city=="CapeTown":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/5359/'#2021-03-17
    if args.city=="Sydney":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/5359/'#2021-03-17
    if args.city=="SanFrancisco":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/12576/'
    if args.city=="NewYork":
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/12576/'
    
    
    
    
    imgs, failed = download_all_tiles(
    base_url,
        15,
        need_to_download,
    )
    # save the images
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(f"{args.output_dir}{args.city_name}"):
        os.makedirs(f"{args.output_dir}{args.city_name}")
    for key, img in imgs.items():
        img.save(f"{args.output_dir}/{args.city_name}/{key}.png")