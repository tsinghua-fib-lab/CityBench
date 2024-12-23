import os
from math import floor 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Polygon

import warnings
warnings.filterwarnings("ignore")

from osgeo import gdal

from pycitydata.sateimg import download_all_tiles
from config import CITY_BOUNDARY, WORLD_POP_DATA_PATH, REMOTE_SENSING_PATH, ARCGIS_TILE_URL, REMOTE_SENSING_ZOOM_LEVEL
from .utils import deg2num, compute_tile_coordinates, create_tile_polygons, num2deg


def download_pop(city_name):
    gdf = gpd.GeoDataFrame({'geometry': [Polygon(CITY_BOUNDARY[city_name])]})
    gdf.crs = {'init': 'epsg:4326'}
    temp_shp = gdf
    temp_shp = temp_shp.to_crs({'init': 'epsg:4326'})
    max_y = deg2num(min(temp_shp.bounds.minx),min(temp_shp.bounds.miny))[1]+5
    min_y = deg2num(max(temp_shp.bounds.maxx),max(temp_shp.bounds.maxy))[1]-5
    max_x = deg2num(max(temp_shp.bounds.maxx),max(temp_shp.bounds.maxy))[0]+5
    min_x = deg2num(min(temp_shp.bounds.minx),min(temp_shp.bounds.miny))[0]-5
    lon_arr, lat_arr,x_arr, y_arr = compute_tile_coordinates(min_x, max_x, min_y, max_y)
    one_street_shp_x_y = create_tile_polygons(lon_arr, lat_arr,x_arr, y_arr)
    # op is replaced by predicate since geopandas 1.0
    intersection = gpd.sjoin(one_street_shp_x_y, temp_shp, op='intersects')
    # intersection = gpd.sjoin(one_street_shp_x_y, temp_shp, predicate='intersects')

    sample_img_list = intersection.y_x
    sample_img_list = list(sample_img_list)

    gdal.AllRegister()
    pop_tiff_path = WORLD_POP_DATA_PATH
    if pop_tiff_path != None:
        pop_dataset = gdal.Open(pop_tiff_path)
        pop_adfGeoTransform = pop_dataset.GetGeoTransform()

    img_indicators_list = []
    pred_pop = []
    for tile in tqdm(sample_img_list):
        one_img_set = {}
        one_img_set['img_name'] = tile
        x=int(tile.split('_')[1].split('.')[0])
        y=int(tile.split('_')[0])
        lat_max,lng_min=num2deg(x,y,zoom=15)
        _,lng_max=num2deg(x+1,y,zoom=15)
        lat_min,_=num2deg(x,y+1,zoom=15)

        lat_mean = (lat_max+lat_min)/2
        lng_mean = (lng_max+lng_min)/2
        one_img_set['lat'] = lat_mean
        one_img_set['lng'] = lng_mean
        
        #worldpop 
        if  pop_tiff_path!=None:
            y_init = int(floor((lat_max-pop_adfGeoTransform[3])/pop_adfGeoTransform[5]))
            x_init = int(floor((lng_min-pop_adfGeoTransform[0])/pop_adfGeoTransform[1]))
            x_end = int(floor((lng_max-pop_adfGeoTransform[0])/pop_adfGeoTransform[1]))
            y_end = int(floor((lat_min-pop_adfGeoTransform[3])/pop_adfGeoTransform[5]))
            band = pop_dataset.GetRasterBand(1)
            data=band.ReadAsArray(x_init,y_init,x_end-x_init,y_end-y_init)
            data[data<0] = 0
            pred_pop.append(np.array(data.sum()).tolist())
            one_img_set['worldpop'] = np.array(data.sum()).tolist()

    img_indicators_df = pd.DataFrame(img_indicators_list)
    img_indicators_df.to_csv(os.path.join(REMOTE_SENSING_PATH, city_name+'_img_indicators.csv'), index=False)


def downlad_rs(city_name, zoom_level):
    """download remote sensing images from """
    try:
        img_info_df = pd.read_csv(os.path.join(REMOTE_SENSING_PATH, city_name+'_img_indicators.csv'))
    except FileNotFoundError:
        print("please first generate input files for downloading data by assigning data_type as rs")
        exit(0)
    need_to_download = img_info_df["img_name"].tolist()

    # 12428
    # time dependent url parameters
    city_url_map = {
        "Beijing": "27659/",    #2021-04-28
        "Shanghai": "16749/",   #2021-10-13
        "Mumbai": "18289/",     #2020-07-01
        "Tokyo": "27659/",      #2021-04-28
        "London": "17825/",     #2022-08-10
        "Paris": "119/",        #2020-10-14
        "Moscow": "9812/",      #2021-02-24
        "SaoPaulo": "29260/",   #2020-12-16
        "Nairobi": "19187/",    #2020-09-23
        "CapeTown": "5359/",    #2021-03-17
        "Sydney": "5359/",      #2021-03-17
        "SanFrancisco": "12576/",#2019-06-05
        "NewYork": "12576/"
    }
    base_url  = ARCGIS_TILE_URL + city_url_map[city_name]
    
    imgs, failed = download_all_tiles(
        base_url,
        zoom_level,
        need_to_download,
    )

    # save the images
    if not os.path.exists(REMOTE_SENSING_PATH):
        os.makedirs(REMOTE_SENSING_PATH)
    if not os.path.exists(f"{REMOTE_SENSING_PATH}/{city_name}"):
        os.makedirs(f"{REMOTE_SENSING_PATH}{city_name}")
    for key, img in imgs.items():
        img.save(f"{REMOTE_SENSING_PATH}/{city_name}/{key}.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--city_name', type=str, default='SanFrancisco', help='city names from {}'.format(CITY_BOUNDARY.keys()))
    parser.add_argument('--data_type', type=str, default='pop-rs', choices=["pop", "rs", 'pop-rs'])
    args = parser.parse_args()

    if args.city_name not in CITY_BOUNDARY.keys():
        print('City {} not found'.format(args.city_name))
        exit(0)

    if args.data_type in ["pop", 'pop-rs']:
        download_pop(city_name=args.city_name)
    if args.data_type in ["rs", 'pop-rs']:
        downlad_rs(city_name=args.city_name, zoom_level=REMOTE_SENSING_ZOOM_LEVEL)
