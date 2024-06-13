import pandas as pd
import math


from math import floor 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import geopandas as gpd

from shapely.geometry import Polygon
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon
from tqdm import tqdm
import gdal

import argparse
from tqdm import tqdm
city_boundary_dict ={
    "SanFrancisco": 
        Polygon([(-122.5099, 37.8076), (-122.5099, 37.6153), (-122.3630, 37.6153), (-122.3630, 37.8076)]),
    "NewYork": 
        Polygon([(-74.0186, 40.7751), (-74.0186, 40.6551), (-73.8068, 40.6551), (-73.8068, 40.7751)]),
    "Beijing": 
        Polygon([(116.1536, 40.0891), (116.1536, 39.7442), (116.6082, 39.7442), (116.6082, 40.0891)]),
    "Shanghai": 
        Polygon([(121.1215, 31.4193), (121.1215, 30.7300), (121.9730, 30.7300), (121.9730, 31.4193)]),
    "Mumbai":
        Polygon([(72.7576, 19.2729), (72.7576, 18.9797), (72.9836, 18.9797), (72.9836, 19.2729)]),    
    "Tokyo":
        Polygon([(139.6005, 35.8712), (139.6005, 35.5859), (139.9713, 35.5859), (139.9713, 35.8712)]),
    "London":
        Polygon([(-0.3159, 51.6146), (-0.3159, 51.3598), (0.1675, 51.3598), (0.1675, 51.6146)]),
    "Paris":
    Polygon([(2.249, 48.9038), (2.249, 48.8115), (2.4239, 48.8115), (2.4239, 48.9038)]),
    "Moscow":

    Polygon([(37.4016, 55.8792), (37.4016, 55.6319), (37.8067, 55.6319), (37.8067, 55.8792)]),
    
    "SaoPaulo":
    Polygon([(-46.8251, -23.4242), (-46.8251, -23.7765), (-46.4365, -23.7765), (-46.4365, -23.4242)]),
    "Nairobi":

    Polygon([(36.6868, -1.1906), (36.6868, -1.3381), (36.9456, -1.3381), (36.9456, -1.1906)]),
    "CapeTown":

    Polygon([(18.3472, -33.8179), (18.3472, -34.0674), (18.6974, -34.0674), (18.6974, -33.8179)]),
    "Sydney":
    Polygon([(150.8382, -33.6450), (150.8382, -34.0447), (151.2982, -34.0447), (151.2982, -33.6450)]),
    
}









def num2deg(x, y, zoom=15):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat_deg = np.rad2deg(lat_rad)
    return lat_deg, lon_deg

def deg2num(lon_deg, lat_deg, zoom=15):
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    return xtile, ytile

def compute_tile_coordinates(min_x, max_x, min_y, max_y):
    x_arr = np.arange(min_x, max_x + 1)
    y_arr = np.arange(min_y, max_y + 1)
    lon_arr, lat_arr = num2deg_batch(x_arr, y_arr)
    return lon_arr, lat_arr,x_arr, y_arr

def num2deg_batch(x_arr, y_arr, zoom=15):
    n = 2.0 ** zoom
    lon_deg_arr = x_arr / n * 360.0 - 180.0
    lat_rad_arr = np.arctan(np.sinh(np.pi * (1 - 2 * y_arr / n)))
    lat_deg_arr = np.rad2deg(lat_rad_arr)
    return lon_deg_arr, lat_deg_arr

def create_tile_polygons(lon_arr, lat_arr,x_arr, y_arr):
    
    tile_gpd= gpd.GeoDataFrame()
    lon_mesh, lat_mesh = np.meshgrid(lon_arr, lat_arr, indexing='ij')
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')
    
    
    vertices = np.array([
        lon_mesh[:-1, :-1], lat_mesh[:-1, :-1],
        lon_mesh[1:, :-1], lat_mesh[1:, :-1],
        lon_mesh[1:, 1:], lat_mesh[1:, 1:],
        lon_mesh[:-1, 1:], lat_mesh[:-1, 1:]
    ])

    vertices = vertices.reshape(4, 2, -1)
    vertices = np.transpose(vertices, axes=(2, 0, 1))
    polygons = [Polygon(p) for p in vertices]
    vertices_x_y = np.array([
        x_mesh[:-1, :-1], y_mesh[:-1, :-1],
        x_mesh[1:, :-1], y_mesh[1:, :-1],
        x_mesh[1:, 1:], y_mesh[1:, 1:],
        x_mesh[:-1, 1:], y_mesh[:-1, 1:]
    ])
    
    vertices_x_y = vertices_x_y.reshape(4, 2, -1)
    vertices_x_y = np.transpose(vertices_x_y, axes=(2, 0, 1))
    y_x = [f"{int(p[0][1])}_{int(p[0][0])}" for p in vertices_x_y]
                            
    
    tile_gpd['geometry'] = polygons
    tile_gpd['y_x'] = y_x
    
    return tile_gpd
def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--city', type=str, default='San Francisco', help='city name')
    
    parser.add_argument('--pop_tiff_path', type=str, default='ppp_2020_1km_Aggregated.tif', help='pop_tiff_path')
    args = parser.parse_args()
    
    
    city_name = args.city
    if city_name not in city_boundary_dict.keys():
        print('City not found')
        exit(0)


        
    gdf = gpd.GeoDataFrame({'geometry': [city_boundary_dict[city_name]]})
    gdf.crs = {'init': 'epsg:4326'}

    temp_shp = gdf

    temp_shp = temp_shp.to_crs({'init': 'epsg:4326'})

    max_y=deg2num(min(temp_shp.bounds.minx),min(temp_shp.bounds.miny))[1]+5
    min_y=deg2num(max(temp_shp.bounds.maxx),max(temp_shp.bounds.maxy))[1]-5
    max_x=deg2num(max(temp_shp.bounds.maxx),max(temp_shp.bounds.maxy))[0]+5
    min_x=deg2num(min(temp_shp.bounds.minx),min(temp_shp.bounds.miny))[0]-5

    temp_y_x=[]




    lon_arr, lat_arr,x_arr, y_arr = compute_tile_coordinates(min_x, max_x, min_y, max_y)
    one_street_shp_x_y = create_tile_polygons(lon_arr, lat_arr,x_arr, y_arr)


    intersection = gpd.sjoin(one_street_shp_x_y, temp_shp, op='intersects')



    sample_img_list = intersection.y_x
    sample_img_list = list(sample_img_list)




    gdal.AllRegister()
    
    
    pop_tiff_path=args.pop_tiff_path
    if pop_tiff_path!=None:
        
        pop_dataset = gdal.Open(pop_tiff_path)  #
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
    img_indicators_df.to_csv(city_name+'_img_indicators.csv',index=False)
