
import numpy as np
import math
import geopandas as gpd
from shapely.geometry import Polygon


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

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
