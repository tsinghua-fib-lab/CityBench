import logging
import os

from geojson import dump

from mosstool.map.osm import RoadNet

bbox = {
    "max_lat": 40.1,
    "min_lat": 39.9,
    "min_lon": 116.5,
    "max_lon": 116.6,
}
# Configure log and store it in the file mapbuilder2.log
logging.basicConfig(
    filename="mapbuilder2.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
os.makedirs("cache", exist_ok=True)
logging.getLogger().addHandler(logging.StreamHandler())
# load configs
rn = RoadNet(
    proj_str="+proj=tmerc +lat_0=39.90611 +lon_0=116.3911",
    max_latitude=bbox["max_lat"],
    min_latitude=bbox["min_lat"],
    max_longitude=bbox["max_lon"],
    min_longitude=bbox["min_lon"],
)

path = "cache/topo.geojson"
rn.create_road_net(path)
