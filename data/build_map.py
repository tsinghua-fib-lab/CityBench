import geojson

from mosstool.map.builder import Builder
from mosstool.map.vis import VisMap
from mosstool.type import Map
from mosstool.util.format_converter import dict2pb

with open("data/geojson/extended_net_with_z.geojson", "r") as f:
    net = geojson.load(f)

builder = Builder(
    net=net,
    proj_str="+proj=tmerc +lat_0=33.9 +lon_0=116.4",
    gen_sidewalk_speed_limit=50 / 3.6,
    road_expand_mode="M",
)
m = builder.build("test")
pb = dict2pb(m, Map())
with open("data/temp/map.pb", "wb") as f:
    f.write(pb.SerializeToString())
vis_map = VisMap(pb)
fc = vis_map.feature_collection
with open("data/temp/map.geojson", "w") as f:
    geojson.dump(fc, f, ensure_ascii=False, indent=2)
deck = vis_map.visualize()
deck.to_html("data/temp/map.html")
