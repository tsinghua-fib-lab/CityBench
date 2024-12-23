from pycitydata.map import Map
from citysim.routing import RoutingClient
import subprocess
from config import MONGODB_URI
def load_map(city_map, cache_dir, routing_path, port):
    m = Map(
            mongo_uri=f"{MONGODB_URI}",
            mongo_db="llmsim",
            mongo_coll=city_map,
            cache_dir=cache_dir,
        )
    route_command = f"{routing_path} -mongo_uri {MONGODB_URI} -map llmsim.{city_map} -cache {cache_dir} -listen localhost:{port}"
    cmd = route_command.split(" ")
    print("loading routing service")
    process = subprocess.Popen(args=cmd, cwd="./")
    routing_client = RoutingClient(f"localhost:{port}")

    return m, process, routing_client