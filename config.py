import os

# general parameters
PROXY = "http://127.0.0.1:10190"
PROXIES = {"http": PROXY, "https": PROXY}
############# TODO 不可对外暴露，仅用于内部测试
MONGODB_URI = ""
###########
MAP_DATA_PATH="citydata/EXP_ORIG_DATA/"
MAP_CACHE_PATH="citydata/map_cache/"
RESOURCE_PATH="citydata/resource/"
ROUTING_PATH="citydata/routing_linux_amd64"
RESULTS_PATH="results/"

# geoqa
GEOQA_SAMPLE_RATIO = 0.1

GEOQA_TASK_MAPPING_v1 = {
    "path": {
        "road_length": "eval_road_length.csv", 
        "road_od": "eval_road_od.csv",
        "road_link": "eval_road_link.csv"
    },
    "node": {
        "poi2coor": "poi2coor.csv",
        "AOI_POI_road4": "AOI_POI_road4.csv",
        "aoi_near": "aoi_near.csv"
    },
    "landmark": {
        "landmark_env": "eval_landmark_env.csv",  
        "landmark_path": "eval_landmark_path.csv"  
    },
    "boundary": {
        "boundary_road": "eval_boundary_road.csv", 
        "AOI_POI_road1": "AOI_POI_road1.csv",
        "AOI_POI_road2": "AOI_POI_road2.csv",
        "AOI_POI_road3": "AOI_POI_road3.csv"
    },
    "districts": {
        "aoi2type": "aoi2type.csv", 
        "type2aoi": "type2aoi.csv",  
        "aoi2addr": "aoi2addr.csv",
        "AOI_POI5": "AOI_POI5.csv",
        "AOI_POI6": "AOI_POI6.csv",
        "road_aoi": "eval_road_aoi.csv"
    },
    "others": {
    "AOI_POI3": "AOI_POI3.csv",
    "AOI_POI4": "AOI_POI4.csv"
    }
}

GEOQA_TASK_MAPPING_v2 = {
    "path": {
        "road_length": "road_length.csv", 
        "road_od": "road_od.csv",
        "road_link": "road_link.csv", 
        "road_arrived_pois": "road_arrived_pois.csv"
    },
    "node": {
        "poi2coor": "poi2coor.csv",
        "poi2addr": "poi2addr.csv",
        "poi2type": "poi2type.csv",
        "type2poi": "type2poi.csv",
        "AOI_POI_road4": "AOI_POI_road4.csv"

    },
    "landmark": {
        "landmark_env": "landmark_env.csv",  
        "landmark_path": "landmark_path.csv"  
    },
    "boundary": {
        "boundary_road": "boundary_road.csv", 
        "aoi_boundary_poi": "aoi_boundary_poi.csv",
        "AOI_POI_road1": "AOI_POI_road1.csv",
        "AOI_POI_road2": "AOI_POI_road2.csv",
        "AOI_POI_road3": "AOI_POI_road3.csv"
    },
    "districts": {
        "aoi2type": "aoi2type.csv", 
        "type2aoi": "type2aoi.csv",  
        "aoi_poi": "aoi_poi.csv", 
        "poi_aoi": "poi_aoi.csv",  
        "aoi_group": "aoi_group.csv", 
        "aoi2addr": "aoi2addr.csv",
        "districts_poi_type": "districts_poi_type.csv",
        "AOI_POI5": "AOI_POI5.csv",
        "AOI_POI6": "AOI_POI6.csv"
    },
    "others": {
    "AOI_POI": "AOI_POI.csv",
    "AOI_POI2": "AOI_POI2.csv",
    "AOI_POI3": "AOI_POI3.csv",
    "AOI_POI4": "AOI_POI4.csv"
    }
}

# mobility prediction
MOBILITY_SAMPLE_RATIO=0.1
# remote sensing
REMOTE_SENSING_PATH="citydata/remote_sensing/"
REMOTE_SENSING_RESULTS_PATH=RESULTS_PATH+"remote_sensing/"
REMOTE_SENSING_ZOOM_LEVEL=15
WORLD_POP_DATA_PATH="{}ppp_2020_1km_Aggregated.tif".format(REMOTE_SENSING_PATH)
# street view
STREET_VIEW_PATH="citydata/street_view/"
SAMPLE_RATIO = 0.2
ARCGIS_TILE_URL = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
# outdoor navigation
# IMAGE_FOLDER = f"citydata/outdoor_navigation_tasks/NEW_StreetView_Images_CUT"
# TODO: 为了方便测试，暂时使用绝对路径
IMAGE_FOLDER = f"/data3/liutianhui/NEW_StreetView_Images_CUT"
SAMPLE_POINT_PATH="citydata/outdoor_navigation_tasks/sample_points/"
STEP = 50
# 控制是否更新该任务
NAVIGATION_UPGRADE = True
REGION_CODE = {
    "SanFrancisco": 90000,
    "NewYork": 20000,
    "Beijing": 10087,
    "Shanghai": 20088,
    "Mumbai": 50000,
    "Tokyo": 70000,
    "London": 30000,
    "Paris": 10000,
    "Moscow": 40000,
    "SaoPaulo": 110000,
    "Nairobi": 60000,
    "CapeTown": 80000,
    "Sydney": 120000
}
# traffic singal
TRIP_DATA_PATH="citydata/trips/"


VLM_API = ["GPT4o", "GPT4omini", "LLama-3.2-90B", "LLama-3.2-11B", "Qwen2-VL-72B", "gpt-4o-mini"]
VLM_MODELS = [
    "QwenVLPlus", "GPT4o", "GPT4o_MINI", "cogvlm2-llama3-chat-19B", "InternVL2-40B", "MiniCPM-Llama3-V-2_5", "llava_next_yi_34b", "llava_next_llama3", "Yi_VL_6B", "Yi_VL_34B", "llava_v1.5_7b", "glm-4v-9b", "InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26B", "Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct", "GPT4omini", "LLama-3.2-90B", "LLama-3.2-11B", "Qwen2-VL-72B"]

VLLM_MODEL_PATH_PREFIX = "/data3/fengjie/init_ckpt/"
VLLM_MODEL_PATH_PREFIX2 = "/data1/citygpt/init_ckpt/multi-modal/"
VLLM_MODEL_PATH = {
    "cogvlm2-llama3-chat-19B": os.path.join(VLLM_MODEL_PATH_PREFIX, "cogvlm2-llama3-chat-19B"),
    "InternVL2-40B": os.path.join(VLLM_MODEL_PATH_PREFIX, "InternVL2-40B"),
    "MiniCPM-Llama3-V-2_5": os.path.join(VLLM_MODEL_PATH_PREFIX, "MiniCPM-Llama3-V-2_5"),
    "llava_next_yi_34b": os.path.join(VLLM_MODEL_PATH_PREFIX, "llava-v1.6-34b-hf"),
    "llava_next_llama3": os.path.join(VLLM_MODEL_PATH_PREFIX, "llama3-llava-next-8b-hf"),
    "llava_v1.5_7b": os.path.join(VLLM_MODEL_PATH_PREFIX2, "llava-1___5-7b-hf"),
    "glm-4v-9b": os.path.join(VLLM_MODEL_PATH_PREFIX2, "glm-4v-9b"),
    "Qwen2-VL-2B-Instruct": os.path.join(VLLM_MODEL_PATH_PREFIX, "Qwen2-VL-2B-Instruct"),
    "Qwen2-VL-7B-Instruct": os.path.join(VLLM_MODEL_PATH_PREFIX2, "Qwen2-VL-7B-Instruct"),
    "Yi_VL_6B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "Yi-VL-6B"),
    "Yi_VL_34B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "Yi-VL-34B"),
    "InternVL2-2B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-2B"),
    "InternVL2-4B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-4B"),
    "InternVL2-8B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-8B"),
    "InternVL2-26B": os.path.join(VLLM_MODEL_PATH_PREFIX2, "InternVL2-26B")
}

LLM_MODELS = [
    "Qwen2-7B", "Qwen2-72B", "Intern2.5-7B", "Intern2.5-20B", 
    "Mistral-7B", "Mixtral-8x22B", "LLama3-8B", "LLama3-70B", "Gemma2-9B", "Gemma2-27B", 
    "DeepSeek-67B", "DeepSeekV2", "GPT3.5-Turbo", "GPT4-Turbo"]

INFER_SERVER = {
    "OpenAI": ["GPT3.5-Turbo", "GPT4-Turbo", "GPT4omini"],
    "DeepInfra": ["Mistral-7B", "Mixtral-8x22B", "LLama3-8B", "LLama3-70B", "Gemma2-9B", "Gemma2-27B", "LLama-3.2-90B", "LLama-3.2-11B"],
    "Siliconflow": ["Qwen2-7B", "Qwen2-72B", "Intern2.5-7B", "Intern2.5-20B", "DeepSeekV2", "Qwen2-VL-72B"],
    "DeepBricks": ["gpt-4o-mini"]
}
LLM_MODEL_MAPPING = {
    "Qwen2-7B":"Qwen/Qwen2-7B-Instruct",
    "Qwen2-72B":"Qwen/Qwen2-72B-Instruct",
    "Intern2.5-7B":"internlm/internlm2_5-7b-chat",
    "Intern2.5-20B":"internlm/internlm2_5-20b-chat",
    "Mistral-7B":"mistralai/Mistral-7B-Instruct-v0.2", 
    "Mixtral-8x22B":"mistralai/Mixtral-8x22B-Instruct-v0.1",
    "LLama3-8B":"meta-llama/Meta-Llama-3-8B-Instruct",
    "LLama3-70B":"meta-llama/Meta-Llama-3-70B-Instruct",
    "Gemma2-9B":"google/gemma-2-9b-it",
    "Gemma2-27B":"google/gemma-2-27b-it",
    "DeepSeekV2":"deepseek-ai/DeepSeek-V2-Chat",
    "GPT3.5-Turbo":"gpt-3.5-turbo-0125",
    "GPT4-Turbo":"gpt-4-turbo-2024-04-09",
    "GPT4omini":"gpt-4o-mini-2024-07-18",
    "GPT4o":"gpt-4o"
}

# 任务运行代码映射
TASK_DEST_MAPPING = {
    # text
    "traffic": "citybench.traffic_signal.run_eval",
    "geoqa": "citybench.geoqa.run_eval",
    "mobility": "citybench.mobility_prediction.run_eval",
    "exploration": "citybench.urban_exploration.eval",
    # visual 
    "population": "citybench.remote_sensing.eval_inference",
    "objects": "citybench.remoet_sensing.eval_inference",
    "geoloc": "citybench.street_view.eval_inference",
    "navigation": "citybench.outdoor_navigation.eval"
    }
# 任务统计代码映射
TASK_METRICS_MAPPING = {
    # text
    "traffic": "citybench.traffic_signal.metrics",
    "geoqa": "citybench.geoqa.metrics",
    "mobility": "citybench.mobility_prediction.metrics",
    "exploration": "citybench.urban_exploration.metrics",
    # visual 
    "population": "citybench.remote_sensing.metrics",
    "objects": "citybench.remoet_sensing.metrics",
    "geoloc": "citybench.street_view.metrics",
    "navigation": "citybench.outdoor_navigation.metrics"
    }
# 任务结果文件
RESULTS_FILE = {
    "traffic": os.path.join(RESULTS_PATH, "signal_results/signal_benchmark_results.csv"),
    "geoqa": os.path.join(RESULTS_PATH, "geo_knowledge_result/geoqa_benchmark_result.csv"),
    "mobility": os.path.join(RESULTS_PATH, "prediction_results/mobility_benchmark_result.csv"),
    "exploration": os.path.join(RESULTS_PATH, "exploration_results/exploration_benchmark_result.csv"),
    "population": os.path.join(RESULTS_PATH, "remote_sensing/population_benchmark_results.csv"),
    "objects": os.path.join(RESULTS_PATH, "remote_sensing/object_benchmark_results.csv"),
    "geoloc": os.path.join(RESULTS_PATH, "street_view/geoloc_benchmark_results.csv"),
    "navigation": os.path.join(RESULTS_PATH, "outdoor_navigation_results/navigation_benchmark_result.csv")
}
# 主表列名选择
METRICS_SELECTION = {
    "traffic": ["Average_Queue_Length", "Throughput"], 
    "geoqa": ["GeoQA_Average_Accuracy"],
    "mobility": ["Acc@1", "F1"],
    "exploration": ["Exploration_Success_Ratio", "Exploration_Average_Steps"],
    "population": ["RMSE", "r2"],
    "objects": ["Infrastructure_Accuracy"],
    "geoloc": ["City_Accuracy", "Acc@25km"],
    "navigation": ["Navigation_Success_Ratio", "Navigation_Average_Distance"]
}

# 原始的城市地图信息
MAP_DICT={
    "Beijing":"map_beijing_20240808",
    "Shanghai":"map_shanghai_20240806",
    "Mumbai":"map_mumbai_20240806",
    "Tokyo":"map_tokyo_20240807",
    "London":"map_london_20240807",
    "Paris":"map_paris_20240808",
    "Moscow":"map_moscow_20240807",
    "NewYork":"map_newyork_20240808",
    "SanFrancisco":"map_san_francisco_20240807",
    "SaoPaulo":"map_san_paulo_20240808",
    "Nairobi":"map_nairobi_20240807",
    "CapeTown":"map_cape_town_20240808",
    "Sydney":"map_sydney_20240807"
}

# 进行交通信号灯控制的局部区域范围
SIGNAL_BOX={
    # 左下 右下 右上 左上
    "Paris": [(2.3373, 48.8527), (2.3525, 48.8527), (2.3525, 48.8599), (2.3373, 48.8599)],
    "NewYork": [(-73.9976, 40.7225), (-73.9877, 40.7225), (-73.9877, 40.7271), (-73.9976, 40.7271)],
    "Shanghai": [(121.4214, 31.2409), (121.4465, 31.2409), (121.4465, 31.2525), (121.4214, 31.2525)],
    "Beijing": [(116.326, 39.9839), (116.3492, 39.9839), (116.3492, 39.9943), (116.326, 39.9943)],
    "Mumbai": [(72.8779, 19.064), (72.8917, 19.064), (72.8917, 19.0749), (72.8779, 19.0749)],
    "London": [(-0.1214, 51.5227), (-0.11, 51.5227), (-0.11, 51.5291), (-0.1214, 51.5291)],
    "SaoPaulo": [(-46.6266, -23.5654), (-46.6102, -23.5654), (-46.6102, -23.5555), (-46.6266, -23.5555)],
    "Nairobi": [(36.8076, -1.2771), (36.819, -1.2771), (36.819, -1.2656), (36.8076, -1.2656)],
    "Sydney": [(151.1860, -33.9276), (151.1948, -33.9276), (151.1948, -33.9188), (151.1860, -33.9188)],
    "SanFrancisco": [(-122.4893,37.7781), (-122.4568,37.7781), (-122.4568, 37.7890), (-122.4893, 37.7890)],
    "Tokyo": [(139.7641,35.6611), (139.7742,35.6611), (139.7742, 35.6668), (139.7641, 35.6668)],
    "Moscow":[(37.3999,55.8388), (37.4447,55.8388), (37.4447, 55.8551), (37.3999, 55.8551)],
    "CapeTown":[(18.5080,-33.9935), (18.5080, -33.9821), (18.5245, -33.9821), (18.5245,-33.9935)]
}

# 城市范围
CITY_BOUNDARY = {
    "SanFrancisco": [(-122.5099, 37.8076), (-122.5099, 37.6153), (-122.3630, 37.6153), (-122.3630, 37.8076)],
    "NewYork": [(-74.0186, 40.7751), (-74.0186, 40.6551), (-73.8068, 40.6551), (-73.8068, 40.7751)],
    "Beijing": [(116.1536, 40.0891), (116.1536, 39.7442), (116.6082, 39.7442), (116.6082, 40.0891)],
    "Shanghai": [(121.1215, 31.4193), (121.1215, 30.7300), (121.9730, 30.7300), (121.9730, 31.4193)],
    "Mumbai": [(72.7576, 19.2729), (72.7576, 18.9797), (72.9836, 18.9797), (72.9836, 19.2729)],    
    "Tokyo": [(139.6005, 35.8712), (139.6005, 35.5859), (139.9713, 35.5859), (139.9713, 35.8712)],
    "London": [(-0.3159, 51.6146), (-0.3159, 51.3598), (0.1675, 51.3598), (0.1675, 51.6146)],
    "Paris": [(2.249, 48.9038), (2.249, 48.8115), (2.4239, 48.8115), (2.4239, 48.9038)],
    "Moscow": [(37.4016, 55.8792), (37.4016, 55.6319), (37.8067, 55.6319), (37.8067, 55.8792)],
    "SaoPaulo": [(-46.8251, -23.4242), (-46.8251, -23.7765), (-46.4365, -23.7765), (-46.4365, -23.4242)],
    "Nairobi": [(36.6868, -1.1906), (36.6868, -1.3381), (36.9456, -1.3381), (36.9456, -1.1906)],
    "CapeTown": [(18.3472, -33.8179), (18.3472, -34.0674), (18.6974, -34.0674), (18.6974, -33.8179)],
    "Sydney": [(150.8382, -33.6450), (150.8382, -34.0447), (151.2982, -34.0447), (151.2982, -33.6450)]
}

