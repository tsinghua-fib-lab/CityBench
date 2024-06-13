class MapConfig:
    """
    与地图有关的Config
    """
    def __init__(self, mongo_uri, mongo_db, mongo_coll, cache_dir, routing_client_addr):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_coll = mongo_coll
        self.cache_dir = cache_dir
        self.routing_client_addr = routing_client_addr

    def map_config_dict(self):
        return {
            "mongo_uri": self.mongo_uri,
            "mongo_db": self.mongo_db,
            "mongo_coll": self.mongo_coll,
            "cache_dir": self.cache_dir,
        }


class Config:
    """
    评估脚本Config
    """
    def __init__(self, region_exp, resource_path, count_limit, evaluate_version, use_english, map_config, env_radius, semantic_radius, max_group, group_size, min_road_length, question_num, dis2corner, step, reason_ques_num, get_limit):
        self.resource_path = resource_path
        self.count_limit = count_limit
        self.region_exp = region_exp
        self.evaluate_version = evaluate_version
        self.use_english = use_english
        self.env_radius = env_radius
        self.semantic_radius = semantic_radius
        self.map_config: MapConfig = map_config
        self.max_group = max_group
        self.group_size = group_size
        self.min_road_length = min_road_length
        self.question_num = question_num
        self.dis2corner = dis2corner
        self.step = step
        self.reason_ques_num = reason_ques_num
        self.get_limit = get_limit
