from .config import Config, MapConfig

config = Config(
    evaluate_version="v82",
    region_exp="newyork",
    resource_path="resource",
    count_limit=100,
    use_english=True,
    semantic_radius=500,
    env_radius=100,
    get_limit=10,
    min_road_length=100,
    group_size=50,
    max_group=5,
    question_num=200,
    dis2corner=50,
    step=12,
    reason_ques_num=500,
    map_config=MapConfig(
        mongo_uri="",
        mongo_db="llmsim",
        mongo_coll="map_newyork_20240512",
        cache_dir="./examples/cache",
        routing_client_addr="localhost:54101"
    ),
)
