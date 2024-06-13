from .config import Config, MapConfig

config = Config(
    evaluate_version="v82",
    resource_path="resource",
    count_limit=1000,
    # simulate_output_file="simulate/logs/output_citywalk_wudaokou_mock_100_10_1_v11.2-eng-chi-eval.jsonl",
    use_english=False,
    semantic_radius=500,
    env_radius=100,
    get_limit=10,
    min_road_length=100,
    region_exp="yuyuantan",
    group_size=50,
    max_group=5,
    question_num=200,
    dis2corner=50,
    step=12,
    reason_ques_num=500,
    map_config=MapConfig(
        mongo_uri="",
        mongo_db="llmsim",
        mongo_coll="map_beijing5ring_withpoi_0424",
        cache_dir="./examples/cache",
        routing_client_addr="localhost:52101"
    ),
)
