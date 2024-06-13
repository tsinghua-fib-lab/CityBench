from multiprocessing import Pool
from llm_mob import main
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--user_cnt', type=int, default=50)
    parser.add_argument('--traj_cnt', type=int, default=10)

    args = parser.parse_args()
    user_cnt = args.user_cnt            # users 
    sample_single_user = args.traj_cnt  # trajectory for each user
    models = ["meta-llama/Meta-Llama-3-8B-Instruct"]
    # models = [
    #     "gpt-3.5", "gpt-4", "meta-llama/Meta-Llama-3-70B-Instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2",
    #     "meta-llama/Meta-Llama-3-8B-Instruct", "deepseek-chat"
    # ]
    cities = [
            "Beijing", "Cape", "London", "Moscow", "Mumbai", "Nairobi", "NewYork" ,"Paris" ,"San", "Sao", "Shanghai", "Sydney","Tokyo"
        ]
    
    para_group = []
    for c in cities:
        for m in models:
            para_group.append([c, m, user_cnt, sample_single_user])

    with Pool(6) as pool:
        results = pool.starmap(main, para_group)
