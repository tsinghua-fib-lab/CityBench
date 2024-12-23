import argparse
from multiprocessing import Pool

from .run_eval import main


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--user_cnt', type=int, default=50)
    parser.add_argument('--traj_cnt', type=int, default=10)

    args = parser.parse_args()
    user_cnt = args.user_cnt            # users 
    sample_single_user = args.traj_cnt  # trajectory for each user
    data_version="mini"
    split_path="citydata/mobility/checkin_split/"
    test_path="citydata/mobility/checkin_test_pk/"
    
    models = ["GPT4o"]
    
    cities = [
            "Beijing", "CapeTown", "London", "Moscow", "Mumbai", "Nairobi", "NewYork" ,"Paris" ,"SanFrancisco", "SaoPaulo", "Shanghai", "Sydney","Tokyo"
        ]
    
    para_group = []
    for c in cities:
        for m in models:
            para_group.append([c, m, user_cnt, sample_single_user, 40, 5, split_path, test_path, data_version])

    with Pool(6) as pool:
        results = pool.starmap(main, para_group)
