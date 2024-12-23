import os
import pandas as pd
import argparse

def get_data(City):
    image_files = os.listdir(f"citydata/street_view/{City}_CUT/")

    # Random sample 500 images for each city and get latitude & longitutde
    city_data = {
        "img_name": [img_name for img_name in image_files[:500]],
        "lat": [img_name.split("&")[1] for img_name in image_files[:500]], 
        "lng": [img_name.split("&")[2] for img_name in image_files[:500]],
    }

    city_df = pd.DataFrame(city_data)
    city_csv_path = f"citydata/street_view/{City}_data.csv"
    city_df.to_csv(city_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='Beijing', help='city name')
    args = parser.parse_args()

    get_data(args.city_name)
