# Prepare tile coordinate and population data for one city
You could use the following command to prepare tile coordinate and population data for one city.

```bash
python prepare_image_and_pop.py --city_name Beijing --pop_tiff_path ppp_2020_1km_Aggregated.tif 
```
ppp_2020_1km_Aggregated.tif is the population data in tiff format. You could download the population data from [WorldPop](https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif).

The `--city_name` is the name of the city, and the `--pop_tiff_path` is the path of the population data in tiff format.

The output will be saved in the city name + '_img_indicators.csv' file.

# Download satellite images
You could use the following command to download the satellite images for one city.

```bash
python download_rs_img.py --city_name Beijing --output_dir images
```
# Run the Multi-Modal Larege Language Model
You could reference streetview/README.md to run the Multi-Modal Larege Language Model.
# Evaluate the model
For the population prediction task, you could use the following command to evaluate the model.

```bash
python eval_pop.py --city_name Beijing --model_name llava34b --img_indicators_csv_path Beijing_img_indicators.csv --output_dir output --jsonl_path YOUR_JSONL_PATH
```
The '--jsonl_path' is the path of the jsonl file that contains the prediction results. The jsonl file should be in the following format.
```json
[
{"img_name": "y_x", "text": "text"}
{"img_name": "y2_x2", "text": "text"}
]
```
As for the other tasks, you could use the following command to evaluate the model.

```bash
python eval_object.py --city_name Beijing --model_name llava34b --img_indicators_csv_path Beijing_img_indicators.csv --output_dir output --jsonl_path YOUR_JSONL_PATH
```
