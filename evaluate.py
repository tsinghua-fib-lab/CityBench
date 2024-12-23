import os
import argparse
import pandas as pd

from config import CITY_BOUNDARY, VLM_MODELS, LLM_MODELS, TASK_DEST_MAPPING, TASK_METRICS_MAPPING, RESULTS_PATH, RESULTS_FILE, METRICS_SELECTION

class Evaluator:
    def __init__(self, city_name, model_name, data_name, task_name) -> None:
        self.city_list = list(CITY_BOUNDARY.keys())
        self.model_list = {"vlm": VLM_MODELS, "llm": LLM_MODELS}
        self.task_list = list(TASK_DEST_MAPPING.keys())
        
        self.city_name_list = city_name.split(",")
        self.model_name_list = model_name.split(",")
        self.task_name_list = task_name.split(",")
        self.data_name = data_name

    def evaluate(self):
        # TODO: run single task or run task sets
        self.multiple_task_wrapper(self.task_name_list, self.model_name_list, self.city_name_list)

    def valid_inputs(self):
        assert self.city_name_list in self.city_list, "City name is not valid"
        assert self.model_name_list in self.model_list, "Model name is not valid"
        assert self.task_name_list in self.task_list, "Task name is not valid"

    def single_task_wrapper(self, task_name, model_name, city_name):
        # run single task 
        task_desc = TASK_DEST_MAPPING[task_name]
        if task_name in ["population", "objects"]:
            eval_scipt = "python -m {} --city_name={} --data_name={} --model_name={} --task_name={}".format(task_desc, city_name, self.data_name, model_name, task_name)
        else:
            eval_scipt = "python -m {} --city_name={} --data_name={} --model_name={}".format(task_desc, city_name, self.data_name, model_name)

        return os.system(eval_scipt)

    def multiple_task_wrapper(self, task_list, model_list, city_list):
        # TODO running multi tasks efficiently
        for task in task_list:
            for model in model_list:
                for city in city_list:
                    self.single_task_wrapper(task, model, city)

    def single_task_metrics(self, task_name):
        # run single task metrics
        task_metric = TASK_METRICS_MAPPING[task_name]
        if task_name in ["population", "objects"]:
            metric_scipt = "python -m {} --task_name={}".format(task_metric, task_name)
        else:
            metric_scipt = "python -m {}".format(task_metric)
        
        return os.system(metric_scipt)
    
    def multiple_task_metrics(self, task_list):
        # run multiple task metrics
        for task in task_list:
            self.single_task_metrics(task)


    def analyze_results(self):
        # 生成所有任务的评估结果
        self.multiple_task_metrics(self.task_list)

    def show_benchmark(self):
        # TODO 直接展示benchmark结果
        data_frames = []

        for task in self.task_name_list:
            if task in RESULTS_FILE:
                df = pd.read_csv(RESULTS_FILE[task])
                selected_columns = METRICS_SELECTION.get(task, [])
                if selected_columns:
                    df = df[selected_columns]
                data_frames.append(df)

        if data_frames:
            merged_df = pd.concat(data_frames, axis=0, ignore_index=True)  # 按行合并
            merged_df_grouped = merged_df.groupby(['Model_Name'], as_index=False).mean()
            output_file = os.path.join(RESULTS_PATH, "benchmark_results.csv")
            merged_df_grouped.to_csv(output_file, index=False)
            print(f"Benchmark results have been saved!")
        else:
            return pd.DataFrame() 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default="Beijing")
    parser.add_argument('--task_name', type=str, default='traffic')
    parser.add_argument('--data_name', type=str, default='all')
    parser.add_argument('--model_name', type=str, default="GPT4o")
    args = parser.parse_args()

    # Evaluator Initialization
    Eval = Evaluator(
        city_name=args.city_name,
        model_name=args.model_name,
        data_name=args.data_name,
        task_name=args.task_name)
    # Running Evalautor 
    Eval.evaluate()
    # Analyze Results
    Eval.analyze_results()
    # Show Results
    Eval.show_benchmark()
