import argparse
from UST_tasks.congestion_prediction.env import CongestionPredictionEnv
from utils.read_utils import load_json, read_csv_to_list
from utils.language_model import LLM as LLM
import wandb

class Congestion_Prediction(object):
    def __init__(self, batch_size, location, task_info_file, llm_path_or_name, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.data = load_json(f"./UST_tasks/congestion_prediction/Data/{location}_12h.json")
        self.roadnet = read_csv_to_list(f"./UST_tasks/congestion_prediction/Data/graph.csv")

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)
        self.sim = CongestionPredictionEnv(self.data, self.roadnet)

        wandb.init(
            project="USTBench-Congestion-Prediction",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        mae, mse, mape, accuracy = self.sim.run(self.llm_agent)

        wandb.log({
            "MAE": mae,
            "MSE": mse,
            "MAPE": mape,
            "Accuracy": accuracy
        })
        wandb.finish()

def main(llm_path_or_name, batch_size, use_reflection, location):
    task_info_file = "./UST_tasks/congestion_prediction/Data/task_info.json"

    algo = Congestion_Prediction(batch_size, location, task_info_file, llm_path_or_name, use_reflection)
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SUMO simulation with autonomous vehicles.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--location", type=str, default="Beijing", help="Location of the simulation.")
    args = parser.parse_args()

    main(args.llm_path_or_name, args.batch_size, args.location)
