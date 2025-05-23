import argparse
from .env import TrafficFlowPredictionEnv
from utils.read_utils import load_json
from utils.language_model import LLM as LLM
import wandb

class Congestion_Prediction(object):
    def __init__(self, batch_size, location, task_info_file, llm_path_or_name, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.data = load_json(f"./UST_tasks/traffic_flow_prediction/Data/{location}.json")

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)
        self.sim = TrafficFlowPredictionEnv(self.data)

        wandb.init(
            project="USTBench-Traffic-OD-Prediction",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        inflow_mae, inflow_mse, inflow_mape, outflow_mae, outflow_mse, outflow_mape = self.sim.run(self.llm_agent)

        wandb.log({
            "Traffic Inflow MAE": inflow_mae,
            "Traffic Inflow MSE": inflow_mse,
            "Traffic Inflow MAPE": inflow_mape,
            "Traffic Outflow MAE": outflow_mae,
            "Traffic Outflow MSE": outflow_mse,
            "Traffic Outflow MAPE": outflow_mape
        })
        wandb.finish()

def main(llm_path_or_name, batch_size, use_reflection, location):
    task_info_file = "./UST_tasks/traffic_flow_prediction/Data/task_info.json"

    algo = Congestion_Prediction(batch_size, location, task_info_file, llm_path_or_name, use_reflection)
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on urban development.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--location", type=str, default="Newyork", help="Location of the prediction.")
    args = parser.parse_args()

    main(args.llm_path_or_name, args.batch_size, args.location)
