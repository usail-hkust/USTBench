import argparse
from .env import UrbanDevelopmentPredictionEnv
from utils.read_utils import load_json
from utils.language_model import LLM as LLM
import wandb

class Congestion_Prediction(object):
    def __init__(self, batch_size, location, task_info_file, llm_path_or_name, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.data = load_json(f"./UST_tasks/urban_development_prediction/Data/{location}.json")

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)
        self.sim = UrbanDevelopmentPredictionEnv(self.data)

        wandb.init(
            project="USTBench-Socio_Economic-Prediction",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        gdp_mae, gdp_mse, gdp_mape, pop_mae, pop_mse, pop_mape = self.sim.run(self.llm_agent)

        wandb.log({
            "GDP MAE": gdp_mae,
            "GDP MSE": gdp_mse,
            "GDP MAPE": gdp_mape,
            "Population MAE": pop_mae,
            "Population MSE": pop_mse,
            "Population MAPE": pop_mape
        })
        wandb.finish()

def main(llm_path_or_name, batch_size, use_reflection, location):
    task_info_file = "./UST_tasks/urban_development_prediction/Data/task_info.json"

    algo = Congestion_Prediction(batch_size, location, task_info_file, llm_path_or_name, use_reflection)
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on urban development.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--location", type=str, default="Guangzhou", help="Location of the prediction.")
    args = parser.parse_args()

    main(args.llm_path_or_name, args.batch_size, args.location)
