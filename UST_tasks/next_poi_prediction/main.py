import argparse
from .env import HumanMobilityPredictionEnv
from utils.read_utils import load_json
from utils.language_model import LLM
import wandb

class HumanMobilityPrediction(object):
    def __init__(self, batch_size, location, task_info_file, llm_path_or_name, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.data = load_json(f"./UST_tasks/human_mobility_prediction/Data/{location}.json")[:3000]
        self.candidate_dict = load_json(f"./UST_tasks/human_mobility_prediction/Data/{location}_candidate_dict.json")
        self.sim = HumanMobilityPredictionEnv(self.data, candidate_dict=self.candidate_dict)

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)

        wandb.init(
            project="USTBench-Next-POI-Prediction",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        precision, mrr, ndcg = self.sim.run(self.llm_agent)

        wandb.log({"precision": precision, "mrr": mrr, "ndcg": ndcg})
        wandb.finish()

def main(llm_path_or_name, batch_size, use_reflection, location):
    task_info_file = "./UST_tasks/human_mobility_prediction/Data/task_info.json"

    algo = HumanMobilityPrediction(batch_size, location, task_info_file, llm_path_or_name, use_reflection)
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SUMO simulation with autonomous vehicles.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to the language model or its name.")
    parser.add_argument("--location", type=str, default="Newyork", help="Location of the simulation.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    args = parser.parse_args()

    main(args.llm_path_or_name, args.batch_size, args.location)
