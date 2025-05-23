import argparse
import os.path
from .env import StationPlacement
from .evaluation_framework import *
import wandb
from tqdm import tqdm

from utils.language_model import LLM
from utils.read_utils import load_json, dump_json

class LLMPlacement(object):
    """
    Callback for saving a model based on the training reward.
    """

    def __init__(self, llm_path_or_name, batch_size, graph_file, node_file, plan_file, task_info_file, location, use_reflection):
        super(LLMPlacement, self).__init__()
        llm_name = os.path.basename(llm_path_or_name)
        self.llm_name = llm_name
        self.location = location

        self.env = StationPlacement(graph_file, node_file, plan_file)

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)

        wandb.init(
            project="USTBench-POI-Placement",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        pbar = tqdm(total=BUDGET, desc="Remaining Budget", dynamic_ncols=True)
        game_over = False
        total_fee_cost = 0.0
        history = []
        init_benefit, init_cost, init_charge_time, init_wait_time, init_travel_time = existing_score_fixed(
            self.env.plan_instance.plan,
            self.env.node_list
        )

        while not game_over:
            install_fee, game_over = self.env.step_llm_agent(self.llm_agent, history)
            total_fee_cost += install_fee
            pbar.n = total_fee_cost
            pbar.refresh()

            benefit, cost, charge_time, wait_time, travel_time = existing_score(
                self.env.plan_instance.plan,
                self.env.node_list
            )
            results = {
                "benefit": benefit,
                "cost": cost,
                "charging_time": charge_time,
                "wait_time": wait_time,
                "travel_time": travel_time,
                "benefit_increase": (benefit - init_benefit) / init_benefit,
                "cost_decrease": (init_cost - cost) / init_cost,
                "charging_time_decrease": (init_charge_time - charge_time) / init_charge_time,
                "wait_decrease": (init_wait_time - wait_time) / init_wait_time,
                "travel_time_decrease": (init_travel_time - travel_time) / init_travel_time
            }
            wandb.log(results)

            if not os.path.exists("./UST_tasks/poi_placement/History/"):
                os.makedirs("./UST_tasks/poi_placement/History/")
            dump_json(history, f"./UST_tasks/poi_placement/History/{self.location}_{self.llm_name}.json")

def main(llm_path_or_name, batch_size, use_reflection, location):
    graph_file = f"./UST_tasks/poi_placement/Data/Graph/{location}/{location}.graphml"
    node_file = f"./UST_tasks/poi_placement/Data/Graph/{location}/nodes_extended_{location}.txt"
    plan_file = f"./UST_tasks/poi_placement/Data/Graph/{location}/existing_plan_{location}.pkl"
    task_info_file = f"./UST_tasks/poi_placement/Data/task_info.json"

    # Run the agent
    algo = LLMPlacement(llm_path_or_name, batch_size, graph_file, node_file, plan_file, task_info_file, location, use_reflection)
    algo.run()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reinforcement learning model training for POI placement.")
    parser.add_argument('--location', default='Qiaonan', type=str, help="Location name (e.g., Qiaonan).")
    parser.add_argument("--llm_path_or_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        type=str, help="The path or name of the LLM to use.")
    parser.add_argument("--batch_size", default=16, type=int, help="The batch size for the LLM.")
    args = parser.parse_args()

    # Run main
    main(args.llm_path_or_name, args.batch_size, args.location)
