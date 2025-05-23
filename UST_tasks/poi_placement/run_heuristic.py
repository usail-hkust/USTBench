import argparse
import os.path
from env import StationPlacement
from evaluation_framework import *
import wandb
from tqdm import tqdm

import sys
sys.path.append("../..")

from utils.read_utils import dump_json

class Placement(object):
    """
    Callback for saving a model based on the training reward.
    """

    def __init__(self, graph_file, node_file, plan_file, location):
        super(Placement, self).__init__()
        self.location = location

        self.env = StationPlacement(graph_file, node_file, plan_file)

        wandb.init(
            project="USTBench-POI-Placement",
            group=f"{self.location}-Heuristic",
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
            install_fee, game_over = self.env.step_heuristic(history, is_negative=False)
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

            if not os.path.exists("./History/"):
                os.makedirs("./History/")
            dump_json(history, f"./History/{self.location}_Heuristic.json")

def main(location):
    graph_file = f"./Data/Graph/{location}/{location}.graphml"
    node_file = f"./Data/Graph/{location}/nodes_extended_{location}.txt"
    plan_file = f"./Data/Graph/{location}/existing_plan_{location}.pkl"

    # Run the agent
    algo = Placement(graph_file, node_file, plan_file, location)
    algo.run()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reinforcement learning model training for POI placement.")
    parser.add_argument('--location', default='Qiaonan', type=str, help="Location name (e.g., Qiaonan).")
    args = parser.parse_args()

    # Run main
    main(args.location)
