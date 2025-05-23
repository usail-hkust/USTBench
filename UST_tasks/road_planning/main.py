import argparse
import os
import sys

import wandb

from utils.language_model import LLM
from UST_tasks.road_planning.env import RoadPlanningEnvironment
from UST_tasks.road_planning.env_utils.utils import *
from utils.read_utils import dump_json, load_json

# Set environment variables and update system path
cwd = os.getcwd()
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.path.join(cwd, 'UST_tasks/road_planning/env_utils'))


class RoadPlanningSimulation:
    def __init__(self, llm_path_or_name, batch_size, slum_name, seed, tmp, root_dir, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.use_reflection = use_reflection
        self.llm_name = llm_name
        self.llm_path_or_name = llm_path_or_name
        self.batch_size = batch_size
        self.slum_name = slum_name
        self.seed = seed
        self.tmp = tmp
        self.root_dir = root_dir

        # Load configuration
        self.cfg = Config('demo', self.slum_name, self.seed, self.tmp, self.root_dir)

        # Initialize environment and language model
        self.env = RoadPlanningEnvironment(self.cfg)
        self.llm_agent = self.create_llm_agent(use_reflection)

    def create_llm_agent(self, use_reflection):
        """Initialize and return the language model agent."""
        task_info = load_json("./UST_tasks/road_planning/Data/task_info.json")
        llm_agent = LLM(self.llm_path_or_name, batch_size=self.batch_size, task_info=task_info, use_reflection=use_reflection)

        return llm_agent

    def run(self):
        """Run the road planning simulation."""
        wandb.init(
            project="USTBench-Road-Planning",
            group=f"{self.llm_name}{'-w/o reflection' if not self.use_reflection else ''}",
            name="Examination"
        )

        game_over = False
        game_start = True
        history = []
        os.makedirs("./UST_tasks/road_planning/History/", exist_ok=True)

        while not game_over:
            game_over = self.env.step(self.llm_agent, game_start, history)
            game_start = False
            dump_json(history, f"./UST_tasks/road_planning/History/{self.llm_name}.json")
        
        ## Number of road segments (NR), average distance (AD), sum of costs (SC)
        NR, AD, SC = self.env.get_metrics()
        wandb.log({
            "number_of_road_segments": NR,
            "average_distance": AD,
            "sum_of_costs": SC
        })

        wandb.finish()

def main(llm_path_or_name, batch_size, use_reflection, slum_name, seed=1,
         tmp=False, root_dir='./USTBench/UST_tasks/road_planning/logs'):
    simulation = RoadPlanningSimulation(llm_path_or_name, batch_size, slum_name, seed, tmp, root_dir, use_reflection)
    simulation.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run road planning simulation.")
    parser.add_argument("--llm-path-or-name", type=str,
                        default="/data/yuanzirui/LLMs/LLama/Meta-Llama-3___1-8B-Instruct",
                        help="Path to the language model or its name.")
    parser.add_argument("--slum-name", type=str, default="CapeTown1", help="Name of the slum for road planning.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--tmp", default=False, action="store_true", help="Use temporary storage.")
    parser.add_argument("--root-dir", type=str, default='./USTBench/UST_tasks/road_planning/logs',
                        help="Root directory for logs and outputs.")

    args = parser.parse_args()
    main(args.llm_path_or_name, args.slum_name, args.batch_size, args.seed, args.tmp, args.root_dir)
