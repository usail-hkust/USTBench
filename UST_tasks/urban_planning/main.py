import argparse
import os

import setproctitle
import warnings

import wandb

from khrylib.utils.torch import *
from UST_tasks.urban_planning.utils.config import Config
from UST_tasks.urban_planning.agents.urban_planning_agent import UrbanPlanningAgent
from utils.read_utils import load_json, dump_json

warnings.simplefilter(action='ignore', category=FutureWarning)


class UrbanPlanningInference:
    def __init__(self, llm_path_or_name, batch_size, cfg, root_dir, tmp, agent_type, mean_action, visualize, only_road, save_video, global_seed, iteration, use_reflection):
        self.llm_path_or_name = llm_path_or_name
        self.use_reflection = use_reflection
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.cfg_name = cfg
        self.tmp = tmp
        self.agent_type = agent_type
        self.mean_action = mean_action
        self.visualize = visualize
        self.only_road = only_road
        self.save_video = save_video
        self.global_seed = global_seed
        self.iteration = iteration

        setproctitle.setproctitle('urban_planning')

        # Load configuration
        self.cfg = Config(self.cfg_name, self.global_seed, self.tmp, self.root_dir, self.agent_type)

        # Set device and seed
        self.device = self.set_device_and_seed()

        # Initialize agent
        self.agent = self.create_agent()

        # Freeze land use if only road planning is required
        if self.only_road:
            self.agent.freeze_land_use()

    def set_device_and_seed(self):
        """Set device and random seed for reproducibility."""
        dtype = torch.float32
        torch.set_default_dtype(dtype)
        device = torch.device('cpu')  # Currently using CPU by default
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        return device

    def create_agent(self):
        """Create and return the UrbanPlanningAgent."""
        checkpoint = int(self.iteration) if self.iteration.isnumeric() else self.iteration
        return UrbanPlanningAgent(cfg=self.cfg, dtype=torch.float32, device=self.device, num_threads=1,
                                  training=False, checkpoint=checkpoint, restore_best_rewards=True)

    def run(self):
        """Run inference for the agent."""
        model_name = os.path.basename(self.llm_path_or_name)
        wandb.init(
            project="USTBench-Urban-Planning",
            group=f"{self.cfg_name}-{model_name}{'-w/o reflection' if not self.use_reflection else ''}",
            name="Examination"
        )
        self.agent.LLM_inference(llm_path_or_name=self.llm_path_or_name, batch_size=self.batch_size,
                                 task_info=load_json("./UST_tasks/urban_planning/cfg/task_info.json"),
                                 num_samples=1, mean_action=self.mean_action, use_reflection=self.use_reflection)
        scores = self.agent.env.score_plan()
        wandb.log({
            "service": scores[1]['life_circle'],
            "ecology": scores[1]['greenness']
        })
        wandb.finish()
        if not os.path.exists("./UST_tasks/urban_planning/History/"):
            os.makedirs("./UST_tasks/urban_planning/History/")
        dump_json(self.agent.llm_history, f"./UST_tasks/urban_planning/History/{self.cfg_name}_{model_name}.json")


def main(llm_path_or_name, batch_size, use_reflection, cfg, root_dir="./UST_tasks/urban_planning/records", tmp=False, agent="rule-centralized", mean_action=False,
         visualize=False, only_road=False, save_video=False, global_seed=2025, iteration="0"):
    inference = UrbanPlanningInference(llm_path_or_name, batch_size, cfg, root_dir, tmp, agent, mean_action, visualize, only_road, save_video, global_seed, iteration, use_reflection)
    inference.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run urban planning inference.")
    parser.add_argument("--llm-path-or-name", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--root-dir", type=str, default="./UST_tasks/urban_planning/records",
                        help="Root directory for logs, summaries, and checkpoints.")
    parser.add_argument("--cfg", type=str, default="hlg", help="Configuration file for RL training.")
    parser.add_argument("--tmp", default=False, action="store_true", help="Use temporary storage.")
    parser.add_argument("--agent", type=str, choices=['rl-sgnn', 'rl-mlp', 'rule-centralized', 'rule-decentralized', 'gsca', 'ga'],
                        default="rule-centralized", help="Agent type.")
    parser.add_argument("--mean-action", default=False, action="store_true", help="Use greedy strategy.")
    parser.add_argument("--visualize", default=False, action="store_true", help="Visualize the planning process.")
    parser.add_argument("--only-road", default=False, action="store_true", help="Visualize only road planning.")
    parser.add_argument("--save-video", default=False, action="store_true", help="Save video of the planning process.")
    parser.add_argument("--global-seed", type=int, default=2025, help="Seed for environment and weight initialization.")
    parser.add_argument("--iteration", type=str, default="0", help='Start iteration (number or "best").')

    args = parser.parse_args()
    main(args.llm_path_or_name, args.batch_size, args.cfg, args.root_dir, args.tmp, args.agent, args.mean_action, args.visualize, args.only_road, args.save_video, args.global_seed, args.iteration)
