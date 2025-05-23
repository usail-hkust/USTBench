import argparse
from .env import Simulation
from utils.read_utils import load_json
from utils.language_model import LLM as LLM
import wandb

class Route_Planning(object):
    def __init__(self, batch_size, location, sumo_config, route_file, road_info_file, adjacency_file, task_info_file, llm_path_or_name, step_size, max_steps, use_reflection):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.location = location

        self.sim = Simulation(
            location=location,
            sumo_config_file=sumo_config,
            route_file=route_file,
            road_info_file=road_info_file,
            adjacency_file=adjacency_file,
            step_size=step_size,
            max_steps=max_steps
        )
        self.sim.initialize()

        # initialize language model
        task_info = load_json(task_info_file)
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)

        wandb.init(
            project="USTBench-Route-Planning",
            group=f"{self.location}-{llm_name}{'-w/o reflection' if not use_reflection else ''}",
            name="Examination"
        )

    def run(self):
        average_travel_time, throughput = self.sim.run(self.llm_agent)

        wandb.log({
            "average_travel_time": average_travel_time,
            "throughput": throughput
        })
        wandb.finish()

def main(llm_path_or_name, batch_size, use_reflection, location, step_size=180.0, max_steps=43200):
    sumo_config = f"./UST_tasks/route_planning/Data/Road_Network/{location}_sumo_config.sumocfg"
    route_file = f"./UST_tasks/route_planning/Data/Road_Network/{location}_od_0.01.rou.alt.xml"
    road_info_file = f"./UST_tasks/route_planning/Data/Road_Network/{location}_road_info.json"
    adjacency_file = f"./UST_tasks/route_planning/Data/Road_Network/{location}_adjacency_info.json"
    task_info_file = "./UST_tasks/route_planning/Data/task_info.json"

    algo = Route_Planning(batch_size, location, sumo_config, route_file, road_info_file, adjacency_file,
                          task_info_file, llm_path_or_name, step_size, max_steps, use_reflection)
    algo.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SUMO simulation with autonomous vehicles.")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--location", type=str, default="Manhattan", help="Location of the simulation.")
    parser.add_argument("--step-size", type=float, default=180.0, help="Simulation step size in seconds.")
    parser.add_argument("--max-steps", type=int, default=43200, help="Maximum number of simulation steps.")
    args = parser.parse_args()

    main(args.llm_path_or_name, args.batch_size, args.location, args.step_size, args.max_steps)
