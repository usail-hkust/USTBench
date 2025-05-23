import sys
import os
import time
import argparse
import wandb
import warnings

from utils.read_utils import load_json, dump_json
from .env_utils import *
from .env import TrafficSignalEnvironment
from utils.language_model import LLM

# Append root directory to system path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


class Simulation:
    """
    Simulation class for running LLM-based traffic signal control.
    Initializes the environment and language model agent, then runs the simulation.
    """
    def __init__(self, llm_path_or_name, batch_size, task_info_file, use_reflection, **simulation_args):
        llm_name = llm_path_or_name.split("/")[-1]
        self.llm_name = llm_name
        self.use_reflection = use_reflection
        self.env = TrafficSignalEnvironment(**simulation_args)
        task_info = load_json(task_info_file)

        # Initialize the language model agent
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)

    def run(self):
        """
        Run the simulation until the environment signals game over.
        Logs the results with wandb and saves the simulation history.
        """
        wandb.init(
            project="USTBench-Traffic-Signal-Control",
            group=f"{self.llm_name}{'-w/o reflection' if not self.use_reflection else ''}",
            name="Examination"
        )

        history_path = f"./UST_tasks/traffic_signal_control/History/{self.llm_name}.json"
        if not os.path.exists(f"./UST_tasks/traffic_signal_control/History"):
            os.makedirs(f"./UST_tasks/traffic_signal_control/History")
        game_over = False
        game_start = True
        history = []

        while not game_over:
            game_over = self.env.step(self.llm_agent, game_start, history)
            game_start = False
            dump_json(history, history_path)

        average_travel_time, average_waiting_time, average_queue_length = self.env.get_metrics()
        wandb.log({
            "average_travel_time": average_travel_time,
            "average_waiting_time": average_waiting_time,
            "average_queue_length": average_queue_length
        })

        wandb.finish()


def parse_args():
    """
    Parse command-line arguments for the simulation.
    """
    parser = argparse.ArgumentParser(description="Run LLM-based Traffic Signal Control Simulation.")
    parser.add_argument("--memo", type=str, default='LLMTSC')
    parser.add_argument("--llm_path_or_name", type=str,
                        default="/data/yuanzirui/LLMs/LLama/Meta-Llama-3___1-8B-Instruct")
    parser.add_argument("--new_max_tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default="LLM-Decision-making")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="newyork_28x7")
    parser.add_argument("--traffic_file", type=str, default="anon_28_7_newyork_real_double.json")
    return parser.parse_args()


def get_simulation_args(memo, new_max_tokens, eightphase, proj_name, num_rounds, dataset, traffic_file):
    """
    Prepare and return simulation arguments based on input parameters.
    """
    traffic_file_list = []
    if dataset == 'jinan':
        count = 3600
        road_net = "3_4"
        traffic_file_list = [
            "anon_3_4_jinan_real.json",
            "anon_3_4_jinan_real_2000.json",
            "anon_3_4_jinan_real_2500.json"
        ]
        template = "Jinan"
    elif dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = [
            "anon_4_4_hangzhou_real.json",
            "anon_4_4_hangzhou_real_5816.json"
        ]
        template = "Hangzhou"
    elif dataset == 'newyork_16x3':
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        template = "NewYork"
    elif dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = [
            "anon_28_7_newyork_real_double.json",
            "anon_28_7_newyork_real_triple.json"
        ]
        template = "NewYork"

    if "24h" in traffic_file:
        count = 86400

    # Validate traffic file
    try:
        if traffic_file not in traffic_file_list:
            raise flowFileException('Flow file does not exist.')
    except flowFileException as e:
        print(e)
        return

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file)

    dic_agent_conf_extra = {
        "NEW_MAX_TOKENS": new_max_tokens,
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "PROJECT_NAME": proj_name,
        "RUN_COUNTS": count,
        "NUM_ROUNDS": num_rounds,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": f"roadnet_{road_net}.json",
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],
        "DIC_REWARD_INFO": {
            'queue_length': -0.25
        }
    }

    if eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = [
            'WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
            'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT'
        ]
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30, 30, 30, 30, 30]
    else:
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join(current_file_path, "model", memo, f"{traffic_file}_{timestamp}"),
        "PATH_TO_WORK_DIRECTORY": os.path.join(current_file_path, "records", memo, f"{traffic_file}_{timestamp}"),
        "PATH_TO_DATA": os.path.join(current_file_path, "Data", template, str(road_net))
    }

    simulation_args = {
        "dic_path": dic_path_extra,
        "dic_agent_conf": dic_agent_conf_extra,
        "dic_traffic_env_conf": merge(dic_traffic_env_conf, dic_traffic_env_conf_extra)
    }
    return simulation_args


def main(llm_path_or_name='/data/yuanzirui/LLMs/LLama/Meta-Llama-3___1-8B-Instruct',
         batch_size=16, use_reflection=True, dataset='newyork_28x7', traffic_file='anon_28_7_newyork_real_double.json',
         memo='LLMTSC', new_max_tokens=16384, eightphase=False, proj_name='LLM-Decision-making', num_rounds=1):
    """
    Main function to initialize simulation arguments, create a Simulation instance, and run the simulation.
    """
    task_info_file = "./UST_tasks/traffic_signal_control/Data/task_info.json"
    simulation_args = get_simulation_args(memo, new_max_tokens, eightphase, proj_name, num_rounds, dataset, traffic_file)
    if simulation_args is None:
        return
    sim = Simulation(llm_path_or_name, batch_size, task_info_file, use_reflection, **simulation_args)
    sim.run()


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
