import os.path
import sys
import random
sys.path.append("../")

from tqdm import tqdm
import networkx as nx

import traci
from .env_utils import parse_rou_file, get_edge_info, get_congestion_level, get_autonomous_vehicle_observation, update_route, get_env_change_text
from utils.read_utils import load_json, dump_json

class Simulation:
    def __init__(self, location, sumo_config_file, route_file, road_info_file, adjacency_file, step_size=1.0, max_steps=1000):
        """
        Initialize the simulation.
        :param sumo_config_file: Path to the SUMO configuration file.
        :param route_file: Path to the route file.
        :param step_size: Simulation step size in seconds.
        :param max_steps: Maximum number of simulation steps.
        """
        self.location = location
        self.sumo_config_file = sumo_config_file
        self.route_file = route_file
        self.road_info_file = road_info_file
        self.adjacency_file = adjacency_file
        self.step_size = step_size
        self.max_steps = max_steps
        self.autonomous_vehicles = set()
        self.edges = None
        self.road_info = None
        self.road_network = None
        self.vehicle_start_times = {}
        self.vehicle_end_times = {}
        self.average_travel_time = 0
        self.total_vehicles = 0
        self.completed_vehicles = 0
        self.history = []
        if not os.path.exists("./UST_tasks/route_planning/History"):
            os.mkdir("./UST_tasks/route_planning/History")

    def initialize(self):
        """
        Start the SUMO simulation and select autonomous vehicles.
        """
        sumo_cmd = ["sumo", "-c", self.sumo_config_file, "--no-warnings", "--ignore-route-errors"]
        traci.start(sumo_cmd)

        # Parse route file and select autonomous vehicles
        all_vehicles = parse_rou_file(self.route_file)
        all_vehicle_ids = [veh_id for veh_id, _, _ in all_vehicles]
        self.total_vehicles = len(all_vehicles)
        self.autonomous_vehicles = set(random.sample(all_vehicle_ids, int(0.02 * self.total_vehicles)))
        print(f"Selected {len(self.autonomous_vehicles)} vehicles as autonomous vehicles.")

        # Initialize edge information
        self.edges = traci.edge.getIDList()
        self.road_info = load_json(self.road_info_file)
        adjacency_matrix = load_json(self.adjacency_file)
        self.road_network = nx.DiGraph()
        for edge in adjacency_matrix:
            road_len = self.road_info[edge]['road_len']
            for neighbor_edge in adjacency_matrix[edge]:
                self.road_network.add_edge(edge, neighbor_edge, weight=road_len)

    def run_heuristic(self):
        """
        Run the SUMO simulation.
        """
        try:
            step = 0.0
            pbar = tqdm(total=self.max_steps, desc="Simulation Progress", unit="steps")
            while step < self.max_steps:
                traci.simulationStep(step)

                # Update simulation info
                vehicle_ids = traci.vehicle.getIDList()
                current_time = traci.simulation.getTime()
                step += self.step_size
                pbar.update(self.step_size)

                # Record start times for new vehicles
                for veh_id in vehicle_ids:
                    if veh_id not in self.vehicle_start_times and veh_id in self.autonomous_vehicles:
                        self.vehicle_start_times[veh_id] = current_time

                # Check for vehicles that have arrived
                arrived_vehicles = traci.simulation.getArrivedIDList()
                for veh_id in arrived_vehicles:
                    if veh_id in self.vehicle_start_times:
                        self.vehicle_end_times[veh_id] = current_time
                        self.completed_vehicles += 1

            # Calculate results
            total_travel_time = sum(
                self.vehicle_end_times[veh_id] - self.vehicle_start_times.get(veh_id, 0)
                for veh_id in self.vehicle_end_times
            )
            self.average_travel_time = total_travel_time / len(self.vehicle_end_times) if self.vehicle_end_times else 0
            print(f"Average travel time: {self.average_travel_time:.2f}s")
            print(f"Network throughput: {self.completed_vehicles}/{self.total_vehicles} vehicles completed.")

        finally:
            print("Simulation ended.")
            traci.close()

            return self.average_travel_time, self.completed_vehicles

    def run(self, llm_agent):
        """
        Run the SUMO simulation.
        """
        update_vehicle_info = None
        if not os.path.exists("./UST_tasks/route_planning/History"):
            os.mkdir("./UST_tasks/route_planning/History")
        history_path = f"./UST_tasks/route_planning/History/{self.location}_{llm_agent.llm_name}.json"

        step = 0.0
        pbar = tqdm(total=self.max_steps, desc="Simulation Progress", unit="steps")
        while step < self.max_steps:
            # Update edge information
            for e in self.edges:
                _, vehicle_num, vehicle_speed, vehicle_length, _ = get_edge_info(e)
                occupancy_rate = vehicle_num / (self.road_info[e]["road_len"] / 8.0)
                alpha = occupancy_rate * 0.08 + 0.02
                min_eta = self.road_info[e]["road_len"] / self.road_info[e]["speed_limit"]
                eta = min_eta * (1 + alpha * vehicle_num)

                self.road_info[e].update({
                    "vehicle_num": vehicle_num,
                    "vehicle_speed": vehicle_speed,
                    "vehicle_length": vehicle_length,
                    "congestion_level": get_congestion_level(occupancy_rate),
                    "min_eta": min_eta,
                    "eta": eta
                })

            # Update simulation info
            vehicle_ids = traci.vehicle.getIDList()
            current_time = traci.simulation.getTime()
            step += self.step_size
            pbar.update(self.step_size)

            # Record start times for new vehicles
            for veh_id in vehicle_ids:
                if veh_id not in self.vehicle_start_times and veh_id in self.autonomous_vehicles:
                    self.vehicle_start_times[veh_id] = current_time

            # Update routes for autonomous vehicles for the first time
            if not update_vehicle_info:
                update_vehicle_info = get_autonomous_vehicle_observation(vehicle_ids, self.autonomous_vehicles,
                                                                         self.road_info, self.road_network)

            # Decision-making
            if update_vehicle_info[0]:
                decisions = llm_agent.hybrid_decision_making_pipeline(update_vehicle_info[1], update_vehicle_info[2])
            else:
                decisions = []

            pass_vehicles = []
            pass_data_texts = []
            pass_decisions = []
            pass_data_analysis = []
            pass_decision_summary = []
            for i, decision_info in enumerate(decisions):
                candidate_roads = update_vehicle_info[2][i].split("/")
                next_road = decision_info["answer"] if decision_info["answer"] and decision_info["answer"] in candidate_roads else None
                data_analysis = decision_info["data_analysis"] if decision_info["data_analysis"] else "N/A"
                decision_summary = decision_info["summary"] if decision_info["summary"] else "N/A"

                if next_road:
                    pass_vehicles.append(update_vehicle_info[0][i])
                    pass_data_texts.append(update_vehicle_info[1][i])
                    pass_decisions.append(decision_info["answer"])
                    pass_data_analysis.append(data_analysis)
                    pass_decision_summary.append(decision_summary)

                    route = update_route(update_vehicle_info[3][i], next_road, update_vehicle_info[4][i])
                    traci.vehicle.setRoute(vehID=update_vehicle_info[0][i], edgeList=[edge for edge in route])

            # Check for vehicles that have arrived
            arrived_vehicles = traci.simulation.getArrivedIDList()
            for veh_id in arrived_vehicles:
                if veh_id in self.vehicle_start_times and veh_id not in self.vehicle_end_times:
                    self.vehicle_end_times[veh_id] = current_time
                    self.completed_vehicles += 1

            # Update simulation
            traci.simulationStep(step)

            # Self-reflection
            vehicle_ids = traci.vehicle.getIDList()
            update_vehicle_info = get_autonomous_vehicle_observation(vehicle_ids, self.autonomous_vehicles,
                                                                     self.road_info, self.road_network)

            self_reflection_data_texts = []
            self_reflection_data_analysis = []
            self_reflection_decisions = []
            self_reflection_decision_summary = []
            self_env_change_texts = []
            for i, veh in enumerate(pass_vehicles):
                for j, updated_vehicle in enumerate(update_vehicle_info[0]):
                    if veh == updated_vehicle:
                        self_reflection_data_texts.append(pass_data_texts[i])
                        self_reflection_data_analysis.append(pass_data_analysis[i])
                        self_reflection_decisions.append(pass_decisions[i])
                        self_reflection_decision_summary.append(pass_decision_summary[i])
                        self_env_change_texts.append(get_env_change_text(pass_decisions[i], update_vehicle_info[5][j]))

            if len(self_env_change_texts) >= 1:
                self_reflections = llm_agent.hybrid_self_reflection_pipeline(
                    self_reflection_data_texts, self_reflection_decisions,
                    self_reflection_decision_summary, self_env_change_texts, update_vehicle_info[2]
                )
                # Log history
                self.history.append({
                    "data_texts": pass_data_texts,
                    "decisions": pass_decisions,
                    "data_analysis": pass_data_analysis,
                    "decision_summary": pass_decision_summary,
                    "env_change_texts": self_env_change_texts,
                    "self_reflections": self_reflections if len(self_reflections) >= 1 else "N/A",
                    "memory": llm_agent.memory
                })
                dump_json(self.history, history_path)

        # Calculate results
        total_travel_time = sum(
            self.vehicle_end_times[veh_id] - self.vehicle_start_times.get(veh_id, 0)
            for veh_id in self.vehicle_end_times
        )
        self.average_travel_time = total_travel_time / len(self.vehicle_end_times) if self.vehicle_end_times else 0
        print(f"Average travel time: {self.average_travel_time:.2f}s")
        print(f"Network throughput: {self.completed_vehicles}/{self.total_vehicles} vehicles completed.")
        print("Simulation ended.")
        traci.close()

        return self.average_travel_time, self.completed_vehicles



