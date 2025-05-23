import sys
import os
from copy import deepcopy
import numpy as np
from .env_utils import *

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


class TrafficSignalEnvironment:
    """A traffic signal control environment that simulates intersections and vehicle movement."""

    def __init__(self, **simulation_args):
        """
        Initialize the traffic signal environment.

        Args:
            simulation_args (dict): Configuration dictionary containing:
                - dic_path: Path configurations
                - dic_agent_conf: Agent configurations
                - dic_traffic_env_conf: Traffic environment configurations
        """
        self.dic_path = simulation_args["dic_path"]
        self.dic_agent_conf = simulation_args["dic_agent_conf"]
        self.dic_traffic_env_conf = simulation_args["dic_traffic_env_conf"]

        # Constants
        self.car_spacing = 9  # meters between cars
        self.seg_num = 10  # number of segments per lane
        self.connectivity = None

        self._initialize_environment()

    def _initialize_environment(self):
        """Set up the initial environment state and configurations."""
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )

        # Initial state setup
        state = self.env.reset()
        self.current_states, _ = self._process_state(state)
        self.intersection_id_list = list(range(len(self.env.list_intersection)))
        self.connectivity = self._init_connectivity()
        self.queue_length_episode = []
        self.waiting_time_episode = []


        # Step configuration
        self.end_step = int(self.dic_traffic_env_conf["RUN_COUNTS"] /
                            self.dic_traffic_env_conf['MIN_ACTION_TIME'])
        self.step_count = 0

    def step(self, llm_agent, game_start, history):
        """
        Execute one simulation step.

        Args:
            llm_agent: The agent making traffic signal decisions
            game_start (bool): Whether this is the first step
            history (list): History of previous steps

        Returns:
            bool: Whether the simulation has ended
        """
        self.step_count += 1
        print(f"Step: {self.step_count}")

        if game_start:
            state = self.env.reset()
            self.current_states, _ = self._process_state(state)
            self.connectivity = self._init_connectivity()

        # Get and process observations
        observation_features = self._get_observation()
        observation_texts = self._observations_to_text(observation_features)

        # Agent decision-making
        answer_form = ["\"ETWT/ELWL/NTST/NLSL\""] * len(observation_texts)
        decision_info = llm_agent.hybrid_decision_making_pipeline(
            observation_texts, answer_form
        )

        # Extract decision information
        decisions = [info["answer"] for info in decision_info]
        data_analyses = [info["data_analysis"] for info in decision_info]
        decision_summaries = [info["summary"] for info in decision_info]

        # Execute environment step
        env_feedback = self._env_step(decisions)
        env_feedback_texts = self._get_env_feedback_text(
            env_feedback, observation_features
        )

        # Agent self-reflection
        self_reflection = llm_agent.hybrid_self_reflection_pipeline(
            observation_texts, decisions,
            decision_summaries, env_feedback_texts, answer_form
        )

        # Update metrics
        self._update_metrics()

        # Update history
        history.append({
            "data_text": observation_texts,
            "data_analysis": data_analyses,
            "answer": decisions,
            "decision_summary": decision_summaries,
            "env_change_text": env_feedback_texts,
            "memory": llm_agent.memory,
            "self_reflection": self_reflection
        })

        return self._check_game_over()

    def _check_game_over(self):
        """Check if the simulation should end."""
        return self.step_count >= self.end_step

    def _env_step(self, decisions):
        """Execute one step in the traffic environment."""
        action_list = [self._selection_to_signal_code(d) for d in decisions]
        next_states, _, done, _ = self.env.step(action_list)
        next_states = self._process_state(next_states)
        self.current_states, _ = next_states
        return self._get_observation()

    def _selection_to_signal_code(self, decision):
        """Convert a letter decision (A/B/C/D) to a signal code index."""
        selection = ['ETWT', 'ELWL', 'NTST', 'NLSL']
        return selection.index(decision) if decision in selection else 0

    def _get_observation(self):
        """Collect and format the current observation of traffic conditions."""
        traffic_conditions = []

        for i in range(len(self.current_states)):
            state = self.current_states[i]
            inter_feature = {}

            for lane in state:
                inter_feature[lane] = {
                    'queue_num': state[lane]['queue_len'],
                    'moving_num': state[lane]['moving_num'],
                    'occupancy': state[lane]['occupancy'],
                    'avg_wait_time': state[lane]['avg_wait_time']
                }

            traffic_conditions.append(inter_feature)

        return {
            "traffic_conditions": traffic_conditions,
            "connectivity": self.connectivity
        }

    def _observations_to_text(self, observation_features):
        """Convert observation features to human-readable text."""
        observation_texts = []

        for i in range(len(observation_features["traffic_conditions"])):
            inter_connectivity = self._get_connectivity_of_inter(i)
            analysis_data = self._analyze_traffic_condition(
                i, observation_features["traffic_conditions"], inter_connectivity
            )
            data_text = self._get_data_text(i, observation_features["traffic_conditions"])
            observation_texts.append(data_text)

        return observation_texts

    def _get_env_feedback_text(self, env_feedback, observation_features):
        """Generate feedback text about environment changes after an action."""
        last_traffic_conditions = observation_features["traffic_conditions"]
        next_traffic_conditions = env_feedback["traffic_conditions"]
        env_feedback_texts = []

        metric_to_text = {
            'queue_num': 'queues',
            'moving_num': 'moving vehicles',
            'occupancy': 'occupancy',
            'avg_wait_time': 'average waiting time'
        }

        for i in range(len(last_traffic_conditions)):
            inter_connectivity = self._get_connectivity_of_inter(i)
            last_analysis = self._analyze_traffic_condition(
                i, last_traffic_conditions, inter_connectivity
            )
            next_analysis = self._analyze_traffic_condition(
                i, next_traffic_conditions, inter_connectivity
            )

            feedback_text = ""
            for loc in ['local', 'upstream', 'downstream']:
                for metric in ['queue_num', 'moving_num', 'occupancy', 'avg_wait_time']:
                    last_value, next_value = self._calculate_metric_values(
                        last_analysis, next_analysis, loc, metric
                    )
                    change = next_value - last_value

                    if change == 0:
                        feedback_text += self._format_no_change_text(loc, metric_to_text[metric])
                    else:
                        feedback_text += self._format_change_text(
                            loc, metric_to_text[metric], change, metric
                        )

            env_feedback_texts.append(feedback_text)

        return env_feedback_texts

    def _calculate_metric_values(self, last_analysis, next_analysis, loc, metric):
        """Calculate metric values for comparison."""
        if metric != 'avg_wait_time':
            last_value = sum(last_analysis[loc][metric].values())
            next_value = sum(next_analysis[loc][metric].values())
        else:
            last_value = self._calculate_weighted_wait_time(last_analysis, loc)
            next_value = self._calculate_weighted_wait_time(next_analysis, loc)

        return last_value, next_value

    def _calculate_weighted_wait_time(self, analysis, loc):
        """Calculate weighted average wait time."""
        wait_times = [
            analysis[loc]['avg_wait_time'][lane] * analysis[loc]['queue_num'][lane]
            for lane in analysis[loc]['queue_num'].keys()
        ]
        total_queues = sum(analysis[loc]['queue_num'].values())

        return sum(wait_times) / total_queues if total_queues != 0 else 0

    def _format_no_change_text(self, location, metric_name):
        """Format text when there's no change in a metric."""
        if location == 'local':
            return f'No changes in {metric_name} at target intersection lanes\n'
        return f'No changes in {location} {metric_name}\n'

    def _format_change_text(self, location, metric_name, change, metric_type):
        """Format text describing a change in a metric."""
        change_value = self._format_change_value(change, metric_type)

        if change > 0:
            change_direction = "Increase"
        else:
            change_direction = "Decrease"

        if location == 'local':
            return f"{change_direction} in {metric_name} at target intersection lanes: {change_value}\n"
        return f"{change_direction} in {location} {metric_name}: {change_value}\n"

    def _format_change_value(self, change, metric_type):
        """Format the numeric value of a change appropriately."""
        if metric_type == 'occupancy':
            return f"{round(abs(change) * 100, 2)}%"
        elif metric_type == 'avg_wait_time':
            return f"{round(abs(change), 2)}"
        return f"{int(abs(change))}"

    def _analyze_traffic_condition(self, inter_id, traffic_conditions, inter_connectivity):
        """Analyze traffic conditions at an intersection and its neighbors."""
        analysis_data = {
            'local': {},
            'upstream': {},
            'downstream': {}
        }

        for loc in analysis_data:
            for metric in ['queue_num', 'moving_num', 'occupancy', 'avg_wait_time']:
                analysis_data[loc][metric] = {}

                if loc == 'local':
                    for lane in traffic_conditions[inter_id]:
                        analysis_data[loc][metric][lane] = traffic_conditions[inter_id][lane][metric]
                else:
                    for idx, lane in inter_connectivity[loc]:
                        key = f'{lane} of {idx}'
                        analysis_data[loc][metric][key] = traffic_conditions[idx][lane][metric]

        return analysis_data

    def _generate_sorted_text_with_value(self, data):
        """Generate sorted text representation of data with values."""
        non_zero_items = [(k, v) for k, v in data.items() if v != 0.0]
        sorted_items = sorted(non_zero_items, key=lambda x: x[1], reverse=True)

        result = []
        current_value = None
        current_keys = []

        for key, value in sorted_items:
            if current_value is None or value == current_value:
                current_keys.append(f"{key} ({value:.2f})")
            else:
                if len(current_keys) > 1:
                    result.append(' = '.join(current_keys))
                else:
                    result.append(current_keys[0])
                current_keys = [f"{key} ({value:.2f})"]
            current_value = value

        if current_keys:
            if len(current_keys) > 1:
                result.append(' = '.join(current_keys))
            else:
                result.append(current_keys[0])

        zero_items = [k for k, v in data.items() if v == 0.0]
        if zero_items:
            result.append('other lanes')

        return ' > '.join(result), len(non_zero_items), 0

    def _analysis_data_to_text(self, data):
        """Convert analysis data to human-readable text."""
        metric_names = {
            'queue_num': 'queues',
            'moving_num': 'moving vehicles',
            'avg_wait_time': 'waiting time',
            'occupancy': 'occupancy'
        }

        analysis_text = ""

        for metric in metric_names:
            for location in data:
                metric_data = data[location][metric]

                # Adjust values for certain metrics
                if metric == 'occupancy':
                    metric_data = {k: v * 100 for k, v in metric_data.items()}
                elif metric == 'avg_wait_time':
                    metric_data = {k: v / 60 for k, v in metric_data.items()}

                sorted_text, non_zero_count, _ = self._generate_sorted_text_with_value(metric_data)

                # Handle special cases for zero values
                if non_zero_count == 0:
                    sorted_text = self._get_zero_value_text(metric)
                elif metric == 'occupancy':
                    sorted_text = sorted_text.replace(")", "%)")

                # Format the output line
                if location in ['upstream', 'downstream']:
                    line = f"- Ranking of {location} {metric_names[metric]}: {sorted_text}\n"
                else:
                    line = f"- Ranking of {metric_names[metric]} at target intersection lanes: {sorted_text}\n"

                analysis_text += line

        return analysis_text

    def _get_zero_value_text(self, metric):
        """Get appropriate text when a metric has zero value."""
        zero_texts = {
            'queue_num': 'no vehicles queuing',
            'moving_num': 'no vehicles moving',
            'avg_wait_time': 'no vehicles waiting',
            'occupancy': 'no vehicles'
        }
        return zero_texts.get(metric, 'no data')

    def _process_state(self, state):
        """Process raw state data into a more usable format."""
        current_states = []
        current_outputs = []

        for i in range(len(state)):
            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])

            statistic_state, statistic_incoming_state, _ = self._get_state_detail_many_seg_all_lane(roads)
            arrive_left_times = self.env.list_intersection[i].dic_vehicle_arrive_leave_time

            for lane in location_all_direction_dict:
                statistic_state[lane]['stay_time'] = {}
                statistic_state[lane]['queue_num'] = statistic_state[lane]['queue_len']
                statistic_state[lane]['moving_num'] = sum(statistic_state[lane]['cells'])
                statistic_state[lane]['occupancy'] = len(statistic_state[lane]['veh2pos']) / (
                        statistic_state[lane]['road_length'] // self.car_spacing
                )

                for veh in statistic_state[lane]['veh2cell']:
                    enter_time = arrive_left_times[veh]["enter_time"]
                    current_time = self.env.current_time
                    statistic_state[lane]['stay_time'][veh] = current_time - enter_time

            current_states.append(statistic_state)
            current_outputs.append(statistic_incoming_state)

        return current_states, current_outputs

    def _get_state_detail_many_seg_all_lane(self, roads):
        """
        Retrieve detailed state information for all lanes divided into segments.

        Args:
            roads (dict): Road configuration dictionary

        Returns:
            tuple: (statistic_state, statistic_incoming_state, mean_speed)
        """
        lane_queues = self.env.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.env.eng.get_lane_vehicles()

        statistic_state = {}
        statistic_state_incoming = {}
        outgoing_lane_speeds = []

        for r in roads:
            location = roads[r]["location"]
            road_length = float(roads[r]["length"])

            if roads[r]["type"] == "outgoing":
                self._process_outgoing_road(
                    r, roads, location, road_length, lane_queues, statistic_state
                )

                # Process vehicle positions for outgoing roads
                straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
                left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
                right_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_right"]]
                lanes = straight_lanes + left_lanes + right_lanes

                for lane in lanes:
                    self._process_lane_vehicles(
                        lane, roads[r], location, road_length,
                        lane_vehicles, statistic_state, outgoing_lane_speeds
                    )
            else:
                # Process incoming roads
                self._process_incoming_road(
                    r, roads, location, road_length, lane_queues,
                    lane_vehicles, statistic_state_incoming
                )

        mean_speed = np.mean(outgoing_lane_speeds) if outgoing_lane_speeds else 0.0
        return statistic_state, statistic_state_incoming, mean_speed

    def _process_outgoing_road(self, road_id, roads, location, road_length, lane_queues, statistic_state):
        """Process data for an outgoing road."""
        road = roads[road_id]
        loc_short = location_dict_short[location]

        if road["go_straight"] is not None:
            queue_len = sum(lane_queues[f"{road_id}_{lane}"] for lane in road["lanes"]["go_straight"])
            statistic_state[f"{loc_short}T"] = self._create_lane_state(road_length, queue_len)

        if road["turn_left"] is not None:
            queue_len = sum(lane_queues[f"{road_id}_{lane}"] for lane in road["lanes"]["turn_left"])
            statistic_state[f"{loc_short}L"] = self._create_lane_state(road_length, queue_len)

        if road["turn_right"] is not None:
            queue_len = sum(lane_queues[f"{road_id}_{lane}"] for lane in road["lanes"]["turn_right"])
            statistic_state[f"{loc_short}R"] = self._create_lane_state(road_length, queue_len)

    def _create_lane_state(self, road_length, queue_len):
        """Create initial state dictionary for a lane."""
        return {
            "cells": [0] * self.seg_num,
            "ql_cells": [0] * self.seg_num,
            "queue_len": queue_len,
            "out_of_lane": 0,
            "veh2cell": {},
            "avg_wait_time": 0.0,
            "wait_time": {},
            "veh2pos": {},
            "road_length": road_length
        }

    def _process_lane_vehicles(self, lane, road, location, road_length,
                               lane_vehicles, statistic_state, outgoing_speeds):
        """Process vehicle data for a specific lane."""
        vehicles = lane_vehicles[lane]
        lane_group = self._get_lane_group(lane, road, location)

        if lane_group == -1:
            return

        lane_key = location_all_direction_dict[lane_group]
        waiting_times = []

        for veh in vehicles:
            veh_info = self.env.eng.get_vehicle_info(veh)
            lane_pos = road_length - float(veh_info["distance"])

            # Update vehicle position data
            statistic_state[lane_key]["veh2pos"][veh] = lane_pos

            # Update segment data
            seg_length = road_length / self.seg_num
            gpt_lane_cell = int(lane_pos // seg_length)
            statistic_state[lane_key]["veh2cell"][veh] = gpt_lane_cell

            # Update waiting time data
            veh_waiting_time = self.env.waiting_vehicle_list[veh][
                'time'] if veh in self.env.waiting_vehicle_list else 0.0
            statistic_state[lane_key]["wait_time"][veh] = veh_waiting_time

            if veh in self.env.waiting_vehicle_list:
                waiting_times.append(veh_waiting_time)

            # Update cell counts
            if gpt_lane_cell >= self.seg_num:
                statistic_state[lane_key]["out_of_lane"] += 1
            else:
                speed = float(veh_info["speed"])
                if speed > 0.1:
                    statistic_state[lane_key]["cells"][gpt_lane_cell] += 1
                    outgoing_speeds.append(speed)
                else:
                    statistic_state[lane_key]["ql_cells"][gpt_lane_cell] += 1

        # Update average wait time
        if waiting_times:
            statistic_state[lane_key]["avg_wait_time"] = np.mean(waiting_times)

    def _get_lane_group(self, lane, road, location):
        """Determine the lane group for a given lane."""
        lane_parts = lane.split('_')
        lane_idx = int(lane_parts[-1])

        if location == "North":
            if lane_idx in road["lanes"]["go_straight"]:
                return 0
            elif lane_idx in road["lanes"]["turn_left"]:
                return 1
            elif lane_idx in road["lanes"]["turn_right"]:
                return 2
        elif location == "South":
            if lane_idx in road["lanes"]["go_straight"]:
                return 3
            elif lane_idx in road["lanes"]["turn_left"]:
                return 4
            elif lane_idx in road["lanes"]["turn_right"]:
                return 5
        elif location == "East":
            if lane_idx in road["lanes"]["go_straight"]:
                return 6
            elif lane_idx in road["lanes"]["turn_left"]:
                return 7
            elif lane_idx in road["lanes"]["turn_right"]:
                return 8
        elif location == "West":
            if lane_idx in road["lanes"]["go_straight"]:
                return 9
            elif lane_idx in road["lanes"]["turn_left"]:
                return 10
            elif lane_idx in road["lanes"]["turn_right"]:
                return 11
        return -1

    def _process_incoming_road(self, road_id, roads, location, road_length,
                               lane_queues, lane_vehicles, statistic_state):
        """Process data for an incoming road."""
        queue_len = sum(lane_queues[f"{road_id}_{lane}"] for lane in range(3))
        output_loc = incoming2output[roads[road_id]['location']]

        statistic_state[output_loc] = {
            "cells": [0] * self.seg_num,
            "ql_cells": [0] * self.seg_num,
            "out_of_lane": 0,
            "veh2cell": {},
            "queue_len": queue_len
        }

        incoming_lanes = [f"{road_id}_{idx}" for idx in range(3)]

        for lane in incoming_lanes:
            self._process_incoming_lane_vehicles(
                lane, location, road_length, lane_vehicles, statistic_state
            )

    def _process_incoming_lane_vehicles(self, lane, location, road_length,
                                        lane_vehicles, statistic_state):
        """Process vehicle data for an incoming lane."""
        vehicles = lane_vehicles[lane]
        lane_group = self._get_incoming_lane_group(location)

        if lane_group == -1:
            return

        lane_key = location_incoming_dict[lane_group]

        for veh in vehicles:
            veh_info = self.env.eng.get_vehicle_info(veh)
            lane_pos = road_length - float(veh_info["distance"])

            seg_length = road_length / self.seg_num
            gpt_lane_cell = int(lane_pos // seg_length)
            statistic_state[lane_key]["veh2cell"][veh] = gpt_lane_cell

            if gpt_lane_cell >= self.seg_num:
                statistic_state[lane_key]["out_of_lane"] += 1
            else:
                if float(veh_info["speed"]) > 0.1:
                    statistic_state[lane_key]["cells"][gpt_lane_cell] += 1
                else:
                    statistic_state[lane_key]["ql_cells"][gpt_lane_cell] += 1

    def _get_incoming_lane_group(self, location):
        """Get lane group for incoming lanes."""
        if location == "North":
            return 1
        elif location == "South":
            return 0
        elif location == "East":
            return 3
        elif location == "West":
            return 2
        return -1

    def _init_connectivity(self):
        """Initialize the connectivity graph between intersections."""
        connectivity = set()

        for inter_id in self.intersection_id_list:
            inter_connectivity = self._get_connectivity_of_inter(inter_id)
            connectivity.update(inter_connectivity['edges'])

        return connectivity

    def _get_connectivity_of_inter(self, inter_id):
        """Get connectivity information for a specific intersection."""
        upstream_relation = {
            'NT': ['North', ['NT', 'EL', 'WR']],
            'NL': ['North', ['NT', 'EL', 'WR']],
            'NR': ['North', ['NT', 'EL', 'WR']],
            'ET': ['East', ['ET', 'SL', 'NR']],
            'EL': ['East', ['ET', 'SL', 'NR']],
            'ER': ['East', ['ET', 'SL', 'NR']],
            'ST': ['South', ['ST', 'WL', 'ER']],
            'SL': ['South', ['ST', 'WL', 'ER']],
            'SR': ['South', ['ST', 'WL', 'ER']],
            'WT': ['West', ['WT', 'NL', 'SR']],
            'WL': ['West', ['WT', 'NL', 'SR']],
            'WR': ['West', ['WT', 'NL', 'SR']]
        }

        downstream_relation = {
            'NT': ['South', ['NR', 'NT', 'NL']],
            'NL': ['East', ['WR', 'WT', 'WL']],
            'NR': ['West', ['ER', 'ET', 'EL']],
            'ET': ['West', ['ER', 'ET', 'EL']],
            'EL': ['South', ['NR', 'NT', 'NL']],
            'ER': ['North', ['SR', 'ST', 'SL']],
            'ST': ['North', ['SR', 'ST', 'SL']],
            'SL': ['West', ['ER', 'ET', 'EL']],
            'SR': ['East', ['WR', 'WT', 'WL']],
            'WT': ['East', ['WR', 'WT', 'WL']],
            'WL': ['North', ['SR', 'ST', 'SL']],
            'WR': ['South', ['NR', 'NT', 'NL']]
        }

        neighbor_list = self._get_neighbor_list(inter_id)
        loc2id = {neighbor['location']: neighbor['id'] for neighbor in neighbor_list}

        lane2lane_edges = []
        upstream_inter_lane_tuple_list = []
        downstream_inter_lane_tuple_list = []

        for lane in location_all_direction_dict:
            # Process upstream connections
            upstream_loc = upstream_relation[lane][0]
            if upstream_loc in loc2id:
                upstream_id = loc2id[upstream_loc]
                for up_lane in upstream_relation[lane][1]:
                    edge_length = self.current_states[upstream_id][up_lane]['road_length']
                    edge_item = (
                        f"{up_lane} of {upstream_id}",
                        f"{lane} of {inter_id}",
                        f"{edge_length}m"
                    )
                    lane2lane_edges.append(edge_item)
                    upstream_inter_lane_tuple_list.append((upstream_id, up_lane))

            # Process downstream connections
            downstream_loc = downstream_relation[lane][0]
            if downstream_loc in loc2id:
                downstream_id = loc2id[downstream_loc]
                for down_lane in downstream_relation[lane][1]:
                    edge_length = self.current_states[inter_id][lane]['road_length']
                    edge_item = (
                        f"{lane} of {inter_id}",
                        f"{down_lane} of {downstream_id}",
                        f"{edge_length}m"
                    )
                    lane2lane_edges.append(edge_item)
                    downstream_inter_lane_tuple_list.append((downstream_id, down_lane))

        return {
            'connect_intersections': list(loc2id.values()),
            'edges': list(set(lane2lane_edges)),
            'upstream': upstream_inter_lane_tuple_list,
            'downstream': downstream_inter_lane_tuple_list
        }

    def _get_neighbor_list(self, inter_id):
        """Get list of neighboring intersections."""
        n_list = []
        inter_name = self.env.list_intersection[inter_id].inter_name
        inter_list = list(self.env.intersection_dict.keys())

        intersection = self.env.intersection_dict[inter_name]
        roads = deepcopy(intersection["roads"])
        road_list = list(roads.keys())

        neighbor_list = [
            inter for inter in self.env.traffic_light_node_dict[inter_name]['neighbor_ENWS']
            if inter
        ]

        road2inter = [r.replace("road", "intersection")[:-2] for r in road_list]
        neighbor2loc = {
            inter: roads[road_list[i]]['location']
            for i, inter in enumerate(road2inter)
            if inter in neighbor_list
        }

        for neighbor_inter_name in neighbor_list:
            n_list.append({
                "id": inter_list.index(neighbor_inter_name),
                "name": neighbor_inter_name,
                "location": neighbor2loc[neighbor_inter_name]
            })

        return n_list

    def _get_data_text(self, inter_id, traffic_conditions):
        """
        Generate human-readable text describing the traffic conditions.

        Args:
            inter_id (int): Intersection ID
            traffic_conditions (list): List of traffic condition dictionaries

        Returns:
            str: Formatted text describing the traffic conditions
        """
        connectivity_data = self._get_connectivity_of_inter(inter_id)
        connected_intersection_ids = connectivity_data['connect_intersections']
        all_inter_ids = [inter_id] + connected_intersection_ids

        data_text = ""

        # Add traffic condition text for each intersection
        for i, idx in enumerate(all_inter_ids):
            inter_condition_text = ''
            empty = True

            if idx == inter_id:
                inter_condition_text += f"Target Intersection {inter_id}:\n"
            else:
                inter_condition_text += f"Neighboring Intersection {idx}:\n"

            for lane, lane_info in traffic_conditions[idx].items():
                if lane_info['occupancy'] != 0:
                    empty = False
                    inter_condition_text += f"- {lane} lane:\n"
                    inter_condition_text += f"  - queue: {lane_info['queue_num']}\n"
                    inter_condition_text += f"  - move: {lane_info['moving_num']}\n"
                    inter_condition_text += f"  - wait_time: {round(lane_info['avg_wait_time'] / 60, 2)}\n"
                    inter_condition_text += f"  - occupancy: {round(lane_info['occupancy'] * 100, 2)}%\n"

            if not empty:
                data_text += inter_condition_text + '\n'

        # Add connectivity information
        data_text += "The connectivity of lanes:\n["
        data_text += ', '.join(
            f"({', '.join(map(str, edge))})"
            for edge in connectivity_data['edges']
        )
        data_text += ']'

        return data_text
    
    def _update_metrics(self):
        queue_length_inter = []
        for inter in self.env.list_intersection:
            queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
        self.queue_length_episode.append(sum(queue_length_inter)) 
        # queue_length_inter = np.array(queue_length_inter)
        # ql_num_matrix = queue_length_inter.reshape(self.num_col, self.num_row)
        # waiting time
        waiting_times = []
        for veh in self.env.waiting_vehicle_list:
            waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
        self.waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)
    
    def _get_average_travel_time(self):
        current_time = self.env.get_current_time()
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else current_time
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        average_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
        return average_travel_time

    def get_metrics(self):
        ## perf at this point
        awt = np.mean(self.waiting_time_episode)
        aql = np.mean(self.queue_length_episode)
        att = self._get_average_travel_time()
        return att, awt, aql
