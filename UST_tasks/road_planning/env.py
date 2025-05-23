from UST_tasks.road_planning.env_utils import RoadEnv
from UST_tasks.road_planning.env_utils.utils import Config
import copy


class RoadPlanningEnvironment:
    """
    Simulation environment for planning road connections.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the simulation environment.

        :param cfg: Configuration dictionary for the environment.
        """
        self.cfg = cfg
        self.env = RoadEnv(cfg)
        self.connectivity = None
        self.NR = -1
        self.game_over = False
        self.step_count = 0
        self.stage = 1
        self.available_roads = []

    def step(self, llm_agent, game_start: bool, history: list) -> bool:
        """
        Execute a single simulation step.

        :param llm_agent: An agent with decision making and self-reflection pipelines.
        :param game_start: Flag indicating if the game has just started.
        :param history: A list that accumulates the history of steps.
        :return: Boolean indicating whether the simulation has ended.
        """
        self.step_count += 1
        print(f"step: {self.step_count}")

        if game_start:
            self.env.reset()

        # Get current observations and available roads
        observations = self.get_observation()
        self.available_roads = [road_data['road_id'] for road_data in observations['available_roads']]
        answer_option_forms = self._create_answer_option_forms()
        data_text = self.get_data_texts()

        # Decision making using the language model agent
        decision_info = llm_agent.hybrid_decision_making_pipeline([data_text], answer_option_forms)
        decision = decision_info[0]["answer"]
        data_analysis = decision_info[0]["data_analysis"]
        decision_summary = decision_info[0]["summary"]

        # Environment step based on decision and get feedback
        env_feedback = self.env_step(observations, decision)
        env_feedback_text = self.get_env_feedback_text(env_feedback)

        # Self-reflection stage
        self_reflection = llm_agent.hybrid_self_reflection_pipeline(
            [data_text], [decision], [decision_summary], [env_feedback_text], answer_option_forms
        )

        # Append current step details to history
        history.append({
            "data_text": data_text,
            "data_analysis": data_analysis,
            "answer": decision,
            "decision_summary": decision_summary,
            "env_change_text": env_feedback_text,
            "memory": llm_agent.memory,
            "self_reflection": self_reflection
        })

        return self.game_over

    def _create_answer_option_forms(self) -> list:
        """
        Create answer option forms based on available roads.

        :return: List of formatted answer options.
        """
        answer_option_form = ["\"" + '/'.join([f"road {road_id}" for road_id in self.available_roads]) + "\""]
        return answer_option_form

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.env.reset()
        self.game_over = False
        self.step_count = 0

    def env_step(self, last_observations: dict, decision: str) -> dict:
        """
        Take an environment step based on the decision and provide feedback.

        :param last_observations: Observations before the step.
        :param decision: Decision string provided by the agent.
        :return: Environment feedback as a dictionary.
        """
        last_available_roads = [road_data['road_id'] for road_data in last_observations['available_roads']]
        last_unconnected_regions = len(last_observations.get('unconnected_regions', []))
        last_avgdist = last_observations['avgdist']
        last_stage = self.stage

        # Parse decision to get the selected road id; default to first available if parsing fails
        selected_road_id = self._parse_decision(decision) if decision else self.available_roads[0]
        if selected_road_id not in self.available_roads:
            selected_road_id = self.available_roads[0]

        next_state, reward, done, info = self.env.step(selected_road_id)
        self.check_stage()
        next_observations = self.get_observation(done)
        if done:
            self.game_over = True

        next_available_roads = [road_data['road_id'] for road_data in next_observations['available_roads']]
        new_available_roads = list(set(next_available_roads) - set(last_available_roads))
        new_roads_feedback = self._get_new_roads_feedback(new_available_roads, last_stage)
        next_unconnected_regions = len(next_observations.get('unconnected_regions', []))
        next_avgdist = next_observations['avgdist']
        unconnected_regions_decrease = last_unconnected_regions - next_unconnected_regions
        avgdist_decrease = last_avgdist - next_avgdist

        if last_stage == 1:
            env_feedback = {
                f"road_{selected_road_id}": {
                    "newly_connected_regions": unconnected_regions_decrease,
                    "potential_highest_connected_regions_via_new_roads": new_roads_feedback
                }
            }
        else:
            env_feedback = {
                f"road_{selected_road_id}": {
                    "distance_reduction_among_regions": avgdist_decrease,
                    "potential_highest_distance_reduction_via_new_roads": new_roads_feedback
                }
            }
        return env_feedback

    def _parse_decision(self, decision: str) -> int:
        """
        Extract the road id from the decision string.

        :param decision: Decision string (e.g., "road_3").
        :return: Parsed road id as integer.
        """
        decision = decision.lower()
        selected_road_id = None
        if '_' in decision:
            try:
                selected_road_id = int(decision.split('_')[-1])
            except ValueError:
                pass
        elif " " in decision:
            try:
                selected_road_id = int(decision.split(' ')[-1])
            except ValueError:
                pass
        return selected_road_id if selected_road_id is not None else self.available_roads[0]

    def get_env_feedback_text(self, env_feedback: dict) -> str:
        """
        Generate a human-readable feedback text from environment feedback.

        :param env_feedback: Feedback dictionary from the environment step.
        :return: Formatted feedback string.
        """
        build_road = list(env_feedback.keys())[0]
        unit_text = 'km' if "distance_reduction_among_regions" in env_feedback[build_road] else ""
        road_id = int(build_road.split('_')[-1])
        keys = list(env_feedback[build_road].keys())
        results_type_text1 = keys[0]
        results_value1 = env_feedback[build_road][results_type_text1]
        results_type_text2 = keys[1]
        results_value2 = env_feedback[build_road][results_type_text2]
        text = (
            f"The feedback of building road {road_id}:\n"
            f"- {results_type_text1}: {results_value1}{unit_text}\n"
            f"- {results_type_text2}: {results_value2}{unit_text}\n"
        )
        return text

    def check_stage(self):
        """
        Update the simulation stage based on environment status.
        """
        if self.stage == 1 and self.env._stage == 'full_connected':
            self.NR = len(self.env._mg.road_edges)
        self.stage = 2 if self.env._stage == 'full_connected' else 1

    def get_observation(self, done=False) -> dict:
        """
        Retrieve and construct the current state observation.

        :return: Observation dictionary with regions, connectivity, and available roads.
        """
        observation = {}
        observation['stage'] = 2 if self.env._stage == 'full_connected' else 1

        mg = self.env._mg
        region_list = mg.inner_facelist
        road_list = mg.road_edges
        edge_list = mg.edge_list
        node_list = mg.node_list
        edge_length_list = mg.edge_length

        # Determine connected regions
        connected_regions = []
        for road in road_list:
            edge_id = edge_list.index(road)
            for region in mg.edge_face_index[edge_id]:
                if region not in connected_regions:
                    connected_regions.append(region)
        observation['connected_regions'] = []
        for region in connected_regions:
            region_id = region_list.index(region)
            region_nodes = [node_list.index(node) for node in region.nodes]
            observation['connected_regions'].append({
                'region_id': region_id,
                'region_nodes': region_nodes
            })

        # Determine unconnected regions (only in stage 1)
        if observation['stage'] == 1:
            unconnected_regions = [region for region in region_list if region not in connected_regions]
            observation['unconnected_regions'] = []
            for region in unconnected_regions:
                region_id = region_list.index(region)
                region_nodes = [node_list.index(node) for node in region.nodes]
                observation['unconnected_regions'].append({
                    'region_id': region_id,
                    'region_nodes': region_nodes
                })

        # Determine potential roads
        potential_roads = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0:
                potential_roads.append(edge)
        observation['available_roads'] = []
        observation['connectivity'] = []

        # Build connectivity information
        for road in road_list:
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
            observation['connectivity'].append((road_nodes[0], road_nodes[1], road_length))

        # Build available roads data
        for road in potential_roads:
            road_id = edge_list.index(road)
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
            if done:
                new_roads_data = []
            else:
                new_roads_data = self.get_new_road_data(road_id, potential_roads)
            road_data = {
                'road_id': road_id,
                'edge': road_nodes,
                'length': road_length,
                'new_roads': new_roads_data
            }
            observation['available_roads'].append(road_data)
        observation['avgdist'] = mg.get_f2f_avgdist()
        return observation

    def get_new_road_data(self, road_id: int, potential_roads: list) -> list:
        """
        Retrieve data for new roads that become available after building a given road.

        :param road_id: The id of the road being built.
        :param potential_roads: List of currently potential roads.
        :return: List of dictionaries containing new road data.
        """
        new_env = copy.deepcopy(self.env)
        mg = new_env._mg
        node_list = mg.node_list
        edge_list = mg.edge_list
        edge_length_list = mg.edge_length
        new_env.step(road_id)
        new_roads_data = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0 and edge not in potential_roads:
                road_data = {
                    'road_id': edge_id,
                    'edge': (node_list.index(edge.nodes[0]), node_list.index(edge.nodes[1])),
                    'length': round(edge_length_list[edge_list.index(edge)], 2)
                }
                new_roads_data.append(road_data)
        return new_roads_data

    def get_data_texts(self) -> str:
        """
        Construct a text summary of the current state for decision making.

        :return: Data text string.
        """
        mg = self.env._mg
        stage = 2 if self.env._stage == 'full_connected' else 1
        region_list = mg.inner_facelist
        road_list = mg.road_edges
        edge_list = mg.edge_list
        node_list = mg.node_list
        edge_length_list = mg.edge_length

        # Determine connected regions
        connected_regions = []
        for road in road_list:
            edge_id = edge_list.index(road)
            for region in mg.edge_face_index[edge_id]:
                if region not in connected_regions:
                    connected_regions.append(region)
        # Determine unconnected regions
        unconnected_regions = [region for region in region_list if region not in connected_regions]

        # Determine potential roads
        potential_roads = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0:
                potential_roads.append(edge)

        # Construct data text
        data_text = f"We are at stage {stage}.\n\n"
        data_text += "Connected regions:\n\n"
        for region in connected_regions:
            region_id = region_list.index(region)
            region_nodes = [node_list.index(node) for node in region.nodes]
            data_text += f"region {region_id}:\n- region_nodes: {region_nodes}\n\n"
        data_text += "Connectivity:\n["
        for i, road in enumerate(road_list):
            if i != 0:
                data_text += ", "
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
            data_text += f"(node {road_nodes[0]}, node {road_nodes[1]}, {road_length}km)"
        data_text += "]\n\n"
        if stage == 1:
            data_text += "Unconnected regions:\n\n"
            for region in unconnected_regions:
                region_id = region_list.index(region)
                region_nodes = [node_list.index(node) for node in region.nodes]
                data_text += f"region {region_id}:\n- region_nodes: {region_nodes}\n\n"
        data_text += "Available roads:\n\n"
        potential_roads_dict = {}
        for road in potential_roads:
            road_id = edge_list.index(road)
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_text = f"(node {node_list.index(road.nodes[0])}, node {node_list.index(road.nodes[1])}, {road_length}km)"
            new_roads_id_list, new_roads_text = self.get_new_roads(road_id, potential_roads)
            potential_roads_dict[road_id] = {
                'distance': edge_length_list[road_id],
                'duration': edge_length_list[road_id]
            }
            if stage == 1:
                potential_roads_dict[road_id]['topology1'] = 0
                potential_roads_dict[road_id]['topology2'] = 0
                road_connected_regions = mg.edge_face_index[road_id]
                new_connected_regions = connected_regions.copy()
                for region in road_connected_regions:
                    if region in unconnected_regions:
                        new_connected_regions.append(region)
                        potential_roads_dict[road_id]['topology1'] += 1
                topology2_list = []
                for new_r_id in new_roads_id_list:
                    topology2 = 0
                    road_connected_regions = mg.edge_face_index[new_r_id]
                    for region in road_connected_regions:
                        if region not in new_connected_regions:
                            topology2 += 1
                    topology2_list.append(topology2)
                potential_roads_dict[road_id]['topology2'] = max(topology2_list) if topology2_list else 0
            else:
                potential_roads_dict[road_id]['topology1'] = self._get_decreased_distance(road_id)
                decreased_distance_list = []
                for new_r_id in new_roads_id_list:
                    decreased_distance = self._get_new_road_decreased_distance_after_road(road_id, new_r_id)
                    decreased_distance_list.append(decreased_distance)
                potential_roads_dict[road_id]['topology2'] = max(decreased_distance_list) if decreased_distance_list else 0

            data_text += f"road {road_id}:\n"
            data_text += f"- road_edge: {road_text}\n"
            data_text += f"- new_roads: {new_roads_text}\n\n"
        return data_text

    def _calc_road_unconnected_region(self, road_id: int) -> int:
        """
        Calculate how many unconnected regions would be linked by constructing the given road.

        :param road_id: ID of the road to test.
        :return: Number of newly connected regions.
        """
        mg = self.env._mg
        region_list = mg.inner_facelist
        road_list = mg.road_edges
        edge_list = mg.edge_list
        node_list = mg.node_list

        connected_regions = []
        for road in road_list:
            edge_id = edge_list.index(road)
            for region in mg.edge_face_index[edge_id]:
                if region not in connected_regions:
                    connected_regions.append(region)
        unconnected_regions = [region for region in region_list if region not in connected_regions]

        road = edge_list[road_id]
        road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
        new_connected_regions = []
        for region in unconnected_regions:
            region_nodes = [node_list.index(node) for node in region.nodes]
            if road_nodes[0] in region_nodes or road_nodes[1] in region_nodes:
                new_connected_regions.append(region)
        return len(new_connected_regions)

    def _get_new_roads_feedback(self, new_available_roads: list, last_stage: int, done: bool = False) -> int:
        """
        Evaluate the maximum potential feedback from newly available roads.

        :param new_available_roads: List of new road ids.
        :param last_stage: The previous simulation stage.
        :param done: Whether the simulation has ended.
        :return: Maximum feedback value.
        """
        feedback_values = []
        for road_id in new_available_roads:
            if last_stage == 1:
                feedback_value = self._calc_road_unconnected_region(road_id)
            else:
                feedback_value = 0 if done else self._get_new_road_decreased_distance(road_id)
            feedback_values.append(feedback_value)
        return max(feedback_values) if feedback_values else 0

    def _get_decreased_distance(self, road_id: int) -> float:
        """
        Calculate the reduction in average distance after constructing a road.

        :param road_id: Road id to test.
        :return: Decreased distance value.
        """
        new_env = copy.deepcopy(self.env)
        mg = new_env._mg
        origin_avgdist = mg.get_f2f_avgdist()
        new_env.step(road_id)
        after_avgdist = mg.get_f2f_avgdist()
        return origin_avgdist - after_avgdist

    def _get_new_road_decreased_distance(self, road_id: list) -> float:
        """
        Calculate the decreased average distance for a new road option.

        :param road_id: The ids of the road being built.
        :return: Decreased distance value.
        """
        origin_avgdist = self.env._mg.get_f2f_avgdist()
        new_env = copy.deepcopy(self.env)
        new_env.step(road_id)
        after_avgdist = new_env._mg.get_f2f_avgdist()
        return origin_avgdist - after_avgdist
    
    def _get_new_road_decreased_distance_after_road(self, road_id, new_road_id):
        """
        Calculate the decreased average distance after building a road.
        :param road_id: The id of the road being built.
        :param new_road_id: The id of the new road being built.
        :return: Decreased distance value.
        """
        origin_avgdist = self.env._mg.get_f2f_avgdist()
        new_env = copy.deepcopy(self.env)
        next_state, reward, done, info = new_env.step(road_id)
        if done:
            return 0
        else:
            orginal_avgdist = new_env._mg.get_f2f_avgdist()
            next_state, reward, done, info = new_env.step(new_road_id)
            after_avgdist = new_env._mg.get_f2f_avgdist()
            return orginal_avgdist - after_avgdist

    def get_decision_making_texts(self, data_text: str) -> str:
        """
        Generate the prompt text for the decision-making process.

        :param data_text: Data summary text.
        :return: Full decision-making prompt text.
        """
        info_text = (
            "## Task Description\n\n"
            "Design a road network to connect unconnected regions to the existing connected regions in an urban area. "
            "Using the provided data, identify the roads that can be built to maximize connectivity.\n\n"
        )
        schema_text = (
            "## Data Schema\n\n"
            "- connected_regions: A list of regions that are already connected.\n"
            "- connectivity: A list of tuples in the format (node_1, node_2, distance), representing undirected roads between nodes with the specified distance (in km).\n"
            "- unconnected_regions: A list of regions that are currently not connected to any other region.\n"
            "- region_nodes: A list of boundary nodes defining the extent of each region.\n"
            "- available_roads: A list of roads that can be built to improve connectivity.\n"
            "- road_edge: A tuple in the format (node_1, node_2, distance), representing an undirected road that can be constructed between two nodes with the specified distance (in km).\n"
            "- new_roads: A list of tuples in the format (node_1, node_2, distance), representing new roads that can be constructed after building the previous road.\n\n"
        )
        stage = 2 if self.env._stage == 'full_connected' else 1
        question_text = (
            "## Question\n\n"
            "Based on the provided data, identify the most suitable road should be built next "
            f"to {'reduce the travel distances among regions' if stage != 1 else 'connect the largest number of unconnected regions'}.\n\n"
        )
        note_text = (
            "## Note\n\n"
            "Let's solve this step by step. Finally, summarize your analysis, and provide your answer in JSON format, like:\n\n"
            "```JSON\n{\n  \"summary\": \"YOUR_SUMMARY\",\n  \"answer\": \"1/2/3/4\"\n}\n```"
        )
        return info_text + schema_text + data_text + question_text + note_text

    def get_analysis_text(self, potential_roads_dict: dict) -> str:
        """
        Generate an analysis text ranking the potential roads based on various metrics.

        :param potential_roads_dict: Dictionary containing potential road metrics.
        :return: Analysis text string.
        """
        stage = 2 if self.env._stage == 'full_connected' else 1
        analysis_text = "## Analysis\n\n"
        if stage == 1:
            analysis_text += "Rank of the available roads by the number of regions that can be connected by them:\n"
            topology2_text = (
                "Rank of the available roads by the average number of unconnected regions that can be linked to the already connected regions "
                "through newly buildable roads after constructing the initial available road:\n"
            )
        else:
            analysis_text += "Rank of the available roads by the effect of reducing travel distance among regions:\n"
            topology2_text = (
                "Rank of the available roads by the average deduction on travel distance of regions through newly buildable roads after constructing the initial available road:\n"
            )
        topology1_rank = self.get_road_id_sorted_by_values(potential_roads_dict, 'topology1')
        analysis_text += self.get_rank_text(topology1_rank, potential_roads_dict, 'topology1')
        analysis_text += topology2_text
        topology2_rank = self.get_road_id_sorted_by_values(potential_roads_dict, 'topology2')
        analysis_text += self.get_rank_text(topology2_rank, potential_roads_dict, 'topology2')
        analysis_text += "Rank of the road lengths:\n"
        distance_rank = self.get_road_id_sorted_by_values(potential_roads_dict, 'distance')
        analysis_text += self.get_rank_text(distance_rank, potential_roads_dict, 'distance')
        return analysis_text

    def get_road_id_sorted_by_values(self, potential_roads_dict: dict, key_name: str) -> list:
        """
        Sort road IDs based on a given metric.

        :param potential_roads_dict: Dictionary with road metrics.
        :param key_name: The key in the dictionary to sort by.
        :return: List of road ids sorted accordingly.
        """
        scores = [(road_id, info[key_name]) for road_id, info in potential_roads_dict.items()]
        reverse_order = key_name in ['topology1', 'topology2']
        sorted_ids = sorted(scores, key=lambda x: x[1], reverse=reverse_order)
        return [road_id for road_id, _ in sorted_ids]

    def get_rank_text(self, sorted_ids: list, potential_roads_dict: dict, key_name: str) -> str:
        """
        Generate a ranking string based on sorted road ids and their corresponding metric.

        :param sorted_ids: List of sorted road ids.
        :param potential_roads_dict: Dictionary with road metrics.
        :param key_name: The key corresponding to the metric.
        :return: Formatted ranking string.
        """
        separator = " < " if key_name in ['distance', 'duration'] else " > "
        rank_text = ""
        for i, road_id in enumerate(sorted_ids):
            if i != len(sorted_ids) - 1:
                current_val = round(potential_roads_dict[road_id][key_name], 2)
                next_val = round(potential_roads_dict[sorted_ids[i + 1]][key_name], 2)
                sep = " = " if current_val == next_val else separator
                rank_text += f"road {road_id}{sep}"
            else:
                rank_text += f"road {road_id}\n\n"
        return rank_text

    def get_new_roads(self, road_id: int, potential_roads: list) -> tuple:
        """
        Get new roads that become available after constructing a given road.

        :param road_id: The id of the road being built.
        :param potential_roads: List of currently potential roads.
        :return: Tuple with list of new road ids and a formatted text description.
        """
        new_env = copy.deepcopy(self.env)
        mg = new_env._mg
        edge_list = mg.edge_list
        new_env.step(road_id)
        new_roads = []
        new_roads_id_list = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0 and edge not in potential_roads:
                new_roads.append(edge)
                new_roads_id_list.append(edge_id)
        node_list = mg.node_list
        new_roads_text = []
        for r_id, r in zip(new_roads_id_list, new_roads):
            node0 = node_list.index(r.nodes[0])
            node1 = node_list.index(r.nodes[1])
            length = round(mg.edge_length[edge_list.index(r)], 2)
            new_roads_text.append(f"(node {node0}, node {node1}, {length}km)")
        return new_roads_id_list, f"[{', '.join(new_roads_text)}]"
    
    def _get_sum_of_costs(self):
        road_list = self.env._mg.road_edges
        edge_list = self.env._mg.edge_list
        edge_length_list = self.env._mg.edge_length
        road_length_list = [edge_length_list[edge_list.index(road)] for road in road_list]
        return sum(road_length_list)

    def get_metrics(self):
        """
        Calculate and return the number of road segments, average distance, and sum of costs.
        :return: Tuple containing NR, AD, and SC."
        """
        NR = self.NR
        AD = self.env._mg.get_f2f_avgdist()
        SC = self._get_sum_of_costs()
        return NR, AD, SC
