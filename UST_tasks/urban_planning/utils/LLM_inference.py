import copy
import json

import numpy as np
import pandas as pd
import torch
from utils.language_model import LLM
import networkx as nx

class LLM_Agent():
    def __init__(self, llm_path_or_name, batch_size, task_info, use_reflection):
        self.llm_agent = LLM(llm_path_or_name, batch_size=batch_size, task_info=task_info, use_reflection=use_reflection)
        self.service_types = ['outside', 'feasible', 'road', 'boundary', 'residential', 'business', 'office', 'green', 'green', 'school', 'hospital', 'hospital', 'recreation']
        self.service_types_with_size = ['outside', 'feasible', 'road', 'boundary', 'residential', 'business', 'office', 'green_l', 'green_s', 'school', 'hospital_l', 'hospital_s', 'recreation']

    def get_1_2_hop_neighbors(self, graph, query_edge):
        sub_graph = nx.Graph()

        one_hop_neighbors = list(graph.neighbors(query_edge))

        two_hop_neighbors = list()
        for neighbor in one_hop_neighbors:
            neighbors_of_neighbor = graph.neighbors(neighbor)
            sub_graph.add_edge(query_edge, neighbor, weight=graph[query_edge][neighbor]['weight'])
            for neighbor_of_neighbor in list(neighbors_of_neighbor):
                sub_graph.add_edge(neighbor, neighbor_of_neighbor,
                                   weight=graph[neighbor][neighbor_of_neighbor]['weight'])

            neighbors_of_neighbor = graph.neighbors(neighbor)
            two_hop_neighbors.extend(list(neighbors_of_neighbor))

        one_hop_neighbors.extend(two_hop_neighbors)
        neighbors = set(one_hop_neighbors)
        neighbors.remove(query_edge)
        return neighbors, sub_graph

    def get_observation(self, placemen_client):
        graph, entity_dict = placemen_client.get_region_graph()
        centroids = entity_dict.geometry.centroid
        residential_regions = entity_dict[entity_dict['type'] == 4]
        service_regions = entity_dict[entity_dict['type'].isin(list(range(5, 13)))]
        feasible_regions = entity_dict[entity_dict['type'] == 1]

        # QA: Topology
        # calculate distance to service
        for i, res_region in residential_regions.iterrows():
            service_distance_dict = {}
            service_distance_dict_center = {}
            for j, service in service_regions.iterrows():
                if service['type'] not in service_distance_dict:
                    service_distance_dict[service['type']] = [nx.shortest_path_length(graph, i, j, weight='weight')]
                    service_distance_dict_center[service['type']] = [centroids[i].distance(centroids[j])]
                else:
                    service_distance_dict[service['type']].append(nx.shortest_path_length(graph, i, j, weight='weight'))
                    service_distance_dict_center[service['type']].append(centroids[i].distance(centroids[j]))

            # find min distance
            for service_id in service_distance_dict:
                if service_id not in service_distance_dict:
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}'] = float('inf')
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}_center'] = float('inf')
                else:
                    min_distance = min(service_distance_dict[service_id])
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}'] = min_distance
                    min_distance_center = min(service_distance_dict_center[service_id])
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}_center'] = min_distance_center

        # find the service farther than 1000m
        for i, res_region in residential_regions.iterrows():
            distant_services = []
            for s in range(5, 13):
                if f'distance_to_{self.service_types[s]}' in res_region and res_region[f'distance_to_{self.service_types[s]}'] > 1000:
                    distant_services.append(self.service_types[s])
            residential_regions.loc[i, 'distant_services'] = json.dumps(distant_services)

        # QA: distance/duration
        for i, res_region in residential_regions.iterrows():
            # find 1-hop neighbors
            neighbors = list(graph.neighbors(i))

            # calculate neighbors' distance
            neighbors_distance = {}
            for neighbor in neighbors:
                neighbors_distance[str(neighbor)] = centroids[i].distance(centroids[neighbor])

            residential_regions.loc[i, 'neighbors_distance'] = json.dumps(neighbors_distance)

        return graph, residential_regions, service_regions, feasible_regions


    def get_region_data_text(self, graph, residential_regions, service_regions, feasible_regions):
        # index regions
        region_idx_dict = {}
        all_regions = pd.concat([residential_regions, service_regions, feasible_regions])
        for i, region in all_regions.iterrows():
            service_type = region['type']
            mapped_type = self.service_types[service_type]  # 获取映射后的类型

            if mapped_type not in region_idx_dict:
                region_idx_dict[mapped_type] = 0
            else:
                region_idx_dict[mapped_type] += 1

            # 生成索引字符串
            idx = region_idx_dict[mapped_type]
            service_idx_str = f"{mapped_type} region {idx}"
            if i in residential_regions.index:
                residential_regions.at[i, "service_idx"] = service_idx_str
            elif i in service_regions.index:
                service_regions.at[i, "service_idx"] = service_idx_str
            else:
                feasible_regions.at[i, "service_idx"] = service_idx_str
            all_regions.at[i, "service_idx"] = service_idx_str

        # observation preparation
        residential_region_texts = []
        for i, res_region in residential_regions.iterrows():
            area = residential_regions.area[i]
            residential_region_texts.append(f"{residential_regions.service_idx[i]}:\n"
                                            f"- area: {round(area/1000000, 4)}km²")

        feasible_region_texts = []
        for i, feasible_region in feasible_regions.iterrows():
            area = feasible_regions.area[i]
            feasible_region_texts.append(f"{feasible_regions.service_idx[i]}:\n"
                                         f"- area: {round(area/1000000, 4)}km²")

        service_region_texts = []
        for i, service_region in service_regions.iterrows():
            area = service_regions.area[i]
            service_region_texts.append(f"{service_regions.service_idx[i]}:\n"
                                        f"- area: {round(area/1000000, 4)}km²")

        edges_str = [f"({all_regions.service_idx[u]}, {all_regions.service_idx[v]}, {round(d['weight'], 2)}m)" for u, v, d in graph.edges(data=True)]
        adj_info_text = f"Connectivity of regions:\n[{', '.join(edges_str)}]"
        region_text = "\n\n".join(feasible_region_texts + residential_region_texts + service_region_texts + [adj_info_text])

        return region_text, all_regions

    def get_env_feedback(self, new_residential_regions, all_regions, residential_regions):
        feedbacks = {}
        for j, residential_region in new_residential_regions.iterrows():
            for service in self.service_types[5:]:
                origin_service_distance = residential_regions[f'distance_to_{service}'][j]
                new_service_distance = new_residential_regions[f'distance_to_{service}'][j]
                if all_regions['service_idx'][j] not in feedbacks:
                    feedbacks[all_regions['service_idx'][j]] = {
                        service: (origin_service_distance - new_service_distance) / origin_service_distance
                    }
                else:
                    feedbacks[all_regions['service_idx'][j]][service] = (
                            (origin_service_distance - new_service_distance) / origin_service_distance)
        return feedbacks

    def select_action(self, env, logger, ir_sub_graph, residential_regions, service_regions, feasible_regions):
        region_text, all_regions = self.get_region_data_text(ir_sub_graph, residential_regions, service_regions, feasible_regions)
        target_edges = []

        land_use_type, _ = env._get_land_use_and_mask()
        land_use = self.service_types[land_use_type['type']]

        # Options
        distance_from_service_to_residential_regions = {}
        for j, res_region in residential_regions.iterrows():
            # data analysis
            analysis_texts = []
            for service_type in self.service_types[5:]:
                analysis_texts.append(f"- {service_type}: {round(res_region[f'distance_to_{service_type}'], 2)}m")

                if service_type not in distance_from_service_to_residential_regions:
                    distance_from_service_to_residential_regions[service_type] = {
                        j: res_region[f'distance_to_{service_type}']}
                else:
                    distance_from_service_to_residential_regions[service_type][j] = res_region[
                        f'distance_to_{service_type}']

        distance_from_target_service_to_residential_regions = sorted(distance_from_service_to_residential_regions[land_use].items(), key=lambda x: -x[1])
        target_residential_region = distance_from_target_service_to_residential_regions[0][0]

        # calculate distance to feasible regions
        gdf, whole_graph = env._plc._get_current_gdf_and_graph()
        # Get node dict
        node_graph2gdf_dict = {node: gdf.index[i] for i, node in enumerate(whole_graph.nodes)}
        node_gdf2node_dict = {gdf_index: node for node, gdf_index in node_graph2gdf_dict.items()}

        feasible_regions = gdf[gdf['type'] == 1]
        distance_to_feasible_regions = []
        for j, fea_region in feasible_regions.iterrows():
            try:
                distance_to_feasible_regions.append([j, nx.shortest_path_length(ir_sub_graph, target_residential_region, j, weight='weight')])
            except nx.NetworkXNoPath:
                distance_to_feasible_regions.append([j, np.inf])
        samples = sorted(distance_to_feasible_regions, key=lambda x: x[1])

        options = [all_regions.service_idx[sample[0]] for sample in samples]
        option_ids = [sample[0] for sample in samples]

        # Environment feedback
        _target_edges = []

        for region in option_ids:
            region_neighbors = whole_graph.neighbors(node_gdf2node_dict[region])
            for neighbor in region_neighbors:
                if gdf['type'][node_graph2gdf_dict[neighbor]] == 13:
                    # graph_region = node_gdf2graph_dict[region]
                    edge = np.array((region, node_graph2gdf_dict[neighbor]))
                    edge_reverse = np.array((node_graph2gdf_dict[neighbor], region))
                    all_edges = np.array(whole_graph.edges)
                    current_graph_nodes_id = gdf.index.to_numpy()
                    all_edges = current_graph_nodes_id[all_edges]
                    edge_idx = np.where((all_edges == edge).all(axis=1))[0]
                    if len(edge_idx) == 0:
                        edge_idx = np.where((all_edges == edge_reverse).all(axis=1))[0]
                    _target_edges.append(edge_idx)
                    break

        available_edges = []
        available_options = []
        available_option_ids = []
        for k, edge_idx in enumerate(_target_edges):
            simulation = copy.deepcopy(env)
            simulation_logger = copy.deepcopy(logger)
            try:
                _ = simulation.step(torch.tensor(edge_idx).long(), simulation_logger, land_use_type=land_use_type['type'])
            except Exception as e:
                continue
            target_edges.append(edge_idx)
            available_edges.append(edge_idx)
            available_options.append(options[k])
            available_option_ids.append(option_ids[k])

        # get the options
        option_text = "[" + ", ".join([f"{all_regions['service_idx'][available_option_ids[i]]}" for i in range(len(available_option_ids))]) + "]"
        data_text = (f"{region_text}\n\n"
                     f"Feasible regions to plan:\n"
                     f"{option_text}\n\n"
                     f"The next service to be built:\n"
                     f"{land_use} (with {round(land_use_type['area'] / 1000000, 4)}km²)")
        answer_option_form = "\"" + "/".join([all_regions['service_idx'][available_option_ids[i]] for i in range(len(available_option_ids))]) + "\""

        # LLM inference
        retry_count = 0
        target_edge = None
        decision_info = None
        while retry_count < 3:
            decision_info = self.llm_agent.hybrid_decision_making_pipeline([data_text], [answer_option_form])[0]
            target_region = decision_info["answer"]
            if target_region in available_options:
                target_edge = available_edges[available_options.index(target_region)]
                break
            else:
                retry_count += 1

        return data_text, target_edge, decision_info, answer_option_form

