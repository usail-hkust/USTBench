import copy
import sys

import numpy as np

sys.path.append("../")

import networkx as nx
import traci
import xml.etree.ElementTree as ET

# 获取边的车道数
def get_edge_lane_info(edge_id, lane_id):
    lane_num = traci.edge.getLaneNumber(edge_id)  # 获取边上的所有车道ID
    vehicle_num = traci.edge.getLastStepVehicleNumber(edge_id)
    vehicle_speed = traci.edge.getLastStepMeanSpeed(edge_id)
    vehicle_length = traci.edge.getLastStepLength(edge_id)
    speed_limit = traci.lane.getMaxSpeed(lane_id)

    # route = traci.simulation.findRoute(fromEdge=edge_id, toEdge=edge_id)
    road_len = traci.lane.getLength(lane_id)

    return lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len, speed_limit


def get_edge_info(edge_id):
    lane_num = traci.edge.getLaneNumber(edge_id)  # 获取边上的所有车道ID
    vehicle_num = traci.edge.getLastStepVehicleNumber(edge_id)
    vehicle_speed = traci.edge.getLastStepMeanSpeed(edge_id)
    vehicle_length = traci.edge.getLastStepLength(edge_id)

    route = traci.simulation.findRoute(fromEdge=edge_id, toEdge=edge_id)
    road_len = route.length

    return lane_num, vehicle_num, vehicle_speed, vehicle_length, road_len


def get_congestion_level(congestion_rate):
    if 0.0 <= congestion_rate <= 0.60:
        return 0
    elif 0.60 < congestion_rate <= 0.70:
        return 1
    elif 0.70 < congestion_rate <= 0.80:
        return 2
    elif 0.80 < congestion_rate <= 0.90:
        return 3
    elif 0.90 < congestion_rate <= 1.0:
        return 4
    else:
        return 5


def parse_rou_file(file_path):
    trips = []
    tree = ET.parse(file_path)
    root = tree.getroot()

    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')  # 获取 vehicle 的 ID
        route_distribution = vehicle.find('routeDistribution')
        if route_distribution is not None:
            route = route_distribution.find('route')
            if route is not None:
                edges = route.get('edges')  # 获取 edges 属性
                if edges:
                    edge_list = edges.split()
                    start_edge = edge_list[0]  # 起点 edge_id
                    end_edge = edge_list[-1]  # 终点 edge_id
                    trips.append((vehicle_id, start_edge, end_edge))
    return trips


def get_1_2_hop_neighbors(graph, query_edge):
    one_hop_neighbors = set(graph.neighbors(query_edge))

    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        two_hop_neighbors.update(graph.neighbors(neighbor))

    return one_hop_neighbors, two_hop_neighbors

def get_subgraph(graph, query_edges):
    sub_graph = graph.subgraph(query_edges)

    return sub_graph

def get_autonomous_vehicle_observation(vehicle_ids, autonomous_vehicles, road_info, road_network):
    update_vehicle_info = [[], [], [], [], [], []]
    for veh_id in vehicle_ids:
        if veh_id in autonomous_vehicles:
            # get observation
            current_edge = traci.vehicle.getRoadID(veh_id)
            end_edge = traci.vehicle.getRoute(veh_id)[-1]
            veh_trip = (veh_id, current_edge, end_edge)
            road_candidates, data_text, answer_option_form = get_observation_text(veh_trip, road_network, road_info)

            if len(road_candidates) < 2:
                continue

            update_vehicle_info[0].append(veh_id)
            update_vehicle_info[1].append(data_text)
            update_vehicle_info[2].append(answer_option_form)
            update_vehicle_info[3].append(current_edge)
            update_vehicle_info[4].append(end_edge)
            update_vehicle_info[5].append(road_candidates)

    return update_vehicle_info


def get_observation(trip, road_network, edge_dict):
    _, start_edge, end_edge = trip

    edge_candidates, _ = get_1_2_hop_neighbors(road_network, start_edge)

    edge_candidates = list(edge_candidates)[:10]
    edge_candidate_info = {}
    for edge_can in edge_candidates:
        congestion_level = edge_dict[edge_can]["congestion_level"]

        try:
            shortest_route_len = nx.dijkstra_path_length(road_network, source=edge_can, target=end_edge)
        except Exception as e:
            continue

        one_hop_neighbors, two_hop_neighbors = get_1_2_hop_neighbors(road_network, edge_can)
        nei_dict = {}
        for nei in one_hop_neighbors:
            if len(nei_dict) >= 10:
                break
            nei_congestion_level = edge_dict[nei]['congestion_level']
            nei_dict[nei] = {
                "hop": 1,
                "congestion_level": nei_congestion_level
            }
        for nei in two_hop_neighbors:
            if len(nei_dict) >= 10:
                break
            nei_congestion_level = edge_dict[nei]['congestion_level']
            nei_dict[nei] = {
                "hop": 2,
                "congestion_level": nei_congestion_level
            }

        edge_candidate_info[edge_can] = {
            "congestion_level": congestion_level,
            "shortest_route_len": shortest_route_len,
            "neighbors": nei_dict
        }

    return edge_candidate_info

def get_observation_text(trip, road_network, edge_dict):
    # get observation
    road_candidates = get_observation(trip, road_network, edge_dict)

    candidate_roads_texts = []
    answer_option_form = "\"" + "/".join([edge_can for edge_can in road_candidates]) + "\""
    for edge_can in road_candidates:
        can_road_text = (f"road: {edge_can}\n"
                         f"- congestion_level: {str(road_candidates[edge_can]['congestion_level'])}\n"
                         f"- shortest_route_length: {str(round(road_candidates[edge_can]['shortest_route_len'], 2))}m\n"
                         f"- road_length: {str(round(edge_dict[edge_can]['road_len'], 2))}m")
        candidate_roads_texts.append(can_road_text)

    candidate_roads_text = ("Candidate roads:\n\n" +
                            "\n\n".join([can_road_text for can_road_text in candidate_roads_texts]))

    # Get subgraph adj & nearby edges
    nearby_roads = []
    for edge_can in road_candidates:
        nearby_roads += list([nei for nei in road_candidates[edge_can]["neighbors"]])
    nearby_roads = list(set(nearby_roads))
    subgraph_roads = set([edge_can for edge_can in road_candidates] + nearby_roads)
    subgraph = get_subgraph(road_network, subgraph_roads)

    nearby_road_texts = []
    for nei_road in nearby_roads:
        road_text = (f"road {nei_road}:\n"
                     f"- congestion_level: {edge_dict[nei_road]['congestion_level']}\n"
                     f"- road_length: {edge_dict[nei_road]['road_len']}m")
        nearby_road_texts.append(road_text)
    nearby_roads_text = ("Nearby roads:\n\n" +
                         "\n\n".join(nearby_road_texts))

    # adjacency info
    edges_str = [f"({u}, {v}, {round(d['weight'], 2)}m)" for u, v, d in subgraph.edges(data=True)]
    adj_info_text = (f"Connectivity:\n"
                     "[" + ", ".join(edges_str) + "]")

    obs_text = candidate_roads_text + "\n\n" + nearby_roads_text + "\n\n" + adj_info_text

    return road_candidates, obs_text, answer_option_form


def update_route(current_edge, next_edge, end_edge):
    current_route = traci.simulation.findRoute(fromEdge=current_edge, toEdge=end_edge)
    try:
        candidate_route = traci.simulation.findRoute(fromEdge=next_edge, toEdge=end_edge)
        new_route = [current_edge] + list(candidate_route.edges)
        return new_route

    except Exception as e:
        print(f"Route Switch Failed: {e}")
        return current_route.edges


def get_env_change_text(current_road, candidate_roads):
    average_congestion_level = np.mean([candidate_roads[road]["congestion_level"] for road in candidate_roads])
    average_route_length = np.mean([candidate_roads[road]["shortest_route_len"] for road in candidate_roads])

    nei_1_congestion = []
    nei_2_congestion = []
    for road in candidate_roads:
        for nei in candidate_roads[road]["neighbors"]:
            if candidate_roads[road]["neighbors"][nei]["hop"] == 1:
                nei_1_congestion.append(candidate_roads[road]["neighbors"][nei]["congestion_level"])
            else:
                nei_2_congestion.append(candidate_roads[road]["neighbors"][nei]["congestion_level"])

    average_nei_1_congestion = np.mean(nei_1_congestion)
    average_nei_2_congestion = np.mean(nei_2_congestion)

    env_change_text = (f"Candidate roads after passing road {current_road}:\n"
                       f"- average_congestion_level: {str(round(average_congestion_level))}\n"
                       f"- average_route_length: {str(round(average_route_length, 2))}m\n"
                       f"- average_1_hop_connected_road_congestion_level: {str(round(average_nei_1_congestion))}\n"
                       f"- average_2_hop_connected_road_congestion_level: {str(round(average_nei_2_congestion))}")

    return env_change_text
