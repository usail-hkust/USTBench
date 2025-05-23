import math
import networkx as nx
import numpy as np
from geopy.distance import geodesic

def format_recent(recent):
    """将最近一次的历史记录格式化为多行字符串"""
    recent_line = ("- trajectory: {trajectory}\n"
                    "- adjacency: {adjacency}\n")
    trajectory_text = "[" + ', '.join([str((f'{visit[3]} {visit[2]}', visit[1], visit[0])) for visit in recent]) + "]"
    trajectory_text = trajectory_text.replace("\'", "")
    graph = nx.Graph()
    for i, visit in enumerate(recent):
        # (poi_1, poi_2, distance)
        if i > 0 and i < len(recent) - 1:
            graph.add_edge(f'{visit[3]} {visit[2]}', f'{recent[i - 1][3]} {recent[i - 1][2]}', weight=visit[4][i-1])
            graph.add_edge(f'{visit[3]} {visit[2]}', f'{recent[i + 1][3]} {recent[i + 1][2]}', weight=visit[4][i + 1])
            if i > 1:
                graph.add_edge(f'{visit[3]} {visit[2]}', f'{recent[i - 2][3]} {recent[i - 2][2]}', weight=visit[4][i-2])
            if i < len(recent) - 2:
                graph.add_edge(f'{visit[3]} {visit[2]}', f'{recent[i + 2][3]} {recent[i + 2][2]}', weight=visit[4][i + 2])
    adjacency_text = [f"({u}, {v}, {round(d['weight'], 2)}m)" for u, v, d in graph.edges(data=True)]
    adjacency_text = "[" + ", ".join(adjacency_text) + "]" if adjacency_text else "N/A"
    recent_line = recent_line.format(
        trajectory=trajectory_text,
        adjacency=adjacency_text
    )
    return recent_line

def format_history(history):
    """将30天的历史记录格式化为多行字符串"""
    history_lines = []
    for day, record in enumerate(history, 1):
        history_line = (f"Day {day}:\n"
                        "- trajectory: {trajectory}\n"
                        "- adjacency: {adjacency}\n")
        trajectory_text = "[" + ', '.join([str((f'{visit[3]} {visit[2]}', visit[1], visit[0])) for visit in record]) + "]"
        trajectory_text = trajectory_text.replace("\'", "")
        graph = nx.Graph()
        for i, visit in enumerate(record):
            # (poi_1, poi_2, distance)
            if i > 0 and i < len(record) - 1:
                graph.add_edge(f'{visit[3]} {visit[2]}', f'{record[i - 1][3]} {record[i - 1][2]}', weight=float(visit[4][i-1][:-1]))
                graph.add_edge(f'{visit[3]} {visit[2]}', f'{record[i + 1][3]} {record[i + 1][2]}', weight=float(visit[4][i + 1][:-1]))
                if i > 1:
                    graph.add_edge(f'{visit[3]} {visit[2]}', f'{record[i - 2][3]} {record[i - 2][2]}', weight=float(visit[4][i-2][:-1]))
                if i < len(record) - 2:
                    graph.add_edge(f'{visit[3]} {visit[2]}', f'{record[i + 2][3]} {record[i + 2][2]}', weight=float(visit[4][i + 2][:-1]))
        adjacency_text = [f"({u}, {v}, {round(d['weight'], 2)}m)" for u, v, d in graph.edges(data=True)]
        history_line = history_line.format(
            trajectory=trajectory_text,
            adjacency="[" + ", ".join(adjacency_text) + "]" if adjacency_text else "N/A"
        )
        history_lines.append(history_line)
    return "\n".join(history_lines)

def get_data_text(raw_data, candidates, candidate_dict):
    data_text = ("History:\n\n"
                 "<history_data>\n\n"
                 "Recent:\n\n"
                 "<recent_data>\n\n"
                 "Candidates:\n\n"
                 "<candidates>")

    history = raw_data["history"]
    recent = raw_data["recent"]

    # crate history text
    histories = format_history(history)
    data_text = data_text.replace("<history_data>", histories)

    # crate recent text
    recent_poi_trajectory = [poi[2] for poi in recent]
    recent_text = format_recent(recent)
    data_text = data_text.replace("<recent_data>", f"- trajectory: {recent_text}")

    # crate candidates text
    candidate_text = []
    for i, candidate in enumerate(candidates):
        distance_to_recent_pois = []
        for poi in recent_poi_trajectory:
            recent_poi_coord = (candidate_dict[str(poi)]['latitude'], candidate_dict[str(poi)]['longitude'])
            candidate_coord = (candidate_dict[str(candidate)]['latitude'], candidate_dict[str(candidate)]['longitude'])
            distance = geodesic(recent_poi_coord, candidate_coord).kilometers
            distance_to_recent_pois.append(f"({candidate_dict[str(candidate)]['category_name']} {candidate}, "
                                           f"{candidate_dict[str(poi)]['category_name']} {poi}, "
                                           f"{round(distance * 1000)}m)")

        candidate_text.append(f"{candidate_dict[str(candidate)]['category_name']} {candidate}:\n"
                              f"- adjacency: [{', '.join(distance_to_recent_pois)}]")
    candidate_text = "\n\n".join(candidate_text)
    data_text = data_text.replace("<candidates>", candidate_text)
    
    return data_text

def get_env_change_text(predictions, answers):
    env_change_texts = []
    for i in range(len(predictions)):
        precision = calculate_metric([predictions[i]['answer']], [answers[i][2]], 'p@k')
        mrr = calculate_metric([predictions[i]['answer']], [answers[i][2]], 'mrr@k')
        ndcg = calculate_metric([predictions[i]['answer']], [answers[i][2]], 'ndcg@k')

        env_change_texts.append(f"Your prediction: {predictions[i]['answer']}.\n"
                                f"Actual: {answers[i][3]} {answers[i][2]}.\n"
                                f"Your prediction's precision@10: {np.mean(precision)}.\n"
                                f"Your prediction's mrr@10: {np.mean(mrr)}.\n"
                                f"Your prediction's ndcg@10: {np.mean(ndcg)}.")

    return env_change_texts

def calculate_metric(predictions, actuals, metric, k=10):
    results = []
    for pred, actual in zip(predictions, actuals):
        if actual in pred[:k]:
            index = pred.index(actual)
            if metric == 'p@k':
                results.append(1)
            elif metric == 'mrr@k':
                results.append(1 / (index + 1))
            elif metric == 'ndcg@k':
                results.append(1 / math.log2(index + 2))
        else:
            results.append(0)
    return results


