import numpy as np

def get_data_text(raw_data, graph):
    data_text = (f"Target Road:\n\n"
                 f"road {raw_data['road_id']}:\n"
                 f"- speed_limit: {raw_data['speed_limit']}\n"
                 f"- congestion_level_in_past_one_hour: {raw_data['recent']}\n"
                 f"- congestion_level_in_past_three_days: {str(raw_data['history'])}\n\n")

    neighbor_data = raw_data['neighbors']['1-hop'] + raw_data['neighbors']['2-hop']
    neighbor_data_text = "Nearby Roads:\n"
    for nei in neighbor_data:
        neighbor_data_text += (f"road: {nei['road_id']}\n"
                               f"- speed_limit: {nei['speed_limit']}\n"
                               f"- congestion_level_in_past_one_hour: {nei['recent']}\n\n")
    data_text += neighbor_data_text

    # adjacency info
    sub_graph = graph.subgraph([raw_data['road_id']] + [nei['road_id'] for nei in neighbor_data])
    edges_str = [f"({u}, {v}, {round(d['weight'], 2)}m)" for u, v, d in sub_graph.edges(data=True)]
    adj_info_text = (f"Connectivity of roads:\n"
                     "[" + ", ".join(edges_str) + "]")
    data_text += adj_info_text
    
    return data_text


def get_env_change_text(predictions, answers):
    env_change_texts = []
    for i in range(len(predictions)):
        higher_num = 0
        lower_num = 0
        correct_num = 0
        for j in range(len(predictions[i]['answer'])):
            if predictions[i]['answer'][j][0] == answers[i][j][0]:
                correct_num += 1
            elif predictions[i]['answer'][j][0] > answers[i][j][0]:
                higher_num += 1
            else:
                lower_num += 1
        env_change_texts.append(f"Your prediction: {predictions[i]['answer']}.\n"
                                f"Actual: {answers[i]}.\n"
                                f"Your prediction has {higher_num} value higher, {lower_num} value lower, and {correct_num} value equal to the actual traffic flow.")

    return env_change_texts


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero_indices = y_true != 0
    y_true = y_true[nonzero_indices]
    y_pred = y_pred[nonzero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true))

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    return correct_count / len(y_true)
