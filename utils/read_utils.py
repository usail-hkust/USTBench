import os
import re
import yaml
import gzip
import json
import csv
import numpy as np

list_json_pattern = r'\[\s*(?:"[^"]*",?\s*)*\]'
list_dict_json_pattern = r'\[(?:\s*\{(?:[^{}]|"[^"]*"|\d+|true|false|null|:|,|\s)*\}\s*,?\s*)*\]'
dict_json_pattern = r'\{(?:(?:"(?:\\.|[^"\\])*")|(?R)|[^{}"])*\}'
markdown_code_pattern = r'```(?:\w+)?\n([\s\S]*?)\n```'

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")

def read_lines(path_to_file):
    data = []
    try:
        with open(path_to_file, 'r') as f:
            for line in f:
                tmp = [float(x) for x in line.strip().split()]
                data.append(tmp)
    except Exception as e:
        raise e

    return data

def dump_ndarray(data, path_to_file):
    try:
        with open(path_to_file, 'wb') as f:
            np.save(f, data)
    except Exception as e:
        raise e

def load_ndarray(path_to_file):
    try:
        with open(path_to_file, 'rb') as f:
            data = np.load(f)
    except Exception as e:
        raise e

    return data

def dump_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for each in data:
                f.write(json.dumps(each) + '\n')
    except Exception as e:
        raise e

def load_ndjson(file, return_type='array'):
    if return_type == 'array':
        return load_ndjson_to_array(file)
    elif return_type == 'dict':
        return load_ndjson_to_dict(file)
    else:
        raise RuntimeError('Unknown return_type: %s' % return_type)

def load_ndjson_to_array(file):
    data = []
    try:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def load_ndjson_to_dict(file):
    data = {}
    try:
        with open(file, 'r') as f:
            for line in f:
                data.update(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def dump_json(data, file, indent=4):
    try:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_dict_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for k, v in data.items():
                line = json.dumps([k, v]) + '\n'
                f.write(line)
    except Exception as e:
        raise e

def load_gzip_json(file):
    try:
        with gzip.open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def get_all_files(dir, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(dir) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(dir, filename) for filename in os.listdir(dir) if os.path.isfile(os.path.join(dir, filename)) and not filename.startswith('.')]

def read_csv_to_list(filename, spliter='\t'):
    data = []
    with open(filename, 'r', newline='', encoding='ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def construct_self_reflection_data(data, prediction_or_decision, task):
    new_question = re.sub(r'## Note.*', '', data['question'], flags=re.DOTALL)
    new_question = (f"{new_question}"
                    f"## {prediction_or_decision}\n\n"
                    f"{data['decision']}\n\n"
                    f"## Reason\n\n"
                    f"{data['summary']}\n\n"
                    f"## Environment Changes\n\n"
                    f"<env_changes>\n\n"
                    f"## Note\n\n"
                    f"- Based on environment changes caused by the prediction, evaluate the correctness of the prediction and identify any weaknesses.\n"
                    f"- Verify if the {prediction_or_decision} and its reasoning are accurate.\n"
                    f"- If the {prediction_or_decision} is incorrect or inaccurate, propose a better alternative with a clear rationale.\n"
                    f"- Let's solve this step by step. Finally, summarize your analysis, and provide your answer in JSON format, like:" +
                    "\n\n```JSON\n{\n\t\"summary\": \"YOUR_SUMMARY\",\n\t\"answer\": \"A/B/C/D\"\n}\n```")

    if prediction_or_decision == 'Prediction':
        new_question = new_question.replace('<env_changes>', 'The prediction is wrong.')
    else:
        if task == 'traffic_signal_control':
            option2signal = {'A': "ETWT", 'B': "ELWL", 'C': "NTST", 'D': "NLSL"}

            # Previous
            pre_conditions = {"queue": [], "wait_time": [], "occupancy": []}
            for inter in data['current_condition']:
                for lane in data['current_condition'][inter]:
                    pre_conditions['queue'].append(data['current_condition'][inter][lane]['queue_num'])
                    pre_conditions['wait_time'].append(data['current_condition'][inter][lane]['avg_wait_time'])
                    pre_conditions['occupancy'].append(data['current_condition'][inter][lane]['occupancy'])
            pre_conditions['queue'] = np.mean(pre_conditions['queue']) if pre_conditions['queue'] else 0.0
            pre_conditions['wait_time'] = np.mean(pre_conditions['wait_time']) if pre_conditions['wait_time'] else 0.0
            pre_conditions['occupancy'] = np.mean(pre_conditions['occupancy']) if pre_conditions['occupancy'] else 0.0

            # New
            new_conditions = {"queue": [], "wait_time": [], "occupancy": []}
            for inter in data['future_condition']["ETWT"]:
                for lane in data['future_condition'][option2signal[data['decision']]][inter]:
                    new_conditions['queue'].append(data['future_condition'][option2signal[data['decision']]][inter][lane]['queue_num'])
                    new_conditions['wait_time'].append(data['future_condition'][option2signal[data['decision']]][inter][lane]['avg_wait_time'])
                    new_conditions['occupancy'].append(data['future_condition'][option2signal[data['decision']]][inter][lane]['occupancy'])
            new_conditions['queue'] = np.mean(new_conditions['queue']) if new_conditions['queue'] else 0.0
            new_conditions['wait_time'] = np.mean(new_conditions['wait_time']) if new_conditions['wait_time'] else 0.0
            new_conditions['occupancy'] = np.mean(new_conditions['occupancy']) if new_conditions['occupancy'] else 0.0

            new_question = new_question.replace(
                '<env_changes>',
                       f"Queue {'increased' if new_conditions['queue'] > pre_conditions['queue'] else 'decreased'} by {round(np.abs(new_conditions['queue'] - pre_conditions['queue']) / pre_conditions['queue'] * 100, 2)}%\n"
                       f"Average wait time {'increased' if new_conditions['wait_time'] > pre_conditions['wait_time'] else 'decreased'} by {round(np.abs(new_conditions['wait_time'] - pre_conditions['wait_time']) / pre_conditions['wait_time'] * 100, 2)}%\n"
                       f"Lane occupancy {'increased' if new_conditions['occupancy'] > pre_conditions['occupancy'] else 'decreased'} by {round(np.abs(new_conditions['occupancy'] - pre_conditions['occupancy']) / pre_conditions['occupancy'] * 100, 2)}%"
            )

        elif task == 'poi_placement':
            option2poi = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            new_question = new_question.replace(
                        '<env_changes>',
                        f"Station coverage {'increased' if data['feedbacks'][option2poi[data['decision']]]['cov_gain'] > 0 else 'decreased'} by {round(np.abs(data['feedbacks'][option2poi[data['decision']]]['cov_gain']) * 100, 2)}%\n"
                        f"Charging time {'increased' if data['feedbacks'][option2poi[data['decision']]]['chg_gain'] < 0 else 'decreased'} by {round(np.abs(data['feedbacks'][option2poi[data['decision']]]['chg_gain']) * 100, 2)}%\n"
                        f"Travel time {'increased' if data['feedbacks'][option2poi[data['decision']]]['travel_gain'] < 0 else 'decreased'} by {round(np.abs(data['feedbacks'][option2poi[data['decision']]]['travel_gain']) * 100, 2)}%\n"
                        f"Waiting time {'increased' if data['feedbacks'][option2poi[data['decision']]]['wait_gain'] < 0 else 'decreased'} by {round(np.abs(data['feedbacks'][option2poi[data['decision']]]['wait_gain']) * 100, 2)}%"
            )

        elif task == 'route_planning':
            new_question = new_question.replace(
                        '<env_changes>',
                        f"Average congestion levels of roads can be selected next:\noption A: {round(data['feedbacks'][0]['average_congestion_level'], 2)}, option B: {round(data['feedbacks'][1]['average_congestion_level'], 2)}, option C: {round(data['feedbacks'][2]['average_congestion_level'], 2)}, option D: {round(data['feedbacks'][3]['average_congestion_level'], 2)}\n\n"
                        f"Average congestion levels of other nearby roads:\noption A: {round(data['feedbacks'][0]['average_congestion_level_of_nearby_roads'], 2)}, option B: {round(data['feedbacks'][1]['average_congestion_level_of_nearby_roads'], 2)}, option C: {round(data['feedbacks'][2]['average_congestion_level_of_nearby_roads'], 2)}, option D: {round(data['feedbacks'][3]['average_congestion_level_of_nearby_roads'], 2)}\n\n"
                        f"Average route lengths of roads can be selected next:\noption A: {round(data['feedbacks'][0]['average_route_length'], 2)}, option B: {round(data['feedbacks'][1]['average_route_length'], 2)}, option C: {round(data['feedbacks'][2]['average_route_length'], 2)}, option D: {round(data['feedbacks'][3]['average_route_length'], 2)}"
            )

        elif task == 'road_planning':
            option2road = {}
            for i, road in enumerate(list(data['environment_feedback'].keys())):
                option2road[chr(ord('A') + i)] = road
            if data['stage'] == 'stage1':
                new_question = new_question.replace(
                        '<env_changes>',
                        f"Newly connected regions: {round(data['environment_feedback'][option2road[data['decision']]]['newly_connected_regions'], 2)}\n"
                        f"Maximum number of regions can be connected next: {round(data['environment_feedback'][option2road[data['decision']]]['potential_highest_connected_regions_via_new_roads'], 2)}"
                )
            elif data['stage'] == 'stage2':
                new_question = new_question.replace(
                        '<env_changes>',
                        f"Distance deduction among regions: {round(data['environment_feedback'][option2road[data['decision']]]['distance_reduction_among_regions'], 2)}\n"
                        f"Maximum distance deduction via building new roads: {round(data['environment_feedback'][option2road[data['decision']]]['potential_highest_distance_reduction_via_new_roads'], 2)}"
                )
        elif task == 'urban_planning':
            option2region = {"A": 0, "B": 1, "C": 2, "D": 3}
            service_gain = {'business': [], 'green': [], 'hospital': [], 'office': [], 'recreation': [], 'school': []}
            option_feedbacks = data['feedbacks'][option2region[data['decision']]]
            for res_region in option_feedbacks:
                for service in service_gain:
                    service_gain[service].append(option_feedbacks[res_region][service])

            service_gain = {k: np.mean(v) for k, v in service_gain.items()}
            new_question = new_question.replace(
                        '<env_changes>',
                        f"Average {'increase' if service_gain['business'] > 0 else 'decrease'} in service accessibility of business: {round(service_gain['business'] * 100, 2)}%\n"
                        f"Average {'increase' if service_gain['green'] > 0 else 'decrease'} in service accessibility of green: {round(service_gain['green'] * 100, 2)}%\n"
                        f"Average {'increase' if service_gain['hospital'] > 0 else 'decrease'} in service accessibility of hospital: {round(service_gain['hospital'] * 100, 2)}%\n"
                        f"Average {'increase' if service_gain['office'] > 0 else 'decrease'} in service accessibility of office: {round(service_gain['office'] * 100, 2)}%\n"
                        f"Average {'increase' if service_gain['recreation'] > 0 else 'decrease'} in service accessibility of recreation: {round(service_gain['recreation'] * 100, 2)}%\n"
                        f"Average {'increase' if service_gain['school'] > 0 else 'decrease'} in service accessibility of school: {round(service_gain['school'] * 100, 2)}%"
            )

    return {'question': new_question, 'answer': data['answer']}




