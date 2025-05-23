import os
import yaml
import gzip
import json
import csv
import numpy as np

list_json_pattern = r'\[\s*(?:"[^"]*",?\s*)*\]'
list_dict_json_pattern = r'\[(?:\s*\{(?:[^{}]|"[^"]*"|\d+|true|false|null|:|,|\s)*\}\s*,?\s*)*\]'
dict_json_pattern = r'\{(?:[^{}]|"[^"]*"|\d+|true|false|null|:|,|\s)*\}'

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

def calculate_centroid(box):
    """计算 bounding box 的中心点"""
    x_min, y_min, x_max, y_max = box
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2
    return (centroid_y, centroid_x)  # 返回 (纬度, 经度) 格式
