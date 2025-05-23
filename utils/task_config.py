import sys
sys.path.append("../")

from UST_tasks.next_poi_prediction.main import main as human_mobility_prediction_main
from UST_tasks.poi_placement.main import main as poi_placement_main
from UST_tasks.route_planning.main import main as route_planning_main
from UST_tasks.congestion_prediction.main import main as congestion_prediction_main
from UST_tasks.socio_economic_prediction.main import main as urban_development_prediction_main
from UST_tasks.traffic_signal_control.main import main as traffic_signal_control_main
from UST_tasks.traffic_od_prediction.main import main as traffic_flow_prediction_main
from UST_tasks.road_planning.main import main as road_planning_main
from UST_tasks.urban_planning.main import main as urban_planning_main
from UST_tasks.question_answering.main import main as question_answering_main

TASK_CONFIG = {
    "human_mobility_prediction": {
        "function": human_mobility_prediction_main,
        "required_args": {
            "location": {"type": str, "description": "Location of the dataset."}
        }
    },
    "poi_placement": {
        "function": poi_placement_main,
        "required_args": {
            "location": {"type": str, "description": "Location name (e.g., Qiaonan)."}
        },
    },
    "congestion_prediction": {
        "function": congestion_prediction_main,
        "required_args": {
            "location": {"type": str, "description": "Location of the dataset."}
        }
    },
    "route_planning": {
        "function": route_planning_main,
        "required_args": {
            "location": {"type": str, "description": "Location of the simulation."}
        },
    },
    "urban_development_prediction": {
        "function": urban_development_prediction_main,
        "required_args": {
            "location": {"type": str, "description": "Location of the dataset."}
        }
    },
    "traffic_signal_control": {
        "function": traffic_signal_control_main,
        "required_args": {
            "dataset": {"type": str, "description": "Road network of the dataset."},
            "traffic_file": {"type": str, "description": "Traffic file of the dataset."}
        }
    },
    "traffic_flow_prediction": {
        "function": traffic_flow_prediction_main,
        "required_args": {
            "location": {"type": str, "description": "Location of the dataset."}
        }
    },
    "road_planning": {
        "function": road_planning_main,
        "required_args": {
            "slum_name": {"type": str, "description": "slum dataset name"},
        }
    },
    "urban_planning": {
        "function": urban_planning_main,
        "required_args": {
            "cfg": {"type": str, "description": "Dataset config file."}
        }
    },
    "question_answering": {
        "function": question_answering_main,
        "required_args": {
            "tasks": {"type": str, "description": "List of tasks to be executed."},
            "datasets": {"type": str, "description": "List of datasets to be used."},
        }
    }
}
