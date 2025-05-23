import copy
import os
import sys
import argparse
import numpy as np
from collections import defaultdict
import wandb

from utils.read_utils import load_json, dump_json
from utils.language_model import LLM


def ensure_dir_exists(directory):
    """Ensure the directory exists; create it if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def sample_data(data, max_samples=3000):
    """Sample data to balance different types of spatial-temporal questions."""
    if 'spatial_temporal_type' not in data[0] or data[0]['spatial_temporal_type'] == 'hybrid':
        return data[:max_samples]  # If no specific type, simply take the first max_samples

    spatial_temporal_question_num = defaultdict(int)
    sampled_data = []

    for sample in data:
        st_type = sample['spatial_temporal_relation']
        if spatial_temporal_question_num[st_type] < 1000:
            spatial_temporal_question_num[st_type] += 1
            sampled_data.append(sample)

    return sampled_data


def evaluate_task(llm, task, datasets, model_name):
    """Evaluate a single task and return its accuracy."""
    print(f"========================== Task: {task} ==========================")

    eval_results = []
    base_log_path = f"./UST_tasks/question_answering/logs/{task}"
    ensure_dir_exists(base_log_path)

    for dataset in datasets:
        wandb.init(
            project="USTBench-Question-Answering",
            group=f"{task}-{dataset}-{llm.llm_name}",
            name="Examination"
        )

        # Load dataset
        data_path = f"./UST_tasks/question_answering/Data/{task}/{dataset}_QA.json"
        data = load_json(data_path)

        # sampled_data = sample_data(data)
        print(f"{len(data)} questions sampled.")

        # Model evaluation
        responses, acc, st_results = llm.evaluate(data, task=dataset)
        eval_results.append(acc)

        # Print evaluation results
        if st_results:
            st_results_ = {}
            for st_type, result in st_results.items():
                print(f"Dataset: {dataset}, ST Type: {st_type}, Acc: {result['accuracy']}")
                st_results_[f"{st_type}_accuracy"] = result
            wandb.log(st_results)
        else:
            print(f"Dataset: {dataset}, Acc: {acc}")
            wandb.log({"accuracy": acc})

        # Save evaluation results
        result_file = f"{base_log_path}/{model_name}_{dataset}_QA.json"
        dump_json(responses, result_file)
        wandb.finish()

    task_acc = np.mean(eval_results)
    print(f"Task overall Acc: {task_acc}")

    return task_acc


def main(llm_path_or_name, batch_size, use_reflection, tasks, datasets):
    """Main function to initialize the LLM and run evaluations."""
    tasks = tasks.split(", ")
    datasets = datasets.split(", ")
    llm = LLM(llm_path_or_name, batch_size=batch_size)
    model_name = os.path.basename(llm_path_or_name)

    print(f"========================== Running Evaluation on Model: {model_name} ==========================")

    # Evaluate all tasks and compute overall accuracy
    all_results = [evaluate_task(llm, task, datasets, model_name) for task in tasks]

    final_acc = np.mean(all_results)
    print(f"========================== Final Overall ==========================")
    print(f"Final overall Acc: {final_acc}")
    print(f"========================== Final Overall ==========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Evaluation")
    parser.add_argument("--llm-path-or-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Path to the language model or its name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the language model.")
    parser.add_argument("--tasks", type=str, default="human_mobility_prediction", help="List of tasks to evaluate, e.g., poi_placement")
    parser.add_argument("--datasets", type=str, default="st_understanding", help="List of datasets for each task, e.g., analysis decision_making")

    args = parser.parse_args()
    main(args.llm_path_or_name, args.batch_size, args.tasks, args.datasets)
