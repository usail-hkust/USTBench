import argparse
import sys
import os
import torch
import numpy as np
import random
from utils.task_config import TASK_CONFIG

# Enhanced validate_args to include type checking and better error messages
def validate_args(task_name, provided_args):
    """Validate that all required arguments for the task are provided and have correct types."""
    required_args = TASK_CONFIG[task_name]["required_args"]
    missing_args = [arg for arg in required_args if arg not in provided_args]

    if missing_args:
        print(f"\nError: Missing required arguments for task '{task_name}': {missing_args}")
        print("Required parameters with types and descriptions:")
        for arg, meta in required_args.items():
            print(f"  --{arg} ({meta['type'].__name__}): {meta['description']}")
        sys.exit(1)

    for arg, value in provided_args.items():
        if arg in required_args:
            expected_type = required_args[arg]["type"]
            try:
                provided_args[arg] = expected_type(value)  # Convert to expected type
            except ValueError:
                print(f"\nError: Argument '{arg}' should be of type {expected_type.__name__}.")
                sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run a specific task using a specified LLM.")
    parser.add_argument("--task", type=str, default="question_answering", help="The task to run (e.g., poi_placement).")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for the simulation.")
    parser.add_argument("--llm_path_or_name", type=str, default="../llm_models/Qwen/Qwen2.5-7B-Instruct", help="The path or name of the LLM to use.")
    parser.add_argument("--use_reflection", type=lambda x: x.lower() == "true", default=True, help="Whether to use reflection or not.")
    parser.add_argument('--seed', type=int, default=1, help="Seed for reproducibility.")
    args, unknown_args = parser.parse_known_args()

    # Parse additional arguments provided via the command line
    additional_args = {arg.lstrip("--"): value for arg, value in zip(unknown_args[0::2], unknown_args[1::2])}

    # Validate task
    if args.task not in TASK_CONFIG:
        print(f"Error: Task '{args.task}' is not recognized. Available tasks: {list(TASK_CONFIG.keys())}")
        sys.exit(1)

    task_config = TASK_CONFIG[args.task]
    validate_args(args.task, additional_args)

    # Set environment variables and seed for reproducibility
    os.environ['PYTHONASHSEED'] = f'{args.seed}'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Dynamically load the task module and function
    task_function = task_config["function"]

    # Pass arguments to the task function
    print(f"Running task '{args.task}' with LLM '{args.llm_path_or_name}' and parameters: {additional_args}")
    print(additional_args)
    task_function(llm_path_or_name=args.llm_path_or_name, batch_size=args.batch_size, use_reflection=args.use_reflection, **additional_args)

if __name__ == "__main__":
    main()

