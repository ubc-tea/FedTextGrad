#!/usr/bin/env python3
"""
main.py

Federated TextGrad Experiment Entrance.

Usage:
    python main.py [options]
"""

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
import numpy as np
from comet_ml import Experiment, OfflineExperiment

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Federated TextGrad Experiment Runner.")
    parser.add_argument("--task", nargs='+', type=str, help="List of task names.")
    parser.add_argument("--evaluation_engine", type=str, default="gpt-4o", help="API for evaluation.")
    parser.add_argument("--test_engine", type=str, default="gpt-3.5-turbo-0125", help="API for testing.")
    parser.add_argument("--batch_size", type=int, default=3, help="Training batch size.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum training epochs.")
    parser.add_argument("--max_steps", type=int, default=3, help="Maximum training steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--do_not_run_larger_model", action="store_true", help="Skip running the larger model.")
    parser.add_argument("--aggregate_method", type=str, default="summarization", help="Context aggregation method [concat, summarization, concat_uid, sum_uid].")
    parser.add_argument("--homo_split_num", type=int, default=3, help="Number of clients in homogeneous setting.")
    parser.add_argument("--comet_mode", type=str, default="offline", choices=["offline", "online"], help="Comet ML logging mode.")
    parser.add_argument("--comet_project_name", type=str, default="fedtextgrad", help="Comet ML project name.")
    parser.add_argument("--comet_log_path", type=str, default="./logs/comet_results/", help="Comet ML log directory.")
    parser.add_argument("--proximal_update", action="store_true", help="Enable proximal update to prevent updating when no accuracy improvement.")
    parser.add_argument("--module", type=str, required=True, help="Module to run [train_debugging, train_centralized, train_homo_fed, train_multi_task, train_hetero_fed].")
    
    return parser.parse_args()

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

def load_and_run_module(module_name: str, module_path: str, args, experiment):
    """
    Load and execute a module dynamically.
    
    Args:
        module_name (str): Name of the module.
        module_path (str): Path to the module file.
        args (argparse.Namespace): Parsed arguments.
        experiment (Experiment or OfflineExperiment): Comet ML experiment object.
    """
    if not os.path.exists(module_path):
        print(f"Error: Module file '{module_path}' not found.")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module '{module_name}' from '{module_path}'.")
        sys.exit(1)
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error executing module '{module_name}': {e}")
        sys.exit(1)
    
    if hasattr(module, 'run_training'):
        module.run_training(args, experiment)
    else:
        print(f"Error: Module '{module_name}' does not contain 'run_training' function.")
        sys.exit(1)

def main():
    """
    Main function to set up the experiment and run the selected training module.
    """
    args = parse_arguments()
    os.environ["COMET_OFFLINE_DIRECTORY"] = args.comet_log_path
    os.makedirs(args.comet_log_path, exist_ok=True)
    
    # Initialize Comet ML experiment
    if args.comet_mode == "offline":
        experiment = OfflineExperiment(project_name=args.comet_project_name)
    else:
        comet_api_key = os.getenv("COMET_API_KEY")
        if not comet_api_key:
            print("Error: COMET_API_KEY not found in environment variables.")
            sys.exit(1)
        experiment = Experiment(api_key=comet_api_key, project_name=args.comet_project_name)
    
    experiment.log_parameters(vars(args))
    set_seed(args.seed)
    
    # Locate and execute the specified module
    module_file = f"{args.module}.py" if not args.module.endswith(".py") else args.module
    module_path = os.path.join(os.getcwd(), module_file)
    
    load_and_run_module(args.module, module_path, args, experiment)
    
    experiment.end()

if __name__ == '__main__':
    main()
