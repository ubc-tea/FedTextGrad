import numpy as np
import sys
import os
import re
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
from copy import deepcopy
from string import Template
from torch.utils.data import random_split
from utils.prompt_template import SUMMARIZATION_TEMPLATE, UID_TEMPLATE, FORMATTING_INSTRUCTION
from eval import eval_dataset
from utils.prompt_complexity import calculate_text_complexity
from pprint import pprint

def run_training(args, experiment):
    """
    Run federated training with multiple clients using TextGrad.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        experiment (Experiment): Comet ML experiment object.
    """
    llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load datasets
    train_set, val_set, test_set, eval_fn = load_task(args.task[0], evaluation_api=llm_api_eval)
    print(f"Whole Train/Val/Test Set Lengths: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    experiment.log_parameters({"train_length": len(train_set), "val_length": len(val_set), "test_length": len(test_set)})
    
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(STARTING_SYSTEM_PROMPT)
    experiment.log_parameter("task_description", STARTING_SYSTEM_PROMPT)
    
    # Test initial model performance
    system_prompt_eval = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=True, role_description="system prompt")
    model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt_eval)

    if not args.do_not_run_larger_model:
        reference = np.mean(eval_dataset(test_set, eval_fn, model_evaluation))
        experiment.log_parameter("0shot_eval_engine_test_acc", reference)

    # Federated setup
    split_num = args.homo_split_num
    optimizer_list, model_list, system_prompt_list = [], [], []

    for _ in range(split_num):
        system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=True, role_description="structured system prompt")
        model = tg.BlackboxLLM(llm_api_test, system_prompt)
        optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])
        system_prompt_list.append(system_prompt)
        model_list.append(model)
        optimizer_list.append(optimizer)
    
    # Split dataset for clients
    client_train_sets = random_split(train_set, [len(train_set) // split_num] * (split_num - 1) + [len(train_set) - (split_num - 1) * (len(train_set) // split_num)])
    client_val_sets = random_split(val_set, [len(val_set) // split_num] * (split_num - 1) + [len(val_set) - (split_num - 1) * (len(val_set) // split_num)])
    train_loaders = [tg.tasks.DataLoader(cts, batch_size=args.batch_size, shuffle=True) for cts in client_train_sets]

    print(f"Client Train/Val/Test Set Lengths: {len(client_train_sets[0])}, {len(client_val_sets[0])}, {len(test_set)}")
    experiment.log_parameters({"client_train_length": len(client_train_sets[0]), "client_val_length": len(client_val_sets[0]), "client_test_length": len(test_set)})
    
    results = {
        "0shot_test_acc": np.mean(eval_dataset(test_set, eval_fn, model_list[0])),
        "0shot_validation_acc": [np.mean(eval_dataset(val_set, eval_fn, model_list[0]))],
        "0shot_prompt": [system_prompt_list[0].get_value()],
        "best_agg_val_acc": None,
        "best_agg_prompt": None,
    }
    
    experiment.log_parameters({
        "0shot_test_engine_test_acc": results["0shot_test_acc"],
        "0shot_test_engine_validation_acc": results["0shot_validation_acc"][-1],
        "0shot_test_engine_prompt": results["0shot_prompt"][-1],
    })
    
    success_update = 0
    update_num = 0

    for epoch in range(args.max_epochs):
        for client_idx in range(split_num):
            print(f"\nTraining Client {client_idx}")
            system_prompt = system_prompt_list[client_idx]
            total_step = 0
            
            for batch_x, batch_y in tqdm(train_loaders[client_idx], desc=f"Client {client_idx} - Epoch {epoch}"):
                optimizer_list[client_idx].zero_grad()
                
                losses = [eval_fn(inputs={"prediction": model_list[client_idx](tg.Variable(x, requires_grad=False, role_description='query or answer variable')), "ground_truth_answer": tg.Variable(int(y), requires_grad=False, role_description='query or answer variable')}) for x, y in zip(batch_x, batch_y)]
                total_loss = tg.sum(losses)
                total_loss.backward()
                optimizer_list[client_idx].step()
                
                experiment.log_metric(f"client_{client_idx}_train_loss", np.mean([float(l.get_value()) for l in losses]), step=total_step)
                total_step += 1
                
                if total_step > args.max_steps:
                    break
    
    # Aggregate results
    agg_val_acc = np.mean(eval_dataset(val_set, eval_fn, model_list[0]))
    if results["best_agg_val_acc"] is None or results["best_agg_val_acc"] < agg_val_acc:
        results["best_agg_val_acc"] = agg_val_acc
        results["best_agg_prompt"] = system_prompt.get_value()
    
    experiment.log_parameter("best_agg_val_acc", results["best_agg_val_acc"])
    
    # Final test accuracy
    results["last_test_acc"] = np.mean(eval_dataset(test_set, eval_fn, model_list[0]))
    experiment.log_parameter("last_test_acc", results["last_test_acc"])
    
    # Save prompts
    for prompt_type, prompt_value in {"last_prompt": system_prompt.get_value(), "best_agg_prompt": results["best_agg_prompt"]}.items():
        filename = os.path.join(args.comet_log_path, f"{args.task[0]}_{prompt_type}.txt")
        with open(filename, "w") as f:
            f.write(prompt_value)
        experiment.log_asset(filename)
    
    print(f"Best Aggregated Validation Accuracy: {results['best_agg_val_acc']}")
    print(f"Final Test Accuracy: {results['last_test_acc']}")
