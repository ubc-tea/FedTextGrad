import os
import re
import sys
from copy import deepcopy
from string import Template
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm
from textgrad.tasks import load_task
import textgrad as tg
from eval import eval_dataset

from utils.prompt_template import SUMMARIZATION_TEMPLATE, UID_TEMPLATE, FORMATTING_INSTRUCTION

# Append parent directory to sys.path to locate local libraries.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def compute_batch_loss(
    model: Any, eval_fn: Any, batch_x: List[Any], batch_y: List[Any]
) -> Tuple[float, List[Any]]:
    """
    Compute the average loss value and collect loss outputs for a batch.

    Args:
        model: The language model (BlackboxLLM) instance.
        eval_fn: Evaluation function for the task.
        batch_x (List[Any]): Batch of input queries.
        batch_y (List[Any]): Batch of corresponding ground truth answers.

    Returns:
        Tuple[float, List[Any]]: The average loss value and a list of loss outputs.
    """
    total_loss = 0
    loss_outputs = []
    for x, y in zip(batch_x, batch_y):
        x_var = tg.Variable(
            x, requires_grad=False, role_description="query to the language model"
        )
        # Convert numpy integers if necessary.
        y_val = int(y) if isinstance(y, np.integer) else y
        y_var = tg.Variable(
            y_val, requires_grad=False, role_description="correct answer for the query"
        )
        response = model(x_var)
        try:
            eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
        except Exception:
            eval_output = eval_fn([x_var, y_var, response])
        loss_outputs.append(eval_output)
        value_str = eval_output.get_value()
        match = re.search(r'<ACCURACY>\s*(\d+)\s*</ACCURACY>', value_str)
        if match:
            value = int(match.group(1))
        else:
            value = int(value_str)
        total_loss += value

    avg_loss = total_loss / len(batch_x)
    return avg_loss, loss_outputs


def run_training(args: Any, experiment: Any) -> None:
    """
    Run the federated heterogeneous training procedure over multiple tasks.

    This function loads tasks, initializes evaluation and test engines, creates
    per-task data loaders, and performs training with periodic prompt aggregation.
    
    Args:
        args: Parsed command-line arguments.
        experiment: Comet ML experiment instance.
    """
    # Initialize engines.
    llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load tasks and associated datasets.
    train_set_list = []
    val_set_list = []
    test_set_list = []
    eval_fn_list = []
    task_name_list = args.task

    for task_name in task_name_list:
        train_set, val_set, test_set, eval_fn = load_task(task_name, evaluation_api=llm_api_eval)
        train_set_list.append(train_set)
        val_set_list.append(val_set)
        test_set_list.append(test_set)
        eval_fn_list.append(eval_fn)

    # Build starting system prompts and data loaders for each task.
    STARTING_SYSTEM_PROMPT_LIST = []
    train_loader_list = []
    for idx, task_name in enumerate(task_name_list):
        print(f"Train/Val/Test Set Lengths for {task_name}: "
              f"{len(train_set_list[idx])}, {len(val_set_list[idx])}, {len(test_set_list[idx])}")
        STARTING_SYSTEM_PROMPT_LIST.append(train_set_list[idx].get_task_description())
        train_loader_list.append(
            tg.tasks.DataLoader(train_set_list[idx], batch_size=args.batch_size, shuffle=True)
        )
        print(STARTING_SYSTEM_PROMPT_LIST[idx])

    experiment.log_parameter("task_description", STARTING_SYSTEM_PROMPT_LIST)

    # Initialize per-task system prompts, evaluation models, test models, and optimizers.
    system_prompt_list = []
    model_evaluation_list = []
    reference_list = []
    model_list = []
    optimizer_list = []
    results = {}

    for idx, task_name in enumerate(task_name_list):
        # 0-shot evaluation using evaluation engine.
        system_prompt = tg.Variable(
            STARTING_SYSTEM_PROMPT_LIST[idx],
            requires_grad=True,
            role_description="system prompt to the language model",
        )
        system_prompt_list.append(system_prompt)
        model_eval = tg.BlackboxLLM(llm_api_eval, system_prompt)
        model_evaluation_list.append(model_eval)

        if not args.do_not_run_larger_model:
            reference = np.mean(eval_dataset(test_set_list[idx], eval_fn_list[idx], model_eval))
            reference_list.append(reference)

        # Build test model for training.
        system_prompt = tg.Variable(
            STARTING_SYSTEM_PROMPT_LIST[idx],
            requires_grad=True,
            role_description=(
                "structured system prompt to a somewhat capable language model that specifies "
                "the behavior and strategies for the QA task"
            ),
        )
        model = tg.BlackboxLLM(llm_api_test, system_prompt)
        model_list.append(model)
        optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])
        optimizer_list.append(optimizer)

        # Log 0-shot performance.
        test_acc = np.mean(eval_dataset(test_set_list[idx], eval_fn_list[idx], model))
        val_acc = np.mean(eval_dataset(val_set_list[idx], eval_fn_list[idx], model))
        results[f"{task_name}_0shot_test_acc"] = test_acc
        results[f"{task_name}_0shot_validation_acc"] = val_acc
        results[f"{task_name}_0shot_prompt"] = system_prompt.get_value()

        experiment.log_parameter(f"{task_name}_0shot_test_engine_test_acc", test_acc)
        experiment.log_parameter(f"{task_name}_0shot_test_engine_validation_acc", val_acc)
        experiment.log_parameter(f"{task_name}_0shot_test_engine_prompt", system_prompt.get_value())

    success_update = 0
    update_num = 0

    print("Federated Heterogeneous Training......")
    # Training loop.
    for epoch in range(args.max_epochs):
        for idx, task_name in enumerate(task_name_list):
            print(f"\nTraining on {task_name}")
            total_step = 0

            while total_step < args.max_steps:
                for steps, (batch_x, batch_y) in enumerate(tqdm(train_loader_list[idx], position=0)):
                    tqdm.write(f"Task {task_name}. Training step {steps}. Epoch {epoch}")
                    optimizer_list[idx].zero_grad()
                    loss_value, losses = compute_batch_loss(
                        model_list[idx], eval_fn_list[idx], batch_x, batch_y
                    )
                    print(f"\nBatch Train Loss Value: {loss_value:.4f}")

                    last_batch_prompt = system_prompt_list[idx].get_value()

                    total_loss = tg.sum(losses)
                    total_loss.backward()
                    optimizer_list[idx].step()

                    # Re-run batch to decide whether to update the prompt.
                    updated_loss_value = 0
                    for x, y in zip(batch_x, batch_y):
                        x_var = tg.Variable(
                            x, requires_grad=False, role_description="query to the language model"
                        )
                        y_val = int(y) if isinstance(y, np.integer) else y
                        y_var = tg.Variable(
                            y_val, requires_grad=False, role_description="correct answer for the query"
                        )
                        response = model_list[idx](x_var)
                        try:
                            eval_output = eval_fn_list[idx](
                                inputs={"prediction": response, "ground_truth_answer": y_var}
                            )
                        except Exception:
                            eval_output = eval_fn_list[idx]([x_var, y_var, response])
                        match = re.search(r'<ACCURACY>\s*(\d+)\s*</ACCURACY>', eval_output.get_value())
                        if match:
                            value = int(match.group(1))
                        else:
                            value = int(eval_output.get_value())
                        updated_loss_value += value
                    updated_loss_value /= len(batch_x)
                    print(f"Updated Batch Train Loss Value: {updated_loss_value}")

                    update_num += 1
                    # Determine if update is accepted based on proximal_update flag.
                    if args.proximal_update:
                        if updated_loss_value <= loss_value and updated_loss_value != 1.0:
                            print("Improving Failure! Dropping updated prompt in this step.")
                            system_prompt_list[idx].set_value(last_batch_prompt)
                        else:
                            print("Improving Success!")
                            success_update += 1
                    else:
                        if updated_loss_value < loss_value:
                            print("Improving Failure! Dropping updated prompt in this step.")
                            system_prompt_list[idx].set_value(last_batch_prompt)
                        else:
                            print("Improving Success!")
                            success_update += 1

                    experiment.log_metric(f"client_{idx}_train_acc", loss_value,
                                          step=((total_step + 1) * (epoch + 1)))
                    experiment.log_metric(f"client_{idx}_updated_train_acc", updated_loss_value,
                                          step=((total_step + 1) * (epoch + 1)))

                    # Synchronize system prompt.
                    system_prompt_list[idx].set_value(system_prompt_list[idx].get_value())
                    total_step += 1
                    if total_step > args.max_steps:
                        break

        # Aggregate prompts across tasks based on the chosen method.
        if args.aggregate_method == "concat":
            concat_prompt = tg.autograd.functional.aggregate(system_prompt_list)
            print(f"\nConcat Prompt: {concat_prompt.get_value()}")
            system_prompt = concat_prompt
        elif args.aggregate_method == "summarization":
            concat_prompt = tg.autograd.functional.aggregate(system_prompt_list)
            print(f"\nConcat Prompt: {concat_prompt.get_value()}")
            summarized_instruction = SUMMARIZATION_TEMPLATE.substitute(prompt=concat_prompt.get_value())
            summarized_prompt = llm_api_eval(summarized_instruction)
            summarized_prompt += FORMATTING_INSTRUCTION
            print(f"\nSummarized Prompt: {summarized_prompt}")
            system_prompt = tg.Variable(
                summarized_prompt,
                requires_grad=True,
                role_description=(
                    "structured system prompt to a somewhat capable language model that specifies "
                    "the behavior and strategies for the QA task"
                )
            )
        elif args.aggregate_method == "sum_uid":
            concat_prompt = tg.autograd.functional.aggregate(system_prompt_list)
            print(f"\nConcat Prompt: {concat_prompt.get_value()}")
            sum_uid_instruction = UID_TEMPLATE.substitute(prompt=concat_prompt.get_value())
            sum_uid_prompt = llm_api_eval(sum_uid_instruction)
            sum_uid_prompt += FORMATTING_INSTRUCTION
            print(f"\nSum UID Prompt: {sum_uid_prompt}")
            system_prompt = tg.Variable(
                sum_uid_prompt,
                requires_grad=True,
                role_description=(
                    "structured system prompt to a somewhat capable language model that specifies "
                    "the behavior and strategies for the QA task"
                )
            )
        else:
            raise ValueError("Not Supported Aggregation Method.")

        # Update each task's system prompt with the aggregated prompt.
        for idx in range(len(task_name_list)):
            system_prompt_list[idx].set_value(system_prompt.get_value())

        # Use the first task's validation set as the representative for aggregation.
        agg_val_acc = np.mean(eval_dataset(val_set_list[0], eval_fn_list[0], model_list[0]))
        if "best_agg_val_acc" not in results:
            results["best_agg_val_acc"] = agg_val_acc
            results["best_agg_prompt"] = system_prompt.get_value()
        if results["best_agg_val_acc"] < agg_val_acc:
            results["best_agg_val_acc"] = agg_val_acc
            results["best_agg_prompt"] = system_prompt.get_value()

    update_success_rate = success_update / update_num if update_num else 0.0
    print(f"\nUpdate Success Rate: {update_success_rate:.4f}")
    experiment.log_parameter("update_success_rate", update_success_rate)

    # Evaluate on the test set using the first task's test set.
    results["last_test_acc"] = np.mean(eval_dataset(test_set_list[0], eval_fn_list[0], model_list[0]))
    experiment.log_parameter("last_test_acc", results["last_test_acc"])

    # Save the last prompt.
    last_prompt = system_prompt.get_value()
    last_prompt_path = Path(args.comet_log_path) / f"{args.task[0]}_last_prompt.txt"
    with last_prompt_path.open("w") as f:
        f.write(last_prompt)
    experiment.log_asset(str(last_prompt_path))

    # Evaluate with the best aggregated prompt.
    system_prompt.set_value(results["best_agg_prompt"])
    results["best_test_acc"] = np.mean(eval_dataset(test_set_list[0], eval_fn_list[0], model_list[0]))
    experiment.log_parameter("best_test_acc", results["best_test_acc"])

    best_prompt = results["best_agg_prompt"]
    best_prompt_path = Path(args.comet_log_path) / f"{args.task[0]}_best_agg_prompt.txt"
    with best_prompt_path.open("w") as f:
        f.write(best_prompt)
    experiment.log_asset(str(best_prompt_path))
