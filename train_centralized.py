import os
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

# Append parent directory to sys.path to locate local libraries.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import textgrad as tg
from textgrad.tasks import load_task
from eval import eval_dataset


def compute_batch_loss_and_outputs(
    model: Any, eval_fn: Any, batch_x: List[Any], batch_y: List[Any]
) -> Tuple[float, List[Any]]:
    """
    Compute the average loss value and collect loss outputs for a batch.

    Args:
        model: The language model wrapped by textgrad.
        eval_fn: Evaluation function that computes the loss.
        batch_x (List[Any]): List of input queries.
        batch_y (List[Any]): List of corresponding ground truth answers.

    Returns:
        Tuple[float, List[Any]]: A tuple containing the average loss value
        and a list of individual loss outputs.
    """
    total_value = 0.0
    outputs = []
    for x, y in zip(batch_x, batch_y):
        # Prepare input and label variables.
        x_var = tg.Variable(x, requires_grad=False, role_description="query to the language model")
        y_val = int(y) if isinstance(y, np.integer) else y
        y_var = tg.Variable(y_val, requires_grad=False, role_description="correct answer for the query")

        # Obtain model response.
        response = model(x_var)
        try:
            eval_output = eval_fn(inputs={"prediction": response, "ground_truth_answer": y_var})
        except Exception:
            eval_output = eval_fn([x_var, y_var, response])
        outputs.append(eval_output)

        # Extract accuracy value from the evaluation output.
        value_str = eval_output.get_value()
        match = re.search(r"<ACCURACY>\s*(\d+)\s*</ACCURACY>", value_str)
        if match:
            value = int(match.group(1))
        else:
            value = int(value_str)
        total_value += value

    average_loss = total_value / len(batch_x)
    return average_loss, outputs


def run_training(args: Any, experiment: Any) -> None:
    """
    Run the training procedure for the Federated TextGrad experiment.

    Args:
        args: Parsed command-line arguments.
        experiment: Comet ML experiment instance.
    """
    # Initialize language model engines.
    llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
    llm_api_test = tg.get_engine(engine_name=args.test_engine)
    tg.set_backward_engine(llm_api_eval, override=True)

    # Load dataset and evaluation function.
    train_set, val_set, test_set, eval_fn = load_task(args.task[0], evaluation_api=llm_api_eval)
    print("Train/Val/Test Set Lengths:", len(train_set), len(val_set), len(test_set))
    dataset_length = {
        "train_length": len(train_set),
        "val_length": len(val_set),
        "test_length": len(test_set),
    }
    experiment.log_parameters(dataset_length)

    # Define the starting system prompt.
    STARTING_SYSTEM_PROMPT = (
        " You will answer a reasoning question. Think step by step. "
        "The last line of your response should be of the following format: "
        "'Answer: $VALUE' where VALUE is a numerical value."
    )
    print(STARTING_SYSTEM_PROMPT)
    experiment.log_parameter("task_description", STARTING_SYSTEM_PROMPT)

    # Prepare data loader.
    train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Setup the model for 0-shot evaluation.
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description="system prompt to the language model",
    )
    model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)
    if not args.do_not_run_larger_model:
        reference = np.mean(eval_dataset(test_set, eval_fn, model_evaluation))
        experiment.log_parameter("0shot_eval_engine_test_acc", reference)

    # Create model for training.
    system_prompt = tg.Variable(
        STARTING_SYSTEM_PROMPT,
        requires_grad=True,
        role_description=(
            "structured system prompt to a somewhat capable language model that specifies "
            "the behavior and strategies for the QA task"
        ),
    )
    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    optimizer = tg.optimizer.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

    # Initial evaluation.
    results = {
        "0shot_test_acc": np.mean(eval_dataset(test_set, eval_fn, model)),
        "validation_acc": [np.mean(eval_dataset(val_set, eval_fn, model))],
        "prompt": [system_prompt.get_value()],
    }
    print("Test Acc:", results["0shot_test_acc"])
    print("Validation Acc:", results["validation_acc"])
    experiment.log_parameter("0shot_test_engine_test_acc", results["0shot_test_acc"])
    experiment.log_parameter("0shot_test_engine_validation_acc", results["validation_acc"][-1])
    experiment.log_parameter("0shot_test_engine_prompt", results["prompt"][-1])

    results["best_validation_acc"] = results["validation_acc"][-1]
    results["best_prompt"] = results["prompt"][-1]

    success_update = 0
    update_num = 0

    # Training loop.
    for epoch in range(args.max_epochs):
        for step, (batch_x, batch_y) in enumerate(tqdm(train_loader, position=0)):
            tqdm.write(f"\nTraining step {step}. Epoch {epoch}")
            optimizer.zero_grad()

            # Compute loss before update.
            loss_value, losses = compute_batch_loss_and_outputs(model, eval_fn, batch_x, batch_y)
            print(f"\nBatch Train Loss Value: {loss_value:.4f}")

            # Save current prompt before update.
            last_batch_prompt = system_prompt.get_value()

            # Backpropagation and optimization step.
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()

            # Recompute loss after update.
            updated_loss_value, _ = compute_batch_loss_and_outputs(model, eval_fn, batch_x, batch_y)
            print(f"Updated Batch Train Loss Value: {updated_loss_value:.4f}")

            update_num += 1
            # Decision: Accept update based on proximal_update flag.
            if args.proximal_update:
                if updated_loss_value <= loss_value and updated_loss_value != 1.0:
                    print("Improving Failure! Reverting to previous prompt.")
                    system_prompt.set_value(last_batch_prompt)
                else:
                    print("Improving Success!")
                    success_update += 1
            else:
                if updated_loss_value < loss_value:
                    print("Improving Failure! Reverting to previous prompt.")
                    system_prompt.set_value(last_batch_prompt)
                else:
                    print("Improving Success!")
                    success_update += 1

            # Log training metrics.
            global_step = (step + 1) * (epoch + 1)
            experiment.log_metric("train_acc", loss_value, step=global_step)
            experiment.log_metric("updated_train_acc", updated_loss_value, step=global_step)

            if step == args.max_steps:
                break

        # Validate after each epoch.
        validation_acc = np.mean(eval_dataset(val_set, eval_fn, model))
        print("Validation Acc:", validation_acc)
        results["validation_acc"].append(validation_acc)
        results["prompt"].append(system_prompt.get_value())

        if results["best_validation_acc"] < validation_acc:
            print("\nUpdate best validation performance.")
            results["best_validation_acc"] = validation_acc
            results["best_prompt"] = system_prompt.get_value()

        print(f"\nGlobal Step: {global_step}")
        experiment.log_text(system_prompt.get_value(), step=global_step)
        experiment.log_metric("validation_acc", validation_acc, step=epoch)

    update_success_rate = success_update / update_num if update_num else 0.0
    print(f"\nUpdate Success Rate: {update_success_rate:.4f}")
    experiment.log_parameter("update_success_rate", update_success_rate)

    # Evaluate on test set.
    results["last_test_acc"] = np.mean(eval_dataset(test_set, eval_fn, model))
    print("Test Acc:", results["last_test_acc"])
    experiment.log_parameter("last_test_acc", results["last_test_acc"])

    # Save the last prompt.
    last_prompt = system_prompt.get_value()
    last_prompt_file = Path(args.comet_log_path) / f"{args.task[0]}_last_prompt.txt"
    with last_prompt_file.open("w") as f:
        f.write(last_prompt)
    experiment.log_asset(str(last_prompt_file))

    # Evaluate using the best prompt.
    system_prompt.set_value(results["best_prompt"])
    results["best_test_acc"] = np.mean(eval_dataset(test_set, eval_fn, model))
    print("Test Acc:", results["best_test_acc"])
    experiment.log_parameter("best_test_acc", results["best_test_acc"])

    # Save the best prompt.
    best_prompt_file = Path(args.comet_log_path) / f"{args.task[0]}_best_prompt.txt"
    with best_prompt_file.open("w") as f:
        f.write(results["best_prompt"])
    experiment.log_asset(str(best_prompt_file))
