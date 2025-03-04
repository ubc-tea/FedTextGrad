import os
import pandas as pd
import platformdirs
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime, MultiFieldTokenParsedEvaluation
from .base import Dataset

import re

def eval_string_based(response_text, correct_answer):
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
    
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == correct_answer else 0.0
    return score


class LiveBenchMath(Dataset):
    def __init__(self, root: str=None, split: str="train", train_ratio=0.8, val_split_ratio=0.2, *args, **kwargs):
        """
        LiveBench dataset with math from HF."""
        dataset = load_dataset("livebench/math")['test']

        train_test_split = dataset.train_test_split(test_size=1-train_ratio)

        # Access the train and test sets
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        valid_size = int(len(train_dataset) * (1-val_split_ratio))
        train_val_split = train_dataset.train_test_split(test_size=1-train_ratio)
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']


        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        assert split in ["train", "valid", "test"]
        if split == "train":
            self.data = train_dataset
        elif split == "valid":
            self.data = val_dataset
        else:  # split == "test"
            self.data = test_dataset
        self.split = split
        self._task_description = "You will answer a mathematics reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
            
    def __getitem__(self, index):
        row = self.data[index]
        question = row["turns"]
        answer = row["ground_truth"]
        question_prompt = f"Question: {question}\n"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will answer a mathematics reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

    def get_default_task_instruction(self):
        return "You will answer a mathematics reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."


class LiveBenchReasoning(Dataset):
    def __init__(self, root: str=None, split: str="train", train_ratio=0.8, val_split_ratio=0.2, *args, **kwargs):
        """
        LiveBench dataset with math from HF."""
        dataset = load_dataset("livebench/reasoning")['test']

        train_test_split = dataset.train_test_split(test_size=1-train_ratio)

        # Access the train and test sets
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']

        valid_size = int(len(train_dataset) * (1-val_split_ratio))
        train_val_split = train_dataset.train_test_split(test_size=1-train_ratio)
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']


        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        assert split in ["train", "valid", "test"]
        if split == "train":
            self.data = train_dataset
        elif split == "valid":
            self.data = val_dataset
        else:  # split == "test"
            self.data = test_dataset
        self.split = split
        # self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value or a word or 'yes/no, yes/no, yes/no'."

    def __getitem__(self, index):
        row = self.data[index]
        question = row["turns"]
        answer = row["ground_truth"]
        question_prompt = f"Question: {question}\n"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value or a word or 'yes/no, yes/no, yes/no'."


if __name__ == "__main__":
    LiveBenchMath()