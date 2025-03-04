import os
import pandas as pd
import platformdirs

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


class ProLLaMA(Dataset):
    def __init__(self, root: str=None, split: str="train", val_split_ratio=0.2, *args, **kwargs):
        """
        MMLU dataset from HF."""
        from datasets import load_dataset
        train_df = pd.read_json(os.path.join(root, "train_split.json"))
        train_df = train_df[train_df['instruction'].str.contains("Determine superfamily", case=False, na=False)]
        test_df = pd.read_json(os.path.join(root, "test_split.json"))
        test_df = test_df[test_df['instruction'].str.contains("Determine superfamily", case=False, na=False)]
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")

        self.root = root
        assert split in ["train", "valid", "test"]
        if split == "train":
            valid_size = int(len(train_df) * (1-val_split_ratio))
            self.data = train_df.sample(n=valid_size, random_state=42)
            self.data.reset_index(drop=True, inplace=True)
        elif split == "valid":
            valid_size = int(len(train_df) * val_split_ratio)
            self.data = train_df.sample(n=valid_size, random_state=42)
            self.data.reset_index(drop=True, inplace=True)
        else:  # split == "test"
            self.data = test_df
            self.data.reset_index(drop=True, inplace=True)
        self.split = split
        self._task_description = 'You will classify the a given amino acid sequence to a superfamily. Think step by step.'
            
    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = row["instruction"] + row["input"]
        answer = row["output"]
        question_prompt = f"Question: {question}\n"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "Given a amino acid sequence. Your goal is to determine its superfamily."

    def get_default_task_instruction(self):
        return "Given a amino acid sequence. Your goal is to determine its superfamily."