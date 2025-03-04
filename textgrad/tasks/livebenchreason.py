import os
import pandas as pd
import platformdirs
from sklearn.model_selection import train_test_split
from datasets import load_dataset

import textgrad as tg
from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime, MultiFieldTokenParsedEvaluation
from .base import Dataset

import re


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1].replace("$", "").replace("fbox","boxed")

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def spatial_process_results(prediction: tg.Variable, ground_truth_answer: tg.Variable, debug=False) -> int:
    prediction = prediction.value
    ground_truth_answer = ground_truth_answer.value
    
    word_to_number = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    bold_words = re.findall(r'\*\*([^\*]+)\*\*', prediction)
    score = 0

    # allow the answer to be within the last 3 bolded words
    words_to_check = []
    for i in range(3):
        if bold_words and len(bold_words) > i:
            words_to_check.append(bold_words[-i-1].strip().lower())

    for i, word in enumerate(words_to_check):
        if word == ground_truth_answer.strip().lower():
            score = 1

        # allow the answer to be the number spelled out
        if word in word_to_number and word_to_number[word] == ground_truth_answer.strip().lower():
            score = 1

        # allow certain cases like "two tetrahedra" == "tetrahedra" and "equilateral triangle" == "triangle"
        # while still disallowing cases like "circle square triangle" == "triangle"
        for answer in ["tetrahedra", "tetrahedron", "triangle", "square"]:
            if ground_truth_answer.strip().lower() == answer and answer in word and len(word) < (2 * len(answer) + 5):
                score = 1

    allow_boxed = True
    if score == 0 and allow_boxed:
        prediction = prediction.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(prediction)
        if last_boxed:
            parsed_answer = remove_boxed(last_boxed)
            if parsed_answer == ground_truth_answer:
                score = 1

    debug = False
    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth_answer.strip().lower())
        if bold_words:
            print("BOLD WORDS:", bold_words[-1].strip().lower())
        print("END OF OUTPUT", prediction[-50:])        

    return score


def web_of_lies_process_results(prediction: tg.Variable, ground_truth_answer: tg.Variable, debug=False) -> int:
    prediction = prediction.value
    ground_truth_answer = ground_truth_answer.value
    # pull out words in bold
    bold_words = re.findall(r'\*\*(.*?)\*\*', prediction)

    if not bold_words:
        if debug:
            print("NO BOLDS, answer", ground_truth, "output", prediction[-50:], )
        return 0

    last_bold = bold_words[-1].lower()

    # Check if last_bold is an exact match of ground_truth
    if last_bold == ground_truth_answer.lower():
        return 1

    # Check if last_bold contains the ground_truth
    if last_bold.count("yes") + last_bold.count("no") == 3 and ground_truth_answer.lower() in last_bold:
        return 1        

    if debug:
        print('FAILED, answer', ground_truth_answer, 'output', last_bold)

    return 0


def zebra_puzzle_process_results(prediction: tg.Variable, ground_truth_answer: tg.Variable) -> int:
    prediction = prediction.value
    ground_truth_answer = ground_truth_answer.value

    # Mapping of numbers to words for 1 to 9
    number_to_word = {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

    # Pull out words in bold
    bold_words = re.findall(r'\*\*\*(\w+)\*\*\*', prediction)

    # Remove any trailing punctuation from the last bold word if exists
    if bold_words:
        if (bold_words[-1].lower() == ground_truth_answer.lower() or
            (bold_words[-1] in number_to_word and number_to_word[bold_words[-1]].lower() == ground_truth_answer.lower())
            or bold_words[-1].lower() + ' movies' == ground_truth_answer.lower()):
            return 1
        else:
            return 0
    else:
        # Split the text into words and remove punctuation.
        words = re.findall(r'\b\w+\b', prediction)
        last_word = words[-1] if words else ''
        # Check if the last bold word is a number and matches the word representation of the ground_truth
        if (last_word.lower() == ground_truth_answer.lower() or
            (last_word in number_to_word and number_to_word[last_word].lower() == ground_truth_answer.lower())
            or last_word.lower() + ' movies' == ground_truth_answer.lower()):
            return 1
        return 0


class LiveBenchReasoning(Dataset):
    def __init__(self, root: str=None, split: str="train", task: str=None, train_ratio=0.8, val_split_ratio=0.2, *args, **kwargs):
        """
        LiveBench dataset with math from HF."""
        dataset = load_dataset("livebench/reasoning")['test']
        train_test_split = dataset.train_test_split(test_size=1-train_ratio)

        if task is not None:
            dataset = dataset.filter(lambda item: item["task"] == task)

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
        self._task_description = "You will answer a reasoning question. Think step by step."

    def __getitem__(self, index):
        row = self.data[index]
        question = row["turns"]
        answer = row["ground_truth"]
        question_prompt = f"Question: {question}\n"
        return question_prompt, answer

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return "You will answer a reasoning question. Think step by step."

