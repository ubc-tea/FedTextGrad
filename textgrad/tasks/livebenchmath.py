import os
import pandas as pd
import platformdirs
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from typing import Dict, List, Optional
import warnings
import signal
import sympy
from sympy.parsing.latex import parse_latex

import textgrad as tg
from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime, MultiFieldTokenParsedEvaluation
from .base import Dataset

import re

def amps_hard_process_results(prediction: tg.Variable, ground_truth_answer: tg.Variable) -> int:
    prediction = prediction.value
    ground_truth_answer = ground_truth_answer.value

    retval = 0

    if isinstance(ground_truth_answer, list):
        ground_truth_answer = ground_truth_answer[-1]
    prediction = prediction.replace("+C","")
    prediction = prediction.replace("+ C", "")
    prediction = prediction.replace("\\\\fbox{", "\\\\boxed{")

    last_boxed = last_boxed_only_string(prediction)
    if last_boxed:
        parsed_answer = normalize_final_answer(remove_boxed(last_boxed))
        if is_equiv(ground_truth_answer, parsed_answer):
            retval = 1

    debug = False
    if retval == 0 and debug:
        print("FAILED", ground_truth_answer, "OUTPUT", prediction[-70:], "\n")
    return retval

def last_boxed_only_string(string: str) -> Optional[str]:
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


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        try:
            parsed_x1 = parse_latex(x1)
            parsed_x2 = parse_latex(x2)
        except (
            sympy.parsing.latex.errors.LaTeXParsingError,
            sympy.SympifyError,
            TypeError,
        ):
            warnings.warn(f"couldn't parse one of {x1} or {x2}")
            return False

        try:
            diff = parsed_x1 - parsed_x2
        except TypeError:
            warnings.warn(f"couldn't subtract {x1} and {x2}")
            return False

        try:
            if sympy.Abs(sympy.simplify(diff)) < 0.001:
                return True
            else:
                return False
        except ValueError:
            warnings.warn(
                f"Had some trouble simplifying when comparing {x1} and {x2}"
            )
    except TimeoutError:
        warnings.warn(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        warnings.warn(e)
        raise
    except Exception as e:
        warnings.warn(f"Failed comparing {x1} and {x2} with {e}")
        return False


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def mathcontest_process_results(prediction: tg.Variable, ground_truth_answer: tg.Variable, question_text: str) -> int:
    prediction = prediction.value
    ground_truth_answer = ground_truth_answer.value

    score = 0
    # the reference answer must be a single capital letter from A to E (I.e., the multiple choice answer)
    if not (isinstance(ground_truth_answer, str) and len(ground_truth_answer) == 1 and 'A' <= ground_truth_answer <= 'E'):
        raise ValueError("amc_answer must be a single capital letter between A and E.")

    # The LLM was prompted to repeat letter answer 5 times, to make it easy to pull out the answer        
    if ground_truth_answer * 4 in prediction:
        score = 1

    allow_boxed = True
    if score == 0 and allow_boxed:
        prediction = prediction.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(prediction)
        if last_boxed:
            parsed_answer = remove_boxed(last_boxed)
            if parsed_answer == ground_truth_answer:
                score = 1

    allow_answer_values = True
    if score == 0 and allow_answer_values:
        value = extract_answer(question_text, ground_truth_answer)
        length_to_check = 20 + len(value)
        if value in prediction[-length_to_check:]:
            score = 1

    debug = False
    if debug:
        # check if the LLM guessed a letter, even if it was wrong
        letter_answer = False
        for letters in ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]:
            if letters in prediction:
                letter_answer = True

        if not letter_answer and score == 0:
            print("INCORRECT")
            print("GROUND TRUTH", ground_truth_answer.strip().lower())
            if last_boxed:
                print("BOXED:", parsed_answer)
            print("END OF OUTPUT", prediction[-50:])      

    return score


def extract_answer(statement, letter):

    pattern = r'\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)'
    matches = re.findall(pattern, statement)
    answers = {match[0]: match[1].strip() for match in matches}
    answer = answers.get(letter, None)

    if not answer or answer == "":
        # this only happens for one question, which is too long for the LLMs to repeat
        answer = "FAILURE"

    answer = answer.strip()
    answer = answer.strip("$")
    answer = answer.strip("~")

    return answer


def last_boxed_only_string(string: str) -> Optional[str]:
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

def match_expression_completions_to_ground_truth(completions, ground_truth):
    num_matches = 0
    for i in range(len(ground_truth)):
        if i not in completions:
            continue

        completion = completions[i].lower().strip().replace(' ' , '')
        comp = ground_truth[i].lower().strip().replace(' ' , '')

        if completion == comp:
            num_matches += 1

    return num_matches/len(ground_truth)

def remove_nonnumeric_chars_at_ends(s):
    start_index = 0
    while start_index < len(s) and not s[start_index].isdigit():
        start_index += 1
    end_index = start_index
    while end_index < len(s) and s[end_index].isdigit():
        end_index += 1

    return s[start_index:end_index], len(s) - (end_index - start_index)

def extract_expression_completions_from_generation(generation):
    # generation has Answer: comma separated list of numbers. I want to extract the last such comma separated list
    split_string = "Answer"
    numbers = [k.strip() for k in generation.split(split_string)[-1].split(',')]

    # the last number may have some extra non-numeric characters at the end. Those need to be removed
    new_numbers = []
    for i, n in enumerate(numbers):
        n, num_removed = remove_nonnumeric_chars_at_ends(n)
        if n != '' and n != "₂":
            new_numbers.append(int(n))
        if (i > 0) and (num_removed > 0):
            break

    numbers = new_numbers
    return numbers

def proof_rearrangement_process_results(prediction: tg.Variable, ground_truth_answer: tg.Variable, edit_distance=False) -> int:
    prediction = prediction.value
    ground_truth_answer = ground_truth_answer.value
    
    ground_truth_answer = [int(n) for n in ground_truth_answer.split(',')]

    completions = extract_expression_completions_from_generation(prediction)

    if edit_distance:
        from Levenshtein import distance
        match = distance(completions, ground_truth_answer)
        frac_matches = 1-(match/max(len(completions), len(ground_truth_answer)))
    else:
        match = [(completions[i] == ground_truth_answer[i]) if i < len(ground_truth_answer) else 0 for i in range(len(completions))]
        frac_matches = sum(match)/len(match) if len(match) > 0 else 0

    return frac_matches

class LiveBenchMath(Dataset):
    def __init__(self, root: str=None, split: str="train", task: str=None, train_ratio=0.8, val_split_ratio=0.2, *args, **kwargs):
        """
        LiveBench dataset with math from HF."""
        dataset = load_dataset("livebench/math")['test']

        if task is not None:
            dataset = dataset.filter(lambda item: item["task"] == task)

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