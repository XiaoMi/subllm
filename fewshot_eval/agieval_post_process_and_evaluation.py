import re
import pandas as pd
from math_equivalence import is_equiv

# define the datasets
english_qa_datasets = ["lsat-ar", "lsat-lr", "lsat-rc", "logiqa-en", "sat-math", "sat-en", "aqua-rat",
                       "sat-en-without-passage", "gaokao-english"]
chinese_qa_datasets = ["logiqa-zh", "jec-qa-kd", "jec-qa-ca", "gaokao-chinese", "gaokao-geography", "gaokao-history",
                       "gaokao-biology", "gaokao-chemistry", "gaokao-physics", "gaokao-mathqa"]
english_cloze_datasets = ["math"]
chinese_cloze_datasets = ["gaokao-mathcloze"]

multi_choice_datasets = ["jec-qa-kd", "jec-qa-ca", "gaokao-physics"]
math_output_datasets = {"gaokao-mathcloze", "math"}


def extract_last_line(string):
    lines = string.split('\n')
    for item in lines[::-1]:
        if item.strip() != "":
            string = item
            break
    return string


def remove_few_shot_prefix(string:str):
    prefix_list = ["The answer is therefore", "答案是", "The answer is"]
    for prefix in prefix_list:
        if string.startswith(prefix):
            string = string[len(prefix):].strip()
        elif prefix in string:
            index = string.rfind(prefix)
            if index >= 0:
                string = string[index + len(prefix):].strip()
    return string


def try_parse_few_shot_qa_single_answer(string, setting_name, language='en'):
    if setting_name == "few-shot-CoT":
        string = extract_last_line(string)
    if language == 'en':
        pattern = "answer is .*?([A-G])"
        match = re.search(pattern, string)
    elif language == 'zh':
        pattern = "答案是.*?([A-G])"
        match = re.search(pattern, string)
    else:
        raise ValueError("Unknown language {0}".format(language))
    if match:
        return match.group(1)
    else:
        return None


def try_parse_few_shot_pattern(string:str, dataset_name, setting_name):
    if setting_name == "few-shot-CoT":
        string = extract_last_line(string)
    if dataset_name in chinese_cloze_datasets:
        return string.startswith("答案是")
    elif dataset_name in english_cloze_datasets:
        return string.startswith("The answer is therefore")
    elif dataset_name in chinese_qa_datasets:
        pattern = "答案是.*?([A-G])"
        match = re.search(pattern, string)
        return match is not None
    elif dataset_name in english_qa_datasets:
        pattern = "answer is .*?([A-G])"
        match = re.search(pattern, string)
        return match is not None
    return False


def parse_few_shot_qa_single_answer(string, setting_name, language='en'):
    answer = try_parse_few_shot_qa_single_answer(string, setting_name, language)
    if answer is None:
        return find_first_capital_letter(string)
    else:
        return answer


def find_first_capital_letter(answer):
    letter_set = {"A", "B", "C", "D", "E", "F"}
    for c in answer:
        if c in letter_set:
            return c
    # print("Can't find capital letter in:", answer)
    return ""


def extract_answer_in_bracket(answer, prefix='【', suffix='】'):
    if prefix not in answer and suffix not in answer:
        # print("doesn't found special tokens in:", answer)
        return ""
    s = answer.index(prefix) + len(prefix)
    t = answer.index(suffix)
    ret = answer[s:t]
    return ret


def parse_math_answer(setting_name, raw_string):
    if setting_name == "few-shot-CoT":
        raw_string = extract_last_line(raw_string)
    # if setting_name == "few-shot-CoT" or setting_name == "few-shot":
    #     raw_string = remove_few_shot_prefix(raw_string)
    #     return raw_string

    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left):-1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
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

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(s):
        first_pattern = "\$(.*)\$"
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if "=" in last_match:
                last_match = last_match.split("=")[-1].lstrip(" ")
        return last_match

    def get_answer_without_dollar_sign(s):
        last_match = None
        if "=" in s:
            last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
            if "\\" in last_match:
                last_match = last_match.split("\\")[0]
        else:
            pattern = "(?:\\$)?\d+(?:\.\d+)?(?![\w\d])"
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match

    raw_string = remove_few_shot_prefix(raw_string)
    if "\\boxed" in raw_string:
        answer = remove_boxed(last_boxed_only_string(raw_string))
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    return answer


def parse_qa_multiple_answer(string, setting_name):
    if setting_name == "few-shot-CoT":
        string = extract_last_line(string)
    pattern = "\(*([A-F])\)*"
    match = re.findall(pattern, string)
    if match:
        return match
    return []


def post_process(dataset_name, setting_name, prediction):
    if dataset_name in english_cloze_datasets or dataset_name in chinese_cloze_datasets:
        return parse_math_answer(setting_name, prediction)

    if dataset_name in ["jec-qa-kd", "jec-qa-ca", "gaokao-physics"]:
        return parse_qa_multiple_answer(prediction, setting_name)

    # all other datasets are QA problems with single answer
    if 'zero-shot' in setting_name:
        answer = find_first_capital_letter(prediction)
        return answer

    # all other datasets are QA problems with single answer and setting_name are few-shot
    language = "en" if dataset_name in english_qa_datasets else "zh"
    if dataset_name in english_qa_datasets or dataset_name in chinese_qa_datasets:
        return parse_few_shot_qa_single_answer(prediction, setting_name, language)
    else:
        raise ValueError(f"Unsupported dataset name {dataset_name}")


class ResultsForHumanSchema(object):
    def __init__(self,
                 index, problem_input, label,
                 model_input="", model_output="",
                 parse_result="",
                 first_stage_output="", second_stage_input="",
                 is_correct=False
                 ):
        self.index = index
        self.problem_input = problem_input
        self.model_input = model_input
        self.model_output = model_output
        self.parse_result = parse_result
        self.label = label
        self.first_stage_output = first_stage_output
        self.second_stage_input = second_stage_input
        self.is_correct = is_correct

    def to_dict(self):
        return {
            "index": self.index,
            "problem_input": self.problem_input,
            "model_input": self.model_input,
            "model_output": self.model_output,
            "parse_result": self.parse_result,
            "label": self.label,
            "is_correct": self.is_correct,
            "first_stage_output": self.first_stage_output,
            "second_stage_input": self.second_stage_input,
        }

    @staticmethod
    def to_tsv(result_list, path):
        result_json = [item.to_dict() for item in result_list]
        table = pd.json_normalize(result_json)
        table.to_excel(path, index=False)


def convert_to_set(item):
    if isinstance(item, list):
        return set(item)
    if isinstance(item, str):
        return {item}
    if item is None:
        return {}
    raise ValueError("Input can't parse:", item)



def evaluate_single_sample(dataset_name, prediction, label):
    if dataset_name in multi_choice_datasets:
        p = convert_to_set(prediction)
        l = convert_to_set(label)
        return p == l
    elif dataset_name in math_output_datasets:
        return is_equiv(prediction, label)
    else:
        return prediction == label

