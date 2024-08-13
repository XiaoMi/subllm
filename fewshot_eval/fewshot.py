# Copyright (C) 2024 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.
#

import argparse
import os
import pandas as pd
import json
import json_lines
import re
import datetime
from agieval_post_process_and_evaluation import post_process, evaluate_single_sample
from modeling.modeling_subllm_inference import SubllmConfig, SubllmModel

eval_args = argparse.ArgumentParser()
eval_args.add_argument("--count_tokens", action="store_true")
eval_args.add_argument("--model_path", type=str, required=True)
eval_args.add_argument("--config_path", type=str, required=True)
eval_args.add_argument("--tokenizer_path", type=str, required=True)
eval_args.add_argument("--result_path", type=str, required=True)

eval_args.add_argument("--task", type=str, required=True)
eval_args.add_argument("--n_shot", type=int, required=True)
eval_args.add_argument("--seed", type=int, required=True)
eval_args.add_argument("--device", type=int, required=False, default=0)
eval_args.add_argument("--batch_size", type=int, required=False, default=4)
eval_args.add_argument("--flash_attn_2", type=bool, default=True)
eval_args.add_argument("--max_length", type=int, default=8192)
eval_args.add_argument("--downsample", action="store_true")
eval_args.add_argument("--print", action="store_true")
eval_args.add_argument("--defined_subject", type=str)
eval_args = eval_args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{eval_args.device}"

from eval_utils import (
    eval_generation_em,
    eval_generation_em_answers,
    exact_match_score,
    exact_match_score_with_multiple_candidates
)

import torch
import json
import numpy as np

from tqdm import tqdm
from transformers.models.llama import LlamaTokenizer
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TASK_DATA_PATH = {
    "agnews": {
        "train": "./eval_data/ag_news_train.jsonl",
        "test": "./eval_data/ag_news_test.jsonl",
    },
    "amazon": {
        "train": "./eval_data/amazon_train.jsonl",
        "test": "./eval_data/amazon_test.jsonl"
    },
    "dbpedia": {
        "train": "./eval_data/dbpedia_train.jsonl",
        "test": "./eval_data/dbpedia_test.jsonl"
    },
    "yelp": {
        "train": "./eval_data/yelp_train.jsonl",
        "test": "./eval_data/yelp_test.jsonl"
    },
    "sst2": {
        "train": "./eval_data/sst2_train.jsonl",
        "test": "./eval_data/sst2_test.jsonl",
    },
    "tweet_hate": {
        "train": "./eval_data/tweet_hate_train.jsonl",
        "test": "./eval_data/tweet_hate_test.jsonl"
    },
    "mmlu":{
        "dev": "./eval_data/MMLU/data/dev",
        "test": "./eval_data/MMLU/data/test",
    },
    "bbh":{
        "train": "./eval_data/BBH/cot-prompts",
        "test": "./eval_data/BBH/bbh",
    },
    "agieval":{
        "zero-shot": "./eval_data/AGIEval/experiment_input/zero-shot/",
        "few-shot": "./eval_data/AGIEval/experiment_input/few-shot/"
    }
}
for task in TASK_DATA_PATH:
    for split in TASK_DATA_PATH[task]:
        if not os.path.exists(TASK_DATA_PATH[task][split]):
            pass
            # raise FileNotFoundError

PRINT = eval_args.print
DOWNSAMPLE = eval_args.downsample


def load_json(path):
    return json.load(open(path, "r"))


def load_jsonl(path, max_line=None):
    with open(path, "r", encoding="utf-8") as fn:
        data = [json.loads(line) for line in fn.readlines()]
        if max_line is not None:
            rng = np.random.RandomState(666)
            data = rng.choice(data, min(max_line, len(data)), replace=False)
    return data


def normalise_pred(pred):
    return pred.strip().split("\n")[0].strip()


class PromptDataset(Dataset):
    def __init__(self, prompt_list, tokenizer, need_tok=True):
        self.data = prompt_list
        self.tokenizer = tokenizer
        self.need_tok = need_tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_dataloader(self, batch_size, max_length):
        def collate_fn(items):
            if self.need_tok:
                batch = [item["prompt"] for item in items]
                return self.tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=max_length, return_token_type_ids=False)
            else:
                return items
        return DataLoader(self, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)


@torch.no_grad()
def generate(model, tokenizer, prompt_list, generation_kwargs,
             max_length, batch_size, task, n_shot, seed):
    predictions = []
    bar = tqdm(total=len(prompt_list), desc=f"{task}-{n_shot}-{seed}")
    prompt_dataset = PromptDataset(prompt_list, tokenizer)
    dataloader = prompt_dataset.get_dataloader(batch_size, max_length)
    
    avg_input_tokens = 0 
    sample_num = 0
    total_input_tokens = 0
    for batch in dataloader:
        model_inputs = batch.to("cuda")
        sample_num += 1
      
        generate_ids = model.generate(**model_inputs, **generation_kwargs)
        pred_ids = generate_ids[:, model_inputs["input_ids"].shape[1]:]
        pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True,return_token_type_ids=False)
        predictions.extend(pred)
        bar.update(len(batch["input_ids"]))

        if PRINT:
            for cur_pred, cur_input in zip(pred, batch):
                print(f"cur_input:{cur_input}, cur_pred:{cur_pred}")
   
    assert len(predictions) == len(prompt_list)
    
    return predictions


def mmlu_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):

    generation_kwargs["max_new_tokens"] = MAX_GEN_TOKENS[task]
    
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(TASK_DATA_PATH[task]["test"])])
    
    choices = ["A", "B", "C", "D"]

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
        
    def format_example(df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
        if k == -1:
            k = df.shape[0]
        for i in range(k):
            prompt += format_example(df, i)
        return prompt 
    
    prompt_list, label_list = [], [] 

    for s_idx, subject in enumerate(subjects): 
        dev_df = pd.read_csv(os.path.join(TASK_DATA_PATH[task]["dev"], subject + "_dev.csv"), header=None)[:n_shot]
        test_df = pd.read_csv(os.path.join(TASK_DATA_PATH[task]["test"], subject + "_test.csv"), header=None)

        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, n_shot)
            
            prompt = train_prompt + prompt_end
            prompt_list.append({"prompt": prompt})

            label = test_df.iloc[i, test_df.shape[1]-1]
            label_list.append(label)
        
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    
    label2str = { x:x for x in choices}
    get_label_call = lambda x: x

    score = metric_acc(predictions, label_list, label2str, get_label_call, normalise_pred=normalise_pred)
    print(f"task: {task}， sample_num: {len(label_list)}, n_shot: {n_shot}, score: {score}")
    return {"score": score, "sample_num": len(label_list)}


def bbh_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, defined_subject):

    generation_kwargs["max_new_tokens"] = MAX_GEN_TOKENS[task]
    
    bbh_multiple_choice_sets = [
        'temporal_sequences',
        'disambiguation_qa',
        'date_understanding',
        'tracking_shuffled_objects_three_objects',
        'penguins_in_a_table',
        'geometric_shapes',
        'snarks',
        'ruin_names',
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_five_objects',
        'logical_deduction_three_objects',
        'hyperbaton',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'movie_recommendation',
        'salient_translation_error_detection',
        'reasoning_about_colored_objects',
    ]
    bbh_free_form_sets = [
        'multistep_arithmetic_two',
        'navigate',
        'dyck_languages',
        'word_sorting',
        'sports_understanding',
        'boolean_expressions',
        'object_counting',
        'formal_fallacies',
        'causal_judgement',
        'web_of_lies',
    ]

    def get_data(train_path, test_path):
        prompt_list, label_list, subject_list = [], [], []
        for file in os.listdir(test_path):
            if not file.startswith(defined_subject):
                continue 
            if not file.endswith(".json"):
                continue 
            subject = file.split('.')[0]
            cot_fpath = os.path.join(train_path, file.replace(".json", ".txt"))
            test_fpath = os.path.join(test_path, file)

            with open(cot_fpath) as fcot, open(test_fpath) as ftest:
                test_data = json.load(ftest)
                cot_prompt = fcot.read().split("-----\n")[1]
                
            for item in test_data["examples"]:
                item.update(
                    {"prompt": f"Follow the given examples and answer the question.\n{cot_prompt}\n\nQ:{item.get('input')}\nA: Let's think step by step.", 'subject': subject}
                )
                prompt_list.append(item)
                
                label = get_label(item)
                label_list.append(label)
                subject_list.append(subject)
                
        return prompt_list, label_list, subject_list

    def get_label(example):
        subject = example["subject"]
        if subject in bbh_multiple_choice_sets:
            return bbh_mcq_postprocess(example["target"])
        return example["target"]
        
    def bbh_mcq_postprocess(text):
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        match = re.search(r'\(([A-Z])\)*', ans)
        if match:
            return match.group(1)
        match = re.search(r'([A-Z])', ans)
        if match:
            return match.group(1)
        return ans

    def bbh_freeform_postprocess(text):
        ans = text
        ans_line = ans.split('answer is ')
        if len(ans_line) != 1:
            ans = ans_line[1].strip()
        ans = ans.split('\n')[0]
        if ans.endswith('.'):
            ans = ans[:-1]
        return ans

    prompt_list, label_list, subject_list = get_data(TASK_DATA_PATH[task]["train"], TASK_DATA_PATH[task]["test"])
    
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    
    now = datetime.datetime.now()
    formatted_time = now.strftime(f"%Y-%m-%d_%H-%M-%S")
    if len(model.decoder.downsamplers) > 0:
        filename = f"bbh_subllm_output_{formatted_time}_{defined_subject}.txt"
    else:
        filename = f"bbh_llama_output_{formatted_time}_{defined_subject}.txt"
    fout = open(filename, 'w')
    pred_proc_list = []
    for pred, label, subject in zip(predictions, label_list, subject_list):
        if subject in bbh_multiple_choice_sets:
            pred = bbh_mcq_postprocess(pred)
        pred = bbh_freeform_postprocess(pred)
        pred_proc_list.append(pred)
        fout.write(pred+"\t"+label+"\t"+subject+"\n")
    fout.close()
    get_label_call = lambda x: x
    
    score = metric_acc(pred_proc_list, label_list, None, get_label_call, normalise_pred=None)
    print(f"task: {task}, sample_num: {len(label_list)}, n_shot: {n_shot}, score: {score}, subject:{defined_subject}")
    return {"score": score, "sample_num": len(label_list), "subject": defined_subject}


def agieval_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, defined_subject):
    generation_kwargs["max_new_tokens"] = MAX_GEN_TOKENS[task]
    
    def get_data(data_dir):
        prompt_list, label_list, dataset_name_list = [], [], []
        for file in os.listdir(data_dir):
            if not file.endswith(".jsonl") or not file.startswith(defined_subject):
                continue 
            with open(os.path.join(data_dir, file), encoding='utf8') as fh:
                
                for line in fh:
                    if line is None:
                        continue
                    sample = json.loads(line)
                    prompt_list.append({"prompt": sample["context"]})
                    label_list.append(sample["label"])
                    dataset_name_list.append(file.split(".")[0])
        return prompt_list, label_list, dataset_name_list
    if n_shot > 0:
        setting_name = "few-shot"
        # raise NotImplementedError
    else:
        setting_name = "zero-shot"

    prompt_list, label_list, dataset_name_list = get_data(TASK_DATA_PATH[task][setting_name]) 
                    
       
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    
    processed_pred_list = []
    for dataset_name, pred in zip(dataset_name_list, predictions):
        proc_pred = post_process(dataset_name, setting_name, pred)
        processed_pred_list.append(proc_pred)
   
    correct_numer = 0
    for i in range(len(processed_pred_list)):
        if evaluate_single_sample(dataset_name_list[i], processed_pred_list[i], label_list[i]):
            correct_numer += 1
    accuracy = correct_numer*1.0 / len(label_list)
    score = accuracy

    print(f"task: {task}, sample_num: {len(label_list)}, n_shot: {n_shot}, score: {score}, subject: defined_subject")
    return {"score": score, "sample_num": len(label_list), "subject": defined_subject}


def get_sampled_demonstrations(train_data, n_shot, seed):
    rng = np.random.RandomState(seed)
    rng.shuffle(train_data)
    return train_data[:n_shot]


def get_agnews_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Article: {item['text'].strip()} Category: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Article: {input_example['text'].strip()} Category:"
    return prompt


def agnews_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 4
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    label2str = {0: "world", 1: "sports", 2: "business", 3: "science"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_agnews_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    print(f"task: {task}， n_shot: {n_shot}, score: {score}")
    return {"score": score}


def get_amazon_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()} Sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()} Sentiment:"
    return prompt


def amazon_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["amazon"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["amazon"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    label2str = {0: "negative", 1: "positive"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_amazon_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_dbpedia_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Title: {item['title'].strip()}. Content: {item['content'].strip()}\nCategory: {label2str[item['label']]}\n\n"
    prompt = prompt + f"Title: {input_example['title'].strip()}. Content: {input_example['content'].strip()}\nCategory:"
    return prompt


def dbpedia_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    names = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation",
             "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork", ]
    generation_kwargs["max_new_tokens"] = 8
    train_data = load_jsonl(TASK_DATA_PATH["dbpedia"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["dbpedia"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 10000, replace=False)
    label2str = {idx: name for idx, name in enumerate(names)}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_dbpedia_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_yelp_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"text: {item['text'].strip()} sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"text: {input_example['text'].strip()} sentiment:"
    return prompt


def yelp_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH["yelp"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["yelp"]["test"])
    if DOWNSAMPLE:
        sample_test_rng = np.random.RandomState(666)
        test_data = sample_test_rng.choice(test_data, 5000, replace=False)
    label2str = {0: "negative", 1: "positive"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_yelp_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_sst2_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"text: {item['text'].strip()} sentiment: {label2str[item['label']]}\n\n"
    prompt = prompt + f"text: {input_example['text'].strip()} sentiment:"
    return prompt


def sst2_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 3
    train_data = load_jsonl(TASK_DATA_PATH[task]["train"])
    test_data = load_jsonl(TASK_DATA_PATH[task]["test"])
    label2str = {0: "positive", 1: "negative"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_sst2_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def get_tweet_hate_prompt(input_example, demonstrations, label2str):
    prompt = ""
    for item in demonstrations:
        prompt = prompt + f"Text: {item['text'].strip()}\nLabel: {label2str[item['label']]}\n"
    prompt = prompt + f"Text: {input_example['text'].strip()} Label:"
    return prompt


def tweet_hate_evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size):
    generation_kwargs["max_new_tokens"] = 6
    train_data = load_jsonl(TASK_DATA_PATH["tweet_hate"]["train"])
    test_data = load_jsonl(TASK_DATA_PATH["tweet_hate"]["test"])
    label2str = {0: "Non-hate", 1: "Hate"}
    get_label_call = lambda x: x["label"]
    demonstrations = get_sampled_demonstrations(train_data, n_shot, seed)
    prompt_list = []
    for idx in range(len(test_data)):
        prompt_list.append({"prompt": get_tweet_hate_prompt(test_data[idx], demonstrations, label2str)})
    predictions = generate(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed)
    score = metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=normalise_pred)
    return {"score": score}


def process_ctx(ctx):
    # ctx = ctx.strip()
    # if ctx[-1] not in [".", "!", "?"]:
    #     ctx = ctx + "."
    # return ctx
    return ctx


def metric_acc(predictions, test_data, label2str, get_label_call, normalise_pred=None, normalise_target=None):
    correct_cnt = 0
    for pred, item in zip(predictions, test_data):
        if normalise_pred is not None:
            pred = normalise_pred(pred)
        if label2str is not None:
            target = label2str[get_label_call(item)]
        else:
            target = item
        if normalise_target is not None:
            target = normalise_target(target)
        # print(f'pred: {pred}, \t target:{target}')
        if target == pred:
            correct_cnt += 1
    acc = correct_cnt / len(test_data) * 100
    return acc



eval_callables = {
    "nq": cbqa_evaluation,
    "tq": cbqa_evaluation,
    "wq": cbqa_evaluation,
    "sst2": sst2_evaluation,
    "agnews": agnews_evaluation,
    "nq_obqa": nq_obqa_evaluation,
    "hotpotqa": hotpotqa_evaluation,
    "amazon": amazon_evaluation,
    "dbpedia": dbpedia_evaluation,
    "yelp": yelp_evaluation,
    "tweet_hate": tweet_hate_evaluation,
    "tweet_offensive": tweet_offensive_evaluation,
    "squad": squad_evaluation,
    "tq_obqa": tq_obqa_evaluation,
    "mmlu": mmlu_evaluation,
    "bbh": bbh_evaluation,
    "agieval": agieval_evaluation,
}

MAX_GEN_TOKENS = {
    "mmlu": 3,
    "bbh": 1024,
    "agieval": 60, 
}

def main():
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "min_length": 1,
        "eos_token_id": 2,
        "use_cache": True,
    }
    results = {
        "task": eval_args.task,
        "n_shot": eval_args.n_shot,
        "seed": eval_args.seed,
        "model": eval_args.model_path
    }
    
    cfg = SubllmConfig().from_json_file(eval_args.config_path)
    cfg.device = torch.cuda.current_device()
    model = SubllmModel(cfg)
    model = model.to(torch.bfloat16).eval()
    
    with open(eval_args.model_path, 'rb') as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.cuda()
    
    print(f'load model done')

    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_path, truncation_side='left', padding_size='left',return_token_type_ids=False,)
    tokenizer.pad_token_id = tokenizer.bos_token_id # TODO
    evaluation = eval_callables[eval_args.task]

    if eval_args.task in ["bbh", "agieval"]:
        score = evaluation(model, tokenizer, generation_kwargs, eval_args.task, eval_args.n_shot, eval_args.seed, eval_args.max_length - MAX_GEN_TOKENS[eval_args.task], eval_args.batch_size, eval_args.defined_subject)
    else:
        score = evaluation(model, tokenizer, generation_kwargs, eval_args.task, eval_args.n_shot, eval_args.seed, eval_args.max_length - MAX_GEN_TOKENS[eval_args.task], eval_args.batch_size)
    results.update(score)
    results = json.dumps(results)
    logger.info(results)
    with open(eval_args.result_path, "a") as fn:
        fn.write(results + "\n")


if __name__ == '__main__':
    main()
