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

import logging, argparse
from typing import List, Optional
import torch
import torch.cuda as cuda
from queue import Queue
from typing import Optional
from decoding import get_logits_processor
import numpy as np
import os 
from copy import deepcopy
from modeling.modeling_subllm_inference import SubllmConfig, SubllmModel
logger = logging.getLogger(__name__)

def hum_convert(value):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024.0
    for i in range(len(units)):
        if (value / size) < 1:
            return "%.2f%s" % (value, units[i])
        value = value / size

class BaseStreamer:
    def put(self, value):
        raise NotImplementedError()
    def end(self):
        raise NotImplementedError()
    
class TextStreamer(BaseStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = False, end_token = '[EOS]', **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.end_token = end_token
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        value = value.tolist()
        if isinstance(value, int):
            value = [value]
        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value)
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif text.endswith(self.end_token):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""
        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


class TextIteratorStreamer(TextStreamer):
    def __init__(
        self, tokenizer, skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

def recaluate_fisrt_token_time(list_of_dicts):
    dict_A = min(list_of_dicts, key=lambda x: int(list(x.keys())[0]))

    for d in list_of_dicts:
        if d is not dict_A:
            first_key = list(d.keys())[0]
            sum_values = sum(value for key, value in dict_A.items() if int(key) < int(first_key))
            d[first_key] += sum_values

    return list_of_dicts

class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device
    def get_tokens(self, prompts):
        return torch.Tensor([self.tokenizer.encode(x, bos=False, eos=False) for x in prompts]).cuda(self.cuda_id)
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k = None,
        min_length = None,
        early_stop = False,
        no_repeat_ngram_size = None,
        repet_penalty = None,
        cache=False,
        streamer: Optional[List[TextIteratorStreamer]] = None,
        token_time = None,
        last_time = None,
        max_input_len = 0,
    ) -> List[str]:
        bsz = len(prompts)
        if streamer is not None:
            assert bsz == len(streamer), "bsz 必须要和 Streamer 的数量相等"
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]

        for i,p in enumerate(prompt_tokens):
            if len(p) > max_input_len:
                prompt_tokens[i] = p[ : max_input_len]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        per_prompt_start = [len(t) for t in prompt_tokens]
        total_len = min(self.model.config.tokens_per_sample, max_gen_len + max_prompt_size)
        print(f'total len : {total_len}')
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_token_id).long().to(self.device)
        
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        input_text_mask = tokens != self.tokenizer.pad_token_type_id
        
        start_pos = min_prompt_size
        prev_pos = 0
        stop_record = {k:0 for k in range(bsz)}

        logits_processor = get_logits_processor(topk=top_k,
                                                min_length=min_length,
                                                repet_penalty=repet_penalty,
                                                no_repeat_ngram_size=no_repeat_ngram_size)

        past_key_values = None
        print("==========================")
        last_time = time.time() # 
        for cur_pos in range(start_pos, total_len):
            # print(f'cur pos: {cur_pos}')
            input_text_mask = tokens != self.tokenizer.pad_token_type_id
            output = self.model(tokens[:, prev_pos:cur_pos], use_cache = cache, past_key_values = past_key_values, attention_mask=input_text_mask)
            logits = output.logits
            past_key_values = output.past_key_values
          
            logits = logits[:, -1, :]
            logits = logits_processor(tokens[:, :cur_pos], logits)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            if token_time is not None:
                torch.cuda.current_stream().synchronize()
                current_time = time.time()  # Current time
                time_diff = (current_time - last_time) * 1000  # Time difference in milliseconds
                # token_time[cur_pos+1] = time_diff
                last_time = current_time  # Update the last time to the current time

            next_token = next_token.reshape(-1)
            prev_pos = cur_pos
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if early_stop and cur_pos > max_prompt_size:
                if torch.all(next_token) == self.tokenizer.eos_token_id or torch.all(next_token) == 0:
                    break
                for i in range(len(next_token)):
                    cur_token = next_token[i].item()
                    if cur_token in [self.tokenizer.eos_token_id] and stop_record[i] == 0:
                        stop_record[i] = 1
                        if streamer is not None:
                            # for bi in range(bsz):
                            streamer[i].end()
                stop_num = sum([v for k,v in stop_record.items()])
                if stop_num == bsz:
                    break
            # streamers
            if streamer is not None:
                for bi in range(bsz):
                    if cur_pos >= per_prompt_start[bi] and stop_record[bi] == 0:
                        streamer[bi].put(next_token[bi].cpu())
                        if token_time is not None:
                            token_time[bi][cur_pos+1] = time_diff


        if streamer is not None:
            for k,v in stop_record.items():
                if v==0:
                    streamer[k].end()
            return None, None
        else:
            decoded = []
            token_num = []
            for i, t in enumerate(tokens.tolist()):
                # cut to max gen len
                t = t[: len(prompt_tokens[i]) + max_gen_len]
                # cut to eos tok if any
                ptp = prompt_tokens[i]
                if self.tokenizer.eos_token_id in t:
                    e = t[len(ptp):].index(self.tokenizer.eos_token_id)
                    t = t[len(ptp): len(ptp) + e]
                    token_num.append(len(t))
                elif 60002 in t:
                    e = t[len(ptp):].index(60002)
                    t = t[len(ptp): len(ptp) + e]
                    token_num.append(len(t))
                else:
                    t = t[len(ptp):]
                    token_num.append(len(t))
                dout = self.tokenizer.decode(t)
                decoded.append(dout)
            return decoded, token_num

    def skip_and_stop(self, x, end_token):
        if "<|endoftext|>" in x:
            x = x.replace("<|endoftext|>", "")
        if end_token in x:
            e = x.index(end_token)
            x = x[:e]
        return x


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def get_inference_args():
    args = argparse.ArgumentParser("Inference Arguments")
    args.add_argument("--model_path", type=str, default="/path/to/mimodel1d3b")
    args.add_argument("--model_config_path", type=str, default="/path/to/config")
    args.add_argument("--tokenizer_path", type=str, default="/path/to/tokenizer")
    args.add_argument("--data_path", type=str, default="/path/to/data")
    args.add_argument("--save_path", type=str, default="/path/to/data")
    args.add_argument("--gradient_checkpointing", action="store_true")
    args.add_argument("--fp16", action="store_true")
    args.add_argument("--bf16", action="store_true")
    args.add_argument("--use_cache", action="store_true")
    args.add_argument("--max_length", type=int, default=128)
    args.add_argument("--top_p", type=float, default=0.9)
    args.add_argument("--temperature", type=float, default=0.7)
    args.add_argument("--repetition_penalty", type=float, default=None)
    args.add_argument("--min_length", type=int, default=None)
    args.add_argument("--no_repeat_ngram_size", type=int, default=None)
    args.add_argument("--timing", action="store_true")
    args.add_argument("--num_avg_times", type=int, default=10)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--max_input_len", type=int, default=1)
    return args.parse_args()

if __name__=="__main__":
    """
    test in transformers == 4.28.1
    """
    import time
    args = get_inference_args()
    from transformers import AutoModel, AutoTokenizer
    # load model
    config = SubllmConfig.from_json_file(args.model_config_path)
    model = SubllmModel(config)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    # load tokenizer     
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, truncation_side='left', padding_size='left', trust_remote_code=True)
    tokenizer.pad_token_id = 0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    if args.fp16:
        model = model.half()
    if args.bf16:
        print('using bfloat16')
        model = model.bfloat16()
    model.config.use_cache = args.use_cache
    model = model.cuda()
    model.eval()
    print("-" * 66)
    
    query_list = []
    if args.data_path.endswith("txt"):
        if args.batch_size == 1:
            with open(args.data_path, 'r') as f:
                query_list.append(f.read().strip())
        else:
            raise NotImplementedError
    elif args.data_path.endswith("json"):
        import json
        with open(args.data_path, 'r') as f:
            data=json.load(f)
            query_list.append(data['text'])

    gen = Generator(model, tokenizer)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    fout = open(args.save_path, 'w')
    streamer_list = []
    for batch_index in range(args.batch_size):
        streamer_list.append(TextIteratorStreamer(tokenizer))

    
    # first prompt for warmup
    gen.generate(
        prompts=[x for x in query_list],
        max_gen_len=args.max_length - args.max_input_len,
        temperature=args.temperature,
        top_p=args.top_p,
        min_length=args.min_length,
        early_stop=True,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repet_penalty=args.repetition_penalty,
        cache=args.use_cache,
        streamer=streamer_list,
        max_input_len=args.max_input_len,
    )
    print("streaming inference:")
    for bsz in range(args.batch_size):
        for s in streamer_list[bsz]:
            print(s, end="", flush=True)
        print("\n", flush=True)
        print("-" * 66, flush=True)
            # time.sleep(0.1)
    
    if args.timing:
        first_token_time = {}
        avg_token_time_array = {}
        for index in range(args.num_avg_times):
            cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            initial_memory = cuda.memory_allocated()
            
            token_time = [{} for i in range(len(query_list))]  
            gen.generate(
                prompts=[x for x in query_list],
                max_gen_len=args.max_length - args.max_input_len,
                temperature=args.temperature,
                top_p=args.top_p,
                min_length=args.min_length,
                early_stop=True,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repet_penalty=args.repetition_penalty,
                cache=args.use_cache,
                streamer=streamer_list,
                token_time=token_time,
                last_time=None,
                max_input_len=args.max_input_len
            )


            print("timing streaming inference:")
            for bsz in range(args.batch_size):
                for s in streamer_list[bsz]:
                    print(s, end="", flush=True, file=fout)

            token_time = recaluate_fisrt_token_time(token_time)
            print("\nToken Generation Time:", flush=True, file=fout)
            for i, val in enumerate(token_time):
                print(f"Sample {i} Token Generation Time (ms): {val}", flush=True, file=fout)


            first_key_values = [list(d.values())[0] for d in token_time]
            first_key_mean = np.mean(first_key_values)
            first_token_time[index] = first_key_mean

            non_first_key_values = [value for d in token_time for key, value in list(d.items())[1:]] # Exclude the first token
            non_first_key_mean = np.mean(non_first_key_values)

            avg_token_time_array[index] = non_first_key_mean
            num_token_avged = len(non_first_key_values)
            print(f"Index {index} Average Token Time (excluding first token) on {num_token_avged} tokens: {non_first_key_mean} ms", file=fout)
            torch.cuda.synchronize()
            peak_memory = cuda.max_memory_allocated()
            used_memory = peak_memory - initial_memory
            print(f'peak_memory - initial_memory = {hum_convert(peak_memory)} - {hum_convert(initial_memory)}, used_memory:{hum_convert(used_memory)}')
            # print(f'peak_memory - initial_memory = {peak_memory} - {initial_memory}, used_memory:{used_memory}')

        avg_values_first_token = list(first_token_time.values())
        first_token_avg_time = np.mean(avg_values_first_token)
        avg_values_token = list(avg_token_time_array.values())
        token_avg_time = np.mean(avg_values_token)
        print(f"Average First Token Time : {first_token_avg_time} ms", file=fout)
        print(f"Average Token Time (excluding first token): {token_avg_time} ms", file=fout)


    fout.close()