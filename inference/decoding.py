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

from transformers.generation import RepetitionPenaltyLogitsProcessor, MinLengthLogitsProcessor, LogitsProcessorList,\
    LogitsWarper, TopKLogitsWarper, NoRepeatNGramLogitsProcessor, ExponentialDecayLengthPenalty


def get_logits_processor(
        eos_token_id = None,
        min_length = None,
        repet_penalty = None,
        topk = None,
        no_repeat_ngram_size = None,
        exp_decay_length_penalty = None,
        min_tokens_to_keep = 1,
        prompt_length = 0
):
    logits_processor = LogitsProcessorList()
    # processor
    if repet_penalty is not None:
        logits_processor.append(
            RepetitionPenaltyLogitsProcessor(
                penalty=repet_penalty
            )
        )
    if no_repeat_ngram_size is not None:
        logits_processor.append(
            NoRepeatNGramLogitsProcessor(
                ngram_size=no_repeat_ngram_size
            )
        )
    if min_length is not None:
        logits_processor.append(
            MinLengthLogitsProcessor(min_length= min_length, eos_token_id=eos_token_id)
        )
    if exp_decay_length_penalty is not None:
        logits_processor.append(
            ExponentialDecayLengthPenalty(
                exponential_decay_length_penalty=exp_decay_length_penalty,
                eos_token_id=eos_token_id,
                input_ids_seq_length=prompt_length
            )
        )
    # warper
    if topk is not None:
        logits_processor.append(
            TopKLogitsWarper(
                top_k=topk, min_tokens_to_keep=min_tokens_to_keep
            )
        )
    return logits_processor