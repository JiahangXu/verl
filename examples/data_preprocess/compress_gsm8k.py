# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/compressed_gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--compress_ratio', default="045")

    args = parser.parse_args()

    token_skip_train = datasets.load_dataset("parquet", data_files=os.path.join(f"/mnt/teamdrive/data/ThinkingLingua/gsm8k_think_token_{args.compress_ratio}.parquet"))["train"]
    test_dataset = datasets.load_dataset('openai/gsm8k', 'main')['test']
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}. "

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = instruction_following + question_raw

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": "openai/gsm8k",
                "prompt": [{
                    'role': 'system',
                    'content': 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.'
                },{
                    'role': 'user',
                    'content': question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn
    

    # add a row to each data item that represents a unique id
    def make_map_fn_ts(split):

        def process_fn(example, idx):
            solution = example.pop('answer')
            messages = example.pop("messages")
            example.pop("ratio")
            data = {
                "data_source": "openai/gsm8k",
                "prompt": messages[:-1],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = token_skip_train.map(function=make_map_fn_ts('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # os.makedirs(args.compress_ratio, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
