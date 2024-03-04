import copy
import json
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from llm_to_slm.train.arguments_dataclass import DataArguments


def make_prompt(instruction: str, input: str="") -> str:
    if input:
        return f"""### Instruction:
{instruction}

### Input:
{input}

### Response:

"""
    
    return f"""### Instruction:
{instruction}

### Response:

"""


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, 
        data_path: str,
        slm_tokenizer: transformers.PreTrainedTokenizer,
        llm_tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        ignore_index:int =-100
    ):
        super(LazySupervisedDataset, self).__init__()
        
        self.ignore_index = ignore_index
        list_data_dict = json.load(open(data_path, "r"))
        
        print("Formatting inputs...Skip in lazy mode")
        self.slm_tokenizer = slm_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.list_data_dict = [i for i in list_data_dict if i["output"] != ""]
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        print("call dataset length!!")
        length_list = []
        for sample in self.list_data_dict:
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']))
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        prompt = make_prompt(self.list_data_dict[i]["instruction"], self.list_data_dict[i]['input'])
        sources = self.slm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.slm_tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        input_ids = self.slm_tokenizer(
            f"{prompt}{self.list_data_dict[i]['output']}{self.slm_tokenizer.eos_token}",
            return_tensors="pt",
            padding="longest",
            max_length=self.slm_tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        labels = copy.deepcopy(input_ids)
        source_len = sources.shape[1]
        labels[0][:source_len] = self.ignore_index

        llm_input_ids = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=input_ids.shape[1], # fit slm token length
            truncation=True,
        ).input_ids

        data_dict = dict(
            input_ids=input_ids,
            llm_input_ids=llm_input_ids,
            labels=labels
        )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                llm_input_ids=data_dict["llm_input_ids"][0],
                labels=data_dict["labels"][0]
            )

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    slm_tokenizer: transformers.PreTrainedTokenizer
    llm_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, llm_input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "llm_input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.slm_tokenizer.pad_token_id
        )
        llm_input_ids = torch.nn.utils.rnn.pad_sequence(
            llm_input_ids,
            batch_first=True,
            padding_value=self.llm_tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        input_ids = input_ids[:, :self.slm_tokenizer.model_max_length]
        llm_input_ids = llm_input_ids[:, :self.llm_tokenizer.model_max_length]
        labels = labels[:, :self.slm_tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            llm_input_ids=llm_input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.slm_tokenizer.pad_token_id),
        )

        return batch
    