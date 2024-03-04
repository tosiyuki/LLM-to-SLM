import pathlib
from typing import Dict

import torch
import transformers

from llm_to_slm.model.llm_to_slm_gpt2 import Llm2SlmGpt2ForCausalLM
from llm_to_slm.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llm_to_slm.train.llm_to_slm_trainer import LlmToSlmTrainer
from llm_to_slm.train.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset


def make_supervised_data_module(
    slm_tokenizer: transformers.PreTrainedTokenizer,
    llm_tokenizer: transformers.PreTrainedTokenizer,
    data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    bnb_model_from_pretrained_args = {}

    if model_args.base_model == "gpt2":
        model = Llm2SlmGpt2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    else:
        print(f"{model_args.base_model} is not found")
        exit(-1)

    model.config.use_cache = False
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    slm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_encoder,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    slm_tokenizer.pad_token = slm_tokenizer.unk_token
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    model.get_model().initialize_llm_modules(
        model_name=model_args.llm_encoder,
        projector_type=model_args.projector_type,
        slm_hidden_size=model.config.hidden_size
    )
    model.get_llm_encoder().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    model.config.tokenizer_padding_side = slm_tokenizer.padding_side
    model.config.tokenizer_model_max_length = training_args.model_max_length

    data_module = make_supervised_data_module(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        data_args=data_args
    )
    trainer = LlmToSlmTrainer(
        model=model,
        tokenizer=slm_tokenizer,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )


if __name__ == '__main__':
    train()