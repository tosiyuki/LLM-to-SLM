from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="gpt2",
                                      metadata={"help": "support only gpt2"})
    model_name_or_path: Optional[str] = field(default="openai-community/gpt2")
    llm_encoder: Optional[str] = field(default="google-t5/t5-small",
                                       metadata={"help": "support only T5"})
    projector_type: Optional[str] = field(default='mlp2x_relu')


@dataclass
class DataArguments:
    data_path: str = field(default="dataset/alpaca_data.json",
                           metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False) # dataset sampler option

    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    output_dir: str = field(default="./output_llava/checkpoints/t5-small-to-gpt2")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=24000)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=1e-3)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine")
    seed: int = field(default=42)
