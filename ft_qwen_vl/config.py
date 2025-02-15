from dataclasses import dataclass, field
from typing import Optional

import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None)
    min_pixels: Optional[int] = field(default=32*28*28)
    max_pixels: Optional[int] = field(default=64*28*28)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    run_name: Optional[str] = field(default="debug_ft_qvl")
    num_train_epochs: Optional[int] = field(default=1)
    batch_size: Optional[int] = field(default=2)
    learning_rate: Optional[float] = field(default=1e-5)
    training_args: Optional[float] = field(default=1e-5)
    
