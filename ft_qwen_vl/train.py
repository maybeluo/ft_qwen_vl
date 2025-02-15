import os
os.environ["WANDB_PROJECT"] = "ft_qwen_vl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json

import torch
from transformers import HfArgumentParser, set_seed
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ft_qwen_vl.config import ModelArguments, DataArguments, TrainingArguments
from ft_qwen_vl.trainer import Qwen2_5_VLTrainer
from ft_qwen_vl.utils.logger import get_custom_logger
GLOG = get_custom_logger()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    set_seed(training_args.seed)
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    GLOG.info(f"local_rank: {local_rank}")
    
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if local_rank == -1 else {"": local_rank},
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, 
        min_pixels=data_args.min_pixels, 
        max_pixels=data_args.max_pixels,
        padding_side="left"
    )
    
    special_tokens_dict = {# 添加特殊token
        'additional_special_tokens': ['<|token_marker|>']
    }
    num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
    GLOG.info(f"Added {num_added_toks} special tokens")
    model.resize_token_embeddings(len(processor.tokenizer))
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    trainer = Qwen2_5_VLTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_args=data_args,
        optimizers=(optimizer, None),  # 优化器，第二个参数是scheduler
    )
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # 保存最终模型
    if local_rank in [-1, 0]:
        final_checkpoint_folder = os.path.join(training_args.output_dir, "final")
        trainer.save_model(final_checkpoint_folder)
        processor.save_pretrained(final_checkpoint_folder)
        GLOG.info(f"Final model saved to: {final_checkpoint_folder}")
        
        # 测试生成
        GLOG.info("Testing generation...")
        with open(data_args.data_path, "r") as fr:
            test_data = json.load(fr)
        for dic in test_data:
            messages = dic["messages"]
            output = trainer.generate(messages)
            GLOG.info(f"Input: {messages}")
            GLOG.info(f"Output: {output}")
            GLOG.info("-" * 50)

if __name__ == "__main__":
    main()