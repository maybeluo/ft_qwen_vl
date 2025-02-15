import os
import torch
from transformers import Trainer
from torch.utils.data import DataLoader
from functools import partial

from ft_qwen_vl.dataset import ToyDataSet, collate_fn
from ft_qwen_vl.utils.vision_process import process_vision_info
from ft_qwen_vl.utils.logger import get_custom_logger
GLOG = get_custom_logger()

class Qwen2_5_VLTrainer(Trainer):
    def __init__(self, model=None, args=None, processor=None, data_args=None, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        self.processor = processor
        self.data_args = data_args

    def train(self, resume_from_checkpoint=None, **kwargs):
        self.model.train()
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        num_items_in_batch = kwargs.get('num_items_in_batch')
        if num_items_in_batch is not None:
            loss = loss * inputs['input_ids'].size(0) / num_items_in_batch
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataset = ToyDataSet(self.data_args.data_path)
        data_collator = partial(collate_fn, processor=self.processor)
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def generate(self, messages):
        self.model.eval()
        with torch.no_grad():
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            # inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            inputs = inputs.to(self.args.device)
            
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return output_text[0] if output_text else ""

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        # 保存模型
        if self.args.save_strategy != "no":
            self.model.save_pretrained(
                checkpoint_folder,
                is_main_process=self.is_world_process_zero(),
                save_function=self.save_model,
            )
            
        # 保存训练状态
        if self.is_world_process_zero():
            torch.save(self.state, os.path.join(checkpoint_folder, "trainer_state.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_folder, "optimizer.pt"))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_folder, "scheduler.pt"))
                
        # 保存processor
        if self.is_world_process_zero():
            self.processor.save_pretrained(checkpoint_folder)
            
        return checkpoint_folder