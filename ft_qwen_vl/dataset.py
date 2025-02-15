import json
import torch
from torch.utils.data import Dataset, DataLoader

# from qwen_vl_utils import process_vision_info
from ft_qwen_vl.utils.vision_process import process_vision_info
from ft_qwen_vl.utils.common import find_assistant_content_sublist_indexes
from ft_qwen_vl.utils.logger import get_custom_logger
GLOG = get_custom_logger()


class ToyDataSet(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, processor):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list, processor.tokenizer):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels = torch.tensor(labels_list, dtype=torch.int64)
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels
    }
