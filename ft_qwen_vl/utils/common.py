import os


def find_assistant_content_sublist_indexes(input_ids, tokenizer):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # Get token IDs for markers
    start_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_marker = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    
    if not start_marker or not end_marker:
        raise ValueError("Could not encode markers properly")
        
    start_marker_len = len(start_marker)
    spans = []
    
    i = 0
    while i < len(input_ids) - start_marker_len:
        # Look for start marker
        if input_ids[i:i + start_marker_len] == start_marker:
            start_idx = i + start_marker_len
            
            # Look for end marker after start position
            for j in range(start_idx, len(input_ids) - len(end_marker) + 1):
                if input_ids[j:j + len(end_marker)] == end_marker:
                    # Include end marker in the span
                    end_idx = j + len(end_marker)
                    spans.append((start_idx, end_idx))
                    i = end_idx - 1
                    break
            else:
                # No end marker found
                raise ValueError(f"Found start marker at {i} but no corresponding end marker")
        i += 1
    
    if not spans:
        raise ValueError("No assistant content spans found")
        
    return spans
