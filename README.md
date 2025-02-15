# Fine-tune Qwen2.5-VL / Qwen2-VL

## Install
clone the repo and install requirements like transformers. To support Qwen2.5-VL, please install transformers by:
```
pip3 install git+https://github.com/huggingface/transformers accelerate
```

To support video input, please install:
```bash
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]
```

## Usage
Prepare your data in the format of Qwen2.5-VL, and put it in the `data` directory, and crate directory `ckpts` to save the model. Then run:
```bash
bash run.sh
```
You can select local mode or distributed mode by comment/uncomment the corresponding lines in `run.sh`.


## Acknowledgement
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [finetune-Qwen2-VL](https://github.com/zhangfaen/finetune-Qwen2-VL/)
