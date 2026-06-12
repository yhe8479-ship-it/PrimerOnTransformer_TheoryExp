# Flexible Transformer Framework Train and Test Demo

## 快速训练全部任务

```bash
python train_all_demos.py
```

这个入口默认每个任务只训练 3 个 epoch，用来快速验证数据流、forward 和 loss 是否能跑通。

## 单独训练某个任务

```bash
python tasks/Encoder-only_tasks/train_text_classification.py --epochs 30
python tasks/Encoder-only_tasks/train_ner.py --epochs 30
python tasks/Decoder-only_tasks/train_decoder_lm.py --epochs 30
python tasks/Decoder-only_tasks/train_dialogue_generation.py --epochs 30
python tasks/Encoder-Decoder_tasks/train_translation.py --epochs 40
python tasks/Encoder-Decoder_tasks/train_summarization.py --epochs 40
```

## 训练产物

训练完成后会在 `run/` 目录下保存：

- `model.pt`
- `tokenizer.json`
- Encoder-Decoder 任务会保存 `src_tokenizer.json` 和 `tgt_tokenizer.json`

## 说明

这是一套教学级小数据 demo。数据量很小，不能追求真实效果，主要用于理解：

- 原始文本如何进入 tokenizer
- 如何 padding 成 batch
- 如何制作输入和标签
- 三种 Transformer 开关如何对应不同任务
- loss 怎么计算
- 权重和 tokenizer 怎么保存

## CPU 运行提示

训练脚本里设置了 `torch.set_num_threads(1)`，这是为了让教学小 demo 在普通 CPU 环境下更稳定，避免某些机器上 PyTorch 多线程调度反而变慢。


## 测试 / 推理

先训练，再测试。

例如：

```bash
python tasks/Encoder-only_tasks/train_text_classification.py --epochs 30
python tasks/Encoder-only_tasks/test_text_classification.py --text "这个电影很好看"
```

6 个任务的测试脚本：

```bash
python tasks/Encoder-only_tasks/test_text_classification.py --text "这个电影很好看"
python tasks/Encoder-only_tasks/test_ner.py --text "小明明天去北京"
python tasks/Decoder-only_tasks/test_decoder_lm.py --prompt "我明天"
python tasks/Decoder-only_tasks/test_dialogue_generation.py --context "你好"
python tasks/Encoder-Decoder_tasks/test_translation.py --text "我明天去北京。"
python tasks/Encoder-Decoder_tasks/test_summarization.py --text "今天北京天气晴朗，很多市民来到公园散步和运动，城市交通整体平稳。"
```

统一测试入口：

```bash
python test_all_demos.py
```

如果还没有训练，测试脚本会提示你先运行对应的 `train_xxx.py`。
