"""
统一测试 6 个任务。

先运行训练：
    python train_all_real_data_demos.py

再运行测试：
    python test_all_real_data_demos.py
"""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent

TEST_COMMANDS = [
    ("tasks/Encoder-only_tasks/test_text_classification.py", ["--text", "这个电影很好看"]),
    ("tasks/Encoder-only_tasks/test_ner.py", ["--text", "小明明天去北京"]),
    ("tasks/Decoder-only_tasks/test_decoder_lm.py", ["--prompt", "我明天"]),
    ("tasks/Decoder-only_tasks/test_dialogue_generation.py", ["--context", "你好"]),
    ("tasks/Encoder-Decoder_tasks/test_translation.py", ["--text", "我明天去北京。"]),
    (
        "tasks/Encoder-Decoder_tasks/test_summarization.py",
        ["--text", "今天北京天气晴朗，很多市民来到公园散步和运动，城市交通整体平稳。"],
    ),
]


if __name__ == "__main__":
    for test_file, args in TEST_COMMANDS:
        print("\n" + "=" * 80)
        print("Testing:", test_file)
        print("=" * 80)

        old_argv = sys.argv[:]
        sys.argv = [test_file] + args
        runpy.run_path(str(PROJECT_ROOT / test_file), run_name="__main__")
        sys.argv = old_argv
