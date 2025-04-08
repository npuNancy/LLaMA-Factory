import os
import json
import numpy as np
import setproctitle
from matplotlib import pyplot as plt


def set_proc_title():
    # 获取conda环境名和用户名
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown_env")
    username = os.environ.get("USER", "unknown_user")

    # 设置进程名格式：conda_env@username
    new_proc_name = f"{conda_env}@{username}"
    setproctitle.setproctitle(new_proc_name)


def plot_train_eval(filepath):
    save_path = os.path.splitext(filepath)[0] + "_plot.png"
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    eval_steps = []
    eval_loss = []

    train_steps = []
    train_loss = []

    for line in lines:
        if not line:
            continue
        if '"loss"' in line:
            data = json.loads(line)
            train_steps.append(data["current_steps"])
            train_loss.append(data["loss"])
        if "eval_loss" in line:
            data = json.loads(line)
            eval_steps.append(data["current_steps"])
            eval_loss.append(data["eval_loss"])

    # 绘制1x2图像
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label="train_loss")
    plt.xlabel("train_steps")
    plt.ylabel("train_loss")
    plt.title("train_loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_loss, label="eval_loss")
    plt.xlabel("eval_steps")
    plt.ylabel("eval_loss")
    plt.title("eval_loss")
    plt.legend()

    # 保存图像
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


if __name__ == "__main__":
    filepath = (
        "/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event_zh2en/trainer_log.jsonl"
    )
    plot_train_eval(filepath)
