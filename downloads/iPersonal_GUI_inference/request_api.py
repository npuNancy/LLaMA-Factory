import re
import os
import time
import json
import base64
import random
import asyncio
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from aiolimiter import AsyncLimiter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def create_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1000000,
    top_p: float = 1.0,
    temperature: float = 1.0,
    presence_penalty: float = 1.0,
    stream: bool = False,
    n: int = 1,
) -> Any:
    """
    Creates a chat completion request.
    创建聊天请求。

    Parameters:
    - api_key (str): API key for authentication.
    - base_url (str): The base URL for the API endpoint.
    - model (str): 要使用的模型的 ID.
    - messages (List[Dict[str, Any]]): 消息列表 [{"role": "system", "content": "You are a helpful assistant."}].
    - max_tokens (int, optional): 在聊天补全中生成的最大标记数。输入标记和生成标记的总长度受模型的上下文长度限制。
    - top_p (float, optional): Controls nucleus sampling. Default is 1.0.
    - temperature (float, optional): 采样温度，介于 0 和 2 之间. Default is 1.0.
    - presence_penalty (float, optional): 种替代温度采样的方法，称为核采样，其中模型考虑具有 top_p 概率质量的标记的结果。所以 0.1 意味着只考虑构成前 10% 概率质量的标记。 我们通常建议改变这个或temperature但不是两者。. Default is 1.0.
    - stream (bool, optional): 默认为 false 如果设置,则像在 ChatGPT 中一样会发送部分消息增量。
    - n (int): 默认为 1,为每个输入消息生成多少个聊天补全选择。
    更多参数参考: https://openai.apifox.cn/api-67883981
    Returns:
    - Any: The response from the chat completion API.
    """

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            timeout=60,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=presence_penalty,
            top_p=top_p,
        )
        if response:
            return response.choices[0].message.content.strip()
        else:
            print(f"OpenAI API 返回空: {response}")
            return None
    except Exception as e:
        print(f"OpenAI API 错误: {e}")
        raise e


@dataclass
class AsyncLLMAPI:
    """
    大模型API的调用类
    """

    ## ChatOpenAI 参数
    base_url: str
    api_key: str  # 每个API的key不一样
    model: str = "gpt-3.5-turbo"

    temperature: float = 1.0
    max_tokens: int = 1000000
    timeout: int = None
    max_retries: int = 1

    ## AsyncLLMAPI 参数
    uid: int = 0
    num_per_second: int = 6  # 限速每秒调用6次
    llm: ChatOpenAI = field(init=False)  # 自动创建的对象，不需要用户传入
    disable_tqdm: bool = False

    def __post_init__(self):
        self.cnt = 0  # 统计每个API被调用了多少次
        # 初始化 llm 对象
        self.llm = self.create_llm()
        # 创建限速器，每秒最多发出 num_per_second 个请求
        self.limiter = AsyncLimiter(self.num_per_second, 1)

    def create_llm(self):
        # 创建 llm 对象
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=2,
        )

    async def __call__(self, messages):
        # 异步协程 限速
        # print(f"API {self.uid} 调用次数: {self.cnt}")
        self.cnt += 1
        async with self.limiter:
            return await self.llm.agenerate([messages])

    @staticmethod
    async def _run_task_with_progress(task, pbar):
        """包装任务以更新进度条"""
        result = await task
        pbar.update(1)
        return result

    @staticmethod
    def async_run(
        llms: List["AsyncLLMAPI"],
        messages_list: List[str],
        keyword: str = "",  # 文件导出名
        output_dir: str = "output",
        chunk_size=10,  # 每次请求多少条数据
    ):

        async def _func(llms, messages_list_chunk):
            """
            异步请求处理一小块数据
            """
            # 调用第 k 个 AsyncLLMAPI 对象, 执行 __call__
            results = [llms[i % len(llms)](messages) for i, messages in enumerate(messages_list_chunk)]
            # 使用 tqdm 创建一个进度条
            with tqdm(total=len(results), disable=True) as pbar:
                # 将多个协程任务交由 asyncio.gather 并发执行
                results = await asyncio.gather(*[AsyncLLMAPI._run_task_with_progress(task, pbar) for task in results])
            return results

        idx = 0
        result_json = []
        tmp_file_list = []
        pbar_total = tqdm(total=len(messages_list))
        # 每完成一块数据的请求则把该块数据保存到csv文件中
        while idx < len(messages_list):
            file = f"{idx}_{keyword}.json"
            file_dir = os.path.join(output_dir, file)
            tmp_file_list.append(file_dir)

            if os.path.exists(file_dir):
                # 用于断点续传
                print(f"{file_dir} already exist! Just skip.")
                with open(file_dir, "r") as f:
                    tmp_json = f.readlines()
            else:
                tmp_data = messages_list[idx : idx + chunk_size]

                loop = asyncio.get_event_loop()
                tmp_result = loop.run_until_complete(_func(llms=llms, messages_list_chunk=tmp_data))
                tmp_result = [item.generations[0][0].text for item in tmp_result]
                tmp_json = [
                    {
                        "query": tmp_data[i],
                        "predict": tmp_result[i],
                    }
                    for i in range(len(tmp_data))
                ]

                # 如果文件夹不存在，则创建
                if not os.path.exists(tmp_folder := os.path.dirname(file_dir)):
                    os.makedirs(tmp_folder)

                # 保存部分内容
                with open(file_dir, "w") as f:
                    json.dump(tmp_json, f, indent=4, ensure_ascii=False)

            ## 将 tmp_json 添加到 result_json 中
            result_json.extend(tmp_json)
            idx += chunk_size
            pbar_total.update(chunk_size)

        # 保存完整的
        with open(os.path.join(output_dir, f"all_{keyword}.json"), "w") as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

        # 删除临时文件, 确保临时文件目录存在
        for file in tmp_file_list:
            if os.path.exists(file):
                os.remove(file)

        return result_json


def gpt_4o():
    api_key = "sk-rxxtMWBqJgFsJLQbDVCl5kSoEXa68OBYJSxNpWMm696yfSbx"
    base_url = "https://xiaoai.plus/v1"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
        {"role": "user", "content": "日本的首都是哪？"},
    ]
    res = create_chat_completion(api_key, base_url, "gpt-4o", messages)
    print(res)


def main(dataset, output_dir, model="gpt-3.5-turbo", chunk_size=10):

    with open(dataset, "r") as f:
        json_data = json.load(f)

    ## 将json_data转换为 messages_list
    # messages_list = [
    #     [SystemMessage(content=message["messages"][0]["content"]),HumanMessage(content=message["messages"][1]["content"])]
    #     for message in json_data
    # ]

    ## 将json_data转换为 messages_list
    messages_list = [message["messages"][:2] for message in json_data]

    llms = [
        AsyncLLMAPI(
            model=model,
            base_url="https://xiaoai.plus/v1",
            api_key="sk-rxxtMWBqJgFsJLQbDVCl5kSoEXa68OBYJSxNpWMm696yfSbx",
            uid=1,
        )
    ]

    result_json = AsyncLLMAPI.async_run(
        llms, messages_list, keyword="tmp", output_dir=output_dir, chunk_size=chunk_size
    )

    result = []
    print(f"{len(messages_list)=}")
    print(f"{len(result_json)=}")
    for idx, message in enumerate(json_data):
        result.append(
            {
                "system": message["messages"][0]["content"],
                "user": message["messages"][1]["content"],
                "label": message["messages"][2]["content"],
                "predict": result_json[idx]["predict"],
            }
        )
    with open(os.path.join(output_dir, "generated_predictions.jsonl"), "w") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    dir = "/data4/yanxiaokai/LLaMA-Factory/data/iPersonal-GUI/stage3/honor_datasets/sft_100_20way/test_20way_read.json"
    dir = "/data4/yanxiaokai/LLaMA-Factory/data/iPersonal-GUI/stage3/honor_datasets/sft_100_20way/test_20way.json"
    dir = "/data4/yanxiaokai/LLaMA-Factory/data/iPersonal-GUI/stage3/honor_datasets/sft_100_20way/test_20way_no_null.json"
    model = "gpt-4o"
    output_dir = f"saves/qwen25vl_7B_stage3/lora/predict_20way_sft_{model}"
    main(dir, output_dir, model=model, chunk_size=100)
