"""
人工MAML:
meta-training:
内层循环：用同一个User_i（Task_i）的support数据训练Model，使其成为Model_i
外层循环：1. 用User_i的query数据，让Model_i做预测，Σi的梯度，用于更新Model

"""

import numpy as np


class MAML(object):
    def __init__(self):
        """
        定义参数，实验中用到10-way，10-shot
        """
        # 假设有 USER_NUM 个用户 等价于 USER_NUM 个 Task
        # 共有10个任务
        self.num_tasks = 80

        # 每个任务的数据量: 10-shot, 10-query
        self.n_way = 20
        self.k_shot = 10
        self.k_query = 10

        # 训练的迭代次数
        self.epochs = 3

        # 每个任务训练的迭代次数
        self.inner_epochs = 5

        self.alpha = 0.0001  # lr

        # 外循环的学习率，用来更新meta模型的\theta
        self.beta = 0.0001

        # meta模型初始化的参数
        self.theta = np.random.normal(size=16).reshape(-1, 1)

    def train(self):
        """
        meta-train
        """
        model = "qwen_2.5_vl_7b"
        for epoch in range(self.epochs):
            loss_sum = []
            for i, task in enumerate(range(self.num_tasks)):
                """
                # 遍历每个用户（任务）
                # 每个任务的数据量: 10-shot, 10-query
                每个用户的数据存于一个json文件
                [
                    {},
                    {},
                ]
                """
                support_x, support_y = "", ""  # json文件中读取, 任选 k_shot 个数据
                query_x, query_y = "", ""  # json文件中读取, 任选 k_query 个数据

                # 内层循环：用同一个User_i（Task_i）的support数据训练Model，使其成为Model_i
                """
                用 support_x, support_y 训练 Model, 训 inner_epochs 轮
                训练结果(lora adapter) 存为 Model_i
                """
                # 用 Model_i 做推理.
                query_y_pred = ""  # query_y_pred = Model_i(query_x)

                # 计算loss
                loss = ""  # loss = loss_fn(query_y_pred, query_y)
                loss_sum.append(loss)

            # 外层 用 loss_avg 更新Model
            loss_avg = np.mean(loss_sum)
