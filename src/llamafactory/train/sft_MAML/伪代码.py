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

    @override
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # tr_loss 是一个张量，用于避免TPUs通过 .item() 同步
        tr_loss = torch.tensor(0.0).to(args.device)

        model.zero_grad()  # 清空梯度
        grad_norm: Optional[float] = None

        ## 这是MAML的外层循环
        ## 这是主要的修改之处
        for epoch in range(epochs_trained, num_train_epochs):
            """
            # 遍历每个用户（任务）
            # 每个任务的数据量: N-shot, K-query
            """

            task_losses = []  # 用于存放每个任务的loss(标量), 最后用于更新Model

            ############### 更新的代码 ###########
            ## 这里保存一份当前模型的参数，用于给后面 Task-Specific Model 初始化
            # 保存当前模型参数（元参数）作为备份
            meta_params = copy.deepcopy(model.state_dict())
            # 复制一份 self.optimizer 作为备份
            meta_optimizer = copy.deepcopy(self.optimizer.state_dict())
            # 复制一个 meta_accelerator
            meta_accelerator = copy.deepcopy(self.accelerator)
            ############### 更新的代码 ###########

            for maml_task in range(self.maml_num_tasks):

                # --- 内循环：在支持集上做几步梯度更新 ---
                # 克隆一份模型供内循环使用，防止直接改动 model 参数
                # 在 MAML 内循环中，使用 model 和 self.optimizer 来训练 task-specific model

                self.__get_gpu_memory(f"在第 {epoch} 个Epoch的第 {maml_task} 个任务. Model 初始化前")
                ############### 更新的代码 ###########
                # Task-Specific Model 初始化
                model.load_state_dict(meta_params)
                self.optimizer.load_state_dict(meta_optimizer)
                # 重置优化器状态
                self.optimizer.state = collections.defaultdict(dict)
                # 重置加速器状态
                self.accelerator = meta_accelerator
                ############### 更新的代码 ###########

                ## 数据加载器
                self.support_dataset, self.query_dataset = self.process_MAML_dataset(
                    self.maml_training_dataset_list[maml_task]
                )
                self.train_dataset = self.support_dataset  # 第maml_task个任务的 支持集
                train_dataloader = self.get_train_dataloader()  # 这个函数会读取 self.train_dataset
                epoch_dataloader = train_dataloader

                #################### 当前task的支持集训练 内循环 START #############################
                step = -1
                update_step = -1
                for i in range(self.maml_inner_epochs):
                    epoch_iterator = iter(epoch_dataloader)
                    total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
                    for _ in range(total_updates):
                        update_step += 1

                        # 获取当前批次的样本和批次中的项目数量
                        batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                        for i, inputs in enumerate(batch_samples):
                            step += 1
                            # 我们明确地不想依赖 `accelerator.accumulate` 来进行生成训练
                            context = ()
                            with context():
                                """
                                training_step() 包含模型的正向传播 和 反向更新, 但没有更新参数
                                这一步应该是用 support_x 和 support_y 来训练该Task的模型
                                """
                                tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                            tr_loss = tr_loss + tr_loss_step

                            # 正常应该调用optimizer.zero_grad()，防止梯度累加干扰当前批次的计算
                            # 但梯度累加也可用于模拟大batch训练
                            self.optimizer.step()  # 更新 task-specific 模型参数

                            model.zero_grad()  # 清空梯度
                            self.state.global_step += 1

                #################### 当前task的支持集训练 内循环 END ##################################

                ###################### 计算 Task-Specific Model 的 损失(带梯度) START ####################
                ## 用当前task的 查询集 计算 Model_i 的loss, 并保存（不在这里做反向传播）
                query_loss_list = []
                query_dataset = self.query_dataset  # 查询集
                query_dataloader = self.get_test_dataloader(query_dataset)  # 获取查询集的 dataloader

                epoch_dataloader = query_dataloader
                step = -1
                epoch_iterator = iter(epoch_dataloader)

                total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
                for _ in range(total_updates):
                    # 获取当前批次的样本和批次中的项目数量
                    for i, inputs in enumerate(batch_samples):
                        step += 1

                        context = ()
                        with context():
                            """
                            training_step() 包含模型的正向传播 和 反向更新, 但没有更新参数
                            这一步应该是用 support_x 和 support_y 来训练该Task的模型
                            """
                            query_loss_step = self.training_step_only_forward(model, inputs, num_items_in_batch)

                        query_loss_list.append(query_loss_step)  # 这种方式占用显存太大
                        model.zero_grad()  # 清空梯度

                query_loss = torch.stack(query_loss_list).mean()
                task_losses.append(query_loss)  # 把当前task的loss保存下来
                ###################### 计算 Task-Specific Model 的 损失(带梯度) START ####################

            ################################
            # 完成了1个epoch内所有Task的训练 #
            ################################

            ############## 还原回 Epoch 开始时的 Model 和 Optimizer ##############
            model.load_state_dict(meta_params)
            self.optimizer.load_state_dict(meta_optimizer)
            self.accelerator = meta_accelerator

            ## 对 model 进行梯度更新, 计算所有task的loss的平均值
            avg_loss = torch.stack(task_losses).mean()  # 将task_losses转换为张量,并计算平均值
            loss = avg_loss

            self.accelerator.backward(loss, **kwargs)  # 反向传播
            self.optimizer.step()  # 更新 task-specific 模型参数
            # self.optimizer.zero_grad() # 清空梯度


def fun():
    # 对以一个 epoch
    task_losses = []
    for maml_task in range(self.maml_num_tasks):
        # 1. 获取该任务的支持集和查询集（你可根据需要分割）
        full_dataset = self.maml_train_dataset_list[maml_task]
        support_dataset, query_dataset = torch.utils.data.random_split(
            full_dataset, [int(0.5 * len(full_dataset)), len(full_dataset) - int(0.5 * len(full_dataset))]
        )

        support_loader = DataLoader(support_dataset, batch_size=self.args.per_device_train_batch_size)
        query_loader = DataLoader(query_dataset, batch_size=self.args.per_device_train_batch_size)

        # 2. 拷贝模型和优化器（用于 task-specific 内循环）
        inner_model = copy.deepcopy(self.model)
        inner_model.to(self.args.device)
        inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=1e-3)

        # 3. 内循环：在支持集上训练 inner_model
        inner_model.train()
        for inner_epoch in range(self.maml_inner_epochs):
            for support_inputs in support_loader:
                support_inputs = self._prepare_inputs(support_inputs)
                inner_optimizer.zero_grad()
                support_loss = self.compute_loss(inner_model, support_inputs)
                support_loss.backward()
                inner_optimizer.step()

        # 4. 外循环：用 inner_model 在查询集上计算损失
        query_loss_total = 0.0
        for query_inputs in query_loader:
            query_inputs = self._prepare_inputs(query_inputs)
            inner_model.eval()
            with torch.no_grad():
                query_loss = self.compute_loss(inner_model, query_inputs)
            query_loss_total += query_loss
        meta_loss = query_loss_total / len(query_loader)

        task_losses.append(meta_loss)

    # 5. 对 meta model 做反向传播，用 query_loss 更新 meta parameters
    loss_avg = np.mean(task_losses)
    self.model.train()
    self.optimizer.zero_grad()
    self.accelerator.backward(loss_avg)
    self.optimizer.step()


"""
下面是现在用的伪代码
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim


def MAML_way_1():
    # 1. 生成随机数据（100个样本）
    spt_inputs = [torch.randn(100, 1) * 10 for i in range(10)]  # 输入特征（-10到10的随机数）
    spt_labels = [
        3 * spt_inputs + 2 + torch.randn(100, 1) * 2 for i in range(10)
    ]  # 带噪声的线性关系（真实参数w=3，b=2）

    qry_inputs = [torch.randn(100, 1) * 10 for i in range(10)]  # 输入特征（-10到10的随机数）
    qry_labels = [
        3 * spt_inputs + 2 + torch.randn(100, 1) * 2 for i in range(10)
    ]  # 带噪声的线性关系（真实参数w=3，b=2）

    # 2. 定义单层线性模型
    meta_model = nn.Sequential(nn.Linear(1, 1))  # 输入维度1，输出维度1

    # 3. 定义损失函数和优化器
    meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.01)  # 随机梯度下降
    criterion = nn.MSELoss()  # 均方误差损失

    for epoch in range(num_epoch := 10):

        for task in range(num_task := 2):
            model = copy.deepcopy(meta_model)
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # 用 support set 训练模型
            for i in range(num_inner_steps := 10):
                for spt_input_idx in range(len(spt_inputs)):
                    spt_input = spt_inputs[spt_input_idx]
                    spt_label = spt_labels[spt_input_idx]

                    optimizer.zero_grad()  # 梯度清零
                    outputs = model(spt_input)  # 正向传播
                    support_loss = criterion(outputs, spt_label)  # 计算损失
                    support_loss.backward()  # 反向传播，计算梯度
                    optimizer.step()  # 更新参数

            # 用 query set 测试模型
            for qry_input_idx in range(len(qry_inputs)):
                qry_input = qry_inputs[qry_input_idx]
                qry_label = qry_labels[qry_input_idx]
                outputs = model(qry_input)  # 正向传播
                query_loss = criterion(outputs, qry_label)  # query_loss带有计算图

                # 用 query_loss 更新 meta-model 参数
                query_loss = query_loss / len(qry_inputs)
                query_loss.backward()  # 反向传播，计算梯度
        meta_optimizer.step()  # 更新参数
        meta_optimizer.zero_grad()  # 梯度清零


def MAML_way_2():
    # 1. 生成随机数据（100个样本）
    spt_inputs = [torch.randn(100, 1) * 10 for i in range(10)]  # 输入特征（-10到10的随机数）
    spt_labels = [
        3 * spt_inputs + 2 + torch.randn(100, 1) * 2 for i in range(10)
    ]  # 带噪声的线性关系（真实参数w=3，b=2）

    qry_inputs = [torch.randn(100, 1) * 10 for i in range(10)]  # 输入特征（-10到10的随机数）
    qry_labels = [
        3 * spt_inputs + 2 + torch.randn(100, 1) * 2 for i in range(10)
    ]  # 带噪声的线性关系（真实参数w=3，b=2）

    # 2. 定义单层线性模型
    meta_model = nn.Sequential(nn.Linear(1, 1))  # 输入维度1，输出维度1

    # 3. 定义损失函数和优化器
    meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.01)  # 随机梯度下降
    criterion = nn.MSELoss()  # 均方误差损失

    for epoch in range(num_epoch := 10):

        for task in range(num_task := 2):
            model = copy.deepcopy(meta_model)
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # 用 support set 训练模型
            for i in range(num_inner_steps := 10):
                for spt_input_idx in range(len(spt_inputs)):
                    spt_input = spt_inputs[spt_input_idx]
                    spt_label = spt_labels[spt_input_idx]

                    optimizer.zero_grad()  # 梯度清零
                    outputs = model(spt_input)  # 正向传播
                    support_loss = criterion(outputs, spt_label)  # 计算损失
                    support_loss.backward()  # 反向传播，计算梯度
                    optimizer.step()  # 更新参数

            # 用 query set 测试模型
            losses = []
            for qry_input_idx in range(len(qry_inputs)):
                qry_input = qry_inputs[qry_input_idx]
                qry_label = qry_labels[qry_input_idx]
                outputs = model(qry_input)  # 正向传播
                query_loss = criterion(outputs, qry_label)  # query_loss带有计算图
                losses.append(query_loss)

            # 用 losses 更新 meta-model 参数
            query_loss = torch.stack(losses).mean() / num_task
            query_loss.backward()  # 反向传播，计算梯度
        meta_optimizer.step()  # 更新参数
        meta_optimizer.zero_grad()  # 梯度清零


def MAML_way_3():
    # 1. 生成随机数据（100个样本）
    spt_inputs = [torch.randn(100, 1) * 10 for i in range(10)]  # 输入特征（-10到10的随机数）
    spt_labels = [
        3 * spt_inputs + 2 + torch.randn(100, 1) * 2 for i in range(10)
    ]  # 带噪声的线性关系（真实参数w=3，b=2）

    qry_inputs = [torch.randn(100, 1) * 10 for i in range(10)]  # 输入特征（-10到10的随机数）
    qry_labels = [
        3 * spt_inputs + 2 + torch.randn(100, 1) * 2 for i in range(10)
    ]  # 带噪声的线性关系（真实参数w=3，b=2）

    # 2. 定义单层线性模型
    meta_model = nn.Sequential(nn.Linear(1, 1))  # 输入维度1，输出维度1

    # 3. 定义损失函数和优化器
    meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.01)  # 随机梯度下降
    criterion = nn.MSELoss()  # 均方误差损失

    for epoch in range(num_epoch := 10):
        losses = []
        for task in range(num_task := 2):
            model = copy.deepcopy(meta_model)
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # 用 support set 训练模型
            for i in range(num_inner_steps := 10):
                for spt_input_idx in range(len(spt_inputs)):
                    spt_input = spt_inputs[spt_input_idx]
                    spt_label = spt_labels[spt_input_idx]

                    optimizer.zero_grad()  # 梯度清零
                    outputs = model(spt_input)  # 正向传播
                    support_loss = criterion(outputs, spt_label)  # 计算损失
                    support_loss.backward()  # 反向传播，计算梯度
                    optimizer.step()  # 更新参数

            # 用 query set 测试模型
            for qry_input_idx in range(len(qry_inputs)):
                qry_input = qry_inputs[qry_input_idx]
                qry_label = qry_labels[qry_input_idx]
                outputs = model(qry_input)  # 正向传播
                query_loss = criterion(outputs, qry_label)  # query_loss带有计算图
                losses.append(query_loss)

        # 用 losses 更新 meta-model 参数
        query_loss = torch.stack(losses).mean()
        query_loss.backward()  # 反向传播，计算梯度
        meta_optimizer.step()  # 更新参数
        meta_optimizer.zero_grad()  # 梯度清零
