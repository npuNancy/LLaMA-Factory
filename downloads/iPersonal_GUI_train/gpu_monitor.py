import os
import sys
import time


class GPUGet:
    def __init__(self, time_interval):
        self.time_interval = time_interval

    def get_gpu_info(self):
        gpu_status = os.popen("nvidia-smi | grep %").read().split("|")[1:]
        gpu_dict = dict()
        for i in range(len(gpu_status) // 4):
            index = i * 4
            gpu_state = str(gpu_status[index].split("   ")[2].strip())
            gpu_power = int(gpu_status[index].split("   ")[-1].split("/")[0].split("W")[0].strip())
            gpu_memory = int(gpu_status[index + 1].split("/")[0].split("M")[0].strip())
            gpu_dict[i] = (gpu_state, gpu_power, gpu_memory)
        return gpu_dict

    def loop_monitor(self, min_gpu_number):
        available_gpus = []
        while True:
            gpu_dict = self.get_gpu_info()
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
                if (
                    gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000
                ):  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
                    gpu_str = (
                        f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
                    )
                    sys.stdout.write(gpu_str)
                    sys.stdout.flush()
                    available_gpus.append(i)
            if len(available_gpus) >= min_gpu_number:
                return available_gpus
            else:
                print(f"当前可用GPU数量为{len(available_gpus)}，小于{min_gpu_number}，继续监控GPU状态...")
                available_gpus = []
                time.sleep(self.time_interval)

    def run_with_gpu_number(self, cmd_parameter, cmd_command, min_gpu_number):
        available_gpus = self.loop_monitor(min_gpu_number)
        gpu_list_str = ",".join(map(str, available_gpus))
        # 构建终端命令
        cmd_parameter = rf"""{cmd_parameter}
                          NUM_GPUS={len(available_gpus)} ; \ """  # 一定要有 `; \ `
        cmd_command = rf"""CUDA_VISIBLE_DEVICES={gpu_list_str} \ 
                         {cmd_command}"""
        command_ = rf"""{cmd_parameter} {cmd_command}"""
        print(command_)
        os.system(command_)

    def loop_monitor_with_special_gpu(self, gpu_list):
        """"""
        while True:
            available_gpus = []
            gpu_dict = self.get_gpu_info()
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
                if i not in gpu_list:
                    continue

                gpu_str = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n"
                if (
                    gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000
                ):  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
                    available_gpus.append(i)

            if available_gpus == gpu_list:
                return available_gpus
            else:
                print(f"当前可用GPU为{available_gpus}, 与{gpu_list}不符，继续监控GPU状态...")
                time.sleep(self.time_interval)

    def run_with_special_gpu(self, cmd_command, gpu_list):
        available_gpus = self.loop_monitor_with_special_gpu(gpu_list)
        print(cmd_command)
        os.system(cmd_command)


if __name__ == "__main__":
    time_interval = 60  # 监控GPU状态的频率，单位秒。
    gpu_get = GPUGet(time_interval)

    gpu_list = [0]  # 需要和 sh脚本 中保持一致
    cmd_command = "./data4/yanxiaokai/LLaMA-Factory/downloads/iPersonal_GUI_train/run_stage3_train.sh"
    gpu_get.run_with_special_gpu(cmd_command, gpu_list)

    # cmd_parameter = r""""""  # 命令会使用到的参数，使用 `;` 连接。
    # cmd_command = r"""echo hello_world"""
    # min_gpu_number = 3  # 最小GPU数量，多于这个数值才会开始执行训练任务。
    # gpu_get.run_with_gpu_number(cmd_parameter, cmd_command, min_gpu_number=3)
