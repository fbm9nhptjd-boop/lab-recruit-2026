# 任务一：环境配置报告
## 1. 环境信息
- Python版本：3.10.21
- 工具：Miniconda
- PyTorch：2.12.0.dev20260408+cu128（CUDA 12.8，GPU加速）
- torchvision：0.27.0.dev20260407+cu128
- 显卡型号：RTX 5070 Laptop
- 虚拟环境路径：E:\miniconda3\envs\myenv

## 2. 验证结果
torch.cuda.is_available() → True
GPU设备正常，可以进行CUDA加速运算。可以正常跑通矩阵运算以及简单神经网络训练。

## 3. 操作记录
1. 安装Miniconda
2. 创建虚拟环境myenv，指定Python3.10
3. 判断电脑显卡支持CUDA，安装CUDA版本PyTorch
4. 编写test_gpu.py，验证GPU可以正常运行
5. VS Code调试，解决解释器切换、受限模式、终端缓存问题
6. 安装Git，配置git的user.name、user.email
7. GitHub创建仓库，克隆仓库到本地电脑

## 4.遇到的报错与问题复盘
### 问题1：VS Code右下角解释器总是跳回别的Python版本
现象：已经手动选中myenv环境，但是右下角会变回其他Python。
解决办法：信任当前项目文件夹，重新手动选择myenv(3.10.21)解释器。

### 问题2：右下角显示环境正确，但是旧终端运行代码报No module named torch
现象：VS Code状态栏环境正确，旧终端里面找不到torch库。
原因：旧终端会保留启动时刻的环境，不会跟随状态栏更新。
解决办法：关闭旧终端，新建一个终端再运行代码。

### 问题3：Restricted Mode 受限模式
现象：顶部蓝色提示Restricted Mode，Python相关功能失效，无法正常切换解释器。
解决办法：点击提示条，选择信任文件夹，解除受限模式。

## 5.如何判断PyTorch版本
我的电脑拥有RTX5070 Laptop独立显卡，硬件支持CUDA，因此选择CUDA版本PyTorch，而不是CPU版本。前往PyTorch官网，根据Miniconda环境复制对应的安装命令执行。

## 6.AI辅助记录
AI提供了环境配置步骤以及GPU测试代码；
AI没有提醒旧终端存在环境缓存这个坑，该问题通过自己新建终端测试排查出来。
