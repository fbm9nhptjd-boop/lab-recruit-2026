# 补充任务A：函数拟合实验报告
## 1.实验目的
使用全连接神经网络拟合正弦函数，观察：
1. 不同训练轮次(10、100、1000)下拟合曲线变化
2. 大、小学习率对训练loss和拟合效果的影响
3. 观察过拟合现象：训练集效果好，测试集效果差

## 2.环境与依赖
- Python:3.10
- Pytorch(GPU CUDA)
- matplotlib、numpy
虚拟环境路径：E:\miniconda3\envs\myenv

运行命令：
```powershell
E:\miniconda3\envs\myenv\python.exe fit_curve.py
