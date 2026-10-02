# lab-recruit-2026

深度视觉农业实验室 2026 招新考核

## 项目结构

```
lab-recruit-2026/
├── task1_env.md              # 任务一：环境配置报告
├── test_gpu.py               # GPU可用性验证脚本
├── task_supplement_A.md      # 实验A：CIFAR-10图像分类报告
├── cnn_cifar10.py            # 实验A：CNN图像分类训练代码
├── cifar10_result.png        # 实验A：训练损失与准确率曲线
├── task_supplement_A_fit.md  # 补充任务A：函数拟合实验报告
├── fit_curve.py              # 补充任务A：函数拟合训练代码
├── fit_lr_small.png          # 补充任务A：小学习率拟合结果
├── fit_lr_big.png            # 补充任务A：大学习率拟合结果
└── .gitignore                # 忽略数据集data/等文件
```

## 环境要求

- Miniconda虚拟环境：myenv（Python 3.10）
- PyTorch 2.x（CUDA版），支持GPU加速
- NVIDIA显卡（本机：RTX 5070 Laptop）
- torchvision、matplotlib、numpy

## 运行方法

```powershell
# GPU验证
E:\miniconda3\envs\myenv\python.exe test_gpu.py

# 实验A：CIFAR-10图像分类（首次运行自动下载数据集）
E:\miniconda3\envs\myenv\python.exe cnn_cifar10.py

# 补充任务A：函数拟合
E:\miniconda3\envs\myenv\python.exe fit_curve.py
```

> 注：数据集下载至 `data/` 目录，已通过 `.gitignore` 排除，不纳入版本管理。
