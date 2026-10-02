import torch

print("===== 显卡信息 =====")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")
print(f"显卡总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

device = torch.device("cuda")
print(f"\n使用设备：{device}")

import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).to(device)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]]).to(device)

model = nn.Linear(1,1).to(device)

loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

print("\n===== 开始训练 =====")
for epoch in range(1000):
    pred = model(x)
    loss = loss_fn(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print(f"轮次{epoch:4d} | 损失: {loss.item():.4f}")

print("\n训练完成！")
test_x = torch.tensor([[5.0]]).to(device)
print(f"输入x=5，预测结果：{model(test_x).item():.2f}")
