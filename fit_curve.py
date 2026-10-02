import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 解决matplotlib中文方框、负号乱码问题
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1.生成模拟数据（代替csv文件）----------------------
np.random.seed(42)
x_np = np.linspace(-2 * np.pi, 2 * np.pi, 200)
# 真实函数 y = sin(x)，增加少量噪声
y_np = np.sin(x_np) + 0.1 * np.random.randn(len(x_np))

# 划分训练集、测试集（用来观察过拟合）
train_idx = np.random.choice(len(x_np), size=120, replace=False)
test_idx = np.setdiff1d(np.arange(len(x_np)), train_idx)

x_train = torch.tensor(x_np[train_idx], dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(-1)
x_test = torch.tensor(x_np[test_idx], dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_np[test_idx], dtype=torch.float32).unsqueeze(-1)

x_all = torch.tensor(x_np, dtype=torch.float32).unsqueeze(-1)

# ---------------------- 2.搭建简单全连接网络 ----------------------
class FitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_and_plot(lr, total_epoch, save_name):
    model = FitNet()
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []
    # 保存特定轮次的预测结果：10，100，1000
    record_epochs = {10, 100, 1000}
    pred_cache = {}

    for epoch in range(total_epoch + 1):
        model.train()
        pred_train = model(x_train)
        loss_train = loss_fn(pred_train, y_train)

        opt.zero_grad()
        loss_train.backward()
        opt.step()

        # 测试集loss
        model.eval()
        with torch.no_grad():
            loss_test = loss_fn(model(x_test), y_test)

        train_loss_list.append(loss_train.item())
        test_loss_list.append(loss_test.item())

        if epoch in record_epochs:
            with torch.no_grad():
                pred_cache[epoch] = model(x_all).cpu().numpy()

        if epoch % 200 == 0:
            print(f"epoch {epoch:4d} | train_loss:{loss_train.item():.4f} | test_loss:{loss_test.item():.4f}")

    # 画图
    plt.figure(figsize=(14, 4))
    for i, ep in enumerate(sorted(record_epochs)):
        plt.subplot(1, 3, i+1)
        plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), s=8, label="训练数据")
        plt.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), s=8, label="测试数据")
        plt.plot(x_np, np.sin(x_np), c="red", label="真实sin(x)")
        plt.plot(x_np, pred_cache[ep], c="orange", label=f"网络拟合(epoch={ep})")
        plt.title(f"训练轮次 {ep} | lr={lr}")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"图片已保存：{save_name}")
    return train_loss_list, test_loss_list

if __name__ == "__main__":
    # 小学习率
    train_and_plot(lr=0.005, total_epoch=1200, save_name="fit_lr_small.png")
    # 大学习率
    train_and_plot(lr=0.2, total_epoch=1200, save_name="fit_lr_big.png")
