import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_svmlight_file  # 用于加载 LIBSVM 数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Optimizers import *
import torch.optim as optim
import matplotlib.pyplot as plt 
import os
from utils import *

seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 修改后的数据加载函数
def load_data(file_path):
    data, target = load_svmlight_file(file_path)
    target[target == -1] = 0  # 将所有 -1 的标签改为 0
    return torch.tensor(data.toarray(), dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# 替换为训练和测试数据集的路径
train_file_path = 'data/LIBSVM/a9a'
test_file_path = 'data/LIBSVM/a9a.t'

# 加载训练和测试数据
X_train, y_train = load_data(train_file_path)
X_test, y_test = load_data(test_file_path)
zeros_column = torch.zeros(X_test.shape[0], 1)  # 创建一个与X_test行数相同的全零列
X_test = torch.cat((X_test, zeros_column), dim=1) 
# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 使用相同的标准化参数
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = y_train.unsqueeze(1)  # 调整 y 的维度以适配模型
y_test = y_test.unsqueeze(1)

# 创建 DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 2. 模型定义
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

optimizers = ["SGD", "MomentumSGD", "Adagrad", "RMSProp", "Adam"]
epochs = 200
# 准备记录每种优化器的性能
losses = {k: [] for k in optimizers}
accuracies = {k: [] for k in optimizers}

# 3. 修改训练循环
for opt_name in optimizers:
    model = LogisticRegressionModel(X_train.shape[1]).to(device)  # 重新初始化模型
    criterion = nn.BCELoss()

    # 针对当前模型重新实例化优化器
    if opt_name == "SGD":
        optimizer = SGDOptimizer(model.parameters(), lr=0.001)
    elif opt_name == "MomentumSGD":
        optimizer = MomentumSGDOptimizer(model.parameters(), lr=0.001, momentum=0.9)
    elif opt_name == "Adagrad":
        optimizer = AdagradOptimizer(model.parameters(), lr=0.01)
    elif opt_name == "RMSProp":
        optimizer = RMSPropOptimizer(model.parameters(), lr=0.001, alpha=0.99)
    elif opt_name == "Adam":
        optimizer = AdamOptimizer(model.parameters(), lr=0.001, beta1=0.9, beta2=0.999)
    '''
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    elif opt_name == "MomentumSGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=0.001)
    elif opt_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))
    '''

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移至相同的设备
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        average_loss = total_loss / total_batches
        losses[opt_name].append(average_loss)

        # 打印平均损失
        print(f'Optimizer: {opt_name}, Epoch [{epoch+1}/{epochs}], Average Loss: {average_loss:.4f}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            # 注意：这里不应该重新赋值给 X_test 和 y_test
            predictions = model(X_test.to(device))
            predicted_classes = predictions.round()
            correct = (predicted_classes == y_test.to(device)).sum().item()
            total = y_test.size(0)
            accuracy = correct / total
            accuracies[opt_name].append(accuracy)

        # 打印测试结果
        print(f'Optimizer: {opt_name}, Epoch: {epoch + 1}, Test Accuracy: {accuracy:.4f}')

#current_dir = os.getcwd()
relative_path = "Result/LIBSVM"
#save_directory = os.path.join(current_dir, relative_path)
# 调用函数来绘制和保存图表
plot_performance(losses, accuracies, optimizers, relative_path)