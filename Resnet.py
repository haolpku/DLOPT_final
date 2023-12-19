import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from Optimizers import *
from utils import *
import tqdm

# 设定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=32)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def opt(model,opt_name):

    if opt_name == "SGD":
        optimizer = SGDOptimizer(model.parameters(), lr=0.005)
    elif opt_name == "MomentumSGD":
        optimizer = MomentumSGDOptimizer(model.parameters(), lr=0.005, momentum=0.9)
    elif opt_name == "Adagrad":
        optimizer = AdagradOptimizer(model.parameters(), lr=0.005)
    elif opt_name == "RMSProp":
        optimizer = RMSPropOptimizer(model.parameters(), lr=0.005, alpha=0.99)
    elif opt_name == "Adam":
        optimizer = AdamOptimizer(model.parameters(), lr=0.005, beta1=0.9, beta2=0.999)

    return optimizer

optimizers=["SGD", "MomentumSGD", "Adagrad", "RMSProp", "Adam"]
losses = {k: [] for k in optimizers}
accuracies = {k: [] for k in optimizers}

for name in optimizers:
    # 初始化 ResNet-18 模型
    net = torchvision.models.resnet18(pretrained=False)
    net.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    optimizer=opt(net,name)
    # 训练模型
    for epoch in range(100):  # loop over the dataset multiple times
        net.train()
        total_loss = 0
        total_batches = 0
        for i, data in tqdm.tqdm(enumerate(trainloader, 0)):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度归零
            optimizer.zero_grad()

            # 前向 + 反向 + 优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            total_loss += loss.item()
            total_batches += 1
        average_loss = total_loss / total_batches
        losses[name].append(average_loss)
        print("epoch:",epoch,"avgloss:",average_loss)

        # 测试模型
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            accuracies[name].append(accuracy)
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
plot_performance(losses, accuracies, optimizers, "/home/wsy0227/codes/AI4S/DLOPT_final-main/DLOPT_final-main/Result/Resnet")