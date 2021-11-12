# 加载python库，包含数据处理与pytorch自己的包
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 
import torchvision
import torchvision.transforms as transforms
import os
import time

# 从同级目录models中加载所有的自定义包
from models import *

# 选定运行设备（CPU或者GPU）
# 如果GPU可用，那么指定第0,1,2...个GPU作为运行设备
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

'''
数据预处理部分
'''
print('==> Preparing data..')
# 处理训练数据，所需要的transform：
# 将不同的处理方法整合（compose）到一个处理序列中
# 包含随机裁剪、随机翻转、numpy格式转tensor格式以及数据归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # 将图片resize到更大的图片
    # transforms.Resize(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 同上
transform_test = transforms.Compose([
    # transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 将从./data数据集文件夹中读取的数据封装成一个CIFAR10类的对象
# train: 是否是训练数据
# download: 如果./data文件夹中没有数据，那么就从默认链接下载数据集
# transform: 上面的transform，用于对数据进行对应的预处理
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# cifar10数据集中包含的完整类别（训练中并没有用到这个classes，只是在这里列出来了）
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model：构建模型结构
# 这里包含了多个网络模型结构，目前使用的是VGG16模型，这个比较常见
# 具体的网络结构定义在models文件夹中，每个文件代表一个网络结构
print('==> Building model..')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
net = VGG('VGG16')

# 将网络转移到目标设备中运行（GPU或者CPU）
net = net.to(device)
# 打印网络结构
print(net)
# 如果设备为GPU，那么设置cudnn
# 这里注释的dataparallel表示并行计算,可以取消注释,但是网络结构会发生改变
# 所以我注释了
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# 定义使用什么样的损失函数(交叉熵损失)
criterion = nn.CrossEntropyLoss()
# 定义优化器: Adam
# net.parameters(): 需要进行反向传播更新的参数
# lr: 学习率（adam会在学习过程中自适应调整lr）
# betas: adam的初始化参数
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# Training
'''
训练主程序
'''
def train(epoch):
    print('Epoch {}/{}'.format(epoch + 1, 200))
    print('-' * 10)
    start_time = time.time()
    # 将网络结构设定为训练模式
    # 官方解释：Sets the module in training mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0 
    
    # 使用迭代器enumerate，迭代地选取trainloader中的数据
    # load一次表示一个batch
    # 返回两个参数：
    # batch_idx：表示批次的index（索引）
    # (inputs, targets): 表示数据与对应的label的组合
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 将数据与标签放置到目标设备上运行（必须要和net所在的设备对应）
        # 要么都在cpu要么都在gpu
        inputs, targets = inputs.to(device), targets.to(device)
        # 将网络中的参数对应的梯度全部清零
        optimizer.zero_grad()
        # 前向传播，根据输入数据进行推断
        outputs = net(inputs)
        # 根据输出结果和lebels计算loss
        loss = criterion(outputs, targets)
        # 反向传播（自动求导，得到需要更新的参数的梯度值）
        loss.backward()
        # 使用优化器更新参数
        optimizer.step()

        # 一个batch的loss值求和
        train_loss += loss.item()
        # 最后一层输出值中最大的值对应的index
        _, predicted = outputs.max(1)
        # 一共有多少个数据
        total += targets.size(0)
        # 统计分类正确的数据的个数
        correct += predicted.eq(targets).sum().item()
    end_time = time.time()
    # 输出一个epoch的训练效果
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

'''
测试主程序
'''
def test(epoch):
    global best_acc

    # 将模型设定为验证模式
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # no_grad()：表示不考虑梯度，也就是模型处于推断状态
    # 这样可以节省memory
    with torch.no_grad():
        # 同上
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # 计算预测准确率
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        
        # 判断checkpoint是否是个文件夹
        if not os.path.isdir('checkpoint-clean'):
            # 如果没有这个文件夹的话，创建一个同名的文件夹
            os.mkdir('checkpoint-clean')
        # 保存网络参数
        # 也可以直接torch.save(net, 'model/path')，这样是直接保存整个模型
        # 包括模型结构和模型采纳数
        torch.save(net.state_dict(), './checkpoint-clean/ckpt.pth')
        best_acc = acc

# python 程序主入口
if __name__ == "__main__":
    # 训练100个epoch（轮次）
    for epoch in range(start_epoch, start_epoch+100):
        # 训练
        train(epoch)
        # 测试
        test(epoch)
