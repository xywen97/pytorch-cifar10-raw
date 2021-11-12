import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np 

import torchvision
import torchvision.transforms as transforms
import cv2

from models import *
from torchvision.transforms import Compose, Normalize, ToTensor

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.Resize(96),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
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
# net = EfficientNetB0()
# net = VGG('VGG16')
# net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

#------------------------------------------------------------------
# Loading weight files to the model and testing them.

net_test = VGG('VGG16')
# print(net_test)

net_test = net_test.to(device)
# net_test = torch.nn.DataParallel(net_test, device_ids=[0])

net_test.load_state_dict(torch.load('./checkpoint-clean/ckpt.pth'))
net_test.eval()

with torch.no_grad():
    # 一共10个类别
    for i in range(10):
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # 这个是numpy的特性，找到targets数组中与i相同的数值对应位置的index
            idxes = np.where(targets.numpy()==i)[0]

            # 初始化一个空的torch tensor，大小为括号中的参数定义
            tmp_inputs = torch.empty(len(idxes),3,32,32)
            tmp_targets = torch.empty(len(idxes))
            # 将筛选出的数据填充到空tensor中
            for j in range(len(idxes)):
                tmp_inputs[j] = inputs[idxes[j]]
                tmp_targets[j] = targets[idxes[j]]
            
            # torch运行需要long类型的数据
            tmp_targets = tmp_targets.type(torch.long)
            # 下面的代码和cifar10.py中的一致
            tmp_inputs, tmp_targets = tmp_inputs.to(device), tmp_targets.to(device)
            outputs = net_test(tmp_inputs)
            loss = criterion(outputs, tmp_targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += tmp_targets.size(0)
            correct += predicted.eq(tmp_targets).sum().item()

        print('class %d: TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (i, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
