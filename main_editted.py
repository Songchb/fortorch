import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

#from utils import progress_bar

# Module
import googlenet


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Traning')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--backend', default='gloo', type=str)
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes (default=1)')
parser.add_argument('--dist-url', type=str, help='url used to set up distributed training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--rank', default=0, type=int, help='current node rank')
args = parser.parse_args()

args.distributed = args.world_size > 1
args.use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('===> training on: ', device)
best_accuracy = 0
start_epoch = 0

# Data
print('===> Preparing Data')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if args.resume:
    print('===> Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('===> Building model...')
    net = Net()
#    net = models.resnet18()
#    net = googlenet.GoogLeNet()

if args.distributed:
    print('===> Distributed Training Mode')
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, rank=args.rank, world_size=args.world_size)

if args.distributed:
    if args.use_cuda:
        print('===> DistributedDataParallel')
        net.to(device)
        net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        print('===> DistributedDataParallelCPU')
        net = torch.nn.parallel.DistributedDataParallelCPU(net)
else:
    if args.use_cuda:
        print('===> DataParallel')
        net.to(device)
        net = torch.nn.parallel.DataParallel(net)

criterion = nn.CrossEntropyLoss().cuda() if args.use_cuda else nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

# Training

def train(epoch):
    print('\nEpoch: %d' % epoch)
    print('Train')
#    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global best_accuracy
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.use_cuda:
            inputs, targets = inputs.to(device), targets.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_accuracy:
            print('\nSaving..')
            state = {
                'net': net.module if args.use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_accuracy = acc

def test(epoch):
    print('\nTest')

    global best_accuracy
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.use_cuda:
            inputs, targets = inputs.to(device), targets.cuda(non_blocking=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_accuracy:
        print('\nSaving..')
        state = {
            'net': net.module if args.use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_accuracy = acc

def validate():
    print('============================> Validating')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if args.use_cuda:
                images, labels = images.to(device), labels.cuda(non_blocking=True)
 
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print('========================================')

if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        test(epoch)
        validate()
