import os
import torch
import torch.nn as nn
import torch.utils.data as D
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from models import resnet, base
import numpy as np
import tensorboard_logger
import torchvision.transforms as transforms
import torchvision.datasets as datasets

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Dynamic ResNet Training')
parser.add_argument('--lr', type=float, default=.1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=164, #350,
        help='total epochs to run')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_devices = torch.cuda.device_count()

def train(epoch):
    rnet.train()

    total = 0
    correct = 0
    train_loss = 0
    total_batch = 0

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        probs = rnet(inputs, True)
        optimizer.zero_grad()
        loss = criterion(probs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = probs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        total_batch += 1

    print('E:%d Train Loss: %.3f Train Acc: %.3f LR %f'
          % (epoch,
             train_loss / total_batch,
             correct / total,
             optimizer.param_groups[0]['lr']))

    tensorboard_logger.log_value('train_acc', correct/total, epoch)
    tensorboard_logger.log_value('train_loss', train_loss / total_batch, epoch)


def test(epoch):
    global best_test_acc
    rnet.eval()

    total = 0
    correct = 0
    test_loss = 0
    total_batch = 0

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        probs = rnet(inputs)
        loss = criterion(probs, targets)

        test_loss += loss.item()
        _, predicted = probs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        total_batch += 1

    print('E:%d Test Loss: %.3f Test Acc: %.3f'
          % (epoch, test_loss / total_batch, correct / total))

    # save best model
    acc = 100.*correct/total

    if acc > best_test_acc:
        best_test_acc = acc
        print('saving best model...')
        state = {
            'net': rnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, 'resnet110.t7')
    tensorboard_logger.log_value('test_acc', acc, epoch)
    tensorboard_logger.log_value('test_loss', test_loss/total_batch, epoch)


def adjust_learning_rate(epoch, stage=[250, 375]):
    order = np.sum(epoch >= np.array(stage))
    lr = args.lr * (0.1 ** order)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf



# dataset and dataloader
train_tf, test_tf = get_transforms()
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
trainloader = D.DataLoader(trainset, batch_size=num_devices*args.batch_size, shuffle=True, num_workers=4)
testloader = D.DataLoader(testset, batch_size=num_devices*args.batch_size, shuffle=False, num_workers=4)
best_test_acc = 0.0

# resnet110
num_layers = 54
rnet = resnet.FlatResNet32(base.BasicBlock, [18, 18, 18], num_classes=10)
rnet.to(device)
if num_devices > 1:
    print('paralleling for multiple GPUs...')
    rnet = nn.DataParallel(rnet)

start_epoch = 0

if args.resume:
    assert os.path.isfile('resnet110.t7'), 'Error: no check-point found!'
    ckpt = torch.load('resnet110.t7')
    rnet.load_state_dict(ckpt['net'])
    best_test_acc = ckpt['acc']
    start_epoch = ckpt['epoch']
else:
    # He's init
    for module in rnet.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

# Loss Fn and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# logger
tensorboard_logger.configure('./log')

for epoch in range(start_epoch+1, args.max_epochs):
    train(epoch)
    test(epoch)
    adjust_learning_rate(epoch)