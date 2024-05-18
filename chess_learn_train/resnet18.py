import time
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import chess_learn_train.Utils
import tensorboard
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torch.utils.data as Data
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # chess_model root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser(description='Params for training. ')
parser.add_argument('--root', type=str, default='C:/Users/jhf12/Documents/graduation_project/chess_robot'
                                                '/chess_learn_train', help='path to data set')
parser.add_argument("--data", type=str, default=ROOT / "datasets", help="dataset path")
parser.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'inference'])
parser.add_argument('--log_path', type=str, default=ROOT / "log/log.pth", help='dir of checkpoints')
parser.add_argument('--black_chess_path', type=str, default=ROOT / "log/black_chess/exp2/black_chess.pth")
parser.add_argument('--red_chess_path', type=str, default=ROOT / "log/red_chess/exp1/red_chess.pth")
parser.add_argument('--restore', type=bool, default=False, help='whether to restore checkpoints')
parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
parser.add_argument('--image_size', type=int, default=28, help='resize image')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num_class', type=int, default=7, choices=range(0, 7))
parser.add_argument("--device", default="cuda:0", help="cuda device, cuda:0 or cpu")
args = parser.parse_args()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('\\')[-2]) >= num_class:  # just get images of the first #num_class
                    break

                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('\\')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('L')
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 看上面的信息是否需要卷积修改，从而满足相加条件
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=True)
    return model


def train():
    writer = SummaryWriter(log_dir='logs')

    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.RandomRotation(180, fill=255),
                                    transforms.ToTensor()])

    train_set = MyDataset("./datasets/train/black_train.txt",
                          num_class=args.num_class, transforms=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    print(device)

    model = resnet18(num_classes=7)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.restore:
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        loss = 0.0
        epoch = 0

    while epoch < args.epoch:
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i == 2:  # every 3 steps
                # 写入TensorBoard日志
                writer.add_scalar('training_loss', running_loss / 3, epoch * len(train_loader) + i)
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 3))
                # running_loss = 0.0

        if epoch % 10 == 9:
            print('Save checkpoint...')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                       args.log_path)
        epoch += 1
    # 关闭SummaryWriter对象
    writer.close()

    print('Finish training')


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


NUM_EPOCHS = 100
DEVICE = args.device
model = resnet18(num_classes=7)

model = model.to(DEVICE)
transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.RandomRotation(180, fill=255),
                                    transforms.ToTensor()])


train_set = MyDataset("./datasets/train/black_train.txt",
                          num_class=args.num_class, transforms=transform)

transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])

test_set = MyDataset("./datasets/test/black_test.txt", num_class=args.num_class, transforms=transform)
valid_loader = DataLoader(test_set, batch_size=args.batch_size)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
# 原先这里选用SGD训练，但是效果很差，换成Adam优化就好了
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

for epoch in range(NUM_EPOCHS):

    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):

        ### PREPARE MINIBATCH
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 300:
            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                  f' Cost: {cost:.4f}')

    # no need to build the computation graph for backprop when computing accuracy
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
        valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Validation Acc.: {valid_acc:.2f}%')

    elapsed = (time.time() - start_time) / 60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')

# if __name__ == '__main__':
    # chess_learn_train.chess_utils.classes_txt("./datasets/train/black",
    #                                     "./datasets/train/black_train.txt", num_class=7)
    # chess_learn_train.chess_utils.classes_txt("./datasets/test/black",
    #                                     "./datasets/test/black_test.txt", num_class=7)
    # if args.mode == 'train':
    #     train()
    # elif args.mode == 'validation':
    #     validation()
    # elif args.mode == 'inference':
    #     for i1 in range(10):
    #         inference()
    # net = resnet18(7)
    # print(net)
