import argparse
import os
import sys
from pathlib import Path
import chess_learn_train.Utils
import torch
from PIL import Image
from torch import nn, optim
import torch.nn.functional as  F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # chess_model root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relatived
# 黑色为True， 红色为False
is_black = False
parser = argparse.ArgumentParser(description='Params for training. ')

parser.add_argument('--root', type=str, default='C:/Users/jhf12/Documents/graduation_project/chess_robot'
                                                '/chess_learn_train', help='path to data set')
parser.add_argument("--data", type=str, default=ROOT / "datasets", help="dataset path")
parser.add_argument('--mode', type=str, default='validation', choices=['train', 'validation', 'inference'])
parser.add_argument('--log_path', type=str, default=ROOT / "log/log.pth", help='dir of checkpoints')
parser.add_argument('--black_chess_path', type=str, default=ROOT / "log/black_chess/exp7/black_chess.pth")
parser.add_argument('--red_chess_path', type=str, default=ROOT / "log/red_chess/exp6/red_chess.pth")
parser.add_argument('--restore', type=bool, default=False, help='whether to restore checkpoints')
# batch_size会影响loss，如果出现loss没有下降的现象，修改batch_size，当前128
parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
parser.add_argument('--image_size', type=int, default=64, help='resize image')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--num_class', type=int, default=7, choices=range(0, 7))
parser.add_argument("--device", default="cuda:0", help="cuda device, cuda:0 or cpu")
args = parser.parse_args()


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


class NetBig(nn.Module):
    def __init__(self):
        super(NetBig, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, args.num_class)
        # self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, args.num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train():
    writer = SummaryWriter(log_dir='logs')
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.RandomRotation(180, fill=255),
                                    transforms.ToTensor()])
    if is_black:
        train_set = MyDataset("./datasets/train/black_train.txt",
                              num_class=args.num_class, transforms=transform)
    else:
        train_set = MyDataset("./datasets/train/red_train.txt",
                              num_class=args.num_class, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    print(device)
    model = NetBig()
    # model = NetSmall()
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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


def validation():
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    if is_black:
        test_set = MyDataset("./datasets/test/black_test.txt", num_class=args.num_class, transforms=transform)
    else:
        test_set = MyDataset("./datasets/test/red_test.txt", num_class=args.num_class, transforms=transform)

    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda:0')
    model = NetBig()
    # model = NetSmall()
    model.to(device)

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    classes = ("0", "1", "2", "3", "4", "5", "6")
    # 0--黑将, 1--黑車, 2--黑馬, 3--黑象, 4--黑士,  5--黑炮,  6--黑卒,
    # 0--红帅,  1--红車, 2--红马, 3--红相, 4--红士, 5--红炮, 6--红兵,
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            for label, prediction in zip(labels, predict):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    print('Accuracy: %.2f%%' % (correct / total * 100))
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))


def inference():
    print('Start inference...')
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    if is_black:
        f = open("./datasets/test/black_test.txt")
    else:
        f = open("./datasets/test/red_test.txt")

    num_line = sum(line.count('\n') for line in f)
    f.seek(0, 0)
    line = int(torch.rand(1).data * num_line - 10)  # -10 for '\n's are more than lines
    while line > 0:
        f.readline()
        line -= 1
    img_path = f.readline().rstrip('\n')
    f.close()
    label = int(img_path.split('\\')[-2])
    print('label:\t%4d' % label)
    input = Image.open(img_path).convert('L')
    input = transform(input)
    input = input.unsqueeze(0)
    model = NetBig()
    # model = NetSmall()
    model.eval()
    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(input)
    _, pred = torch.max(output.data, 1)
    print('predict:\t%4d' % pred)


def single_chess_black_recognize(single_black_img):
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    pil_image = Image.fromarray(single_black_img)
    trans_img = transform(pil_image)
    trans_img = trans_img.unsqueeze(0)
    model = NetBig()
    # model = NetSmall()
    model.eval()
    checkpoint = torch.load(args.black_chess_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(trans_img)
    _, pred = torch.max(output.data, 1)
    # 0--黑将, 1--黑車, 2--黑馬, 3--黑象, 4--黑士,  5--黑炮,  6--黑卒
    return pred


def single_chess_red_recognize(single_red_img):
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    pil_image = Image.fromarray(single_red_img)
    trans_img = transform(pil_image)
    trans_img = trans_img.unsqueeze(0)
    model = NetBig()
    # model = NetSmall()
    model.eval()
    checkpoint = torch.load(args.red_chess_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(trans_img)
    _, pred = torch.max(output.data, 1)
    # 0--红帅,  1--红車, 2--红马, 3--红相, 4--红士, 5--红炮, 6--红兵,
    return pred


if __name__ == '__main__':
    if is_black:
        chess_learn_train.Utils.classes_txt("./datasets/train/black",
                                            "./datasets/train/black_train.txt", num_class=7)
        chess_learn_train.Utils.classes_txt("./datasets/test/black",
                                            "./datasets/test/black_test.txt", num_class=7)
    else:
        chess_learn_train.Utils.classes_txt("./datasets/train/red",
                                            "./datasets/train/red_train.txt", num_class=7)
        chess_learn_train.Utils.classes_txt("./datasets/test/red",
                                            "./datasets/test/red_test.txt", num_class=7)

    if args.mode == 'train':
        train()
    elif args.mode == 'validation':
        validation()
    elif args.mode == 'inference':
        for i1 in range(10):
            inference()
