import torch
from torch import nn
import time
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import PIL.Image as Image
import csv
from sklearn.preprocessing import LabelEncoder


class MyDataset(data.Dataset):

    def __init__(self, csv_path):
        self.filenames = []
        self.labels = []
        current_directory = os.getcwd()
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if row[1] == 'label':
                        continue
                    self.labels.append(row[1])
                    self.filenames.append(
                        os.path.join(current_directory, row[0]))
                except:
                    self.labels.append("")
                    self.filenames.append(
                        os.path.join(current_directory, row[0]))

            # 打印列表中元素不同的个数
            print(len(set(self.labels)))

            # 使用LabelEncoder将字符串转换为数字
            encoder = LabelEncoder()
            labels_encoded = encoder.fit_transform(self.labels)

            # 然后将numpy数组转换为张量
            self.labels = torch.tensor(labels_encoded)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        data = self.preprocess(image)
        return data, label

    def __len__(self):
        return len(self.filenames)

    def preprocess(self, data):
        transform_train_list = [
            # transforms.Resize((self.opt.h, self.opt.w), interpolation=3),
            # transforms.Pad(self.opt.pad, padding_mode='edge'),
            # transforms.RandomCrop((self.opt.h, self.opt.w)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform_train_list)(data)


train_dataset = MyDataset('train.csv')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = MyDataset('test.csv')
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4)


net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(256 * 5 * 5, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 176),
)


def togpu(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(device)


train_dataset = MyDataset("train.csv")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4
)


batch_size = 128  # 设置批量大小为128
lr, num_epochs = 0.02, 10


def train(net):  # 定义训练函数，输入参数为网络模型
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 如果CUDA可用，使用CUDA设备，否则使用CPU
    net = togpu(net)  # 将网络模型移动到GPU（如果可用）
    print("training on", device)  # 打印正在使用的设备
    net = net.to(device)  # 将网络模型移动到正在使用的设备
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr
    )  # 使用SGD优化器，输入参数为网络模型的参数和学习率
    loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    for epoch in range(num_epochs):  # 对每一轮训练进行循环
        train_l_sum, train_acc_sum, n, start = (
            0.0,
            0.0,
            0,
            time.time(),
        )  # 初始化损失和准确率的累计值，以及计时器
        for X, y in train_loader:  # 对训练数据进行循环
            X, y = togpu(X), togpu(y)  # 将数据移动到GPU（如果可用）
            y_hat = net(X)  # 使用网络模型对输入数据进行预测
            l = loss(y_hat, y)  # 计算预测结果和真实标签之间的损失
            optimizer.zero_grad()  # 清零优化器的梯度
            l.backward()  # 对损失进行反向传播，计算梯度
            optimizer.step()  # 使用优化器更新网络模型的参数
            train_l_sum += l.cpu().item()  # 累加损失
            train_acc_sum += (y_hat.argmax(dim=1) ==
                              y).sum().cpu().item()  # 累加准确率
            n += y.shape[0]  # 累加样本数量
        print(
            "epoch %d, loss %.4f, train acc %.3f, time %.1f sec"
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time() - start)
        )  # 打印每轮训练的损失、准确率和时间


train(net)  # 调用训练函数，开始训练


def save_model(net, path):
    torch.save(net.state_dict(), path)
    print("saved:", path)


save_model(net, "./alexnet.pth")
